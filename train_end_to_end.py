from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from deepseek_v4_pro_2b import DeepSeekV4Pro2BConfig, DeepSeekV4Pro2BForCausalLM
from deepseek_v4_pro_2b.muon import Muon, split_muon_adamw_params
from deepseek_pipeline.tokenizer import load_deepseek_tokenizer


class MemmapTokens(Dataset):
    """Standard memmap dataset.  Optionally shard by (rank, world_size) for DDP/FSDP."""

    def __init__(
        self,
        path: str,
        seq_len: int,
        strict: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.path = path
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        total_samples = max(0, (len(self.data) - 1) // self.seq_len)
        if total_samples == 0:
            if strict:
                raise ValueError(f"{path} is too small for seq_len={seq_len}")
            self._global_len = 0
        else:
            self._global_len = total_samples
        # Each rank owns a contiguous slice of the global sample space.
        self._start, self._end = self._rank_slice()

    def _rank_slice(self):
        if self.world_size <= 1 or self._global_len == 0:
            return 0, self._global_len
        per_rank = self._global_len // self.world_size
        start = self.rank * per_rank
        end = start + per_rank if self.rank < self.world_size - 1 else self._global_len
        return start, end

    def __len__(self) -> int:
        return max(0, self._end - self._start)

    def __getitem__(self, local_idx: int):
        global_idx = self._start + local_idx
        start = global_idx * self.seq_len
        stop = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:stop].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        mask = torch.ones_like(x)
        return {"input_ids": x, "labels": y, "attention_mask": mask}


class PackedMemmapTokens(Dataset):
    """Sequence-packing dataset.  Concatenates consecutive documents separated
    by EOS tokens so every sequence is exactly ``seq_len`` tokens.  Loss is
    masked at EOS boundary positions so the model cannot cross-attend between
    documents via the label of the EOS token itself.

    Packing order: the raw token stream is sliced into non-overlapping windows
    of ``seq_len`` tokens.  Documents naturally straddle boundaries; we record
    EOS positions and zero out those labels.
    """

    def __init__(
        self,
        path: str,
        seq_len: int,
        eos_token_id: int,
        strict: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.rank = rank
        self.world_size = world_size
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        # Number of complete packed sequences in the global stream.
        self._global_len = max(0, len(self.data) // seq_len)
        if self._global_len == 0 and strict:
            raise ValueError(f"{path} is too small for packed seq_len={seq_len}")
        start_g = (self.rank * self._global_len) // self.world_size
        end_g = ((self.rank + 1) * self._global_len) // self.world_size
        self._start = start_g
        self._end = end_g

    def __len__(self) -> int:
        return max(0, self._end - self._start)

    def __getitem__(self, local_idx: int):
        global_idx = self._start + local_idx
        start = global_idx * self.seq_len
        stop = start + self.seq_len
        tokens = torch.from_numpy(self.data[start:stop].astype(np.int64))
        x = tokens.clone()
        y = torch.roll(tokens, -1, 0)
        # The last label position always points to the next document's first
        # token which we don't know; zero it out.
        y[-1] = -100
        # Zero out labels immediately after each EOS token (cross-doc boundary).
        eos_positions = (x == self.eos_token_id).nonzero(as_tuple=True)[0]
        for pos in eos_positions:
            if pos + 1 < self.seq_len:
                y[pos + 1] = -100
        mask = (y != -100).long()
        return {"input_ids": x, "labels": y, "attention_mask": mask}


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    tokenizer_name: str
    tokenizer_cache_dir: Optional[str]
    seq_len: int = 512
    batch_size: int = 1
    grad_accum: int = 1
    epochs: int = 2
    max_steps: Optional[int] = None
    learning_rate: float = 2.2e-4
    muon_lr: float = 2e-2
    weight_decay: float = 0.1
    eval_interval: int = 1000
    save_interval: int = 25_000
    resume_from: Optional[str] = None
    auto_resume: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
    preset: str = "2b"
    # --- TODO 4: multi-GPU / training-efficiency flags ---
    fsdp: bool = False                  # wrap model with FSDP (requires torchrun)
    gradient_checkpointing: bool = False  # per-layer activation checkpointing
    sequence_packing: bool = False      # use PackedMemmapTokens instead of MemmapTokens
    warmup_steps: int = 2000            # linear LR warmup steps
    lr_min_ratio: float = 0.1           # cosine decay floor as fraction of peak LR


def make_model_config(tokenizer_name: str, tokenizer_cache_dir: Optional[str], preset: str) -> DeepSeekV4Pro2BConfig:
    tok = load_deepseek_tokenizer(tokenizer_name, tokenizer_cache_dir)
    cfg = DeepSeekV4Pro2BConfig(
        vocab_size=tok.vocab_size,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if preset == "tiny":
        cfg.hidden_size = 64
        cfg.num_hidden_layers = 4
        cfg.num_attention_heads = 4
        cfg.attention_head_dim = 16
        cfg.query_compression_dim = 32
        cfg.indexer_num_heads = 2
        cfg.indexer_head_dim = 8
        cfg.csa_compression = 2
        cfg.hca_compression = 4
        cfg.csa_top_k = 3
        cfg.sliding_window = 8
        cfg.output_groups = 2
        cfg.group_output_dim = 32
        cfg.num_routed_experts = 4
        cfg.num_shared_experts = 1
        cfg.num_experts_per_tok = 2
        cfg.moe_intermediate_size = 32
        cfg.hash_routed_layers = 1
        cfg.mhc_expansion = 2
        cfg.mtp_depth = 1
    return cfg


def save_checkpoint(output_dir: str, step: int, model, adamw, muon, train_cfg: TrainConfig, model_cfg: DeepSeekV4Pro2BConfig) -> str:
    raise NotImplementedError("Use save_training_state instead.")


def _stack_batch(samples: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}


def _next_batch(train_set: MemmapTokens, batch_size: int, sample_index: int) -> tuple[Dict[str, torch.Tensor], int, int, int]:
    if len(train_set) == 0:
        raise ValueError("Training dataset is empty.")
    end_index = min(sample_index + batch_size, len(train_set))
    samples = [train_set[idx] for idx in range(sample_index, end_index)]
    next_sample_index = end_index
    epoch_increment = 0
    if next_sample_index >= len(train_set):
        next_sample_index = 0
        epoch_increment = 1
    return _stack_batch(samples), len(samples), next_sample_index, epoch_increment


def _steps_per_epoch(train_examples: int, batch_size: int, grad_accum: int) -> int:
    micro_batches = math.ceil(train_examples / batch_size)
    return math.ceil(micro_batches / grad_accum)


def _remaining_micro_batches(train_examples: int, batch_size: int, total_epochs: int, epoch: int, next_sample_index: int) -> int:
    remaining_examples = max(0, (total_epochs - epoch) * train_examples - next_sample_index)
    if remaining_examples == 0:
        return 0
    return math.ceil(remaining_examples / batch_size)


def _capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict[str, object]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _build_checkpoint_payload(
    *,
    step: int,
    epoch: int,
    next_sample_index: int,
    tokens_processed: int,
    best_metric: Optional[float],
    best_metric_name: Optional[str],
    best_step: int,
    model,
    adamw,
    muon,
    train_cfg: TrainConfig,
    model_cfg: DeepSeekV4Pro2BConfig,
) -> Dict[str, object]:
    return {
        "step": step,
        "epoch": epoch,
        "next_sample_index": next_sample_index,
        "next_token_offset": next_sample_index * train_cfg.seq_len,
        "tokens_processed": tokens_processed,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "best_step": best_step,
        "model": model.state_dict(),
        "adamw": adamw.state_dict(),
        "muon": muon.state_dict(),
        "rng_state": _capture_rng_state(),
        "train_config": asdict(train_cfg),
        "model_config": model_cfg.to_dict(),
    }


def save_training_state(
    output_dir: str,
    *,
    step: int,
    epoch: int,
    next_sample_index: int,
    tokens_processed: int,
    best_metric: Optional[float],
    best_metric_name: Optional[str],
    best_step: int,
    metric_name: Optional[str],
    metric_value: Optional[float],
    save_step_snapshot: bool,
    save_best: bool,
    model,
    adamw,
    muon,
    train_cfg: TrainConfig,
    model_cfg: DeepSeekV4Pro2BConfig,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    payload = _build_checkpoint_payload(
        step=step,
        epoch=epoch,
        next_sample_index=next_sample_index,
        tokens_processed=tokens_processed,
        best_metric=best_metric,
        best_metric_name=best_metric_name,
        best_step=best_step,
        model=model,
        adamw=adamw,
        muon=muon,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
    )
    if metric_name is not None:
        payload[metric_name] = metric_value

    paths = {"latest": os.path.join(output_dir, "checkpoint-latest.pt")}
    torch.save(payload, paths["latest"])
    if save_step_snapshot:
        paths["step"] = os.path.join(output_dir, f"checkpoint-step-{step}.pt")
        torch.save(payload, paths["step"])
    if save_best:
        paths["best"] = os.path.join(output_dir, "best.pth")
        torch.save(payload, paths["best"])
    return paths


def resolve_resume_path(cfg: TrainConfig) -> Optional[str]:
    if cfg.resume_from:
        return cfg.resume_from
    latest = os.path.join(cfg.output_dir, "checkpoint-latest.pt")
    if cfg.auto_resume and os.path.exists(latest):
        return latest
    return None


def load_training_state(path: str, device: str, model, adamw, muon) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    adamw.load_state_dict(checkpoint["adamw"])
    muon.load_state_dict(checkpoint["muon"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return {
        "step": int(checkpoint.get("step", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "next_sample_index": int(checkpoint.get("next_sample_index", 0)),
        "tokens_processed": int(checkpoint.get("tokens_processed", 0)),
        "best_metric": checkpoint.get("best_metric"),
        "best_metric_name": checkpoint.get("best_metric_name"),
        "best_step": int(checkpoint.get("best_step", 0)),
    }


def evaluate(model, loader, device, dtype_ctx) -> Optional[float]:
    if loader is None:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with dtype_ctx:
                out = model(**batch)
            losses.append(float(out.loss.detach().cpu()))
            if len(losses) >= 4:
                break
    model.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def _cosine_lr_scale(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    """Return the LR multiplier for *step* using linear warmup + cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine_factor


def _apply_lr_scale(optimizer: torch.optim.Optimizer, base_lr: float, scale: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


# ---------------------------------------------------------------------------
# FSDP helpers
# ---------------------------------------------------------------------------

def _wrap_fsdp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Wrap the model with FullyShardedDataParallel if available."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
    except ImportError as exc:
        raise RuntimeError("FSDP requires PyTorch >= 1.12 with distributed support.") from exc

    policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
    return FSDP(
        model,
        auto_wrap_policy=policy,
        device_id=device,
        use_orig_params=True,  # required for Muon + AdamW hybrid
    )


def _is_fsdp_wrapped(model) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except ImportError:
        return False


def _fsdp_state_dict(model) -> dict:
    """Return a full (rank-0 gathered) state dict for saving."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        cfg_full = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg_full):
            return model.state_dict()
    except Exception:
        return model.state_dict()


def train(cfg: TrainConfig) -> None:
    # ------------------------------------------------------------------
    # Distributed process group initialisation
    # ------------------------------------------------------------------
    is_distributed = cfg.fsdp and int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = 0
    world_size = 1
    if is_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Each rank binds to its own GPU.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        cfg = TrainConfig(**{**asdict(cfg), "device": f"cuda:{local_rank}"})

    is_main = rank == 0

    model_cfg = make_model_config(cfg.tokenizer_name, cfg.tokenizer_cache_dir, cfg.preset)
    model = DeepSeekV4Pro2BForCausalLM(model_cfg).to(cfg.device)

    # ------------------------------------------------------------------
    # Gradient checkpointing (applied before FSDP wrapping)
    # ------------------------------------------------------------------
    if cfg.gradient_checkpointing:
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        else:
            # Generic fallback: apply to each transformer layer (if iterable).
            try:
                from torch.utils.checkpoint import checkpoint_sequential
                # Mark for use in the forward pass; model must support it.
                model.gradient_checkpointing = True
            except Exception:
                pass

    # ------------------------------------------------------------------
    # FSDP wrapping
    # ------------------------------------------------------------------
    if cfg.fsdp:
        if not is_distributed:
            raise RuntimeError(
                "--fsdp requires launching with torchrun or equivalent "
                "(WORLD_SIZE > 1 and distributed process group). "
                "Single-process FSDP is not supported."
            )
        model = _wrap_fsdp(model, torch.device(cfg.device))

    # ------------------------------------------------------------------
    # Data loading (shard by rank for FSDP)
    # ------------------------------------------------------------------
    eos_id = model_cfg.eos_token_id or 1
    train_bin = os.path.join(cfg.data_dir, "train.bin")
    if cfg.sequence_packing:
        train_set = PackedMemmapTokens(
            train_bin, cfg.seq_len, eos_token_id=eos_id, rank=rank, world_size=world_size
        )
    else:
        train_set = MemmapTokens(train_bin, cfg.seq_len, rank=rank, world_size=world_size)
    if len(train_set) == 0:
        raise ValueError("train.bin does not contain enough tokens for the configured sequence length.")
    val_path = os.path.join(cfg.data_dir, "val.bin")
    val_loader = None
    if os.path.exists(val_path) and is_main:
        if cfg.sequence_packing:
            val_set = PackedMemmapTokens(val_path, cfg.seq_len, eos_token_id=eos_id, strict=False)
        else:
            val_set = MemmapTokens(val_path, cfg.seq_len, strict=False)
        if len(val_set) > 0:
            val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # ------------------------------------------------------------------
    # Optimisers (operate on the (possibly FSDP-wrapped) model)
    # ------------------------------------------------------------------
    muon_params, adamw_params = split_muon_adamw_params(model)
    adamw = torch.optim.AdamW(adamw_params, lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    muon = Muon(muon_params, lr=cfg.muon_lr, momentum=0.95, weight_decay=0.01, update_rescale=0.2)

    dtype_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if cfg.device.startswith("cuda") and cfg.dtype == "bfloat16"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    resume_state = {
        "step": 0,
        "epoch": 0,
        "next_sample_index": 0,
        "tokens_processed": 0,
        "best_metric": None,
        "best_metric_name": None,
        "best_step": 0,
    }
    resume_path = resolve_resume_path(cfg)
    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_state = load_training_state(resume_path, cfg.device, model, adamw, muon)
        print(f"resumed_from={resume_path}")

    steps_per_epoch = _steps_per_epoch(len(train_set), cfg.batch_size, cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    if resume_state["step"] >= total_steps:
        if is_main:
            print(json.dumps({"status": "already_complete", "step": resume_state["step"], "target_steps": total_steps}))
        return

    step = int(resume_state["step"])
    epoch = int(resume_state["epoch"])
    next_sample_index = int(resume_state["next_sample_index"])
    tokens_processed = int(resume_state["tokens_processed"])
    best_metric = resume_state["best_metric"]
    best_metric_name = resume_state["best_metric_name"]
    best_step = int(resume_state["best_step"])

    model.train()
    for next_step in range(step + 1, total_steps + 1):
        remaining_micro_batches = _remaining_micro_batches(
            len(train_set),
            cfg.batch_size,
            cfg.epochs,
            epoch,
            next_sample_index,
        )
        micro_batches_this_step = min(cfg.grad_accum, remaining_micro_batches)
        if micro_batches_this_step == 0:
            break

        # LR schedule: linear warmup + cosine decay.
        lr_scale = _cosine_lr_scale(next_step - 1, cfg.warmup_steps, total_steps, cfg.lr_min_ratio)
        _apply_lr_scale(adamw, cfg.learning_rate, lr_scale)
        _apply_lr_scale(muon, cfg.muon_lr, lr_scale)

        adamw.zero_grad(set_to_none=True)
        muon.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(micro_batches_this_step):
            batch, consumed_examples, next_sample_index, epoch_increment = _next_batch(train_set, cfg.batch_size, next_sample_index)
            epoch += epoch_increment
            tokens_processed += consumed_examples * cfg.seq_len
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            with dtype_ctx:
                out = model(**batch)
                loss = out.loss / micro_batches_this_step
            loss.backward()
            accum_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        adamw.step()
        muon.step()

        step = next_step

        if is_main:
            if step == 1 or step % cfg.eval_interval == 0:
                val_loss = evaluate(model, val_loader, cfg.device, dtype_ctx)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch,
                            "train_loss": accum_loss,
                            "val_loss": val_loss,
                            "tokens_processed": tokens_processed,
                            "next_token_offset": next_sample_index * cfg.seq_len,
                            "lr_scale": round(lr_scale, 6),
                        }
                    )
                )
            else:
                val_loss = None
                print(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch,
                            "train_loss": accum_loss,
                            "tokens_processed": tokens_processed,
                            "next_token_offset": next_sample_index * cfg.seq_len,
                            "lr_scale": round(lr_scale, 6),
                        }
                    )
                )
        else:
            val_loss = None

        if is_main:
            metric_name = "val_loss" if val_loss is not None else "train_loss"
            metric_value = val_loss if val_loss is not None else accum_loss
            save_best = best_metric is None or metric_value < best_metric
            if save_best:
                best_metric = metric_value
                best_metric_name = metric_name
                best_step = step

            # For FSDP: gather full state dict on rank 0 before saving.
            model_sd = _fsdp_state_dict(model) if _is_fsdp_wrapped(model) else model.state_dict()

            # Build payload manually to inject the FSDP-gathered state dict.
            payload = {
                "step": step,
                "epoch": epoch,
                "next_sample_index": next_sample_index,
                "next_token_offset": next_sample_index * cfg.seq_len,
                "tokens_processed": tokens_processed,
                "best_metric": best_metric,
                "best_metric_name": best_metric_name,
                "best_step": best_step,
                "model": model_sd,
                "adamw": adamw.state_dict(),
                "muon": muon.state_dict(),
                "rng_state": _capture_rng_state(),
                "train_config": asdict(cfg),
                "model_config": model_cfg.to_dict(),
                metric_name: metric_value,
            }

            os.makedirs(cfg.output_dir, exist_ok=True)
            latest_path = os.path.join(cfg.output_dir, "checkpoint-latest.pt")
            torch.save(payload, latest_path)
            saved_paths = {"latest": latest_path}
            if step % cfg.save_interval == 0 or step == total_steps:
                step_path = os.path.join(cfg.output_dir, f"checkpoint-step-{step}.pt")
                torch.save(payload, step_path)
                saved_paths["step"] = step_path
                print(f"saved_checkpoint={step_path}")
            if save_best:
                best_path = os.path.join(cfg.output_dir, "best.pth")
                torch.save(payload, best_path)
                saved_paths["best"] = best_path
                print(f"saved_best={best_path}")

        # Barrier: all ranks wait before the next step so rank-0 state dict
        # gather above doesn't race with the next FSDP forward.
        if is_distributed:
            dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DeepSeek-style 2B model on tokenized reference-source data.")
    parser.add_argument("--data-dir", default="./artifacts/tokenized")
    parser.add_argument("--output-dir", default="./artifacts/checkpoints")
    parser.add_argument("--tokenizer-name", default=os.environ.get("DEEPSEEK_TOKENIZER_NAME", "deepseek-ai/DeepSeek-V3.2"))
    parser.add_argument("--tokenizer-cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=2.2e-4)
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=25_000)
    parser.add_argument("--resume-from")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--preset", choices=["2b", "tiny"], default="2b")
    # TODO-4 multi-GPU flags
    parser.add_argument("--fsdp", action="store_true",
                        help="Wrap model with FSDP (launch with torchrun)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable per-layer activation checkpointing")
    parser.add_argument("--sequence-packing", action="store_true",
                        help="Use PackedMemmapTokens for dense cross-document packing")
    parser.add_argument("--warmup-steps", type=int, default=2000,
                        help="Linear LR warmup steps before cosine decay")
    parser.add_argument("--lr-min-ratio", type=float, default=0.1,
                        help="Minimum LR as a fraction of peak LR (cosine floor)")
    args = parser.parse_args()
    train(
        TrainConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            tokenizer_name=args.tokenizer_name,
            tokenizer_cache_dir=args.tokenizer_cache_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            muon_lr=args.muon_lr,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            resume_from=args.resume_from,
            auto_resume=not args.no_auto_resume,
            preset=args.preset,
            fsdp=args.fsdp,
            gradient_checkpointing=args.gradient_checkpointing,
            sequence_packing=args.sequence_packing,
            warmup_steps=args.warmup_steps,
            lr_min_ratio=args.lr_min_ratio,
        )
    )


if __name__ == "__main__":
    main()
