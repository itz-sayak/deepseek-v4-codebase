#!/usr/bin/env python3
"""Speculative decode grid sweep: depth × temperature.

Sweeps:
  - Temperature: [0.2, 0.4, 0.6, 0.8]  (all with adaptive K)
  - Self-spec depth: [12, 16, 20, 24]   (layers shared with target)

Design for efficiency
---------------------
Target is built and prefilled exactly ONCE.  For each depth the draft model is
built (sharing target layer objects) and prefilled ONCE; all four temperature
variants are then measured by cloning the already-computed states, so no extra
prefill cost per temperature point.

Outputs
-------
artifacts/sweep_speculative_grid.json   – every cell with full stats
artifacts/sweep_speculative_ranked.json – cells sorted by speculative tok/s descending
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from aether_2b import (
    Aether2BConfig,
    Aether2BForCausalLM,
    build_self_spec_draft_model,
)
from aether_2b.serving import Aether2BServingEngine
from aether_2b.speculative import SpeculativeDecoder, SpecDecodeSummary
from aether_pipeline.serving import PytorchAttentionBackend


# ---------------------------------------------------------------------------
# Helpers (inlined from benchmark_speculative to avoid arg coupling)
# ---------------------------------------------------------------------------

def _bytes_to_mib(value: int) -> float:
    return value / (1024 ** 2)


def _make_prompt_tokens(length: int, vocab_size: int, seed: int) -> List[int]:
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(3, vocab_size, (length,), generator=rng).tolist()


def _apply_yarn(cfg: Aether2BConfig, factor: float) -> None:
    cfg.rope_scaling_type = "yarn"
    cfg.rope_scaling_factor = factor


def _prefill(engine: Aether2BServingEngine, prompt_ids: List[int], mhc_chunk: int):
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    t0 = time.perf_counter()
    state = engine.chunked_fast_prefill(prompt_ids, mhc_chunk_size=mhc_chunk)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
        peak_mib = _bytes_to_mib(torch.cuda.max_memory_allocated(engine.device))
    else:
        peak_mib = None
    return state, peak_mib, time.perf_counter() - t0


def _spec_decode(
    decoder: SpeculativeDecoder,
    target_state,
    draft_state,
    prompt_ids: List[int],
    n_tokens: int,
) -> Tuple[SpecDecodeSummary, float]:
    output_ids = list(prompt_ids)
    last_token = int(prompt_ids[-1])
    summary = SpecDecodeSummary(output_ids=output_ids)
    new_tokens = 0
    t0 = time.perf_counter()
    while new_tokens < n_tokens:
        remaining = n_tokens - new_tokens
        orig_k = decoder.draft_steps
        decoder.draft_steps = min(orig_k, remaining)
        if decoder._shared_layer_fusion_depth > 0:
            result, target_state, draft_state = decoder._spec_round_shared_fused(
                target_state, draft_state, last_token
            )
        else:
            result, target_state, draft_state = decoder._spec_round(
                target_state, draft_state, last_token
            )
        decoder.draft_steps = orig_k
        for tok in result.accepted_tokens + [result.bonus_token]:
            output_ids.append(tok)
            new_tokens += 1
            if new_tokens >= n_tokens:
                break
        last_token = output_ids[-1]
        summary.total_proposed += result.draft_proposed
        summary.total_accepted += len(result.accepted_tokens)
        summary.total_rounds += 1
    summary.output_ids = output_ids
    return summary, time.perf_counter() - t0


def _baseline_decode(
    engine: Aether2BServingEngine, prefill_state, n_tokens: int, rounds: int, warmups: int
) -> Dict:
    times = []
    for i in range(warmups + rounds):
        state = prefill_state.clone()
        logits = state.last_logits
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            next_tok = int(logits.argmax(-1).item())
            logits, state = engine.step_token(next_tok, state)
        if i >= warmups:
            times.append(time.perf_counter() - t0)
    m = statistics.mean(times) if times else 0.0
    return {
        "mean_s": m,
        "std_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "mean_tok_s": n_tokens / m if m > 0 else 0.0,
        "rounds": len(times),
    }


def _spec_rounds(
    decoder: SpeculativeDecoder,
    target_prefill_state,
    draft_prefill_state,
    prompt_ids: List[int],
    n_tokens: int,
    rounds: int,
    warmups: int,
) -> Dict:
    times, acceptance_rates, proposed_list, accepted_list = [], [], [], []
    for i in range(warmups + rounds):
        ts = target_prefill_state.clone()
        ds = draft_prefill_state.clone()
        summary, elapsed = _spec_decode(decoder, ts, ds, prompt_ids, n_tokens)
        if i < warmups:
            continue
        times.append(elapsed)
        acceptance_rates.append(summary.mean_acceptance_rate)
        proposed_list.append(summary.total_proposed)
        accepted_list.append(summary.total_accepted)
    m = statistics.mean(times) if times else 0.0
    return {
        "mean_s": m,
        "std_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "mean_tok_s": n_tokens / m if m > 0 else 0.0,
        "mean_acceptance_rate": statistics.mean(acceptance_rates) if acceptance_rates else 0.0,
        "total_proposed_mean": statistics.mean(proposed_list) if proposed_list else 0.0,
        "total_accepted_mean": statistics.mean(accepted_list) if accepted_list else 0.0,
        "rounds": len(times),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid sweep: self-spec depth × temperature")
    parser.add_argument("--source-tokens", type=int, default=262144)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--depths", type=int, nargs="+", default=[12, 16, 20, 24])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--draft-steps", type=int, default=6)
    parser.add_argument("--min-draft-steps", type=int, default=1)
    parser.add_argument("--max-draft-steps", type=int, default=8)
    parser.add_argument("--adapt-up-threshold", type=float, default=0.90)
    parser.add_argument("--adapt-down-threshold", type=float, default=0.60)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--mhc-chunk-size", type=int, default=4096)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--yarn-scaling", action="store_true")
    parser.add_argument("--draft-quant-bits", type=int, default=None, choices=[4, 8],
                        help="Apply TurboQuant KV-cache quantisation to the draft engine "
                             "(4 = 4-bit, 8 = 8-bit, omit = bf16). "
                             "Reduces draft KV-cache HBM ~4× at 4-bit, making self-spec "
                             "viable at long context.")
    parser.add_argument("--output-grid-json", default="artifacts/sweep_speculative_grid.json")
    parser.add_argument("--output-ranked-json", default="artifacts/sweep_speculative_ranked.json")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    backend = PytorchAttentionBackend()

    # ---- build & shard target model once ----
    target_cfg = Aether2BConfig()
    if args.yarn_scaling:
        factor = max(1.0, args.source_tokens / target_cfg.max_position_embeddings)
        _apply_yarn(target_cfg, factor)

    print(f"[sweep] Building target model …", flush=True)
    torch.manual_seed(0)
    target_model = Aether2BForCausalLM(target_cfg).eval().to(device=device, dtype=dtype)
    target_engine = Aether2BServingEngine(target_model, backend=backend, device=device.type)
    if device.type == "cuda" and args.num_gpus > 1:
        target_engine.shard_across_gpus(args.num_gpus)
        print(f"[sweep] Target sharded across {args.num_gpus} GPUs", flush=True)

    # ---- prefill target once ----
    prompt_ids = _make_prompt_tokens(args.source_tokens, target_cfg.vocab_size, seed=1234 + args.source_tokens)
    print(f"[sweep] Prefilling target at {args.source_tokens} tokens …", flush=True)
    target_state, target_peak_mib, target_prefill_time = _prefill(
        target_engine, prompt_ids, args.mhc_chunk_size
    )
    print(f"[sweep] Target prefill done in {target_prefill_time:.1f}s, peak {target_peak_mib:.0f} MiB", flush=True)

    # ---- baseline decode (once) ----
    print(f"[sweep] Measuring baseline decode …", flush=True)
    baseline = _baseline_decode(
        target_engine, target_state, args.decode_tokens,
        args.benchmark_rounds, args.warmup_rounds
    )
    print(f"[sweep] Baseline: {baseline['mean_tok_s']:.2f} tok/s", flush=True)

    grid: List[Dict] = []

    for depth in args.depths:
        if depth <= 0 or depth >= target_cfg.num_hidden_layers:
            print(f"[sweep] Skipping depth={depth} (out of range [1, {target_cfg.num_hidden_layers - 1}])", flush=True)
            continue

        # ---- build draft at this depth ----
        print(f"\n[sweep] Depth={depth}: building draft model …", flush=True)
        draft_cfg = Aether2BConfig()
        if args.yarn_scaling:
            _apply_yarn(draft_cfg, factor)

        # NOTE: build_self_spec_draft_model shares ALL modules (layers, embed,
        # norm, lm_head) with the target.  Do NOT call .to() on the draft — it
        # would move shared layers back to a single device and corrupt the
        # target model's multi-GPU sharding.  Shared modules are already on the
        # correct devices inherited from the sharded target.
        #
        # Set _layer_devices on the draft model BEFORE constructing the serving
        # engine.  The engine skips model.to() when _layer_devices is present,
        # preserving multi-GPU placement.
        draft_model = build_self_spec_draft_model(target_model, depth)
        if device.type == "cuda" and args.num_gpus > 1:
            target_layer_devices = getattr(target_engine.model.model, "_layer_devices", None)
            if target_layer_devices is not None:
                draft_model.model._layer_devices = list(target_layer_devices[:depth])
        draft_engine = Aether2BServingEngine(
            draft_model, backend=backend, device=device.type,
            turbo_quant_bits=args.draft_quant_bits,
        )
        if args.draft_quant_bits is not None:
            print(f"[sweep] Depth={depth}: draft KV-cache quantised to {args.draft_quant_bits}-bit", flush=True)

        # ---- prefill draft once at this depth ----
        print(f"[sweep] Depth={depth}: prefilling draft …", flush=True)
        draft_state, draft_peak_mib, draft_prefill_time = _prefill(
            draft_engine, prompt_ids, args.mhc_chunk_size
        )
        print(f"[sweep] Depth={depth}: draft prefill done in {draft_prefill_time:.1f}s, peak {draft_peak_mib:.0f} MiB", flush=True)

        # ---- sweep temperatures ----
        for temperature in args.temperatures:
            print(f"[sweep] depth={depth}, temp={temperature} …", flush=True)
            decoder = SpeculativeDecoder(
                target_engine,
                draft_engine,
                draft_steps=args.draft_steps,
                temperature=temperature,
                adaptive_draft_steps=True,
                min_draft_steps=args.min_draft_steps,
                max_draft_steps=args.max_draft_steps,
                adapt_up_threshold=args.adapt_up_threshold,
                adapt_down_threshold=args.adapt_down_threshold,
            )
            result = _spec_rounds(
                decoder,
                target_state,
                draft_state,
                prompt_ids,
                args.decode_tokens,
                args.benchmark_rounds,
                args.warmup_rounds,
            )
            speedup = result["mean_tok_s"] / baseline["mean_tok_s"] if baseline["mean_tok_s"] > 0 else 0.0
            record = {
                "depth": depth,
                "temperature": temperature,
                "draft_quant_bits": args.draft_quant_bits,
                "baseline_tok_s": round(baseline["mean_tok_s"], 3),
                "spec_tok_s": round(result["mean_tok_s"], 3),
                "speedup": round(speedup, 3),
                "acceptance_rate": round(result["mean_acceptance_rate"], 4),
                "total_proposed_mean": result["total_proposed_mean"],
                "total_accepted_mean": result["total_accepted_mean"],
                "spec_mean_s": round(result["mean_s"], 3),
                "spec_std_s": round(result["std_s"], 3),
                "prefill_target_s": round(target_prefill_time, 2),
                "prefill_draft_s": round(draft_prefill_time, 2),
                "target_peak_mib": round(target_peak_mib or 0, 1),
                "draft_peak_mib": round(draft_peak_mib or 0, 1),
            }
            grid.append(record)
            print(
                f"  → spec {result['mean_tok_s']:.2f} tok/s  α={result['mean_acceptance_rate']:.3f}  speedup={speedup:.3f}×",
                flush=True,
            )

        # Free draft resources before building the next depth
        del draft_engine, draft_model, draft_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- ranked output ----
    ranked = sorted(grid, key=lambda x: x["spec_tok_s"], reverse=True)
    best = ranked[0] if ranked else {}

    payload = {
        "sweep_config": {
            "source_tokens": args.source_tokens,
            "decode_tokens": args.decode_tokens,
            "depths_swept": args.depths,
            "temperatures_swept": args.temperatures,
            "draft_steps": args.draft_steps,
            "adaptive_k": True,
            "min_draft_steps": args.min_draft_steps,
            "max_draft_steps": args.max_draft_steps,
            "warmup_rounds": args.warmup_rounds,
            "benchmark_rounds": args.benchmark_rounds,
            "num_gpus": args.num_gpus,
            "dtype": args.dtype,
            "yarn_scaling": args.yarn_scaling,
        },
        "baseline_tok_s": round(baseline["mean_tok_s"], 3),
        "best_config": best,
        "grid": grid,
    }

    ranked_payload = {
        "sweep_config": payload["sweep_config"],
        "baseline_tok_s": payload["baseline_tok_s"],
        "best_config": best,
        "ranked": ranked,
    }

    Path(args.output_grid_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.output_ranked_json).write_text(json.dumps(ranked_payload, indent=2), encoding="utf-8")
    print(json.dumps(ranked_payload, indent=2))


if __name__ == "__main__":
    main()
