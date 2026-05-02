#!/usr/bin/env python3
"""Speculative decoding benchmark for Aether-2B.

Measures greedy target-only decode versus greedy speculative decode using a
separate draft model. The benchmark is designed to exercise the real draft-path
integration, so the draft model must share the target tokenizer vocab size.

Usage
-----
    python scripts/benchmark_speculative.py \
        --source-tokens 65536 131072 262144 \
        --draft-model tiny \
        --target-model 2b \
        --yarn-scaling \
        --output-json /tmp/spec_bench.json
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
from aether_pipeline.serving import CudaSparseAttentionBackend, PytorchAttentionBackend


def _bytes_to_mib(value: int) -> float:
    return value / (1024 ** 2)


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


def _device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _backend_from_name(name: str):
    if name == "pytorch":
        return PytorchAttentionBackend()
    if name == "cuda":
        return CudaSparseAttentionBackend()
    raise ValueError(f"Unsupported backend '{name}'")


def _tiny_config(vocab_size: int) -> Aether2BConfig:
    return Aether2BConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        attention_head_dim=16,
        query_compression_dim=32,
        indexer_num_heads=2,
        indexer_head_dim=8,
        csa_compression=2,
        hca_compression=4,
        csa_top_k=3,
        sliding_window=8,
        rope_dim=8,
        output_groups=2,
        group_output_dim=32,
        num_routed_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        hash_routed_layers=1,
        mhc_expansion=2,
        mtp_depth=1,
    )


def _build_config(model_kind: str, vocab_size: Optional[int] = None) -> Aether2BConfig:
    if model_kind == "2b":
        return Aether2BConfig()
    if model_kind == "tiny":
        return _tiny_config(vocab_size or Aether2BConfig().vocab_size)
    raise ValueError(f"Unsupported model kind '{model_kind}'")


def _apply_yarn_scaling(config: Aether2BConfig, scale_factor: float) -> None:
    config.rope_scaling_type = "yarn"
    config.rope_scaling_factor = scale_factor


def _make_prompt_tokens(length: int, vocab_size: int, seed: int) -> List[int]:
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(3, vocab_size, (length,), generator=rng).tolist()


def _greedy_decode_from_state(
    engine: Aether2BServingEngine,
    state,
    max_new_tokens: int,
) -> Tuple[List[int], float, object]:
    if state.last_logits is None:
        raise RuntimeError("Prefill state must include logits")

    output_ids: List[int] = []
    logits = state.last_logits
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        next_token = int(logits.argmax(dim=-1).item())
        output_ids.append(next_token)
        logits, state = engine.step_token(next_token, state)
    elapsed = time.perf_counter() - start
    return output_ids, elapsed, state


def _speculative_decode_from_states(
    decoder: SpeculativeDecoder,
    target_state,
    draft_state,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
) -> Tuple[SpecDecodeSummary, float, object, object]:
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")

    output_ids: List[int] = list(prompt_ids)
    last_token = int(prompt_ids[-1])
    summary = SpecDecodeSummary(output_ids=output_ids)
    new_tokens = 0
    start = time.perf_counter()

    while new_tokens < max_new_tokens:
        remaining = max_new_tokens - new_tokens
        draft_steps = min(decoder.draft_steps, remaining)
        original_steps = decoder.draft_steps
        decoder.draft_steps = draft_steps
        if getattr(decoder, "_shared_layer_fusion_depth", 0) > 0:
            result, target_state, draft_state = decoder._spec_round_shared_fused(target_state, draft_state, last_token)
        else:
            result, target_state, draft_state = decoder._spec_round(target_state, draft_state, last_token)
        decoder.draft_steps = original_steps

        new_tokens_in_round = result.accepted_tokens + [result.bonus_token]
        for token in new_tokens_in_round:
            output_ids.append(token)
            new_tokens += 1
            if new_tokens >= max_new_tokens:
                break
        last_token = output_ids[-1]

        summary.total_proposed += result.draft_proposed
        summary.total_accepted += len(result.accepted_tokens)
        summary.total_rounds += 1

    elapsed = time.perf_counter() - start
    summary.output_ids = output_ids
    return summary, elapsed, target_state, draft_state


def _prefill_and_peak(
    engine: Aether2BServingEngine,
    prompt_ids: Sequence[int],
    mhc_chunk_size: int,
) -> Tuple[object, Optional[float], float]:
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    start = time.perf_counter()
    state = engine.chunked_fast_prefill(prompt_ids, mhc_chunk_size=mhc_chunk_size)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
        peak_mib = _bytes_to_mib(torch.cuda.max_memory_allocated(engine.device))
    else:
        peak_mib = None
    elapsed = time.perf_counter() - start
    return state, peak_mib, elapsed


def _measure_decode_rounds(
    engine: Aether2BServingEngine,
    prefilled_state,
    decode_tokens: int,
    rounds: int,
    warmups: int,
) -> Dict[str, object]:
    times: List[float] = []
    generated_lengths: List[int] = []
    for round_idx in range(warmups + rounds):
        state = prefilled_state.clone()
        _, elapsed, final_state = _greedy_decode_from_state(engine, state, decode_tokens)
        if round_idx < warmups:
            continue
        times.append(elapsed)
        generated_lengths.append(final_state.token_count - prefilled_state.token_count)
    return {
        "mean_s": statistics.mean(times) if times else 0.0,
        "std_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "mean_tok_s": (decode_tokens / statistics.mean(times)) if times and statistics.mean(times) > 0 else 0.0,
        "generated_tokens": generated_lengths[0] if generated_lengths else 0,
        "rounds": len(times),
    }


def _measure_speculative_rounds(
    decoder: SpeculativeDecoder,
    target_prefill_state,
    draft_prefill_state,
    prompt_ids: Sequence[int],
    decode_tokens: int,
    target_greedy_output: Sequence[int],
    rounds: int,
    warmups: int,
) -> Dict[str, object]:
    times: List[float] = []
    acceptance_rates: List[float] = []
    proposed: List[int] = []
    accepted: List[int] = []
    outputs_match_target: Optional[bool] = None

    for round_idx in range(warmups + rounds):
        target_state = target_prefill_state.clone()
        draft_state = draft_prefill_state.clone()
        summary, elapsed, final_target_state, final_draft_state = _speculative_decode_from_states(
            decoder,
            target_state,
            draft_state,
            prompt_ids,
            decode_tokens,
        )
        if round_idx < warmups:
            continue
        times.append(elapsed)
        acceptance_rates.append(summary.mean_acceptance_rate)
        proposed.append(summary.total_proposed)
        accepted.append(summary.total_accepted)
        outputs_match_target = outputs_match_target if outputs_match_target is not None else True
        outputs_match_target = outputs_match_target and (summary.output_ids[len(prompt_ids):] == list(target_greedy_output))
        del final_target_state, final_draft_state

    mean_time = statistics.mean(times) if times else 0.0
    return {
        "mean_s": mean_time,
        "std_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "mean_tok_s": (decode_tokens / mean_time) if mean_time > 0 else 0.0,
        "mean_acceptance_rate": statistics.mean(acceptance_rates) if acceptance_rates else 0.0,
        "total_proposed_mean": statistics.mean(proposed) if proposed else 0.0,
        "total_accepted_mean": statistics.mean(accepted) if accepted else 0.0,
        "rounds": len(times),
        "matches_target_greedy": bool(outputs_match_target),
    }


def _estimate_combined_peak_mib(
    target_engine: Aether2BServingEngine,
    draft_engine: Aether2BServingEngine,
    prompt_ids: Sequence[int],
    mhc_chunk_size: int,
) -> Dict[str, float]:
    target_state, target_peak_mib, _ = _prefill_and_peak(target_engine, prompt_ids, mhc_chunk_size)
    del target_state
    gc.collect()
    if target_engine.device.type == "cuda":
        torch.cuda.empty_cache()

    draft_state, draft_peak_mib, _ = _prefill_and_peak(draft_engine, prompt_ids, mhc_chunk_size)
    del draft_state
    gc.collect()
    if draft_engine.device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "target_peak_mib": float(target_peak_mib or 0.0),
        "draft_peak_mib": float(draft_peak_mib or 0.0),
        "combined_peak_mib": float((target_peak_mib or 0.0) + (draft_peak_mib or 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding speedup")
    parser.add_argument("--target-model", choices=["2b", "tiny"], default="2b")
    parser.add_argument("--draft-model", choices=["tiny", "self"], default="tiny")
    parser.add_argument(
        "--self-spec-layers",
        type=int,
        default=None,
        help="If set, build a shared-layer self-spec draft from the first N target layers",
    )
    parser.add_argument("--source-tokens", type=int, nargs="+", default=[65536, 131072, 262144])
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--draft-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--benchmark-rounds", type=int, default=10)
    parser.add_argument("--mhc-chunk-size", type=int, default=4096)
    parser.add_argument("--backend", choices=["pytorch", "cuda"], default="pytorch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default=None)
    parser.add_argument("--yarn-scaling", action="store_true", help="Enable YaRN RoPE scaling for target and draft models")
    parser.add_argument("--adaptive-k", action="store_true", help="Enable adaptive draft-step control during generation rounds")
    parser.add_argument("--min-draft-steps", type=int, default=1)
    parser.add_argument("--max-draft-steps", type=int, default=None)
    parser.add_argument("--adapt-up-threshold", type=float, default=0.90)
    parser.add_argument("--adapt-down-threshold", type=float, default=0.60)
    parser.add_argument("--gpu-budget-gib", type=float, default=48.0)
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of CUDA GPUs for model layer sharding")
    parser.add_argument("--draft-quant-bits", type=int, default=None, choices=[4, 8],
                        help="TurboQuant KV-cache quantisation for the draft engine (4 or 8 bit). "
                             "Reduces draft cache HBM bandwidth; recommended at long context.")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    device = _device_from_name(args.device)
    if args.backend == "cuda" and device.type != "cuda":
        raise ValueError("CUDA backend requires a CUDA device")
    backend = _backend_from_name(args.backend)
    dtype = _dtype_from_name(args.dtype) if args.dtype is not None else (torch.bfloat16 if device.type == "cuda" else torch.float32)

    target_cfg = _build_config(args.target_model)
    draft_cfg = _build_config(args.draft_model if args.draft_model != "self" else args.target_model, vocab_size=target_cfg.vocab_size)

    if args.self_spec_layers is not None and args.draft_model == "self":
        raise ValueError("--self-spec-layers cannot be combined with --draft-model self")

    max_requested_tokens = max(args.source_tokens)
    rope_scaling_factor = max(1.0, max_requested_tokens / target_cfg.max_position_embeddings)
    if args.yarn_scaling:
        _apply_yarn_scaling(target_cfg, rope_scaling_factor)
        _apply_yarn_scaling(draft_cfg, rope_scaling_factor)

    torch.manual_seed(0)
    target_model = Aether2BForCausalLM(target_cfg).eval().to(device=device, dtype=dtype)
    if args.draft_model == "self":
        draft_model = target_model
    elif args.self_spec_layers is not None:
        draft_model = build_self_spec_draft_model(target_model, args.self_spec_layers).to(device=device, dtype=dtype)
    else:
        draft_model = Aether2BForCausalLM(draft_cfg).eval().to(device=device, dtype=dtype)

    target_engine = Aether2BServingEngine(target_model, backend=backend, device=device.type)
    draft_engine = target_engine if args.draft_model == "self" else Aether2BServingEngine(
        draft_model, backend=backend, device=device.type,
        turbo_quant_bits=args.draft_quant_bits,
    )

    if device.type == "cuda" and args.num_gpus > 1:
        target_engine.shard_across_gpus(args.num_gpus)
        # For shared-layer self-spec, draft model reuses target layer module
        # objects. Re-sharding draft independently would move shared modules to
        # conflicting devices and break target prefill.
        if draft_engine is not target_engine and args.self_spec_layers is None:
            draft_engine.shard_across_gpus(args.num_gpus)
        elif draft_engine is not target_engine and args.self_spec_layers is not None:
            target_layer_devices = getattr(target_engine.model.model, "_layer_devices", None)
            if target_layer_devices is not None:
                draft_engine.model.model._layer_devices = list(target_layer_devices[: len(draft_engine.model.model.layers)])

    decoder = SpeculativeDecoder(
        target_engine,
        draft_engine,
        draft_steps=args.draft_steps,
        temperature=args.temperature,
        adaptive_draft_steps=args.adaptive_k,
        min_draft_steps=args.min_draft_steps,
        max_draft_steps=args.max_draft_steps,
        adapt_up_threshold=args.adapt_up_threshold,
        adapt_down_threshold=args.adapt_down_threshold,
    )

    budget_cap_context = 2 * target_cfg.max_position_embeddings
    effective_source_tokens = list(args.source_tokens)
    budget_report: Dict[str, float] = {"target_peak_mib": 0.0, "draft_peak_mib": 0.0, "combined_peak_mib": 0.0}
    budget_capped = False

    if device.type == "cuda":
        probe_tokens = _make_prompt_tokens(max_requested_tokens, target_cfg.vocab_size, seed=42)
        budget_report = _estimate_combined_peak_mib(target_engine, draft_engine, probe_tokens, args.mhc_chunk_size)
        if budget_report["combined_peak_mib"] > args.gpu_budget_gib * 1024.0 and max_requested_tokens > budget_cap_context:
            effective_source_tokens = [min(length, budget_cap_context) for length in effective_source_tokens]
            budget_capped = True

    results = []
    for requested_tokens, source_tokens in zip(args.source_tokens, effective_source_tokens):
        prompt_ids = _make_prompt_tokens(source_tokens, target_cfg.vocab_size, seed=1234 + source_tokens)

        target_prefill_state, target_prefill_peak_mib, target_prefill_time_s = _prefill_and_peak(
            target_engine, prompt_ids, args.mhc_chunk_size
        )
        target_reference_output, _, _ = _greedy_decode_from_state(
            target_engine,
            target_prefill_state.clone(),
            args.decode_tokens,
        )
        if args.draft_model == "self":
            draft_prefill_state = target_prefill_state
            draft_prefill_peak_mib = target_prefill_peak_mib
            draft_prefill_time_s = target_prefill_time_s
        else:
            draft_prefill_state, draft_prefill_peak_mib, draft_prefill_time_s = _prefill_and_peak(
                draft_engine, prompt_ids, args.mhc_chunk_size
            )

        baseline = _measure_decode_rounds(
            target_engine,
            target_prefill_state,
            args.decode_tokens,
            args.benchmark_rounds,
            args.warmup_rounds,
        )
        speculative = _measure_speculative_rounds(
            decoder,
            target_prefill_state,
            draft_prefill_state,
            prompt_ids,
            args.decode_tokens,
            target_reference_output,
            args.benchmark_rounds,
            args.warmup_rounds,
        )

        results.append(
            {
                "requested_source_tokens": requested_tokens,
                "effective_source_tokens": source_tokens,
                "budget_capped": budget_capped and requested_tokens != source_tokens,
                "prefill": {
                    "target_time_s": target_prefill_time_s,
                    "draft_time_s": draft_prefill_time_s,
                    "target_peak_mib": target_prefill_peak_mib,
                    "draft_peak_mib": draft_prefill_peak_mib,
                },
                "baseline_decode": baseline,
                "speculative_decode": speculative,
            }
        )

        del target_prefill_state
        if args.draft_model != "self":
            del draft_prefill_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "self_spec_layers": args.self_spec_layers,
        "adaptive_k": args.adaptive_k,
        "min_draft_steps": args.min_draft_steps,
        "max_draft_steps": args.max_draft_steps,
        "adapt_up_threshold": args.adapt_up_threshold,
        "adapt_down_threshold": args.adapt_down_threshold,
        "draft_steps": args.draft_steps,
        "temperature": args.temperature,
        "rope_scaling": "yarn" if args.yarn_scaling else "none",
        "rope_scaling_factor": rope_scaling_factor if args.yarn_scaling else 1.0,
        "budget_gib": args.gpu_budget_gib,
        "num_gpus": args.num_gpus,
        "budget_cap_context": budget_cap_context,
        "budget_estimate_mib": budget_report,
        "budget_capped": budget_capped,
        "source_tokens": args.source_tokens,
        "effective_source_tokens": effective_source_tokens,
        "results": results,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
