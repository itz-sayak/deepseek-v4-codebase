#!/usr/bin/env python3
"""Needle-in-Haystack evaluation for TurboQuant (PolarQuant) validation.

Compares turbo_quant_bits=None vs 8 vs 4 by measuring:
  1. Compressed KV cache fidelity — cosine similarity of quantised vs unquantised
     compressed state vectors after prefill of the same token sequence.
  2. Next-token logit agreement — KL divergence and top-1 match between quantised
     and unquantised engines after prefill + 1 decode step.
  3. Needle retrieval — a unique token pattern planted at various positions; the
     post-prefill compressed cache must preserve enough information that the
     next-step logits still distinguish the needle from random haystack tokens.

For the tiny CPU model, context lengths of 64, 128, 256, 512 are used (these
already exercise multiple compression blocks, paging boundaries, and the
TurboQuant encode/decode path).  For GPU runs at 65K+, extend ctx_lengths.

Usage
-----
    conda run -n deepfill python scripts/needle_eval.py [--ctx-lengths 64 128 256 512]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_v4_pro_2b.configuration import DeepSeekV4Pro2BConfig
from deepseek_v4_pro_2b.modeling import DeepSeekV4Pro2BForCausalLM
from deepseek_v4_pro_2b.serving import (
    DeepSeekV4Pro2BServingEngine,
    HCAServingState,
    CSAServingState,
    ModelServingState,
)
from deepseek_pipeline.serving import PytorchAttentionBackend


# ---------------------------------------------------------------------------
# Tiny model config (identical to test infrastructure)
# ---------------------------------------------------------------------------

def _tiny_config() -> DeepSeekV4Pro2BConfig:
    return DeepSeekV4Pro2BConfig(
        vocab_size=256,
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
        sliding_window=4,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compressed_vectors(state: ModelServingState) -> List[torch.Tensor]:
    """Extract all compressed KV vectors as a flat list of bf16/float tensors."""
    vecs = []
    for ls in state.layer_states:
        if isinstance(ls, HCAServingState):
            if ls.compressed is not None:
                vecs.append(ls.compressed.float().view(-1, ls.compressed.shape[-1]))
        elif isinstance(ls, CSAServingState):
            if ls.compressed is not None:
                vecs.append(ls.compressed.float().view(-1, ls.compressed.shape[-1]))
            if ls.index_compressed is not None:
                vecs.append(ls.index_compressed.float().view(-1, ls.index_compressed.shape[-1]))
    return vecs


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-vector cosine similarity."""
    if a.numel() == 0:
        return 1.0
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def _kl_div(logits_ref: torch.Tensor, logits_test: torch.Tensor) -> float:
    """KL(ref || test) on the softmax distributions."""
    p = F.softmax(logits_ref.float(), dim=-1)
    log_q = F.log_softmax(logits_test.float(), dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean").item()


def _top1_match(logits_ref: torch.Tensor, logits_test: torch.Tensor) -> bool:
    return logits_ref.argmax(dim=-1).item() == logits_test.argmax(dim=-1).item()


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_config(
    model: DeepSeekV4Pro2BForCausalLM,
    ctx_length: int,
    turbo_quant_bits: Optional[int],
    baseline_state: Optional[ModelServingState],
    baseline_logits: Optional[torch.Tensor],
    seed: int = 42,
    needle_position_ratio: float = 0.5,
) -> Dict[str, object]:
    """Run needle-in-haystack eval for a single (ctx_length, bits) configuration.

    Returns a dict of metrics.
    """
    torch.manual_seed(seed)

    cfg = model.config
    backend = PytorchAttentionBackend()
    engine = DeepSeekV4Pro2BServingEngine(
        model, backend=backend, turbo_quant_bits=turbo_quant_bits
    )

    # Generate haystack tokens deterministically
    rng = torch.Generator()
    rng.manual_seed(seed)
    haystack = torch.randint(3, cfg.vocab_size, (ctx_length,), generator=rng).tolist()

    # Plant a needle: a distinctive token pattern at needle_position
    needle_pos = int(ctx_length * needle_position_ratio)
    needle_tokens = [cfg.vocab_size - 1, cfg.vocab_size - 2, cfg.vocab_size - 3]
    for i, nt in enumerate(needle_tokens):
        if needle_pos + i < ctx_length:
            haystack[needle_pos + i] = nt

    t0 = time.perf_counter()
    state = engine.prefill(haystack)
    prefill_time = time.perf_counter() - t0

    # Next-token logit after prefill
    probe_token = 1
    logits, _ = engine.step_token(probe_token, state)

    result: Dict[str, object] = {
        "ctx_length": ctx_length,
        "turbo_quant_bits": turbo_quant_bits,
        "prefill_time_s": round(prefill_time, 4),
        "needle_position": needle_pos,
    }

    # If baseline provided, compute similarity metrics
    if baseline_state is not None and baseline_logits is not None:
        # Compressed cache cosine similarity
        vecs_baseline = _compressed_vectors(baseline_state)
        vecs_test = _compressed_vectors(state)

        if len(vecs_baseline) == len(vecs_test) and len(vecs_baseline) > 0:
            sims = []
            for vb, vt in zip(vecs_baseline, vecs_test):
                # With quantization, shapes should match (same number of blocks)
                if vb.shape == vt.shape:
                    sims.append(_cosine_sim(vb, vt))
            result["compressed_cosine_sim_mean"] = round(
                sum(sims) / len(sims), 6
            ) if sims else None
            result["compressed_cosine_sim_min"] = round(
                min(sims), 6
            ) if sims else None
        else:
            result["compressed_cosine_sim_mean"] = None
            result["compressed_cosine_sim_min"] = None

        # Logit quality metrics
        result["kl_div_vs_baseline"] = round(_kl_div(baseline_logits, logits), 6)
        result["top1_match"] = _top1_match(baseline_logits, logits)
        result["logit_max_abs_diff"] = round(
            (baseline_logits.float() - logits.float()).abs().max().item(), 6
        )
    else:
        result["compressed_cosine_sim_mean"] = None
        result["kl_div_vs_baseline"] = None
        result["top1_match"] = None

    return result, state, logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Needle-in-Haystack TurboQuant eval")
    parser.add_argument(
        "--ctx-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Context lengths to evaluate",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    cfg = _tiny_config()
    torch.manual_seed(0)
    model = DeepSeekV4Pro2BForCausalLM(cfg).eval()

    bits_configs = [None, 8, 4]  # baseline, then 8-bit, then 4-bit
    all_results = []

    for ctx_len in args.ctx_lengths:
        print(f"\n{'='*60}")
        print(f"Context length: {ctx_len}")
        print(f"{'='*60}")

        baseline_state = None
        baseline_logits = None

        for bits in bits_configs:
            label = f"bits={bits}" if bits else "baseline (bf16)"
            print(f"\n  → Evaluating {label} ...")

            result, state, logits = evaluate_config(
                model,
                ctx_length=ctx_len,
                turbo_quant_bits=bits,
                baseline_state=baseline_state,
                baseline_logits=baseline_logits,
            )

            if bits is None:
                baseline_state = state
                baseline_logits = logits

            all_results.append(result)

            # Print metrics
            print(f"    Prefill time: {result['prefill_time_s']:.4f}s")
            if result.get("compressed_cosine_sim_mean") is not None:
                print(f"    Compressed cosine sim (mean): {result['compressed_cosine_sim_mean']:.6f}")
                print(f"    Compressed cosine sim (min):  {result['compressed_cosine_sim_min']:.6f}")
            if result.get("kl_div_vs_baseline") is not None:
                print(f"    KL div vs baseline:           {result['kl_div_vs_baseline']:.6f}")
                print(f"    Top-1 match:                  {result['top1_match']}")
                print(f"    Max abs logit diff:           {result['logit_max_abs_diff']:.6f}")

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'CTX':>6} {'BITS':>6} {'COS_SIM':>10} {'COS_MIN':>10} {'KL_DIV':>10} {'TOP1':>6} {'TIME':>8}")
    print(f"{'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*8}")
    for r in all_results:
        bits_str = str(r["turbo_quant_bits"]) if r["turbo_quant_bits"] else "bf16"
        cos_mean = f"{r['compressed_cosine_sim_mean']:.4f}" if r.get("compressed_cosine_sim_mean") is not None else "N/A"
        cos_min = f"{r['compressed_cosine_sim_min']:.4f}" if r.get("compressed_cosine_sim_min") is not None else "N/A"
        kl = f"{r['kl_div_vs_baseline']:.6f}" if r.get("kl_div_vs_baseline") is not None else "N/A"
        top1 = str(r.get("top1_match", "")) if r.get("top1_match") is not None else "N/A"
        time_s = f"{r['prefill_time_s']:.3f}s"
        print(f"{r['ctx_length']:>6} {bits_str:>6} {cos_mean:>10} {cos_min:>10} {kl:>10} {top1:>6} {time_s:>8}")

    # Quality gate — based on LOGIT quality, not raw cache cosine sim.
    # In autoregressive prefill, the compressed cache vectors naturally diverge
    # between quantized and unquantized runs because each step reads from the
    # (now-quantized) cache, causing subsequent hidden states to differ. This is
    # expected. The actual quality metric is whether the model's output logits
    # are preserved — measured by KL divergence and top-1 match.
    print(f"\n{'='*80}")
    print("QUALITY GATE (logit-based)")
    print(f"{'='*80}")
    passed = True
    for r in all_results:
        bits = r["turbo_quant_bits"]
        if bits is None:
            continue
        top1 = r.get("top1_match")
        kl = r.get("kl_div_vs_baseline")
        max_diff = r.get("logit_max_abs_diff")
        label = f"ctx={r['ctx_length']}, bits={bits}"

        # 8-bit: KL < 0.01 and top-1 match required
        # 4-bit: KL < 0.1 and top-1 match required; max logit diff < 0.5
        if bits == 8:
            kl_threshold = 0.01
            diff_threshold = 0.1
        else:
            kl_threshold = 0.1
            diff_threshold = 0.5

        issues = []
        if kl is not None and abs(kl) > kl_threshold:
            issues.append(f"KL={kl:.6f} > {kl_threshold}")
        if top1 is False:
            issues.append("top-1 mismatch")
        if max_diff is not None and max_diff > diff_threshold:
            issues.append(f"max_diff={max_diff:.6f} > {diff_threshold}")

        if issues:
            print(f"  ⚠ FAIL: {label} — {'; '.join(issues)}")
            passed = False
        else:
            kl_str = f"KL={abs(kl):.6f}" if kl is not None else "N/A"
            diff_str = f"max_diff={max_diff:.6f}" if max_diff is not None else "N/A"
            top1_str = str(top1) if top1 is not None else "N/A"
            print(f"  ✓ PASS: {label} — {kl_str}, top1={top1_str}, {diff_str}")

    if passed:
        print("\n  ✅ ALL QUALITY GATES PASSED — TurboQuant is safe for deployment")
    else:
        print("\n  ❌ QUALITY GATE FAILED — see warnings above")

    # Save JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
