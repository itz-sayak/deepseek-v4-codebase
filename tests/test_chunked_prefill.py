"""Tests for chunked_fast_prefill and chunked mHC forward.

Validates that:
  1. chunked_forward on ManifoldConstrainedHyperConnection matches standard forward.
  2. chunked_forward on DeepSeekV4Pro2BModel matches standard forward.
  3. chunked_fast_prefill produces identical serving state to fast_prefill.
  4. chunked_fast_prefill + step_token produces identical logits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_v4_pro_2b.configuration import DeepSeekV4Pro2BConfig
from deepseek_v4_pro_2b.modeling import (
    DeepSeekV4Pro2BForCausalLM,
    ManifoldConstrainedHyperConnection,
    HCAAttention,
    CSAAttention,
    DeepSeekMoE,
)
from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine
from deepseek_pipeline.serving import PytorchAttentionBackend


def tiny_config():
    return DeepSeekV4Pro2BConfig(
        vocab_size=128,
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
# ManifoldConstrainedHyperConnection.chunked_forward correctness
# ---------------------------------------------------------------------------

def test_mhc_chunked_forward_matches_standard_attn():
    """chunked_forward with HCA sublayer matches standard forward."""
    cfg = tiny_config()
    torch.manual_seed(0)
    mhc = ManifoldConstrainedHyperConnection(cfg, HCAAttention(cfg))
    mhc.eval()

    torch.manual_seed(42)
    state = torch.randn(1, 32, cfg.mhc_expansion, cfg.hidden_size)

    with torch.no_grad():
        out_std, _ = mhc.forward(state)
        out_chunked, _ = mhc.chunked_forward(state, chunk_size=8)

    assert out_std.shape == out_chunked.shape
    assert torch.allclose(out_std, out_chunked, atol=1e-4), (
        f"chunked_forward diverged: max_diff={(out_std - out_chunked).abs().max():.6f}"
    )


def test_mhc_chunked_forward_matches_standard_moe():
    """chunked_forward with DeepSeekMoE sublayer matches standard forward."""
    cfg = tiny_config()
    torch.manual_seed(0)
    mhc = ManifoldConstrainedHyperConnection(cfg, DeepSeekMoE(cfg, layer_idx=0))
    mhc.eval()

    torch.manual_seed(42)
    state = torch.randn(1, 16, cfg.mhc_expansion, cfg.hidden_size)
    token_ids = torch.randint(0, cfg.vocab_size, (1, 16))

    with torch.no_grad():
        out_std, bl_std = mhc.forward(state, token_ids=token_ids)
        out_chunked, bl_chunked = mhc.chunked_forward(state, chunk_size=4, token_ids=token_ids)

    assert out_std.shape == out_chunked.shape
    assert torch.allclose(out_std, out_chunked, atol=1e-4), (
        f"chunked MoE diverged: max_diff={(out_std - out_chunked).abs().max():.6f}"
    )


def test_mhc_chunked_forward_falls_back_for_short_seq():
    """When seq_len <= chunk_size, chunked_forward is identical to forward."""
    cfg = tiny_config()
    torch.manual_seed(0)
    mhc = ManifoldConstrainedHyperConnection(cfg, HCAAttention(cfg))
    mhc.eval()

    torch.manual_seed(42)
    state = torch.randn(1, 8, cfg.mhc_expansion, cfg.hidden_size)

    with torch.no_grad():
        out_std, _ = mhc.forward(state)
        out_chunked, _ = mhc.chunked_forward(state, chunk_size=32)

    assert torch.allclose(out_std, out_chunked, atol=1e-5)


# ---------------------------------------------------------------------------
# Model-level chunked_forward correctness
# ---------------------------------------------------------------------------

def test_model_chunked_forward_matches_standard():
    """DeepSeekV4Pro2BModel.chunked_forward matches standard forward."""
    cfg = tiny_config()
    torch.manual_seed(0)
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 32))

    with torch.no_grad():
        hidden_std, bal_std = model.model(input_ids)
        hidden_chunked, bal_chunked = model.model.chunked_forward(input_ids, mhc_chunk_size=8)

    assert hidden_std.shape == hidden_chunked.shape
    assert torch.allclose(hidden_std, hidden_chunked, atol=1e-3), (
        f"Model chunked_forward diverged: max_diff="
        f"{(hidden_std - hidden_chunked).abs().max():.6f}"
    )


# ---------------------------------------------------------------------------
# Serving engine: chunked_fast_prefill correctness
# ---------------------------------------------------------------------------

def test_chunked_fast_prefill_matches_fast_prefill():
    """chunked_fast_prefill produces identical serving state to fast_prefill."""
    cfg = tiny_config()
    torch.manual_seed(0)
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    model.eval()

    backend = PytorchAttentionBackend()
    token_ids = list(range(3, 35))

    engine_std = DeepSeekV4Pro2BServingEngine(model, backend=backend)
    state_std = engine_std.fast_prefill(token_ids)

    engine_chunked = DeepSeekV4Pro2BServingEngine(model, backend=backend)
    state_chunked = engine_chunked.chunked_fast_prefill(token_ids, mhc_chunk_size=8)

    assert state_std.token_count == state_chunked.token_count
    assert torch.allclose(state_std.last_logits, state_chunked.last_logits, atol=1e-3), (
        f"chunked last_logits diverged: max_diff="
        f"{(state_std.last_logits - state_chunked.last_logits).abs().max():.6f}"
    )


def test_chunked_fast_prefill_step_token_logits_match():
    """After chunked_fast_prefill + step_token, logits match fast_prefill + step_token."""
    cfg = tiny_config()
    torch.manual_seed(0)
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    model.eval()

    backend = PytorchAttentionBackend()
    token_ids = list(range(3, 35))
    next_token = torch.tensor([7])

    engine_std = DeepSeekV4Pro2BServingEngine(model, backend=backend)
    state_std = engine_std.fast_prefill(token_ids)
    logits_std, _ = engine_std.step_token(next_token, state_std)

    engine_chunked = DeepSeekV4Pro2BServingEngine(model, backend=backend)
    state_chunked = engine_chunked.chunked_fast_prefill(token_ids, mhc_chunk_size=8)
    logits_chunked, _ = engine_chunked.step_token(next_token, state_chunked)

    assert logits_std.shape == logits_chunked.shape
    assert torch.allclose(logits_std, logits_chunked, atol=1e-3), (
        f"step_token logits diverged: max_diff="
        f"{(logits_std - logits_chunked).abs().max():.6f}"
    )


def test_chunked_fast_prefill_empty_input():
    """chunked_fast_prefill([]) returns an empty state."""
    cfg = tiny_config()
    torch.manual_seed(0)
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    model.eval()
    engine = DeepSeekV4Pro2BServingEngine(model, backend=PytorchAttentionBackend())
    state = engine.chunked_fast_prefill([], mhc_chunk_size=8)
    assert state.token_count == 0


def test_chunked_fast_prefill_short_sequence_identical():
    """When len(token_ids) < mhc_chunk_size, fallback to standard forward."""
    cfg = tiny_config()
    torch.manual_seed(0)
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    model.eval()

    backend = PytorchAttentionBackend()
    token_ids = [5, 6, 7, 8]  # shorter than any reasonable chunk size

    engine1 = DeepSeekV4Pro2BServingEngine(model, backend=backend)
    state1 = engine1.fast_prefill(token_ids)

    engine2 = DeepSeekV4Pro2BServingEngine(model, backend=backend)
    state2 = engine2.chunked_fast_prefill(token_ids, mhc_chunk_size=4096)

    assert state1.token_count == state2.token_count
    assert torch.allclose(state1.last_logits, state2.last_logits, atol=1e-5)
