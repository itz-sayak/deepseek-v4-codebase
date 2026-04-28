"""Tests for PolarQuant (TurboQuant Stage 1).

Mandatory testing ladder before production deployment:
  8-bit → validate (cosine sim > 0.999, needle-in-haystack)
  4-bit → validate (cosine sim > 0.99)
  3-bit → ONLY if 4-bit passes (not implemented, below-spec)

These tests cover:
  - WHT self-inverse property
  - 8-bit round-trip cosine similarity on D=128 (attn_head_dim)
  - 4-bit round-trip cosine similarity on D=128
  - Round-trip on D=64 (index_head_dim)
  - Disabled (bits=None) path via serving engine is a no-op
  - PolarQuant encode/decode shape contracts

Architecture note: compressed KV vectors may have directional structure from
the n_expand geometry.  The cosine similarity thresholds here are MINIMUM bars;
run needle-in-haystack eval before enabling 4-bit in production.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_v4_pro_2b.turbo_quant import PolarQuant, _wht


# ---------------------------------------------------------------------------
# WHT correctness
# ---------------------------------------------------------------------------

def test_wht_self_inverse():
    """_wht(_wht(x)) == x up to floating-point tolerance."""
    torch.manual_seed(0)
    x = torch.randn(4, 16, 128)
    reconstructed = _wht(_wht(x))
    # float32 butterfly; expect < 1e-5 absolute error
    assert torch.allclose(x, reconstructed, atol=1e-5), \
        f"WHT is not self-inverse; max error = {(x - reconstructed).abs().max():.2e}"


def test_wht_self_inverse_d64():
    torch.manual_seed(1)
    x = torch.randn(2, 8, 64)
    reconstructed = _wht(_wht(x))
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_wht_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        _wht(torch.randn(3, 12))


# ---------------------------------------------------------------------------
# PolarQuant 8-bit
# ---------------------------------------------------------------------------

def _cosine_sim_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-vector cosine similarity over all leading dims."""
    a_flat = a.view(-1, a.shape[-1]).float()
    b_flat = b.view(-1, b.shape[-1]).float()
    sims = F.cosine_similarity(a_flat, b_flat, dim=-1)
    return sims.mean().item()


@pytest.mark.parametrize("d", [128, 64])
def test_polar_quant_8bit_roundtrip(d):
    """8-bit PolarQuant round-trip cosine similarity > 0.999 on random vectors."""
    torch.manual_seed(42)
    pq = PolarQuant(bits=8)
    # Simulate compressed KV blocks [B=1, T_blocks=64, D]
    x = torch.randn(1, 64, d, dtype=torch.bfloat16)
    data, scale = pq.encode(x)

    # data must be int8
    assert data.dtype == torch.int8, f"Expected int8, got {data.dtype}"
    # scale must be float32
    assert scale.dtype == torch.float32, f"Expected float32 scale, got {scale.dtype}"
    # scale shape: [..., 1]
    assert scale.shape == (1, 64, 1)
    # data shape: same as x for 8-bit
    assert data.shape == x.shape

    x_hat = pq.decode(data, scale)
    assert x_hat.dtype == torch.bfloat16
    assert x_hat.shape == x.shape

    sim = _cosine_sim_mean(x.float(), x_hat.float())
    assert sim > 0.999, f"8-bit cosine similarity {sim:.6f} < 0.999 (D={d})"


@pytest.mark.parametrize("d", [128, 64])
def test_polar_quant_4bit_roundtrip(d):
    """4-bit PolarQuant round-trip cosine similarity > 0.99 on random vectors."""
    torch.manual_seed(42)
    pq = PolarQuant(bits=4)
    x = torch.randn(1, 64, d, dtype=torch.bfloat16)
    data, scale = pq.encode(x)

    # data must be uint8, D//2 per vector (packed)
    assert data.dtype == torch.uint8, f"Expected uint8, got {data.dtype}"
    assert data.shape == (1, 64, d // 2), f"Expected packed shape {(1, 64, d//2)}, got {data.shape}"
    assert scale.dtype == torch.float32
    assert scale.shape == (1, 64, 1)

    x_hat = pq.decode(data, scale)
    assert x_hat.dtype == torch.bfloat16
    assert x_hat.shape == x.shape

    sim = _cosine_sim_mean(x.float(), x_hat.float())
    assert sim > 0.99, f"4-bit cosine similarity {sim:.6f} < 0.99 (D={d})"


# ---------------------------------------------------------------------------
# Different-seed diagonals (different dims get different diagonals)
# ---------------------------------------------------------------------------

def test_polar_quant_different_dims_use_different_diagonals():
    pq = PolarQuant(bits=8)
    diag128 = pq._diag(128, torch.device("cpu"))
    diag64 = pq._diag(64, torch.device("cpu"))
    # They should differ in content — not just size
    assert not torch.equal(diag128[:64], diag64)


# ---------------------------------------------------------------------------
# encode/decode are consistent across multiple calls (deterministic diagonal)
# ---------------------------------------------------------------------------

def test_polar_quant_encode_decode_deterministic():
    torch.manual_seed(7)
    pq = PolarQuant(bits=8)
    x = torch.randn(1, 16, 128, dtype=torch.bfloat16)

    data1, scale1 = pq.encode(x)
    data2, scale2 = pq.encode(x)

    assert torch.equal(data1, data2), "encode is not deterministic"
    assert torch.equal(scale1, scale2), "encode scale is not deterministic"


# ---------------------------------------------------------------------------
# PolarQuant with structured (non-isotropic) inputs — stress-test for n_expand
# ---------------------------------------------------------------------------

def test_polar_quant_8bit_structured_input():
    """Validate 8-bit on vectors with strong directional structure (worst case for WHT)."""
    torch.manual_seed(0)
    pq = PolarQuant(bits=8)
    # Simulate expansion-biased vectors: strong first-component dominance
    x = torch.zeros(1, 32, 128, dtype=torch.bfloat16)
    x[:, :, 0] = 10.0   # near-singular direction
    x += 0.01 * torch.randn_like(x)

    data, scale = pq.encode(x)
    x_hat = pq.decode(data, scale)

    sim = _cosine_sim_mean(x.float(), x_hat.float())
    # Structured inputs should still achieve high similarity at 8-bit
    assert sim > 0.999, f"8-bit structured input cosine similarity {sim:.6f} < 0.999"


def test_polar_quant_4bit_structured_input():
    """4-bit structured input: directional bias from n_expand architecture."""
    torch.manual_seed(0)
    pq = PolarQuant(bits=4)
    x = torch.zeros(1, 32, 128, dtype=torch.bfloat16)
    x[:, :, 0] = 10.0
    x += 0.01 * torch.randn_like(x)

    data, scale = pq.encode(x)
    x_hat = pq.decode(data, scale)

    sim = _cosine_sim_mean(x.float(), x_hat.float())
    # Structured inputs may degrade at 4-bit; threshold intentionally 0.95 not 0.99
    # If this fails, do NOT enable 4-bit on production mHC compressed state.
    assert sim > 0.95, (
        f"4-bit structured input cosine similarity {sim:.6f} < 0.95 — "
        "strong directional structure is not adequately decorrelated by WHT; "
        "DO NOT enable 4-bit quantization on this architecture without further validation."
    )


# ---------------------------------------------------------------------------
# Invalid bits raises
# ---------------------------------------------------------------------------

def test_polar_quant_invalid_bits():
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        PolarQuant(bits=3)


# ---------------------------------------------------------------------------
# Serving engine integration: turbo_quant_bits=None is a no-op
# ---------------------------------------------------------------------------

def test_serving_engine_no_quant_is_noop():
    """With turbo_quant_bits=None, compressed state stays bf16 and scale is None."""
    from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine, HCAServingState
    from deepseek_v4_pro_2b.configuration import DeepSeekV4Pro2BConfig
    from deepseek_v4_pro_2b.modeling import DeepSeekV4Pro2BForCausalLM
    from deepseek_pipeline.serving import PytorchAttentionBackend

    cfg = DeepSeekV4Pro2BConfig(
        num_hidden_layers=2, hidden_size=64, num_attention_heads=2,
        attention_head_dim=16, query_compression_dim=32, indexer_num_heads=2,
        indexer_head_dim=8, mhc_expansion=2, hca_compression=4, csa_compression=4,
        csa_top_k=4, sliding_window=8, rope_dim=8, output_groups=2,
        group_output_dim=32, num_routed_experts=4, num_shared_experts=1,
        num_experts_per_tok=2, moe_intermediate_size=64, hash_routed_layers=1,
        vocab_size=256,
    )
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    engine = DeepSeekV4Pro2BServingEngine(
        model, backend=PytorchAttentionBackend(), turbo_quant_bits=None
    )
    assert engine._polar_quant is None

    # A round-trip through _quant_append should leave data as bf16 with no scale.
    dummy = torch.randn(1, 1, 16)
    data, scale = engine._quant_append(None, None, dummy)
    assert scale is None
    assert data.dtype == dummy.dtype


def test_serving_engine_8bit_quant_produces_int8():
    """With turbo_quant_bits=8, _quant_append produces int8 data + float32 scale."""
    from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine
    from deepseek_v4_pro_2b.configuration import DeepSeekV4Pro2BConfig
    from deepseek_v4_pro_2b.modeling import DeepSeekV4Pro2BForCausalLM
    from deepseek_pipeline.serving import PytorchAttentionBackend

    cfg = DeepSeekV4Pro2BConfig(
        num_hidden_layers=2, hidden_size=64, num_attention_heads=2,
        attention_head_dim=16, query_compression_dim=32, indexer_num_heads=2,
        indexer_head_dim=8, mhc_expansion=2, hca_compression=4, csa_compression=4,
        csa_top_k=4, sliding_window=8, rope_dim=8, output_groups=2,
        group_output_dim=32, num_routed_experts=4, num_shared_experts=1,
        num_experts_per_tok=2, moe_intermediate_size=64, hash_routed_layers=1,
        vocab_size=256,
    )
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    engine = DeepSeekV4Pro2BServingEngine(
        model, backend=PytorchAttentionBackend(), turbo_quant_bits=8
    )
    assert engine._polar_quant is not None
    assert engine._polar_quant.bits == 8

    dummy = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    data, scale = engine._quant_append(None, None, dummy)
    assert data.dtype == torch.int8
    assert scale is not None and scale.dtype == torch.float32

    # Read back should give bf16.
    recovered = engine._read_compressed(data, scale)
    assert recovered.dtype in [torch.bfloat16, torch.float32]
    assert recovered.shape == dummy.shape
