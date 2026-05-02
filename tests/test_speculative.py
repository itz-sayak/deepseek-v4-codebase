"""Tests for speculative decoding.

Key correctness properties:
  1. Self-speculative (same engine for target and draft) → falls back to
     standard sampling, producing correct output.
  2. Greedy speculative with a perfect draft (draft == target) → 100% acceptance
     rate, output matches pure greedy.
  3. SpecDecodeSummary stats are internally consistent.
  4. EOS stopping works correctly.
  5. Output length respects max_new_tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aether_2b.configuration import Aether2BConfig
from aether_2b.modeling import Aether2BForCausalLM
from aether_2b.serving import Aether2BServingEngine
from aether_2b.speculative import SpeculativeDecoder, SpecDecodeSummary, build_self_spec_draft_model
from aether_pipeline.serving import PytorchAttentionBackend


def tiny_config():
    return Aether2BConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        query_compression_dim=16,
        indexer_num_heads=2,
        indexer_head_dim=8,
        csa_compression=2,
        hca_compression=2,
        csa_top_k=2,
        sliding_window=4,
        rope_dim=8,
        output_groups=2,
        group_output_dim=16,
        num_routed_experts=2,
        num_shared_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        hash_routed_layers=1,
        mhc_expansion=2,
        mhc_sinkhorn_iters=2,
        mtp_depth=1,
    )


def shared_vocab_tiny_config():
    cfg = tiny_config()
    cfg.vocab_size = Aether2BConfig().vocab_size
    return cfg


def make_engine(model):
    return Aether2BServingEngine(model, backend=PytorchAttentionBackend())


# ---------------------------------------------------------------------------
# Basic import and construction
# ---------------------------------------------------------------------------

def test_speculative_decoder_constructs():
    cfg = tiny_config()
    torch.manual_seed(0)
    model = Aether2BForCausalLM(cfg).eval()
    engine = make_engine(model)
    decoder = SpeculativeDecoder(engine, engine, draft_steps=3, temperature=0.0)
    assert decoder.draft_steps == 3
    assert decoder._self_spec is True


def test_speculative_rejects_vocab_mismatch():
    target_cfg = tiny_config()
    draft_cfg = shared_vocab_tiny_config()
    torch.manual_seed(0)
    target_model = Aether2BForCausalLM(target_cfg).eval()
    draft_model = Aether2BForCausalLM(draft_cfg).eval()
    target_engine = make_engine(target_model)
    draft_engine = make_engine(draft_model)

    with pytest.raises(ValueError, match="matching vocab_size"):
        SpeculativeDecoder(target_engine, draft_engine, draft_steps=3, temperature=0.0)


# ---------------------------------------------------------------------------
# SpecDecodeSummary stats
# ---------------------------------------------------------------------------

def test_spec_decode_summary_acceptance_rate():
    s = SpecDecodeSummary(output_ids=list(range(10)),
                          total_proposed=20, total_accepted=15, total_rounds=5)
    assert abs(s.mean_acceptance_rate - 0.75) < 1e-6
    assert abs(s.effective_speedup - 2.0) < 1e-6


def test_spec_decode_summary_empty():
    s = SpecDecodeSummary(output_ids=[])
    assert s.mean_acceptance_rate == 0.0
    assert s.effective_speedup == 1.0


# ---------------------------------------------------------------------------
# Output length respects max_new_tokens
# ---------------------------------------------------------------------------

def test_speculative_generate_length():
    """Output never exceeds prompt + max_new_tokens."""
    cfg = tiny_config()
    torch.manual_seed(42)
    model = Aether2BForCausalLM(cfg).eval()
    engine = make_engine(model)
    decoder = SpeculativeDecoder(engine, engine, draft_steps=3, temperature=0.0)

    prompt = [1, 2, 3, 4]
    max_new = 8
    result = decoder.generate(prompt, max_new_tokens=max_new)
    new_tokens = len(result.output_ids) - len(prompt)
    assert new_tokens == max_new, f"Expected {max_new} new tokens, got {new_tokens}"


def test_real_draft_speculative_is_deterministic_and_accepts_tokens():
    cfg = shared_vocab_tiny_config()
    torch.manual_seed(11)
    target_model = Aether2BForCausalLM(cfg).eval()
    draft_model = Aether2BForCausalLM(cfg).eval()
    draft_model.load_state_dict(target_model.state_dict())

    target_engine = make_engine(target_model)
    draft_engine = make_engine(draft_model)
    decoder = SpeculativeDecoder(target_engine, draft_engine, draft_steps=4, temperature=0.0)

    prompt = list(range(3, 515))
    max_new = 25

    first = decoder.generate(prompt, max_new_tokens=max_new)
    second = decoder.generate(prompt, max_new_tokens=max_new)

    assert first.output_ids == second.output_ids
    assert len(first.output_ids) == len(prompt) + max_new
    assert first.total_proposed > 0
    assert first.total_accepted == first.total_proposed
    assert first.mean_acceptance_rate == 1.0


# ---------------------------------------------------------------------------
# EOS stopping
# ---------------------------------------------------------------------------

def test_speculative_generate_eos_stops():
    """Generation stops when EOS is produced."""
    cfg = tiny_config()
    torch.manual_seed(0)
    model = Aether2BForCausalLM(cfg).eval()
    engine = make_engine(model)

    # Use temperature=0 (greedy) for determinism; with eos=argmax of first logit
    decoder = SpeculativeDecoder(engine, engine, draft_steps=2, temperature=0.0)
    prompt = [1, 2]

    # First, find what token the model emits to use as EOS
    state = engine.prefill(prompt)
    logits, _ = engine.step_token(prompt[-1], state)
    eos_id = int(logits[0].argmax().item())

    result = decoder.generate(prompt, max_new_tokens=20, eos_token_id=eos_id)
    # At least one new token, and EOS should appear
    assert len(result.output_ids) > len(prompt)
    assert eos_id in result.output_ids[len(prompt):]


# ---------------------------------------------------------------------------
# Greedy self-speculative matches engine.generate output
# ---------------------------------------------------------------------------

def test_self_speculative_greedy_matches_standard_generate():
    """Self-spec greedy decoding matches the engine's own generate() in output."""
    cfg = tiny_config()
    torch.manual_seed(7)
    model = Aether2BForCausalLM(cfg).eval()
    engine = make_engine(model)

    prompt = [3, 5, 7, 9]
    max_new = 4

    # Standard greedy via engine.generate
    std_output = engine.generate(prompt, max_new_tokens=max_new, temperature=0.0)

    # Self-spec greedy (same engine → fallback path)
    decoder = SpeculativeDecoder(engine, engine, draft_steps=3, temperature=0.0)
    spec_result = decoder.generate(prompt, max_new_tokens=max_new)

    assert spec_result.output_ids == std_output, (
        f"Self-spec output {spec_result.output_ids} != standard {std_output}"
    )


# ---------------------------------------------------------------------------
# Perfect-draft speculative: draft == target, expect 100% acceptance
# ---------------------------------------------------------------------------

def test_perfect_draft_full_acceptance():
    """When draft model == target model, greedy spec-decode accepts all K tokens."""
    cfg = tiny_config()
    torch.manual_seed(3)
    model = Aether2BForCausalLM(cfg).eval()
    engine_target = make_engine(model)
    engine_draft = make_engine(model)

    # With identical models and greedy decoding all proposals should be accepted
    decoder = SpeculativeDecoder(
        engine_target, engine_draft, draft_steps=3, temperature=0.0
    )
    prompt = [1, 4, 9]
    result = decoder.generate(prompt, max_new_tokens=6)

    # All rounds should have 100% acceptance (assuming no edge case)
    if result.total_rounds > 0:
        assert result.total_accepted == result.total_proposed, (
            f"Expected 100% acceptance: proposed={result.total_proposed} "
            f"accepted={result.total_accepted}"
        )


# ---------------------------------------------------------------------------
# Output ids include the prompt prefix
# ---------------------------------------------------------------------------

def test_output_ids_start_with_prompt():
    cfg = tiny_config()
    torch.manual_seed(1)
    model = Aether2BForCausalLM(cfg).eval()
    engine = make_engine(model)
    decoder = SpeculativeDecoder(engine, engine, draft_steps=2, temperature=0.0)

    prompt = [10, 20, 30]
    result = decoder.generate(prompt, max_new_tokens=4)
    assert result.output_ids[:len(prompt)] == prompt, (
        "Output prefix must match the input prompt"
    )


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_empty_prompt_raises():
    cfg = tiny_config()
    torch.manual_seed(0)
    model = Aether2BForCausalLM(cfg).eval()
    engine = make_engine(model)
    decoder = SpeculativeDecoder(engine, engine, draft_steps=3, temperature=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        decoder.generate([], max_new_tokens=5)


def test_step_token_with_hidden_matches_step_token_logits():
    """Serving helper used by shared-layer fusion preserves step_token logits."""
    cfg = tiny_config()
    torch.manual_seed(13)
    model = Aether2BForCausalLM(cfg).eval()
    engine = make_engine(model)

    prompt = [2, 4, 6]
    state_a = engine.prefill(prompt)
    state_b = state_a.clone()

    token = 7
    logits_ref, _ = engine.step_token(token, state_a)
    logits_hidden, _, _ = engine.step_token_with_hidden(token, state_b)
    assert torch.allclose(logits_ref, logits_hidden, atol=1e-6, rtol=1e-6)


def test_generate_speculative_self_spec_layers_integrated_path_runs():
    """ServingEngine.generate_speculative supports shared-layer self-spec integration."""
    cfg = tiny_config()
    cfg.num_hidden_layers = 4
    torch.manual_seed(17)
    model = Aether2BForCausalLM(cfg).eval()
    engine = make_engine(model)

    prompt = [1, 3, 5, 7]
    out_ids = engine.generate_speculative(
        token_ids=prompt,
        max_new_tokens=4,
        self_spec_layers=2,
        draft_steps=2,
        temperature=0.0,
    )
    assert out_ids[: len(prompt)] == prompt
    assert len(out_ids) == len(prompt) + 4


def test_shared_layer_fusion_depth_detected_for_shared_prefix_draft():
    """Speculative decoder detects shared-layer draft/target prefix for fused verify path."""
    cfg = tiny_config()
    cfg.num_hidden_layers = 4
    torch.manual_seed(19)
    target_model = Aether2BForCausalLM(cfg).eval()
    draft_model = build_self_spec_draft_model(target_model, draft_layers=2)
    target_engine = make_engine(target_model)
    draft_engine = make_engine(draft_model)

    decoder = SpeculativeDecoder(target_engine, draft_engine, draft_steps=2, temperature=0.0)
    assert decoder._shared_layer_fusion_depth == 2
