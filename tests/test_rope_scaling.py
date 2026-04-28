import os
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_pipeline.serving import PytorchAttentionBackend
from deepseek_v4_pro_2b.configuration import DeepSeekV4Pro2BConfig
from deepseek_v4_pro_2b.modeling import DeepSeekV4Pro2BForCausalLM, get_rope_freqs
from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine


def _base_config(**overrides) -> DeepSeekV4Pro2BConfig:
    values = dict(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        attention_head_dim=16,
        query_compression_dim=32,
        indexer_num_heads=2,
        indexer_head_dim=8,
        csa_compression=2,
        hca_compression=4,
        csa_top_k=2,
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
        max_position_embeddings=65536,
        rope_scaling_type="none",
        rope_scaling_factor=1.0,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_mscale=0.1,
    )
    values.update(overrides)
    return DeepSeekV4Pro2BConfig(**values)


def _base_rope_freqs(config: DeepSeekV4Pro2BConfig) -> torch.Tensor:
    half_dim = config.rope_dim // 2
    freqs = torch.arange(half_dim, dtype=torch.float32)
    return 1.0 / (config.rope_theta ** (freqs / half_dim))


def test_get_rope_freqs_none_matches_base():
    config = _base_config(rope_dim=16, rope_scaling_type="none")
    actual = get_rope_freqs(512, config)
    expected = _base_rope_freqs(config)
    torch.testing.assert_close(actual, expected)


def test_get_rope_freqs_linear_scales_all_frequencies():
    config = _base_config(rope_dim=16, rope_scaling_type="linear", rope_scaling_factor=4.0)
    actual = get_rope_freqs(512, config)
    expected = _base_rope_freqs(config) / 4.0
    torch.testing.assert_close(actual, expected)


def test_get_rope_freqs_yarn_keeps_short_wavelengths_and_scales_long_wavelengths():
    config = _base_config(rope_dim=64, rope_scaling_type="yarn", rope_scaling_factor=4.0)
    base = _base_rope_freqs(config)
    actual = get_rope_freqs(512, config)
    wavelen = (2.0 * torch.pi) / base
    short_wavelen = config.max_position_embeddings / config.yarn_beta_fast
    long_wavelen = config.max_position_embeddings / config.yarn_beta_slow

    keep_mask = wavelen <= short_wavelen
    scale_mask = wavelen >= long_wavelen
    ramp_mask = ~(keep_mask | scale_mask)

    if keep_mask.any():
        torch.testing.assert_close(actual[keep_mask], base[keep_mask])
    if scale_mask.any():
        torch.testing.assert_close(actual[scale_mask], base[scale_mask] / config.rope_scaling_factor)
    if ramp_mask.any():
        ramp_values = actual[ramp_mask]
        lower = base[ramp_mask] / config.rope_scaling_factor
        upper = base[ramp_mask]
        assert torch.all(ramp_values <= upper + 1e-6)
        assert torch.all(ramp_values >= lower - 1e-6)


def test_yarn_model_forward_runs_and_produces_finite_logits():
    config = _base_config(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        query_compression_dim=16,
        indexer_num_heads=2,
        indexer_head_dim=8,
        csa_compression=2,
        hca_compression=4,
        csa_top_k=2,
        sliding_window=4,
        rope_dim=8,
        output_groups=2,
        group_output_dim=16,
        moe_intermediate_size=32,
        rope_scaling_type="yarn",
        rope_scaling_factor=4.0,
    )
    model = DeepSeekV4Pro2BForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    with torch.no_grad():
        output = model(input_ids)
    assert output.logits.shape == (1, 32, config.vocab_size)
    assert torch.isfinite(output.logits).all()
    assert output.balance_loss is not None


@pytest.mark.skipif(
    os.environ.get("DEEPSEEK_RUN_ROPE_NEEDLE") != "1",
    reason="Long-context YaRN needle diagnostic is opt-in",
)
def test_yarn_needle_top1_match():
    torch.manual_seed(0)
    source_lengths = (65536, 131072)
    results = []
    for ctx_length in source_lengths:
        base_cfg = _base_config(
            vocab_size=256,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            query_compression_dim=16,
            indexer_num_heads=2,
            indexer_head_dim=8,
            csa_compression=2,
            hca_compression=4,
            csa_top_k=2,
            sliding_window=4,
            rope_dim=8,
            output_groups=2,
            group_output_dim=16,
            moe_intermediate_size=32,
            rope_scaling_type="none",
            rope_scaling_factor=1.0,
        )
        yarn_cfg = replace(base_cfg, rope_scaling_type="yarn", rope_scaling_factor=max(1.0, ctx_length / base_cfg.max_position_embeddings))

        base_model = DeepSeekV4Pro2BForCausalLM(base_cfg).eval()
        yarn_model = DeepSeekV4Pro2BForCausalLM(yarn_cfg).eval()
        yarn_model.load_state_dict(base_model.state_dict())

        base_engine = DeepSeekV4Pro2BServingEngine(base_model, backend=PytorchAttentionBackend(), device="cpu")
        yarn_engine = DeepSeekV4Pro2BServingEngine(yarn_model, backend=PytorchAttentionBackend(), device="cpu")

        rng = torch.Generator().manual_seed(1234)
        haystack = torch.randint(3, base_cfg.vocab_size, (ctx_length,), generator=rng).tolist()
        needle_pos = ctx_length // 2
        needle_tokens = [base_cfg.vocab_size - 1, base_cfg.vocab_size - 2, base_cfg.vocab_size - 3]
        for offset, token in enumerate(needle_tokens):
            if needle_pos + offset < ctx_length:
                haystack[needle_pos + offset] = token

        base_state = base_engine.prefill(haystack)
        yarn_state = yarn_engine.prefill(haystack)

        probe_token = 1
        base_logits, _ = base_engine.step_token(probe_token, base_state)
        yarn_logits, _ = yarn_engine.step_token(probe_token, yarn_state)
        top1_match = base_logits.argmax(dim=-1).item() == yarn_logits.argmax(dim=-1).item()
        results.append((ctx_length, top1_match, base_logits.argmax(dim=-1).item(), yarn_logits.argmax(dim=-1).item()))

    for ctx_length, top1_match, base_top1, yarn_top1 in results:
        print(f"ctx={ctx_length} top1_match={top1_match} base_top1={base_top1} yarn_top1={yarn_top1}")
