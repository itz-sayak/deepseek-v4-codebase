from __future__ import annotations

from .configuration import Aether2BConfig


def estimate_config_parameters(config: Aether2BConfig) -> int:
    """Approximate parameter count without allocating tensors."""

    d = config.hidden_size
    c = config.attention_head_dim
    h = config.num_attention_heads
    dc = config.query_compression_dim
    g = config.output_groups
    dg = config.group_output_dim
    n = config.mhc_expansion

    embedding = config.vocab_size * d
    lm_head = 0 if config.tie_word_embeddings else config.vocab_size * d

    q_path = d * dc + dc * h * c
    grouped_out = h * c * dg + g * dg * d
    common_attn = q_path + grouped_out + d * c + h
    hca = common_attn + 2 * d * c + config.hca_compression * c
    csa = (
        common_attn
        + 4 * d * c
        + 2 * config.csa_compression * c
        + 4 * d * config.indexer_head_dim
        + 2 * config.csa_compression * config.indexer_head_dim
        + dc * config.indexer_num_heads * config.indexer_head_dim
        + d * config.indexer_num_heads
    )

    expert = 3 * d * config.moe_intermediate_size
    moe = (config.num_routed_experts + config.num_shared_experts) * expert + d * config.num_routed_experts

    # Pure-dense mode: single SwiGLU per layer, no router.
    dense_ffn = 3 * d * config.dense_ffn_intermediate_size  # gate + up + down projections

    # Dynamic mHC generators for attention and MoE sublayers.
    mhc = (n * d) * n + (n * d) * (n * n) + (n * d) * n + n + n * n + n + 3

    layers = 0
    for i in range(config.num_hidden_layers):
        layers += hca if config.attention_type(i) == "hca" else csa
        layers += dense_ffn if config.dense_ffn_intermediate_size > 0 else moe
        layers += 2 * mhc

    norms = d * (1 + 2 * config.num_hidden_layers * 4)
    mtp = config.mtp_depth * (d + d * d)
    return embedding + lm_head + layers + norms + mtp
