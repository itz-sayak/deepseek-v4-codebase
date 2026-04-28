from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class DeepSeekV4Pro2BConfig:
    """Scaled DeepSeek-V4-Pro-style configuration.

    The paper's Pro model is a 1.6T total parameter MoE. This configuration keeps
    its architectural ratios and mechanisms, but scales the model to roughly the
    2B total-parameter class by reducing depth, hidden width, heads, and experts.
    """

    vocab_size: int = 65536
    hidden_size: int = 1536
    num_hidden_layers: int = 26
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    query_compression_dim: int = 512
    indexer_num_heads: int = 16
    indexer_head_dim: int = 64
    csa_compression: int = 4
    hca_compression: int = 128
    csa_top_k: int = 256
    sliding_window: int = 128
    rope_dim: int = 64
    rope_theta: float = 10000.0
    max_position_embeddings: int = 65536
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 0.1
    output_groups: int = 4
    group_output_dim: int = 384
    num_routed_experts: int = 16
    num_shared_experts: int = 1
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 832
    hash_routed_layers: int = 3
    mhc_expansion: int = 4
    mhc_sinkhorn_iters: int = 20
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    router_bias_update_speed: float = 0.001
    balance_loss_weight: float = 0.0001
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.3
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def attention_type(self, layer_idx: int) -> str:
        if layer_idx < 2:
            return "hca"
        return "csa" if (layer_idx - 2) % 2 == 0 else "hca"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "DeepSeekV4Pro2BConfig":
        return cls(**values)
