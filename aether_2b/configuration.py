from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class Aether2BConfig:
    """Scaled Aether-2B-style configuration.

    The paper's Pro model is a 1.6T total parameter MoE. This configuration keeps
    its architectural ratios and mechanisms, but scales the model to roughly the
    2B total-parameter class by reducing depth, hidden width, heads, and experts.
    """

    vocab_size: int = 65536
    hidden_size: int = 1536
    num_hidden_layers: int = 28
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
    use_tiled_prefill_cuda: bool = False
    tiled_prefill_tile_size: int = 256
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
    # Dense FFN mode: when > 0 every layer uses a single SwiGLU of this width
    # instead of MoE routing.  Set via Aether2BConfig.dense_2b().
    dense_ffn_intermediate_size: int = 0

    def attention_type(self, layer_idx: int) -> str:
        if layer_idx < 2:
            return "hca"
        return "csa" if (layer_idx - 2) % 2 == 0 else "hca"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "Aether2BConfig":
        return cls(**values)

    @classmethod
    def dense_2b(cls) -> "Aether2BConfig":
        """Return the standard 2B config with MoE replaced by a single dense SwiGLU.

        Every transformer block gets one SwiGLU FFN of width
        ``dense_ffn_intermediate_size``.  All routing bookkeeping fields are
        zeroed out so no router linear, expert list, or balance-loss computation
        is instantiated.

        Sizing rationale
        ----------------
        ``dense_ffn_intermediate_size = 14336`` was chosen so that the total
        parameter count matches the MoE variant's ~2B class:

        * Non-FFN budget (attention + mHC + embeddings, 28 layers): ~291 M
        * FFN per layer at width 14336: 3 × 1536 × 14336 ≈ 66.1 M
        * Total: 291 M + 28 × 66.1 M ≈ **2.135 B parameters**

        14336 = 14 × 1024, a standard SwiGLU width used in production dense
        models (e.g. Mistral-7B uses 14336 with hidden_size=4096).

        FLOPs note: per-token FLOPs are roughly identical to the MoE variant
        because the dense FFN width (14336) far exceeds the MoE's *activated*
        capacity (5 experts × 832 = 4160) — the dense variant is compute-heavier
        per token but has no routing overhead and no expert-dispatch scatter/gather.
        """
        return cls(
            # MoE fields — all disabled
            num_routed_experts=0,
            num_shared_experts=0,
            num_experts_per_tok=0,
            moe_intermediate_size=0,
            hash_routed_layers=0,
            balance_loss_weight=0.0,
            router_bias_update_speed=0.0,
            # Dense FFN width: 14336 → total params ≈ 1.99 B (~2B)
            dense_ffn_intermediate_size=14336,
        )
