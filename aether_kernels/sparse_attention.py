from __future__ import annotations

import torch

from .loader import load_aether_cuda_kernels


def sparse_sink_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    ext = load_aether_cuda_kernels()
    return ext.sparse_sink_attention(q, kv, topk_indices, sink, float(softmax_scale))


def tiled_prefill_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    softmax_scale: float,
    tile_size: int = 256,
) -> torch.Tensor:
    """Tiled prefill attention that streams source tokens in chunks of *tile_size*
    to avoid materialising the full attention matrix for long sequences.

    Tensor contract
    ---------------
    q    : [B, T, H, D]  target tokens (T > 1 for prefill)
    kv   : [B, S, 2, D]  source KV entries
    sink : scalar or [H] per-head sink logit
    """
    ext = load_aether_cuda_kernels()
    return ext.tiled_prefill_attention(
        q.contiguous(),
        kv.contiguous(),
        sink.contiguous(),
        float(softmax_scale),
        int(tile_size),
    )


def csa_indexer_topk(
    index_q: torch.Tensor,
    index_kv: torch.Tensor,
    index_weight: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Fused CSA lightning indexer: compute dot-product scores and return top-k indices.

    Tensor contract
    ---------------
    index_q      : [B, IH, IC]  float32  compressed indexer queries
    index_kv     : [B, S, IC]   float32  compressed KV index embeddings
    index_weight : [B, IH]      float32  per-head routing weights (relu applied internally)
    k            : int          number of entries to select
    returns      : [B, k]       int64    selected source indices
    """
    ext = load_aether_cuda_kernels()
    return ext.csa_indexer_topk(
        index_q.float().contiguous(),
        index_kv.float().contiguous(),
        index_weight.float().contiguous(),
        int(k),
    )


def hca_compress(
    buffer_c: torch.Tensor,
    buffer_z: torch.Tensor,
) -> torch.Tensor:
    """Fused HCA compression: softmax-weighted reduction of one compression window.

    Tensor contract
    ---------------
    buffer_c : [B, M, D]  raw KV vectors for the window  (M = hca_compression)
    buffer_z : [B, M, D]  corresponding unnormalised logit weights
    returns  : [B, D]     single compressed entry (RoPE rotation applied in Python)
    """
    ext = load_aether_cuda_kernels()
    return ext.hca_compress(buffer_c.contiguous(), buffer_z.contiguous())
