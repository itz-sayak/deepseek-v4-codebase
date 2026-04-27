from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - keeps static tooling usable without torch.
    raise ModuleNotFoundError("deepseek_v4_pro_2b requires PyTorch to run") from exc

from .configuration import DeepSeekV4Pro2BConfig


@dataclass
class DeepSeekV4Output:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    mtp_logits: Optional[List[torch.Tensor]] = None
    balance_loss: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.rms_norm is a fused CUDA kernel — avoids materialising a float32
        # copy of x (which would be 2× the tensor size), saving peak memory.
        return F.rms_norm(x, self.weight.shape, self.weight.to(x.dtype), self.eps)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, rope_dim: int, theta: float) -> torch.Tensor:
    if rope_dim <= 0:
        return x
    rope_dim = min(rope_dim, x.size(-1))
    if rope_dim % 2:
        rope_dim -= 1
    pass_part, rope_part = x[..., :-rope_dim], x[..., -rope_dim:]
    freqs = torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (freqs / rope_dim))
    angles = positions.float().unsqueeze(-1) * inv_freq
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(x.dtype)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(x.dtype)
    if positions.ndim == 1 and rope_part.ndim == 4:
        cos = cos.view(1, positions.numel(), 1, rope_dim)
        sin = sin.view(1, positions.numel(), 1, rope_dim)
    elif positions.ndim == 1 and rope_part.ndim == 3:
        cos = cos.view(1, positions.numel(), rope_dim)
        sin = sin.view(1, positions.numel(), rope_dim)
    else:
        while cos.ndim < rope_part.ndim:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
    return torch.cat([pass_part, rope_part * cos + _rotate_half(rope_part) * sin], dim=-1)


def sink_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    dropout_p: float,
    training: bool,
    kv_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    logits = torch.einsum("bhd,bsd->bhs", q, k) * scale
    if kv_mask is not None:
        logits = logits.masked_fill(~kv_mask[:, None, :], float("-inf"))
    max_real = logits.max(dim=-1, keepdim=True).values
    max_all = torch.maximum(max_real, sink.view(1, -1, 1).to(logits.dtype))
    exp_logits = torch.exp(logits - max_all)
    sink_exp = torch.exp(sink.view(1, -1, 1).to(logits.dtype) - max_all)
    weights = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + sink_exp)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return torch.einsum("bhs,bsd->bhd", weights, v)


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if not labels.ne(-100).any():
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


class GroupedOutputProjection(nn.Module):
    def __init__(self, config: DeepSeekV4Pro2BConfig):
        super().__init__()
        assert config.num_attention_heads % config.output_groups == 0
        heads_per_group = config.num_attention_heads // config.output_groups
        group_in = heads_per_group * config.attention_head_dim
        self.groups = nn.ModuleList(
            nn.Linear(group_in, config.group_output_dim, bias=False)
            for _ in range(config.output_groups)
        )
        self.out_proj = nn.Linear(config.output_groups * config.group_output_dim, config.hidden_size, bias=False)
        self.output_groups = config.output_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.output_groups, dim=-2)
        projected = [proj(chunk.flatten(-2)) for proj, chunk in zip(self.groups, chunks)]
        return self.out_proj(torch.cat(projected, dim=-1))


class CompressedAttentionBase(nn.Module):
    def __init__(self, config: DeepSeekV4Pro2BConfig):
        super().__init__()
        self.config = config
        self.q_down = nn.Linear(config.hidden_size, config.query_compression_dim, bias=False)
        self.q_up = nn.Linear(
            config.query_compression_dim,
            config.num_attention_heads * config.attention_head_dim,
            bias=False,
        )
        self.q_norm = RMSNorm(config.attention_head_dim, config.rms_norm_eps)
        self.kv_norm = RMSNorm(config.attention_head_dim, config.rms_norm_eps)
        self.output = GroupedOutputProjection(config)
        self.window_kv = nn.Linear(config.hidden_size, config.attention_head_dim, bias=False)
        self.sink = nn.Parameter(torch.zeros(config.num_attention_heads))

    def _token_mask(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones(hidden_states.size(0), hidden_states.size(1), device=hidden_states.device, dtype=torch.bool)
        return attention_mask.to(torch.bool)

    def _queries(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.q_down(hidden_states)
        q = self.q_up(latent)
        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.config.num_attention_heads, self.config.attention_head_dim)
        positions = torch.arange(seq_len, device=hidden_states.device)
        q = apply_rope(q, positions, self.config.rope_dim, self.config.rope_theta)
        return self.q_norm(q), latent

    def _window_entries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden_states.size(1), device=hidden_states.device)
        kv = self.window_kv(hidden_states)
        kv = apply_rope(kv, positions, self.config.rope_dim, self.config.rope_theta)
        return self.kv_norm(kv)

    def _finish(self, q: torch.Tensor, kv_per_token: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for t, (kv, kv_mask) in enumerate(kv_per_token):
            if kv.numel() == 0:
                outs.append(torch.zeros(q.size(0), q.size(2), q.size(3), device=q.device, dtype=q.dtype))
                continue
            out = sink_attention(
                q[:, t],
                kv,
                kv,
                self.sink,
                self.config.attention_dropout,
                self.training,
                kv_mask=kv_mask,
            )
            neg_pos = torch.full((q.size(0), q.size(2)), -t, device=q.device, dtype=torch.long)
            out = apply_rope(out, neg_pos, self.config.rope_dim, self.config.rope_theta)
            outs.append(out)
        return self.output(torch.stack(outs, dim=1))


class HCAAttention(CompressedAttentionBase):
    def __init__(self, config: DeepSeekV4Pro2BConfig):
        super().__init__(config)
        self.kv_proj = nn.Linear(config.hidden_size, config.attention_head_dim, bias=False)
        self.z_proj = nn.Linear(config.hidden_size, config.attention_head_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.hca_compression, config.attention_head_dim))

    def _compress(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.config.hca_compression
        full = hidden_states.size(1) // m
        if full == 0:
            empty = hidden_states.new_empty(hidden_states.size(0), 0, self.config.attention_head_dim)
            return empty, empty
        h = hidden_states[:, : full * m]
        c = self.kv_proj(h).view(h.size(0), full, m, -1)
        z = self.z_proj(h).view(h.size(0), full, m, -1) + self.bias.view(1, 1, m, -1)
        weights = torch.softmax(z.float(), dim=2).to(c.dtype)
        comp = (weights * c).sum(dim=2)
        positions = (torch.arange(full, device=h.device) * m + (m - 1)).long()
        comp = apply_rope(comp, positions, self.config.rope_dim, self.config.rope_theta)
        return self.kv_norm(comp), positions

    # Chunk size for the T-dimension during prefill.  Keeps peak memory bounded
    # for long contexts by avoiding materialising [B, T, H, nb] all at once.
    _PREFILL_CHUNK: int = 256

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, _ = self._queries(hidden_states)            # [B, T, H, D]
        comp, _ = self._compress(hidden_states)        # [B, nb, D]
        window = self._window_entries(hidden_states)   # [B, T, D]
        token_mask = self._token_mask(hidden_states, attention_mask)  # [B, T]

        B, T, H, D = q.shape
        M = self.config.hca_compression
        W = self.config.sliding_window
        nb = comp.size(1)
        scale = D ** -0.5
        t_idx = torch.arange(T, device=hidden_states.device)
        w_idx = torch.arange(W, device=hidden_states.device)

        # Left-pad window once for sliding-window extraction.
        pad_w = F.pad(window, (0, 0, W - 1, 0))       # [B, T+W-1, D]

        if nb > 0:
            block_idx = torch.arange(nb, device=hidden_states.device)
            full_blocks = token_mask.sum(dim=-1) // M  # [B]
            # valid_g: [B, 1, nb]  (independent of t — computed once)
            valid_g = block_idx.view(1, 1, nb) < full_blocks.view(B, 1, 1)
        else:
            block_idx = None
            valid_g = None

        sink_v = self.sink.view(1, 1, H, 1)
        out_chunks: List[torch.Tensor] = []

        for t_start in range(0, T, self._PREFILL_CHUNK):
            t_end = min(t_start + self._PREFILL_CHUNK, T)
            t_ch = t_idx[t_start:t_end]           # [C]
            C = t_end - t_start
            q_ch = q[:, t_start:t_end]            # [B, C, H, D]

            # ── sliding window for this chunk: [B, C, W, D] ──────────────
            win_ch = pad_w[:, t_start:t_start + C + W - 1, :].unfold(1, W, 1).permute(0, 1, 3, 2)
            actual_pos_ch = t_ch.view(C, 1) - (W - 1) + w_idx.view(1, W)  # [C, W]
            basic_v = (actual_pos_ch >= 0) & (actual_pos_ch < T)
            win_mask_ch = basic_v.unsqueeze(0) & token_mask[:, actual_pos_ch.clamp(0, T - 1)]

            # ── global logits for this chunk ──────────────────────────────
            if nb > 0:
                causal_g_ch = block_idx.view(1, 1, nb) < (t_ch // M).view(1, C, 1)  # [1,C,nb]
                gm_ch = causal_g_ch & valid_g                                        # [B,C,nb]
                g_logits = torch.einsum("bthd,bsd->bths", q_ch, comp) * scale
                g_logits = g_logits.masked_fill(~gm_ch.unsqueeze(2), float("-inf"))
            else:
                g_logits = q_ch.new_full((B, C, H, 0), float("-inf"))

            # ── window logits for this chunk ──────────────────────────────
            w_logits = torch.einsum("bthd,btwd->bthw", q_ch, win_ch) * scale
            w_logits = w_logits.masked_fill(~win_mask_ch.unsqueeze(2), float("-inf"))

            # ── softmax with sink ─────────────────────────────────────────
            all_l = torch.cat([g_logits, w_logits], dim=-1)
            sv = sink_v.to(all_l.dtype)
            mx = torch.maximum(all_l.amax(dim=-1, keepdim=True), sv)
            exp_l = torch.exp(all_l - mx)
            wts = exp_l / (exp_l.sum(dim=-1, keepdim=True) + torch.exp(sv - mx))

            # ── weighted sum ──────────────────────────────────────────────
            out_c = (torch.einsum("bths,bsd->bthd", wts[:, :, :, :nb], comp)
                     if nb > 0 else q_ch.new_zeros(B, C, H, D))
            out_c = out_c + torch.einsum("bthw,btwd->bthd", wts[:, :, :, nb:], win_ch)
            # Inverse RoPE per chunk (position -t for each t in chunk)
            out_c = apply_rope(out_c, -t_ch, self.config.rope_dim, self.config.rope_theta)
            out_chunks.append(out_c)

        return self.output(torch.cat(out_chunks, dim=1))


class CSAAttention(CompressedAttentionBase):
    def __init__(self, config: DeepSeekV4Pro2BConfig):
        super().__init__(config)
        c = config.attention_head_dim
        self.kv_a = nn.Linear(config.hidden_size, c, bias=False)
        self.kv_b = nn.Linear(config.hidden_size, c, bias=False)
        self.z_a = nn.Linear(config.hidden_size, c, bias=False)
        self.z_b = nn.Linear(config.hidden_size, c, bias=False)
        self.bias_a = nn.Parameter(torch.zeros(config.csa_compression, c))
        self.bias_b = nn.Parameter(torch.zeros(config.csa_compression, c))
        self.index_kv_a = nn.Linear(config.hidden_size, config.indexer_head_dim, bias=False)
        self.index_kv_b = nn.Linear(config.hidden_size, config.indexer_head_dim, bias=False)
        self.index_z_a = nn.Linear(config.hidden_size, config.indexer_head_dim, bias=False)
        self.index_z_b = nn.Linear(config.hidden_size, config.indexer_head_dim, bias=False)
        self.index_bias_a = nn.Parameter(torch.zeros(config.csa_compression, config.indexer_head_dim))
        self.index_bias_b = nn.Parameter(torch.zeros(config.csa_compression, config.indexer_head_dim))
        self.index_q_up = nn.Linear(
            config.query_compression_dim,
            config.indexer_num_heads * config.indexer_head_dim,
            bias=False,
        )
        self.index_weight = nn.Linear(config.hidden_size, config.indexer_num_heads, bias=False)

    def _overlap_compress(
        self,
        ca: torch.Tensor,
        cb: torch.Tensor,
        za: torch.Tensor,
        zb: torch.Tensor,
        bias_a: torch.Tensor,
        bias_b: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, dim = ca.shape
        m = self.config.csa_compression
        blocks = seq_len // m
        out = []
        zero_c = ca.new_zeros(bsz, m, dim)
        neg_z = za.new_full((bsz, m, dim), -torch.inf)
        for i in range(blocks):
            a_slice = slice(i * m, (i + 1) * m)
            if i == 0:
                c_cat = torch.cat([ca[:, a_slice], zero_c], dim=1)
                z_cat = torch.cat([za[:, a_slice] + bias_a, neg_z], dim=1)
            else:
                b_slice = slice((i - 1) * m, i * m)
                c_cat = torch.cat([ca[:, a_slice], cb[:, b_slice]], dim=1)
                z_cat = torch.cat([za[:, a_slice] + bias_a, zb[:, b_slice] + bias_b], dim=1)
            weights = torch.softmax(z_cat.float(), dim=1).to(c_cat.dtype)
            out.append((weights * c_cat).sum(dim=1))
        if not out:
            return ca.new_empty(bsz, 0, dim)
        return torch.stack(out, dim=1)

    def _compress_main(self, hidden_states: torch.Tensor) -> torch.Tensor:
        comp = self._overlap_compress(
            self.kv_a(hidden_states),
            self.kv_b(hidden_states),
            self.z_a(hidden_states),
            self.z_b(hidden_states),
            self.bias_a,
            self.bias_b,
        )
        positions = torch.arange(comp.size(1), device=comp.device) * self.config.csa_compression
        comp = apply_rope(comp, positions, self.config.rope_dim, self.config.rope_theta)
        return self.kv_norm(comp)

    def _compress_index(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._overlap_compress(
            self.index_kv_a(hidden_states),
            self.index_kv_b(hidden_states),
            self.index_z_a(hidden_states),
            self.index_z_b(hidden_states),
            self.index_bias_a,
            self.index_bias_b,
        )

    _PREFILL_CHUNK: int = 256

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, latent = self._queries(hidden_states)          # [B, T, H, D]
        comp = self._compress_main(hidden_states)         # [B, nb, D]
        index_keys = self._compress_index(hidden_states)  # [B, nb, IC]
        index_q = self.index_q_up(latent).view(
            hidden_states.size(0), hidden_states.size(1),
            self.config.indexer_num_heads, self.config.indexer_head_dim,
        )                                                  # [B, T, IH, IC]
        index_w = self.index_weight(hidden_states)         # [B, T, IH]
        window = self._window_entries(hidden_states)       # [B, T, D]
        token_mask = self._token_mask(hidden_states, attention_mask)  # [B, T]

        B, T, H, D = q.shape
        M = self.config.csa_compression
        W = self.config.sliding_window
        nb = comp.size(1)
        scale = D ** -0.5
        t_idx = torch.arange(T, device=hidden_states.device)
        w_idx = torch.arange(W, device=hidden_states.device)

        pad_w = F.pad(window, (0, 0, W - 1, 0))  # [B, T+W-1, D]

        # Precompute valid_g: [B, 1, nb] — which blocks exist (padding guard).
        if nb > 0:
            block_idx = torch.arange(nb, device=hidden_states.device)
            full_blocks = token_mask.sum(dim=-1) // M     # [B]
            valid_g = block_idx.view(1, 1, nb) < full_blocks.view(B, 1, 1)  # [B, 1, nb]
        else:
            block_idx = None
            valid_g = None

        K = min(self.config.csa_top_k, nb) if nb > 0 else 0
        sink_v = self.sink.view(1, 1, H, 1)
        out_chunks: List[torch.Tensor] = []

        for t_start in range(0, T, self._PREFILL_CHUNK):
            t_end = min(t_start + self._PREFILL_CHUNK, T)
            t_ch = t_idx[t_start:t_end]     # [C]
            C = t_end - t_start
            q_ch = q[:, t_start:t_end]      # [B, C, H, D]

            # ── sliding window for this chunk ─────────────────────────────
            win_ch = pad_w[:, t_start:t_start + C + W - 1, :].unfold(1, W, 1).permute(0, 1, 3, 2)
            actual_pos_ch = t_ch.view(C, 1) - (W - 1) + w_idx.view(1, W)   # [C, W]
            basic_v = (actual_pos_ch >= 0) & (actual_pos_ch < T)
            win_mask_ch = basic_v.unsqueeze(0) & token_mask[:, actual_pos_ch.clamp(0, T - 1)]

            # ── top-k selection for this chunk ────────────────────────────
            if K > 0:
                # Score: [B, C, IH, nb] → reduce IH → [B, C, nb]
                s = torch.einsum("btic,bsc->btis",
                                 index_q[:, t_start:t_end], index_keys)   # [B,C,IH,nb]
                s = (F.relu(s) * index_w[:, t_start:t_end].unsqueeze(-1)).sum(dim=2)  # [B,C,nb]
                causal_g_ch = block_idx.view(1, 1, nb) < (t_ch // M).view(1, C, 1)
                bm_ch = causal_g_ch & valid_g                             # [B, C, nb]
                s = s.masked_fill(~bm_ch, float("-inf"))
                topk_ch = torch.topk(s, k=K, dim=-1).indices              # [B, C, K]
                sel_mask_ch = bm_ch.gather(2, topk_ch)                    # [B, C, K]
                b_arange = torch.arange(B, device=comp.device).view(B, 1, 1).expand(B, C, K)
                sel_kv_ch = comp[b_arange, topk_ch]                       # [B, C, K, D]

                g_logits = torch.einsum("bthd,btkd->bthk", q_ch, sel_kv_ch) * scale
                g_logits = g_logits.masked_fill(~sel_mask_ch.unsqueeze(2), float("-inf"))
            else:
                g_logits = q_ch.new_full((B, C, H, 0), float("-inf"))
                sel_kv_ch = comp.new_empty(B, C, 0, D)

            # ── window logits ─────────────────────────────────────────────
            w_logits = torch.einsum("bthd,btwd->bthw", q_ch, win_ch) * scale
            w_logits = w_logits.masked_fill(~win_mask_ch.unsqueeze(2), float("-inf"))

            # ── softmax with sink ─────────────────────────────────────────
            all_l = torch.cat([g_logits, w_logits], dim=-1)               # [B,C,H,K+W]
            sv = sink_v.to(all_l.dtype)
            mx = torch.maximum(all_l.amax(dim=-1, keepdim=True), sv)
            exp_l = torch.exp(all_l - mx)
            wts = exp_l / (exp_l.sum(dim=-1, keepdim=True) + torch.exp(sv - mx))

            # ── weighted sum ──────────────────────────────────────────────
            out_c = (torch.einsum("bthk,btkd->bthd", wts[:, :, :, :K], sel_kv_ch)
                     if K > 0 else q_ch.new_zeros(B, C, H, D))
            out_c = out_c + torch.einsum("bthw,btwd->bthd", wts[:, :, :, K:], win_ch)
            out_c = apply_rope(out_c, -t_ch, self.config.rope_dim, self.config.rope_theta)
            out_chunks.append(out_c)

        return self.output(torch.cat(out_chunks, dim=1))


class ManifoldConstrainedHyperConnection(nn.Module):
    def __init__(self, config: DeepSeekV4Pro2BConfig, sublayer: nn.Module):
        super().__init__()
        self.config = config
        self.sublayer = sublayer
        n = config.mhc_expansion
        d = config.hidden_size
        flat = n * d
        self.norm = RMSNorm(flat, config.rms_norm_eps)
        self.w_pre = nn.Linear(flat, n, bias=False)
        self.w_res = nn.Linear(flat, n * n, bias=False)
        self.w_post = nn.Linear(flat, n, bias=False)
        self.s_pre = nn.Parameter(torch.zeros(n))
        self.s_res = nn.Parameter(torch.eye(n))
        self.s_post = nn.Parameter(torch.zeros(n))
        self.alpha_pre = nn.Parameter(torch.tensor(1e-3))
        self.alpha_res = nn.Parameter(torch.tensor(1e-3))
        self.alpha_post = nn.Parameter(torch.tensor(1e-3))

    def _sinkhorn(self, raw: torch.Tensor) -> torch.Tensor:
        m = torch.exp(raw.clamp(min=-20.0, max=20.0))
        for _ in range(self.config.mhc_sinkhorn_iters):
            m = m / m.sum(dim=-2, keepdim=True).clamp_min(1e-12)
            m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return m

    def forward(
        self,
        state: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, n, d = state.shape
        flat = self.norm(state.reshape(bsz, seq_len, n * d))
        a_raw = self.alpha_pre * self.w_pre(flat) + self.s_pre
        b_raw = self.alpha_res * self.w_res(flat).view(bsz, seq_len, n, n) + self.s_res
        c_raw = self.alpha_post * self.w_post(flat) + self.s_post
        a = torch.sigmoid(a_raw)
        b = self._sinkhorn(b_raw)
        c = 2.0 * torch.sigmoid(c_raw)
        x = torch.einsum("btn,btnd->btd", a, state)
        if isinstance(self.sublayer, DeepSeekMoE):
            y, balance_loss = self.sublayer(x, token_ids=token_ids, token_mask=attention_mask)
        else:
            y = self.sublayer(x, attention_mask=attention_mask)
            balance_loss = None
        mixed = torch.einsum("btij,btjd->btid", b, state)
        updated = mixed + c.unsqueeze(-1) * y.unsqueeze(2)
        return updated, balance_loss

    def chunked_forward(
        self,
        state: torch.Tensor,
        chunk_size: int,
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Memory-efficient forward that chunks the T dimension for the mHC.

        The mHC pre-mix (norm, a, b, c computations) and post-mix (mixed residual
        + update) are per-position: position t's a, b, c depend only on state[:, t].
        These can be computed in T-dimension chunks of size ``chunk_size``, avoiding
        materialising the full ``[B, T, n*D]`` flat tensor (which is the OOM source
        at 262K+ context on 24 GB GPUs).

        The sublayer (attention or MoE) still receives the full ``[B, T, D]`` input
        ``x``, because:
          - Attention is causal and needs the full context (it already chunks
            internally via ``_PREFILL_CHUNK``).
          - MoE processes each position independently — chunking would be possible
            but unnecessary since ``[B, T, D]`` is much smaller than ``[B, T, n, D]``.

        Memory reduction:
          - Standard forward peak: ``[B, T, n*D]`` (flat) + ``[B, T, n, n]`` (b_raw)
            + ``[B, T, n, D]`` (mixed) ≈ 3 × [B, T, n, D]
          - Chunked peak: ``[B, C, n*D]`` (flat_c) + ``[B, T, D]`` (x accumulator)
            + ``[B, C, n, D]`` (mixed_c) ≈ [B, C, n, D] + [B, T, D]
          At T=262144, n=4, D=1536, C=4096: standard = ~18 GB, chunked = ~0.8 GB

        Parameters
        ----------
        state : [B, T, n, D]
            mHC state tensor from the previous layer / embedding.
        chunk_size : int
            Number of T-positions to process at once for the mHC operations.
            Typical values: 2048–8192.  Attention chunk size is separate
            (``_PREFILL_CHUNK`` on the sublayer).
        """
        bsz, seq_len, n, d = state.shape

        if seq_len <= chunk_size:
            return self.forward(state, token_ids=token_ids, attention_mask=attention_mask)

        # Phase 1: compute x = einsum("btn,btnd->btd", a, state) for each chunk.
        # x is [B, T, D] — much smaller than state [B, T, n, D]; safe to materialise fully.
        x_parts: List[torch.Tensor] = []
        for t_start in range(0, seq_len, chunk_size):
            t_end = min(t_start + chunk_size, seq_len)
            state_c = state[:, t_start:t_end]              # [B, C, n, D] — view, no copy
            flat_c = self.norm(state_c.reshape(bsz, t_end - t_start, n * d))
            a_raw_c = self.alpha_pre * self.w_pre(flat_c) + self.s_pre
            a_c = torch.sigmoid(a_raw_c)                   # [B, C, n]
            x_c = torch.einsum("btn,btnd->btd", a_c, state_c)  # [B, C, D]
            x_parts.append(x_c)
            # flat_c, a_raw_c, a_c freed at end of loop iteration

        x = torch.cat(x_parts, dim=1)  # [B, T, D]
        del x_parts

        # Phase 2: run the sublayer on the full x — attention is causal over all T.
        if isinstance(self.sublayer, DeepSeekMoE):
            y, balance_loss = self.sublayer(x, token_ids=token_ids, token_mask=attention_mask)
        else:
            y = self.sublayer(x, attention_mask=attention_mask)
            balance_loss = None
        del x  # [B, T, D] freed

        # Phase 3: compute the residual update in chunks.
        #   mixed_c = einsum("btij,btjd->btid", b_c, state_c)
        #   updated_c = mixed_c + c_c.unsqueeze(-1) * y_c.unsqueeze(2)
        updated_parts: List[torch.Tensor] = []
        for t_start in range(0, seq_len, chunk_size):
            t_end = min(t_start + chunk_size, seq_len)
            state_c = state[:, t_start:t_end]              # [B, C, n, D] — view
            flat_c = self.norm(state_c.reshape(bsz, t_end - t_start, n * d))
            b_raw_c = self.alpha_res * self.w_res(flat_c).view(bsz, t_end - t_start, n, n) + self.s_res
            c_raw_c = self.alpha_post * self.w_post(flat_c) + self.s_post
            b_c = self._sinkhorn(b_raw_c)
            c_c = 2.0 * torch.sigmoid(c_raw_c)
            y_c = y[:, t_start:t_end]                      # [B, C, D] — view
            mixed_c = torch.einsum("btij,btjd->btid", b_c, state_c)  # [B, C, n, D]
            updated_c = mixed_c + c_c.unsqueeze(-1) * y_c.unsqueeze(2)
            updated_parts.append(updated_c)
            # flat_c, b_raw_c, c_raw_c, b_c, c_c, mixed_c freed

        del y  # [B, T, D] freed
        updated = torch.cat(updated_parts, dim=1)  # [B, T, n, D]
        del updated_parts

        return updated, balance_loss


class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekV4Pro2BConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.router = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(config.num_routed_experts), persistent=False)
        self.experts = nn.ModuleList(
            SwiGLU(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.num_routed_experts)
        )
        self.shared_experts = nn.ModuleList(
            SwiGLU(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.num_shared_experts)
        )

    def _hash_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.config.num_experts_per_tok, device=token_ids.device)
        return (token_ids.unsqueeze(-1) + offsets) % self.config.num_routed_experts

    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        flat_mask = None if token_mask is None else token_mask.reshape(-1).to(torch.bool)
        if self.layer_idx < self.config.hash_routed_layers and token_ids is not None:
            top_idx = self._hash_indices(token_ids.reshape(-1))
            top_weight = torch.full_like(top_idx, 1.0 / self.config.num_experts_per_tok, dtype=x.dtype)
            probs = F.one_hot(top_idx, self.config.num_routed_experts).float().mean(dim=1)
        else:
            logits = self.router(flat) + self.router_bias
            affinity = torch.sqrt(F.softplus(logits.float()))
            top_weight, top_idx = torch.topk(affinity, self.config.num_experts_per_tok, dim=-1)
            top_weight = (top_weight / top_weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(x.dtype)
            probs = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        out = flat.new_zeros(flat.shape)
        for expert_idx, expert in enumerate(self.experts):
            mask = top_idx == expert_idx
            if not mask.any():
                continue
            rows, slots = mask.nonzero(as_tuple=True)
            out.index_add_(0, rows, expert(flat.index_select(0, rows)) * top_weight[rows, slots].unsqueeze(-1))
        for expert in self.shared_experts:
            out = out + expert(flat)
        if flat_mask is not None:
            out = out * flat_mask.unsqueeze(-1).to(out.dtype)
            if flat_mask.any():
                load = probs[flat_mask].mean(dim=0)
            else:
                return out.view(bsz, seq_len, hidden), out.new_tensor(0.0)
        else:
            load = probs.mean(dim=0)
        balance_loss = self.config.num_routed_experts * (load * load).sum() - 1.0
        return out.view(bsz, seq_len, hidden), balance_loss


class DeepSeekV4Block(nn.Module):
    def __init__(self, config: DeepSeekV4Pro2BConfig, layer_idx: int):
        super().__init__()
        attn_cls = CSAAttention if config.attention_type(layer_idx) == "csa" else HCAAttention
        self.attn = ManifoldConstrainedHyperConnection(config, attn_cls(config))
        self.moe = ManifoldConstrainedHyperConnection(config, DeepSeekMoE(config, layer_idx))

    def forward(
        self,
        state: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state, _ = self.attn(state, token_ids=token_ids, attention_mask=attention_mask)
        state, balance_loss = self.moe(state, token_ids=token_ids, attention_mask=attention_mask)
        return state, balance_loss if balance_loss is not None else state.new_tensor(0.0)

    def chunked_forward(
        self,
        state: torch.Tensor,
        chunk_size: int,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory-efficient forward using chunked mHC operations."""
        state, _ = self.attn.chunked_forward(state, chunk_size, token_ids=token_ids, attention_mask=attention_mask)
        state, balance_loss = self.moe.chunked_forward(state, chunk_size, token_ids=token_ids, attention_mask=attention_mask)
        return state, balance_loss if balance_loss is not None else state.new_tensor(0.0)


class DeepSeekV4Pro2BModel(nn.Module):
    def __init__(self, config: DeepSeekV4Pro2BConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(DeepSeekV4Block(config, i) for i in range(config.num_hidden_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post = nn.Parameter(torch.ones(config.mhc_expansion) / config.mhc_expansion)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        state = hidden.unsqueeze(2).expand(-1, -1, self.config.mhc_expansion, -1).contiguous()
        balance_losses = []
        _layer_devs = getattr(self, '_layer_devices', None)
        for i, layer in enumerate(self.layers):
            # Multi-GPU support: move state + inputs to the layer's device.
            if _layer_devs is not None:
                dev = _layer_devs[i]
                if state.device != dev:
                    state = state.to(dev)
                    input_ids = input_ids.to(dev)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(dev)
            state, balance = layer(state, token_ids=input_ids, attention_mask=attention_mask)
            if attention_mask is not None:
                state = state * attention_mask[:, :, None, None].to(state.dtype)
            balance_losses.append(balance)
        # Return state to primary device (embed_tokens device) for final ops.
        primary = self.embed_tokens.weight.device
        if state.device != primary:
            state = state.to(primary)
        weights = torch.softmax(self.post, dim=0)
        hidden = torch.einsum("n,btnd->btd", weights, state)
        return self.final_norm(hidden), torch.stack([b.to(primary) for b in balance_losses]).mean()

    def chunked_forward(
        self,
        input_ids: torch.Tensor,
        mhc_chunk_size: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory-efficient forward that chunks mHC operations along the T dimension.

        Identical to ``forward()`` except each ``DeepSeekV4Block`` uses
        ``chunked_forward(chunk_size=mhc_chunk_size)`` instead of ``forward()``.
        This keeps peak memory at ``O(chunk_size × n_expand × D)`` per layer
        instead of ``O(T × n_expand × D)``.

        At T=262K, n_expand=4, D=1536 (full 2B model):
          - Standard forward peak: 2 × 1 × 262144 × 4 × 1536 × 2B ≈ 25.2 GB
          - Chunked (C=4096) peak: 2 × 1 × 4096 × 4 × 1536 × 2B + O(T×D) ≈ 1.2 GB
        """
        hidden = self.embed_tokens(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        state = hidden.unsqueeze(2).expand(-1, -1, self.config.mhc_expansion, -1).contiguous()
        balance_losses = []
        _layer_devs = getattr(self, '_layer_devices', None)
        for i, layer in enumerate(self.layers):
            if _layer_devs is not None:
                dev = _layer_devs[i]
                if state.device != dev:
                    state = state.to(dev)
                    input_ids = input_ids.to(dev)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(dev)
            state, balance = layer.chunked_forward(
                state, mhc_chunk_size, token_ids=input_ids, attention_mask=attention_mask
            )
            if attention_mask is not None:
                state = state * attention_mask[:, :, None, None].to(state.dtype)
            balance_losses.append(balance)
        primary = self.embed_tokens.weight.device
        if state.device != primary:
            state = state.to(primary)
        weights = torch.softmax(self.post, dim=0)
        hidden = torch.einsum("n,btnd->btd", weights, state)
        return self.final_norm(hidden), torch.stack([b.to(primary) for b in balance_losses]).mean()


class MTPHead(nn.Module):
    def __init__(self, config: DeepSeekV4Pro2BConfig, lm_head: nn.Linear):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.proj(self.norm(hidden_states)))


class DeepSeekV4Pro2BForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekV4Pro2BConfig):
        super().__init__()
        self.config = config
        self.model = DeepSeekV4Pro2BModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.mtp_heads = nn.ModuleList(MTPHead(config, self.lm_head) for _ in range(config.mtp_depth))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> DeepSeekV4Output:
        hidden, balance_loss = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)
        mtp_logits = [head(hidden) for head in self.mtp_heads]
        loss = None
        if labels is not None:
            shift_labels = labels[:, 1:].clone()
            if attention_mask is not None:
                shift_labels = shift_labels.masked_fill(~attention_mask[:, 1:].to(torch.bool), -100)
            loss = masked_cross_entropy(logits[:, :-1], shift_labels)
            for depth, extra_logits in enumerate(mtp_logits, start=2):
                if input_ids.size(1) > depth:
                    mtp_labels = labels[:, depth:].clone()
                    if attention_mask is not None:
                        mtp_labels = mtp_labels.masked_fill(~attention_mask[:, depth:].to(torch.bool), -100)
                    mtp_loss = masked_cross_entropy(extra_logits[:, :-depth], mtp_labels)
                    loss = loss + self.config.mtp_loss_weight * mtp_loss
            loss = loss + self.config.balance_loss_weight * balance_loss
        return DeepSeekV4Output(logits=logits, loss=loss, mtp_logits=mtp_logits, balance_loss=balance_loss)

    @torch.no_grad()
    def estimate_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
