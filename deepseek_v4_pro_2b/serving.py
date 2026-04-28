from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from deepseek_kernels.paged_kv_allocator import PageHandle, PagedKVAllocator
from deepseek_pipeline.serving import (
    AttentionKernelBackend,
    CudaSparseAttentionBackend,
    LongContextServingManager,
    PytorchAttentionBackend,
    SWACacheMode,
)

from .configuration import DeepSeekV4Pro2BConfig
from .modeling import CSAAttention, HCAAttention, DeepSeekV4Pro2BForCausalLM, apply_rope, rope_attention_scale
from .turbo_quant import PolarQuant


def _clone_optional(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if x is None else x.detach().clone()


def _cpu_optional(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if x is None else x.detach().cpu()


def _append_seq(existing: Optional[torch.Tensor], new_value: torch.Tensor) -> torch.Tensor:
    if new_value.ndim == 2:
        new_value = new_value.unsqueeze(1)
    if existing is None:
        return new_value
    return torch.cat([existing, new_value], dim=1)


def _tail(existing: Optional[torch.Tensor], length: int) -> Optional[torch.Tensor]:
    if existing is None or length <= 0:
        return None
    return existing[:, -length:]


def _trim(existing: Optional[torch.Tensor], max_len: int) -> Optional[torch.Tensor]:
    if existing is None or existing.size(1) <= max_len:
        return existing
    return existing[:, -max_len:]


def _pack_shared_kv(values: torch.Tensor) -> torch.Tensor:
    return torch.stack([values, values], dim=2)


def _full_indices(batch_size: int, source_tokens: int, device: torch.device) -> torch.Tensor:
    return torch.arange(source_tokens, device=device, dtype=torch.long).view(1, 1, source_tokens).expand(batch_size, 1, source_tokens)


@dataclass
class PagedPrefixState:
    handle: PageHandle
    page_count: int


@dataclass
class HCAServingState:
    token_count: int = 0
    loaded_prefix_tokens: int = 0
    compressed: Optional[torch.Tensor] = None
    # PolarQuant: per-vector float16 scale for compressed; None = unquantized (bf16).
    compressed_scale: Optional[torch.Tensor] = None
    window: Optional[torch.Tensor] = None
    buffer_c: Optional[torch.Tensor] = None
    buffer_z: Optional[torch.Tensor] = None

    def clone(self) -> "HCAServingState":
        return HCAServingState(
            token_count=self.token_count,
            loaded_prefix_tokens=self.loaded_prefix_tokens,
            compressed=_clone_optional(self.compressed),
            compressed_scale=_clone_optional(self.compressed_scale),
            window=_clone_optional(self.window),
            buffer_c=_clone_optional(self.buffer_c),
            buffer_z=_clone_optional(self.buffer_z),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": "hca",
            "token_count": self.token_count,
            "loaded_prefix_tokens": self.loaded_prefix_tokens,
            "compressed": _cpu_optional(self.compressed),
            "compressed_scale": _cpu_optional(self.compressed_scale),
            "window": _cpu_optional(self.window),
            "buffer_c": _cpu_optional(self.buffer_c),
            "buffer_z": _cpu_optional(self.buffer_z),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object], device: torch.device) -> "HCAServingState":
        return cls(
            token_count=int(payload["token_count"]),
            loaded_prefix_tokens=int(payload.get("loaded_prefix_tokens", 0)),
            compressed=None if payload.get("compressed") is None else payload["compressed"].to(device),
            compressed_scale=None if payload.get("compressed_scale") is None else payload["compressed_scale"].to(device),
            window=None if payload.get("window") is None else payload["window"].to(device),
            buffer_c=None if payload.get("buffer_c") is None else payload["buffer_c"].to(device),
            buffer_z=None if payload.get("buffer_z") is None else payload["buffer_z"].to(device),
        )


@dataclass
class CSAServingState:
    token_count: int = 0
    loaded_prefix_tokens: int = 0
    compressed: Optional[torch.Tensor] = None
    # PolarQuant scale tensors; None = unquantized (bf16).
    compressed_scale: Optional[torch.Tensor] = None
    index_compressed: Optional[torch.Tensor] = None
    index_compressed_scale: Optional[torch.Tensor] = None
    window: Optional[torch.Tensor] = None
    curr_a_c: Optional[torch.Tensor] = None
    curr_a_z: Optional[torch.Tensor] = None
    curr_b_c: Optional[torch.Tensor] = None
    curr_b_z: Optional[torch.Tensor] = None
    curr_index_a_c: Optional[torch.Tensor] = None
    curr_index_a_z: Optional[torch.Tensor] = None
    curr_index_b_c: Optional[torch.Tensor] = None
    curr_index_b_z: Optional[torch.Tensor] = None
    prev_b_c: Optional[torch.Tensor] = None
    prev_b_z: Optional[torch.Tensor] = None
    prev_index_b_c: Optional[torch.Tensor] = None
    prev_index_b_z: Optional[torch.Tensor] = None

    def clone(self) -> "CSAServingState":
        return CSAServingState(
            token_count=self.token_count,
            loaded_prefix_tokens=self.loaded_prefix_tokens,
            compressed=_clone_optional(self.compressed),
            compressed_scale=_clone_optional(self.compressed_scale),
            index_compressed=_clone_optional(self.index_compressed),
            index_compressed_scale=_clone_optional(self.index_compressed_scale),
            window=_clone_optional(self.window),
            curr_a_c=_clone_optional(self.curr_a_c),
            curr_a_z=_clone_optional(self.curr_a_z),
            curr_b_c=_clone_optional(self.curr_b_c),
            curr_b_z=_clone_optional(self.curr_b_z),
            curr_index_a_c=_clone_optional(self.curr_index_a_c),
            curr_index_a_z=_clone_optional(self.curr_index_a_z),
            curr_index_b_c=_clone_optional(self.curr_index_b_c),
            curr_index_b_z=_clone_optional(self.curr_index_b_z),
            prev_b_c=_clone_optional(self.prev_b_c),
            prev_b_z=_clone_optional(self.prev_b_z),
            prev_index_b_c=_clone_optional(self.prev_index_b_c),
            prev_index_b_z=_clone_optional(self.prev_index_b_z),
        )

    def to_dict(self) -> Dict[str, object]:
        fields = {
            "type": "csa",
            "token_count": self.token_count,
            "loaded_prefix_tokens": self.loaded_prefix_tokens,
        }
        for name in (
            "compressed",
            "compressed_scale",
            "index_compressed",
            "index_compressed_scale",
            "window",
            "curr_a_c",
            "curr_a_z",
            "curr_b_c",
            "curr_b_z",
            "curr_index_a_c",
            "curr_index_a_z",
            "curr_index_b_c",
            "curr_index_b_z",
            "prev_b_c",
            "prev_b_z",
            "prev_index_b_c",
            "prev_index_b_z",
        ):
            fields[name] = _cpu_optional(getattr(self, name))
        return fields

    @classmethod
    def from_dict(cls, payload: Dict[str, object], device: torch.device) -> "CSAServingState":
        kwargs: Dict[str, object] = {
            "token_count": int(payload["token_count"]),
            "loaded_prefix_tokens": int(payload.get("loaded_prefix_tokens", 0)),
        }
        for name in (
            "compressed",
            "compressed_scale",
            "index_compressed",
            "index_compressed_scale",
            "window",
            "curr_a_c",
            "curr_a_z",
            "curr_b_c",
            "curr_b_z",
            "curr_index_a_c",
            "curr_index_a_z",
            "curr_index_b_c",
            "curr_index_b_z",
            "prev_b_c",
            "prev_b_z",
            "prev_index_b_c",
            "prev_index_b_z",
        ):
            value = payload.get(name)
            kwargs[name] = None if value is None else value.to(device)
        return cls(**kwargs)


LayerServingState = Union[HCAServingState, CSAServingState]


@dataclass
class ModelServingState:
    token_count: int
    layer_states: List[LayerServingState]
    last_logits: Optional[torch.Tensor] = None
    paged_prefix: Optional[PagedPrefixState] = None

    def clone(self) -> "ModelServingState":
        return ModelServingState(
            token_count=self.token_count,
            layer_states=[layer.clone() for layer in self.layer_states],
            last_logits=_clone_optional(self.last_logits),
            paged_prefix=self.paged_prefix,
        )

    def to_swa_dict(self) -> Dict[str, object]:
        return {
            "token_count": self.token_count,
            "last_logits": _cpu_optional(self.last_logits),
            "layers": [layer.to_dict() for layer in self.layer_states],
        }

    def to_compressed_dict(self, reusable_tokens: int) -> Dict[str, object]:
        layers: List[Dict[str, object]] = []
        for layer in self.layer_states:
            if isinstance(layer, HCAServingState):
                layers.append(
                    {
                        "type": "hca",
                        "compressed": _cpu_optional(layer.compressed),
                        "compressed_scale": _cpu_optional(layer.compressed_scale),
                    }
                )
            else:
                layers.append(
                    {
                        "type": "csa",
                        "compressed": _cpu_optional(layer.compressed),
                        "compressed_scale": _cpu_optional(layer.compressed_scale),
                        "index_compressed": _cpu_optional(layer.index_compressed),
                        "index_compressed_scale": _cpu_optional(layer.index_compressed_scale),
                    }
                )
        return {"token_count": reusable_tokens, "layers": layers}


class DeepSeekV4Pro2BServingEngine:
    def __init__(
        self,
        model: DeepSeekV4Pro2BForCausalLM,
        backend: Optional[AttentionKernelBackend] = None,
        prefix_manager: Optional[LongContextServingManager] = None,
        paged_allocator: Optional[PagedKVAllocator] = None,
        device: Optional[Union[str, torch.device]] = None,
        turbo_quant_bits: Optional[int] = None,
    ) -> None:
        self.model = model.eval()
        self.config: DeepSeekV4Pro2BConfig = model.config
        self.device = torch.device(device or next(model.parameters()).device)
        self.model.to(self.device)
        if prefix_manager is not None:
            self.prefix_manager = prefix_manager
            self.backend = prefix_manager.backend
        else:
            chosen_backend = backend
            if chosen_backend is None:
                cuda_backend = CudaSparseAttentionBackend()
                chosen_backend = cuda_backend if cuda_backend.available() else PytorchAttentionBackend()
            self.backend = chosen_backend
            self.prefix_manager = None
        self.paged_allocator = paged_allocator
        # PolarQuant: None = disabled, PolarQuant(8) = 8-bit, PolarQuant(4) = 4-bit.
        # Always validate 8-bit correctness before enabling 4-bit.
        self._polar_quant: Optional[PolarQuant] = PolarQuant(turbo_quant_bits) if turbo_quant_bits is not None else None
        self._hca_layer_indices = [
            idx for idx, block in enumerate(self.model.model.layers) if isinstance(block.attn.sublayer, HCAAttention)
        ]
        self._csa_layer_indices = [
            idx for idx, block in enumerate(self.model.model.layers) if isinstance(block.attn.sublayer, CSAAttention)
        ]
        self._hca_blocks_per_page = max(1, self._block_tokens() // self.config.hca_compression)
        self._csa_blocks_per_page = max(1, self._block_tokens() // self.config.csa_compression)

    def _new_state(self) -> ModelServingState:
        layer_states: List[LayerServingState] = []
        for block in self.model.model.layers:
            attn = block.attn.sublayer
            if isinstance(attn, HCAAttention):
                layer_states.append(HCAServingState())
            else:
                layer_states.append(CSAServingState())
        return ModelServingState(token_count=0, layer_states=layer_states)

    # ------------------------------------------------------------------
    # PolarQuant helpers — no-ops when turbo_quant_bits is None
    # ------------------------------------------------------------------

    def _quant_append(
        self,
        existing_data: Optional[torch.Tensor],
        existing_scale: Optional[torch.Tensor],
        new_bf16: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Append a new block to the compressed KV cache.

        If PolarQuant is enabled, *new_bf16* is quantized before appending so
        the accumulated tensor stays in int8/uint8.  The scale tensor is grown
        in parallel.  If disabled, this is just ``_append_seq`` on bf16 data.
        """
        if self._polar_quant is None:
            return _append_seq(existing_data, new_bf16), None
        new_data, new_scale = self._polar_quant.encode(new_bf16)
        data = _append_seq(existing_data, new_data)
        scale = _append_seq(existing_scale, new_scale)
        return data, scale

    def _read_compressed(
        self,
        data: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return *data* as a floating-point tensor, dequantizing if a *scale* is present.

        The output dtype matches the model's compute dtype (e.g. float32 on CPU,
        bf16 on GPU) so downstream einsum/matmul ops don't hit dtype mismatches.
        """
        if data is None:
            return None
        if scale is None or self._polar_quant is None:
            return data  # already in the model's compute dtype
        model_dtype = self.model.lm_head.weight.dtype
        return self._polar_quant.decode(data, scale, out_dtype=model_dtype)

    def _compressed_to_bf16_for_paging(
        self,
        data: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        start: int,
        end: int,
    ) -> Optional[torch.Tensor]:
        """Slice [start:end] from the compressed cache and return in model dtype.

        Paged allocator stores tensors in the model's weight dtype; we dequantize
        before flushing to pages so the allocator format is not affected by
        whether TurboQuant is active.
        """
        if data is None or data.size(1) < end:
            return None
        sliced = data[:, start:end]
        if scale is None or self._polar_quant is None:
            return sliced
        sliced_scale = scale[:, start:end]
        model_dtype = self.model.lm_head.weight.dtype
        return self._polar_quant.decode(sliced, sliced_scale, out_dtype=model_dtype)

    def _block_tokens(self) -> int:
        return math.lcm(self.config.csa_compression, self.config.hca_compression)

    def allocator_page_shape(self) -> Dict[str, Tuple[int, ...]]:
        page_shape: Dict[str, Tuple[int, ...]] = {}
        for layer_idx in self._hca_layer_indices:
            page_shape[f"layer{layer_idx}.hca"] = (self._hca_blocks_per_page, self.config.attention_head_dim)
        for layer_idx in self._csa_layer_indices:
            page_shape[f"layer{layer_idx}.csa"] = (self._csa_blocks_per_page, self.config.attention_head_dim)
            page_shape[f"layer{layer_idx}.csa_index"] = (self._csa_blocks_per_page, self.config.indexer_head_dim)
        return page_shape

    def free_state(self, state: Optional[ModelServingState]) -> None:
        if state is None or self.paged_allocator is None or state.paged_prefix is None:
            return
        self.paged_allocator.free(state.paged_prefix.handle)
        state.paged_prefix = None

    def _sinkhorn(self, mhc, raw: torch.Tensor) -> torch.Tensor:
        return mhc._sinkhorn(raw)

    def _mhc_mix(self, mhc, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n, d = state.shape
        flat = mhc.norm(state.reshape(bsz, n * d))
        a_raw = mhc.alpha_pre * mhc.w_pre(flat) + mhc.s_pre
        b_raw = mhc.alpha_res * mhc.w_res(flat).view(bsz, n, n) + mhc.s_res
        c_raw = mhc.alpha_post * mhc.w_post(flat) + mhc.s_post
        a = torch.sigmoid(a_raw)
        b = self._sinkhorn(mhc, b_raw)
        c = 2.0 * torch.sigmoid(c_raw)
        x = torch.einsum("bn,bnd->bd", a, state)
        return a, b, c, x

    def _finish_mhc(self, state: torch.Tensor, b: torch.Tensor, c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("bij,bjd->bid", b, state)
        return mixed + c.unsqueeze(-1) * y.unsqueeze(1)

    def _query(self, attn, x: torch.Tensor, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = attn.q_down(x)
        q = attn.q_up(latent).view(x.size(0), 1, self.config.num_attention_heads, self.config.attention_head_dim)
        positions = torch.tensor([position], device=x.device, dtype=torch.long)
        q = apply_rope(q, positions, self.config.rope_dim, self.config.rope_theta, config=self.config)
        return attn.q_norm(q), latent

    def _window_entry(self, attn, x: torch.Tensor, position: int) -> torch.Tensor:
        entry = attn.window_kv(x).unsqueeze(1)
        positions = torch.tensor([position], device=x.device, dtype=torch.long)
        entry = apply_rope(entry, positions, self.config.rope_dim, self.config.rope_theta, config=self.config)
        return attn.kv_norm(entry)

    def _run_backend(self, q: torch.Tensor, values: torch.Tensor, sink: torch.Tensor, logit_scale: float = 1.0) -> torch.Tensor:
        if values is None or values.size(1) == 0:
            return q.new_zeros(q.size(0), 1, q.size(2), q.size(3))
        # The CUDA sparse-attention kernel requires all tensors on self.device
        # (cuda:0).  When layers are sharded to cuda:1 for prefill, q/values/sink
        # arrive here on cuda:1; move them, call the backend, and move the result
        # back so that downstream ops (RoPE, output projection) stay on cuda:1.
        orig_device = q.device
        if orig_device != self.device:
            q = q.to(self.device)
            values = values.to(self.device)
            sink = sink.to(self.device)
        kv = _pack_shared_kv(values)
        idx = _full_indices(q.size(0), values.size(1), q.device)
        result = self.backend.sparse_attention(q, kv, idx, sink, q.size(-1) ** -0.5 * logit_scale)
        if orig_device != self.device:
            result = result.to(orig_device)
        return result

    def _post_attention(self, attn, out: torch.Tensor, position: int) -> torch.Tensor:
        neg_pos = torch.full((out.size(0), out.size(1)), -position, device=out.device, dtype=torch.long)
        out = apply_rope(out, neg_pos, self.config.rope_dim, self.config.rope_theta, config=self.config)
        return attn.output(out.unsqueeze(1)).squeeze(1)

    def _paged_prefix_blocks(self, state: ModelServingState, layer_idx: int) -> int:
        if state.paged_prefix is None:
            return 0
        if layer_idx in self._hca_layer_indices:
            return state.paged_prefix.page_count * self._hca_blocks_per_page
        return state.paged_prefix.page_count * self._csa_blocks_per_page

    def _visible_hca_blocks(self, model_state: ModelServingState, state: HCAServingState, layer_idx: int) -> int:
        paged_blocks = self._paged_prefix_blocks(model_state, layer_idx)
        resident_blocks = 0 if state.compressed is None else state.compressed.size(1)
        return min(paged_blocks + resident_blocks, state.token_count // self.config.hca_compression)

    def _materialize_paged_prefix(self, model_state: ModelServingState, layer_idx: int, kind: str) -> Optional[torch.Tensor]:
        if self.paged_allocator is None or model_state.paged_prefix is None or model_state.paged_prefix.page_count == 0:
            return None
        key = f"layer{layer_idx}.{kind}"
        pages = self.paged_allocator.load_pages(model_state.paged_prefix.handle, (0, model_state.paged_prefix.page_count))
        if not pages:
            return None
        return torch.cat([page[key].to(self.device) for page in pages], dim=0).unsqueeze(0)

    def _hca_emit(self, attn: HCAAttention, state: HCAServingState, absolute_token: int) -> None:
        weights = torch.softmax(state.buffer_z.float(), dim=1).to(state.buffer_c.dtype)
        comp = (weights * state.buffer_c).sum(dim=1, keepdim=True)
        pos = torch.tensor([absolute_token], device=comp.device, dtype=torch.long)
        comp = apply_rope(comp, pos, self.config.rope_dim, self.config.rope_theta, config=self.config)
        comp = attn.kv_norm(comp)
        state.compressed, state.compressed_scale = self._quant_append(
            state.compressed, state.compressed_scale, comp
        )
        state.buffer_c = None
        state.buffer_z = None

    def _attention_step_hca(self, attn: HCAAttention, model_state: ModelServingState, state: HCAServingState, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        position = state.token_count
        q, _ = self._query(attn, x, position)
        current_local = self._window_entry(attn, x, position)
        prev_window = _tail(state.window, max(self.config.sliding_window - 1, 0))
        sources = current_local if prev_window is None else torch.cat([prev_window, current_local], dim=1)
        visible = self._visible_hca_blocks(model_state, state, layer_idx)
        if visible:
            paged_prefix = self._materialize_paged_prefix(model_state, layer_idx, "hca")
            # Dequantize resident compressed cache before use.
            resident = self._read_compressed(state.compressed, state.compressed_scale)
            prefix_parts = [part for part in [paged_prefix, resident] if part is not None]
            prefix = torch.cat(prefix_parts, dim=1) if prefix_parts else None
            if prefix is not None:
                sources = torch.cat([prefix[:, :visible], sources], dim=1)
        out = self._run_backend(q, sources, attn.sink, rope_attention_scale(self.config)).squeeze(1)
        y = self._post_attention(attn, out, position)

        state.window = _trim(_append_seq(state.window, current_local), self.config.sliding_window)
        c = attn.kv_proj(x).unsqueeze(1)
        slot = 0 if state.buffer_z is None else state.buffer_z.size(1)
        z = (attn.z_proj(x) + attn.bias[slot]).unsqueeze(1)
        state.buffer_c = _append_seq(state.buffer_c, c)
        state.buffer_z = _append_seq(state.buffer_z, z)
        state.token_count += 1
        if state.buffer_c is not None and state.buffer_c.size(1) == self.config.hca_compression:
            if state.token_count > state.loaded_prefix_tokens:
                self._hca_emit(attn, state, state.token_count - 1)
            else:
                state.buffer_c = None
                state.buffer_z = None
        return y

    def _emit_csa_block(
        self,
        attn: CSAAttention,
        state: CSAServingState,
        absolute_token: int,
    ) -> None:
        dim = state.curr_a_c.size(-1)
        idx_dim = state.curr_index_a_c.size(-1)
        m = self.config.csa_compression
        if state.prev_b_c is None:
            prev_b_c = state.curr_a_c.new_zeros(state.curr_a_c.size(0), m, dim)
            prev_b_z = state.curr_a_z.new_full((state.curr_a_z.size(0), m, dim), float("-inf"))
            prev_index_b_c = state.curr_index_a_c.new_zeros(state.curr_index_a_c.size(0), m, idx_dim)
            prev_index_b_z = state.curr_index_a_z.new_full((state.curr_index_a_z.size(0), m, idx_dim), float("-inf"))
        else:
            prev_b_c = state.prev_b_c
            prev_b_z = state.prev_b_z
            prev_index_b_c = state.prev_index_b_c
            prev_index_b_z = state.prev_index_b_z

        main_c = torch.cat([state.curr_a_c, prev_b_c], dim=1)
        main_z = torch.cat([state.curr_a_z + attn.bias_a, prev_b_z + attn.bias_b], dim=1)
        main_w = torch.softmax(main_z.float(), dim=1).to(main_c.dtype)
        main_comp = (main_w * main_c).sum(dim=1, keepdim=True)
        pos = torch.tensor([absolute_token - (m - 1)], device=main_comp.device, dtype=torch.long)
        main_comp = apply_rope(main_comp, pos, self.config.rope_dim, self.config.rope_theta, config=self.config)
        main_comp = attn.kv_norm(main_comp)
        state.compressed, state.compressed_scale = self._quant_append(
            state.compressed, state.compressed_scale, main_comp
        )

        index_c = torch.cat([state.curr_index_a_c, prev_index_b_c], dim=1)
        index_z = torch.cat([state.curr_index_a_z + attn.index_bias_a, prev_index_b_z + attn.index_bias_b], dim=1)
        index_w = torch.softmax(index_z.float(), dim=1).to(index_c.dtype)
        index_comp = (index_w * index_c).sum(dim=1, keepdim=True)
        state.index_compressed, state.index_compressed_scale = self._quant_append(
            state.index_compressed, state.index_compressed_scale, index_comp
        )

        state.prev_b_c = state.curr_b_c
        state.prev_b_z = state.curr_b_z
        state.prev_index_b_c = state.curr_index_b_c
        state.prev_index_b_z = state.curr_index_b_z
        state.curr_a_c = None
        state.curr_a_z = None
        state.curr_b_c = None
        state.curr_b_z = None
        state.curr_index_a_c = None
        state.curr_index_a_z = None
        state.curr_index_b_c = None
        state.curr_index_b_z = None

    def _visible_csa_blocks(self, model_state: ModelServingState, state: CSAServingState, layer_idx: int) -> int:
        paged_blocks = self._paged_prefix_blocks(model_state, layer_idx)
        resident_blocks = 0 if state.compressed is None else state.compressed.size(1)
        return min(paged_blocks + resident_blocks, state.token_count // self.config.csa_compression)

    def _attention_step_csa(self, attn: CSAAttention, model_state: ModelServingState, state: CSAServingState, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        position = state.token_count
        q, latent = self._query(attn, x, position)
        current_local = self._window_entry(attn, x, position)
        prev_window = _tail(state.window, max(self.config.sliding_window - 1, 0))
        local_sources = current_local if prev_window is None else torch.cat([prev_window, current_local], dim=1)
        visible = self._visible_csa_blocks(model_state, state, layer_idx)
        global_values = None
        if visible:
            full_index = self._materialize_paged_prefix(model_state, layer_idx, "csa_index")
            # Dequantize resident index cache before scoring.
            resident_index = self._read_compressed(state.index_compressed, state.index_compressed_scale)
            if resident_index is not None:
                full_index = resident_index if full_index is None else torch.cat([full_index, resident_index], dim=1)
            # Dequantize resident KV cache before gathering.
            full_compressed = self._materialize_paged_prefix(model_state, layer_idx, "csa")
            resident_compressed = self._read_compressed(state.compressed, state.compressed_scale)
            if resident_compressed is not None:
                full_compressed = resident_compressed if full_compressed is None else torch.cat([full_compressed, resident_compressed], dim=1)
            index_q = attn.index_q_up(latent).view(x.size(0), 1, self.config.indexer_num_heads, self.config.indexer_head_dim)
            index_q = index_q[:, 0]
            score = torch.einsum("bhc,bsc->bhs", index_q, full_index[:, :visible])
            score = (F.relu(score) * attn.index_weight(x).unsqueeze(-1)).sum(dim=1)
            k = min(self.config.csa_top_k, visible)
            idx = torch.topk(score, k=k, dim=-1).indices
            gather = idx.unsqueeze(-1).expand(-1, -1, self.config.attention_head_dim)
            global_values = torch.gather(full_compressed[:, :visible], dim=1, index=gather)
        sources = local_sources if global_values is None else torch.cat([global_values, local_sources], dim=1)
        out = self._run_backend(q, sources, attn.sink, rope_attention_scale(self.config)).squeeze(1)
        y = self._post_attention(attn, out, position)

        state.window = _trim(_append_seq(state.window, current_local), self.config.sliding_window)
        state.curr_a_c = _append_seq(state.curr_a_c, attn.kv_a(x).unsqueeze(1))
        state.curr_a_z = _append_seq(state.curr_a_z, attn.z_a(x).unsqueeze(1))
        state.curr_b_c = _append_seq(state.curr_b_c, attn.kv_b(x).unsqueeze(1))
        state.curr_b_z = _append_seq(state.curr_b_z, attn.z_b(x).unsqueeze(1))
        state.curr_index_a_c = _append_seq(state.curr_index_a_c, attn.index_kv_a(x).unsqueeze(1))
        state.curr_index_a_z = _append_seq(state.curr_index_a_z, attn.index_z_a(x).unsqueeze(1))
        state.curr_index_b_c = _append_seq(state.curr_index_b_c, attn.index_kv_b(x).unsqueeze(1))
        state.curr_index_b_z = _append_seq(state.curr_index_b_z, attn.index_z_b(x).unsqueeze(1))
        state.token_count += 1
        if state.curr_a_c is not None and state.curr_a_c.size(1) == self.config.csa_compression:
            if state.token_count > state.loaded_prefix_tokens:
                self._emit_csa_block(attn, state, state.token_count - 1)
            else:
                state.prev_b_c = state.curr_b_c
                state.prev_b_z = state.curr_b_z
                state.prev_index_b_c = state.curr_index_b_c
                state.prev_index_b_z = state.curr_index_b_z
                state.curr_a_c = None
                state.curr_a_z = None
                state.curr_b_c = None
                state.curr_b_z = None
                state.curr_index_a_c = None
                state.curr_index_a_z = None
                state.curr_index_b_c = None
                state.curr_index_b_z = None
        return y

    def _block_step(self, block, model_state: ModelServingState, layer_state: LayerServingState, layer_idx: int, token_id: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        attn_mhc = block.attn
        _, b_attn, c_attn, x_attn = self._mhc_mix(attn_mhc, state)
        if isinstance(layer_state, HCAServingState):
            y_attn = self._attention_step_hca(attn_mhc.sublayer, model_state, layer_state, layer_idx, x_attn)
        else:
            y_attn = self._attention_step_csa(attn_mhc.sublayer, model_state, layer_state, layer_idx, x_attn)
        state = self._finish_mhc(state, b_attn, c_attn, y_attn)

        moe_mhc = block.moe
        _, b_moe, c_moe, x_moe = self._mhc_mix(moe_mhc, state)
        y_moe, _ = moe_mhc.sublayer(x_moe.unsqueeze(1), token_ids=token_id.unsqueeze(1), token_mask=None)
        return self._finish_mhc(state, b_moe, c_moe, y_moe.squeeze(1))

    @torch.no_grad()
    def step_token(self, token_id: Union[int, torch.Tensor], state: Optional[ModelServingState] = None) -> Tuple[torch.Tensor, ModelServingState]:
        if state is None:
            state = self._new_state()
        token = token_id if isinstance(token_id, torch.Tensor) else torch.tensor([token_id], device=self.device, dtype=torch.long)
        if token.ndim == 0:
            token = token.view(1)
        if token.ndim != 1:
            raise ValueError("Serving engine currently expects a single-token batch vector")
        hidden = self.model.model.embed_tokens(token).unsqueeze(1).expand(-1, self.config.mhc_expansion, -1).contiguous()
        _layer_devs = getattr(self.model.model, '_layer_devices', None)
        for layer_idx, (block, layer_state) in enumerate(zip(self.model.model.layers, state.layer_states)):
            if _layer_devs is not None:
                dev = _layer_devs[layer_idx]
                if hidden.device != dev:
                    hidden = hidden.to(dev)
                    token = token.to(dev)
                self._move_layer_state_to_device(layer_state, dev)
            hidden = self._block_step(block, state, layer_state, layer_idx, token, hidden)
            if _layer_devs is not None:
                self._move_layer_state_to_device(layer_state, self.device)
        if _layer_devs is not None and hidden.device != self.device:
            hidden = hidden.to(self.device)
            token = token.to(self.device)
        weights = torch.softmax(self.model.model.post, dim=0)
        hidden = torch.einsum("n,bnd->bd", weights, hidden)
        hidden = self.model.model.final_norm(hidden)
        logits = self.model.lm_head(hidden)
        state.token_count += 1
        self._flush_live_page_if_ready(state)
        state.last_logits = logits
        return logits, state

    @torch.no_grad()
    def prefill(self, token_ids: Sequence[int], state: Optional[ModelServingState] = None) -> ModelServingState:
        if state is None:
            state = self._new_state()
        for token_id in token_ids:
            _, state = self.step_token(int(token_id), state)
        return state

    def _layer_state_from_payload(self, payload: Dict[str, object]) -> LayerServingState:
        if payload["type"] == "hca":
            return HCAServingState.from_dict(payload, self.device)
        return CSAServingState.from_dict(payload, self.device)

    def _state_from_swa_payload(self, payload: Dict[str, object]) -> ModelServingState:
        return ModelServingState(
            token_count=int(payload["token_count"]),
            layer_states=[self._layer_state_from_payload(layer) for layer in payload["layers"]],
            last_logits=None if payload.get("last_logits") is None else payload["last_logits"].to(self.device),
        )

    def _materialize_state_for_snapshot(self, state: ModelServingState) -> ModelServingState:
        snapshot = state.clone()
        if snapshot.paged_prefix is None:
            return snapshot
        for layer_idx, layer_state in enumerate(snapshot.layer_states):
            if isinstance(layer_state, HCAServingState):
                paged = self._materialize_paged_prefix(state, layer_idx, "hca")
                if paged is not None:
                    layer_state.compressed = paged if layer_state.compressed is None else torch.cat([paged, layer_state.compressed], dim=1)
            else:
                paged_main = self._materialize_paged_prefix(state, layer_idx, "csa")
                if paged_main is not None:
                    layer_state.compressed = paged_main if layer_state.compressed is None else torch.cat([paged_main, layer_state.compressed], dim=1)
                paged_index = self._materialize_paged_prefix(state, layer_idx, "csa_index")
                if paged_index is not None:
                    layer_state.index_compressed = paged_index if layer_state.index_compressed is None else torch.cat([paged_index, layer_state.index_compressed], dim=1)
        snapshot.paged_prefix = None
        return snapshot

    def _ensure_paged_handle(self, state: ModelServingState) -> None:
        if self.paged_allocator is None or state.paged_prefix is not None:
            return
        state.paged_prefix = PagedPrefixState(
            handle=self.paged_allocator.allocate(seq_id=f"state-{id(state)}"),
            page_count=0,
        )

    def _build_paged_prefix(self, payload: Dict[str, object]) -> Optional[PagedPrefixState]:
        if self.paged_allocator is None:
            return None
        page_count = int(payload["token_count"]) // self._block_tokens()
        if page_count == 0:
            return None
        handle = self.paged_allocator.allocate(seq_id=f"prefix-{id(payload)}")
        for page_idx in range(page_count):
            page_tensors = {
                key: torch.zeros(shape, dtype=self.model.lm_head.weight.dtype, device=self.device)
                for key, shape in self.allocator_page_shape().items()
            }
            for layer_idx, layer_payload in enumerate(payload["layers"]):
                if layer_payload["type"] == "hca":
                    start = page_idx * self._hca_blocks_per_page
                    end = start + self._hca_blocks_per_page
                    page_tensors[f"layer{layer_idx}.hca"].copy_(layer_payload["compressed"][0, start:end].to(self.device))
                else:
                    start = page_idx * self._csa_blocks_per_page
                    end = start + self._csa_blocks_per_page
                    page_tensors[f"layer{layer_idx}.csa"].copy_(layer_payload["compressed"][0, start:end].to(self.device))
                    page_tensors[f"layer{layer_idx}.csa_index"].copy_(layer_payload["index_compressed"][0, start:end].to(self.device))
            self.paged_allocator.append_page(handle, page_tensors)
        return PagedPrefixState(handle=handle, page_count=page_count)

    def _flush_live_page_if_ready(self, state: ModelServingState) -> None:
        if self.paged_allocator is None:
            return
        if state.token_count == 0 or (state.token_count % self._block_tokens()) != 0:
            return

        # First pass: validate ALL layers have the expected number of blocks before
        # touching anything.  An early return here is safe because no state has changed.
        for layer_state in state.layer_states:
            if isinstance(layer_state, HCAServingState):
                if layer_state.compressed is None or layer_state.compressed.size(1) < self._hca_blocks_per_page:
                    return
            else:
                if (layer_state.compressed is None
                        or layer_state.compressed.size(1) < self._csa_blocks_per_page
                        or layer_state.index_compressed is None
                        or layer_state.index_compressed.size(1) < self._csa_blocks_per_page):
                    return

        # Second pass: copy into a new page tensor (all layers are confirmed ready).
        page_tensors = {
            key: torch.zeros(shape, dtype=self.model.lm_head.weight.dtype, device=self.device)
            for key, shape in self.allocator_page_shape().items()
        }
        for layer_idx, layer_state in enumerate(state.layer_states):
            if isinstance(layer_state, HCAServingState):
                bf16_slice = self._compressed_to_bf16_for_paging(
                    layer_state.compressed, layer_state.compressed_scale, 0, self._hca_blocks_per_page
                )
                if bf16_slice is not None:
                    page_tensors[f"layer{layer_idx}.hca"].copy_(bf16_slice[0])
            else:
                bf16_slice = self._compressed_to_bf16_for_paging(
                    layer_state.compressed, layer_state.compressed_scale, 0, self._csa_blocks_per_page
                )
                if bf16_slice is not None:
                    page_tensors[f"layer{layer_idx}.csa"].copy_(bf16_slice[0])
                idx_slice = self._compressed_to_bf16_for_paging(
                    layer_state.index_compressed, layer_state.index_compressed_scale, 0, self._csa_blocks_per_page
                )
                if idx_slice is not None:
                    page_tensors[f"layer{layer_idx}.csa_index"].copy_(idx_slice[0])

        self._ensure_paged_handle(state)
        self.paged_allocator.append_page(state.paged_prefix.handle, page_tensors)
        state.paged_prefix.page_count += 1

        # Third pass: trim the flushed leading blocks from each layer.
        for layer_state in state.layer_states:
            if isinstance(layer_state, HCAServingState):
                layer_state.compressed = layer_state.compressed[:, self._hca_blocks_per_page:]
                if layer_state.compressed.size(1) == 0:
                    layer_state.compressed = None
                if layer_state.compressed_scale is not None:
                    layer_state.compressed_scale = layer_state.compressed_scale[:, self._hca_blocks_per_page:]
                    if layer_state.compressed_scale.size(1) == 0:
                        layer_state.compressed_scale = None
            else:
                layer_state.compressed = layer_state.compressed[:, self._csa_blocks_per_page:]
                if layer_state.compressed.size(1) == 0:
                    layer_state.compressed = None
                if layer_state.compressed_scale is not None:
                    layer_state.compressed_scale = layer_state.compressed_scale[:, self._csa_blocks_per_page:]
                    if layer_state.compressed_scale.size(1) == 0:
                        layer_state.compressed_scale = None
                layer_state.index_compressed = layer_state.index_compressed[:, self._csa_blocks_per_page:]
                if layer_state.index_compressed.size(1) == 0:
                    layer_state.index_compressed = None
                if layer_state.index_compressed_scale is not None:
                    layer_state.index_compressed_scale = layer_state.index_compressed_scale[:, self._csa_blocks_per_page:]
                    if layer_state.index_compressed_scale.size(1) == 0:
                        layer_state.index_compressed_scale = None

    def _inject_compressed_prefix(self, state: ModelServingState, payload: Dict[str, object]) -> None:
        loaded_prefix_tokens = int(payload["token_count"])
        paged_prefix = self._build_paged_prefix(payload)
        state.paged_prefix = paged_prefix
        for layer_state, saved in zip(state.layer_states, payload["layers"]):
            layer_state.loaded_prefix_tokens = loaded_prefix_tokens
            if paged_prefix is None:
                layer_state.compressed = None if saved.get("compressed") is None else saved["compressed"].to(self.device)
                layer_state.compressed_scale = None if saved.get("compressed_scale") is None else saved["compressed_scale"].to(self.device)
                if isinstance(layer_state, CSAServingState):
                    layer_state.index_compressed = None if saved.get("index_compressed") is None else saved["index_compressed"].to(self.device)
                    layer_state.index_compressed_scale = None if saved.get("index_compressed_scale") is None else saved["index_compressed_scale"].to(self.device)
            else:
                layer_state.compressed = None
                layer_state.compressed_scale = None
                if isinstance(layer_state, CSAServingState):
                    layer_state.index_compressed = None
                    layer_state.index_compressed_scale = None

    def _snapshot_for_periodic(self, state: ModelServingState) -> Dict[str, object]:
        return self._materialize_state_for_snapshot(state).to_swa_dict()

    @torch.no_grad()
    def build_prefix_cache(self, token_ids: Sequence[int]) -> object:
        if self.prefix_manager is None:
            raise RuntimeError("Prefix caching requires a LongContextServingManager")
        if len(token_ids) == 0:
            raise ValueError("Cannot cache an empty prefix")
        store = self.prefix_manager.cache_store
        checkpoint_token = None
        if store.swa_mode == SWACacheMode.PERIODIC:
            reusable = store.layout.full_block_prefix(len(token_ids))
            checkpoint_token = reusable if reusable <= store.checkpoint_stride else reusable - (reusable % store.checkpoint_stride)

        state = self._new_state()
        periodic_snapshot = None
        for idx, token_id in enumerate(token_ids, start=1):
            _, state = self.step_token(int(token_id), state)
            if checkpoint_token is not None and idx == checkpoint_token:
                periodic_snapshot = self._snapshot_for_periodic(state)

        reusable = store.layout.full_block_prefix(len(token_ids))
        compressed = self._materialize_state_for_snapshot(state).to_compressed_dict(reusable)
        if store.swa_mode == SWACacheMode.ZERO:
            swa_payload = None
        elif store.swa_mode == SWACacheMode.PERIODIC:
            if periodic_snapshot is None:
                periodic_snapshot = self._snapshot_for_periodic(state)
            swa_payload = periodic_snapshot
        else:
            swa_payload = state.to_swa_dict()
        return self.prefix_manager.save_prefix(list(map(int, token_ids)), compressed, swa_state=swa_payload)

    @torch.no_grad()
    def prefill_with_reuse(self, token_ids: Sequence[int]) -> ModelServingState:
        if self.prefix_manager is None:
            return self.prefill(token_ids)
        reuse = self.prefix_manager.reuse_prefix(list(map(int, token_ids)))
        if reuse is None:
            return self.prefill(token_ids)

        mode = SWACacheMode(reuse.restore_plan.swa_mode)
        if mode == SWACacheMode.FULL and reuse.swa_state is not None:
            state = self._state_from_swa_payload(reuse.swa_state)
            self._inject_compressed_prefix(state, reuse.compressed_cache)
            return state

        if reuse.swa_state is not None:
            state = self._state_from_swa_payload(reuse.swa_state)
        else:
            state = self._new_state()
            state.token_count = reuse.restore_plan.recompute_start_token
            for layer_state in state.layer_states:
                layer_state.token_count = reuse.restore_plan.recompute_start_token
        self._inject_compressed_prefix(state, reuse.compressed_cache)
        for layer_state in state.layer_states:
            if mode == SWACacheMode.ZERO:
                layer_state.token_count = reuse.restore_plan.recompute_start_token
        for token_id in token_ids[reuse.restore_plan.recompute_start_token :]:
            _, state = self.step_token(int(token_id), state)
        return state

    # ------------------------------------------------------------------
    # Host offload / restore for inactive sequences
    # ------------------------------------------------------------------

    def offload_swa_to_host(self, state: ModelServingState) -> None:
        """Move sliding-window and partial compression buffer tensors to host RAM.

        These are the bounded "live" tensors (window = O(W), partial buffers =
        O(compression_factor)) that are NOT paged by the allocator.  For a
        high-concurrency serving setup with many inactive sequences, offloading
        them frees device memory between decode steps.  Call ``restore_swa_to_device``
        before the next ``step_token`` on this state.
        """
        _SWA_ATTRS_HCA = ("window", "buffer_c", "buffer_z")
        _SWA_ATTRS_CSA = (
            "window",
            "curr_a_c", "curr_a_z", "curr_b_c", "curr_b_z",
            "curr_index_a_c", "curr_index_a_z", "curr_index_b_c", "curr_index_b_z",
            "prev_b_c", "prev_b_z", "prev_index_b_c", "prev_index_b_z",
        )
        for layer_state in state.layer_states:
            attrs = _SWA_ATTRS_HCA if isinstance(layer_state, HCAServingState) else _SWA_ATTRS_CSA
            for attr in attrs:
                v = getattr(layer_state, attr)
                if v is not None and v.device.type != "cpu":
                    setattr(layer_state, attr, v.cpu())

    def restore_swa_to_device(self, state: ModelServingState) -> None:
        """Move window and partial buffer tensors back to the serving device.

        Must be called after ``offload_swa_to_host`` before the next ``step_token``.
        """
        _SWA_ATTRS_HCA = ("window", "buffer_c", "buffer_z")
        _SWA_ATTRS_CSA = (
            "window",
            "curr_a_c", "curr_a_z", "curr_b_c", "curr_b_z",
            "curr_index_a_c", "curr_index_a_z", "curr_index_b_c", "curr_index_b_z",
            "prev_b_c", "prev_b_z", "prev_index_b_c", "prev_index_b_z",
        )
        for layer_state in state.layer_states:
            attrs = _SWA_ATTRS_HCA if isinstance(layer_state, HCAServingState) else _SWA_ATTRS_CSA
            for attr in attrs:
                v = getattr(layer_state, attr)
                if v is not None and v.device != self.device:
                    setattr(layer_state, attr, v.to(self.device))

    # ------------------------------------------------------------------
    # Multi-GPU sharding
    # ------------------------------------------------------------------

    def _move_layer_state_to_device(
        self, layer_state: "LayerServingState", device: torch.device
    ) -> None:
        """Move all tensor fields of a per-layer serving state to *device* in-place."""
        for attr, val in vars(layer_state).items():
            if isinstance(val, torch.Tensor) and val.device != device:
                setattr(layer_state, attr, val.to(device))

    def shard_across_gpus(self, num_gpus: int) -> None:
        """Distribute transformer layers evenly across *num_gpus* CUDA devices.

        Splits the 26 blocks: the first half stays on ``cuda:0``, the second
        half moves to ``cuda:1`` (and so on).  ``fast_prefill`` and
        ``step_token`` handle per-layer device transfers automatically once
        sharding is active.  Call once after construction, before the first
        prefill.
        """
        if num_gpus < 2:
            return
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        actual = min(num_gpus, available)
        if actual < 2:
            return
        n = len(self.model.model.layers)
        per_gpu = (n + actual - 1) // actual
        devices: List[torch.device] = []
        for i, layer in enumerate(self.model.model.layers):
            gpu_id = min(i // per_gpu, actual - 1)
            dev = torch.device(f"cuda:{gpu_id}")
            layer.to(dev)
            devices.append(dev)
        self.model.model._layer_devices = devices

    # ------------------------------------------------------------------
    # State extraction helpers for fast_prefill
    # ------------------------------------------------------------------

    def _extract_hca_state_from_hidden(
        self,
        attn: "HCAAttention",
        state: HCAServingState,
        x: torch.Tensor,
        N: int,
    ) -> None:
        """Populate ``state`` from the batched sublayer input ``x`` [B, N, hidden].

        This is the fast-path alternative to accumulating state step-by-step.
        The produced state is numerically identical to what ``step_token`` builds.
        """
        M = self.config.hca_compression
        W = self.config.sliding_window
        full_blocks = N // M
        remainder = N % M

        # Compressed blocks — identical to what _hca_emit produces step-by-step.
        if full_blocks > 0:
            comp, _ = attn._compress(x)      # [B, full_blocks, D]
            if comp.size(1) > 0:
                state.compressed, state.compressed_scale = self._quant_append(None, None, comp)
            else:
                state.compressed = None
                state.compressed_scale = None
        else:
            state.compressed = None
            state.compressed_scale = None

        # Window: last min(N, W) window-KV entries.
        window_all = attn._window_entries(x)  # [B, N, D]
        state.window = window_all[:, max(0, N - W):]

        # Partial buffer: tokens in the last incomplete compression block.
        if remainder > 0:
            x_partial = x[:, full_blocks * M:]            # [B, remainder, D]
            state.buffer_c = attn.kv_proj(x_partial)
            state.buffer_z = attn.z_proj(x_partial) + attn.bias[:remainder]
        else:
            state.buffer_c = None
            state.buffer_z = None

        state.token_count = N
        state.loaded_prefix_tokens = 0

    def _extract_csa_state_from_hidden(
        self,
        attn: "CSAAttention",
        state: CSAServingState,
        x: torch.Tensor,
        N: int,
    ) -> None:
        """Populate ``state`` from the batched sublayer input ``x`` [B, N, hidden].

        Numerically identical to the step-by-step accumulation in
        ``_attention_step_csa`` + ``_emit_csa_block``.
        """
        M = self.config.csa_compression
        W = self.config.sliding_window
        full_blocks = N // M
        remainder = N % M

        # Full compressed blocks.
        if full_blocks > 0:
            comp_main = attn._compress_main(x)    # [B, full_blocks, D]
            comp_idx = attn._compress_index(x)    # [B, full_blocks, index_D]
            if comp_main.size(1) > 0:
                state.compressed, state.compressed_scale = self._quant_append(None, None, comp_main)
                state.index_compressed, state.index_compressed_scale = self._quant_append(None, None, comp_idx)
            else:
                state.compressed = None
                state.compressed_scale = None
                state.index_compressed = None
                state.index_compressed_scale = None
        else:
            state.compressed = None
            state.compressed_scale = None
            state.index_compressed = None
            state.index_compressed_scale = None

        # Window.
        window_all = attn._window_entries(x)
        state.window = window_all[:, max(0, N - W):]

        # Partial A/B buffers for the current (incomplete) block.
        if remainder > 0:
            x_partial = x[:, full_blocks * M:]
            state.curr_a_c = attn.kv_a(x_partial)
            state.curr_a_z = attn.z_a(x_partial)
            state.curr_b_c = attn.kv_b(x_partial)
            state.curr_b_z = attn.z_b(x_partial)
            state.curr_index_a_c = attn.index_kv_a(x_partial)
            state.curr_index_a_z = attn.index_z_a(x_partial)
            state.curr_index_b_c = attn.index_kv_b(x_partial)
            state.curr_index_b_z = attn.index_z_b(x_partial)
        else:
            state.curr_a_c = state.curr_a_z = None
            state.curr_b_c = state.curr_b_z = None
            state.curr_index_a_c = state.curr_index_a_z = None
            state.curr_index_b_c = state.curr_index_b_z = None

        # Previous B buffer: the LAST complete block's B-projections, needed by
        # the NEXT _emit_csa_block call for the overlap-compression scheme.
        if full_blocks > 0:
            x_last_b = x[:, (full_blocks - 1) * M: full_blocks * M]
            state.prev_b_c = attn.kv_b(x_last_b)
            state.prev_b_z = attn.z_b(x_last_b)
            state.prev_index_b_c = attn.index_kv_b(x_last_b)
            state.prev_index_b_z = attn.index_z_b(x_last_b)
        else:
            state.prev_b_c = state.prev_b_z = None
            state.prev_index_b_c = state.prev_index_b_z = None

        state.token_count = N
        state.loaded_prefix_tokens = 0

    def _flush_all_completed_pages(self, state: ModelServingState) -> None:
        """Flush ALL complete token-block pages to the paged allocator.

        Called once after ``fast_prefill`` extracts serving state from the batch
        forward pass.  The step-by-step ``_flush_live_page_if_ready`` only flushes
        one page at a time (called after every ``_block_tokens()`` steps); here we
        flush all ``N // _block_tokens()`` pages at once.
        """
        if self.paged_allocator is None:
            return

        num_pages = state.token_count // self._block_tokens()
        if num_pages == 0:
            return

        self._ensure_paged_handle(state)
        page_shape = self.allocator_page_shape()
        dtype = self.model.lm_head.weight.dtype

        for page_idx in range(num_pages):
            page_tensors = {
                key: torch.zeros(shape, dtype=dtype, device=self.device)
                for key, shape in page_shape.items()
            }
            for layer_idx, layer_state in enumerate(state.layer_states):
                s = page_idx * self._hca_blocks_per_page if isinstance(layer_state, HCAServingState) else page_idx * self._csa_blocks_per_page
                if isinstance(layer_state, HCAServingState):
                    e = s + self._hca_blocks_per_page
                    bf16_slice = self._compressed_to_bf16_for_paging(
                        layer_state.compressed, layer_state.compressed_scale, s, e
                    )
                    if bf16_slice is not None:
                        page_tensors[f"layer{layer_idx}.hca"].copy_(bf16_slice[0])
                else:
                    e = s + self._csa_blocks_per_page
                    bf16_slice = self._compressed_to_bf16_for_paging(
                        layer_state.compressed, layer_state.compressed_scale, s, e
                    )
                    if bf16_slice is not None:
                        page_tensors[f"layer{layer_idx}.csa"].copy_(bf16_slice[0])
                    idx_slice = self._compressed_to_bf16_for_paging(
                        layer_state.index_compressed, layer_state.index_compressed_scale, s, e
                    )
                    if idx_slice is not None:
                        page_tensors[f"layer{layer_idx}.csa_index"].copy_(idx_slice[0])
            self.paged_allocator.append_page(state.paged_prefix.handle, page_tensors)
            state.paged_prefix.page_count += 1

        # Remove flushed blocks from the in-memory tensors.
        for layer_state in state.layer_states:
            if isinstance(layer_state, HCAServingState):
                if layer_state.compressed is not None:
                    flushed = num_pages * self._hca_blocks_per_page
                    rest = layer_state.compressed[:, flushed:]
                    layer_state.compressed = rest if rest.size(1) > 0 else None
                if layer_state.compressed_scale is not None:
                    flushed = num_pages * self._hca_blocks_per_page
                    rest = layer_state.compressed_scale[:, flushed:]
                    layer_state.compressed_scale = rest if rest.size(1) > 0 else None
            else:
                if layer_state.compressed is not None:
                    flushed = num_pages * self._csa_blocks_per_page
                    rest = layer_state.compressed[:, flushed:]
                    layer_state.compressed = rest if rest.size(1) > 0 else None
                if layer_state.compressed_scale is not None:
                    flushed = num_pages * self._csa_blocks_per_page
                    rest = layer_state.compressed_scale[:, flushed:]
                    layer_state.compressed_scale = rest if rest.size(1) > 0 else None
                if layer_state.index_compressed is not None:
                    flushed = num_pages * self._csa_blocks_per_page
                    rest = layer_state.index_compressed[:, flushed:]
                    layer_state.index_compressed = rest if rest.size(1) > 0 else None
                if layer_state.index_compressed_scale is not None:
                    flushed = num_pages * self._csa_blocks_per_page
                    rest = layer_state.index_compressed_scale[:, flushed:]
                    layer_state.index_compressed_scale = rest if rest.size(1) > 0 else None

    # ------------------------------------------------------------------
    # Fast batched prefill (uses model's causal forward pass)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fast_prefill(self, token_ids: Sequence[int]) -> ModelServingState:
        """Prefill using the model's batched forward pass instead of token-by-token loops.

        Why this is faster
        ------------------
        ``prefill()`` calls ``step_token()`` N times, each of which:
          1. Does one Python function-call stack round-trip per layer (O(N·L) frames).
          2. Launches each linear projection as a [B=1, 1, D] matmul — very low GPU
             utilisation.
        ``fast_prefill()`` runs a *single* ``model.forward()`` call on all N tokens at
        once, giving:
          1. All mHC linear layers run as [B, N, D] matmuls — N times more efficient.
          2. All KV projections (kv_proj, z_proj, kv_a, kv_b, etc.) run in batch.
          3. Python loop depth reduced from O(N·L) to O(L) for the mHC update.
        The inner Python loop inside ``HCAAttention.forward()`` / ``CSAAttention.forward()``
        still runs N iterations for the causal per-token attention — this is the remaining
        bottleneck for very large N. Replacing it with the tiled_prefill CUDA kernel would
        make it O(N / tile_size) CUDA calls; that change is tracked in TODO item 1 (kernel
        integration into the training-mode forward).

        Correctness guarantee
        ---------------------
        For position `t`, the mHC pre-mix (a, b, c) depends ONLY on the current token's
        embedding/state, not on position t-1.  The ``ManifoldConstrainedHyperConnection``
        forward processes each position independently.  Therefore the batch forward pass
        at position t produces numerically identical hidden states to ``step_token`` at
        the same position — the only shared context comes through the compressed KV cache,
        which is also computed identically by the batch attention's causal mask.

        Parameters
        ----------
        token_ids : sequence of ints
            The prompt token IDs.  Processing starts from position 0.

        Returns
        -------
        ModelServingState ready for ``step_token``-based decode.
        """
        N = len(token_ids)
        if N == 0:
            return self._new_state()

        tokens = torch.tensor([list(token_ids)], device=self.device, dtype=torch.long)

        # Pre-create state so hooks populate it *during* the forward pass.
        # This eliminates accumulating L×[B,T,D] tensors in captured_x, which
        # cost ~13 GB at 128 K context and was the main single-GPU OOM source.
        state = self._new_state()

        hooks = []
        for block, layer_state in zip(self.model.model.layers, state.layer_states):
            attn = block.attn.sublayer
            is_hca = isinstance(layer_state, HCAServingState)

            def _make_hook(a, ls, hca):
                def _hook(module, inp, _output):
                    # inp[0] = mHC-premixed hidden states [B, T, D] for this layer.
                    x = inp[0]
                    if hca:
                        self._extract_hca_state_from_hidden(a, ls, x, N)
                    else:
                        self._extract_csa_state_from_hidden(a, ls, x, N)
                    # For sharded models, extracted tensors live on the layer's
                    # GPU; normalise them back to the primary serving device.
                    if x.device != self.device:
                        self._move_layer_state_to_device(ls, self.device)
                return _hook

            hooks.append(block.attn.sublayer.register_forward_hook(
                _make_hook(attn, layer_state, is_hca)
            ))

        try:
            # Call the backbone only — avoids materialising lm_head over all T
            # tokens (vocab projection [B, T, vocab_size] is ~4 GB at T=65 K).
            hidden, _ = self.model.model(input_ids=tokens)
        finally:
            for h in hooks:
                h.remove()

        state.token_count = N
        # Apply lm_head only on the last token to save memory.
        state.last_logits = self.model.lm_head(hidden[:, -1, :].to(self.device)).detach()

        # Flush all complete pages to the paged allocator if one is configured.
        if self.paged_allocator is not None:
            self._flush_all_completed_pages(state)

        return state

    @torch.no_grad()
    def chunked_fast_prefill(
        self,
        token_ids: Sequence[int],
        mhc_chunk_size: int = 4096,
    ) -> ModelServingState:
        """Memory-efficient fast prefill that chunks mHC operations along T.

        This is the fix for the 262K+ OOM caused by the mHC state tensor
        ``[B, T, n_expand, D]`` peaking at ``2 × state_size`` per layer during
        the standard ``forward()`` (the old ``mixed`` + ``state`` allocation).

        Uses ``model.model.chunked_forward()`` which processes the mHC pre-mix
        and post-mix in T-chunks of ``mhc_chunk_size``, while the attention
        sublayer still sees the full T for correct causal attention.

        Memory budget comparison (B=1, T=262144, n=4, D=1536, bf16):
          - ``fast_prefill``: ~25.2 GB peak (2× state tensor)
          - ``chunked_fast_prefill(mhc_chunk_size=4096)``: ~1.2 GB peak + O(T×D)

        Parameters
        ----------
        token_ids : sequence of ints
            The prompt token IDs.
        mhc_chunk_size : int
            Chunk size for the mHC T-dimension operations.  Default 4096.
            Smaller = less peak memory but more Python loop iterations.

        Returns
        -------
        ModelServingState ready for ``step_token``-based decode.
        """
        N = len(token_ids)
        if N == 0:
            return self._new_state()

        tokens = torch.tensor([list(token_ids)], device=self.device, dtype=torch.long)

        # Pre-create state so hooks populate it during the forward pass.
        state = self._new_state()

        hooks = []
        for block, layer_state in zip(self.model.model.layers, state.layer_states):
            attn = block.attn.sublayer
            is_hca = isinstance(layer_state, HCAServingState)

            def _make_hook(a, ls, hca):
                def _hook(module, inp, _output):
                    x = inp[0]
                    if hca:
                        self._extract_hca_state_from_hidden(a, ls, x, N)
                    else:
                        self._extract_csa_state_from_hidden(a, ls, x, N)
                    if x.device != self.device:
                        self._move_layer_state_to_device(ls, self.device)
                return _hook

            hooks.append(block.attn.sublayer.register_forward_hook(
                _make_hook(attn, layer_state, is_hca)
            ))

        try:
            hidden, _ = self.model.model.chunked_forward(
                input_ids=tokens, mhc_chunk_size=mhc_chunk_size
            )
        finally:
            for h in hooks:
                h.remove()

        state.token_count = N
        state.last_logits = self.model.lm_head(hidden[:, -1, :].to(self.device)).detach()

        if self.paged_allocator is not None:
            self._flush_all_completed_pages(state)

        return state

    @torch.no_grad()
    def generate(
        self,
        token_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_prefix_cache: bool = True,
    ) -> List[int]:
        if len(token_ids) == 0:
            raise ValueError("Generation requires at least one prompt token")
        state = self.prefill_with_reuse(token_ids) if use_prefix_cache else self.prefill(token_ids)
        if state.last_logits is None:
            raise RuntimeError("Prefill did not produce logits")
        generated = list(map(int, token_ids))
        logits = state.last_logits
        for _ in range(max_new_tokens):
            next_logits = logits
            if temperature <= 0:
                next_token = int(next_logits.argmax(dim=-1).item())
            else:
                scaled = next_logits / temperature
                if top_k is not None and top_k < scaled.size(-1):
                    values, indices = torch.topk(scaled, k=top_k, dim=-1)
                    probs = torch.softmax(values, dim=-1)
                    sample = torch.multinomial(probs, num_samples=1)
                    next_token = int(indices.gather(-1, sample).item())
                else:
                    probs = torch.softmax(scaled, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_token)
            logits, state = self.step_token(next_token, state)
        return generated
