from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol

import torch

from deepseek_kernels.loader import cuda_extension_unavailable_reason
from deepseek_kernels.sparse_attention import sparse_sink_attention


class SWACacheMode(str, Enum):
    FULL = "full"
    PERIODIC = "periodic"
    ZERO = "zero"


@dataclass(frozen=True)
class HybridCacheLayout:
    csa_compression: int
    hca_compression: int
    window_size: int
    num_layers: int

    @property
    def block_tokens(self) -> int:
        return math.lcm(self.csa_compression, self.hca_compression)

    @property
    def csa_tokens_per_block(self) -> int:
        return self.block_tokens // self.csa_compression

    @property
    def hca_tokens_per_block(self) -> int:
        return self.block_tokens // self.hca_compression

    def full_block_prefix(self, token_count: int) -> int:
        return token_count - (token_count % self.block_tokens)


@dataclass
class PrefixCacheMetadata:
    prefix_hash: str
    prefix_token_count: int
    reusable_token_count: int
    swa_mode: str
    window_size: int
    num_layers: int
    csa_compression: int
    hca_compression: int
    checkpoint_stride: int
    files: Dict[str, str] = field(default_factory=dict)


@dataclass
class RestorePlan:
    reusable_prefix_tokens: int
    recompute_start_token: int
    compressed_prefix_tokens: int
    needs_tail_recompute: bool
    swa_mode: str
    load_swa_checkpoint_from: Optional[int] = None
    restore_last_window_tokens: Optional[int] = None


class AttentionKernelBackend(Protocol):
    """Backend abstraction for serving-time compressed/sparse attention kernels."""

    def name(self) -> str: ...

    def sparse_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        sink: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor: ...


class PytorchAttentionBackend:
    """Fallback backend. Correctness path, not the 1M-token fast path.

    Tensor contract:
    - q: [batch, target_tokens, num_heads, head_dim]
    - kv: [batch, source_tokens, 2, head_dim] where kv[:, :, 0] is K and kv[:, :, 1] is V
    - topk_indices: [batch, target_tokens, top_k]
    - sink: scalar, [num_heads], or broadcastable to [batch, target_tokens, num_heads, 1]
    """

    def name(self) -> str:
        return "pytorch"

    def sparse_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        sink: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        if q.ndim != 4:
            raise ValueError(f"Expected q to have shape [B, T, H, D], got {tuple(q.shape)}")
        if kv.ndim != 4 or kv.size(-2) != 2:
            raise ValueError(f"Expected kv to have shape [B, S, 2, D], got {tuple(kv.shape)}")
        if topk_indices.ndim != 3:
            raise ValueError(f"Expected topk_indices to have shape [B, T, K], got {tuple(topk_indices.shape)}")
        if q.size(0) != kv.size(0) or q.size(-1) != kv.size(-1):
            raise ValueError("q and kv batch/head dimensions do not match")
        if topk_indices.size(0) != q.size(0) or topk_indices.size(1) != q.size(1):
            raise ValueError("topk_indices must align with q batch and target-token dimensions")

        source = kv.unsqueeze(1).expand(-1, q.size(1), -1, -1, -1)
        index = topk_indices[..., None, None].expand(-1, -1, -1, kv.size(-2), kv.size(-1))
        gathered = torch.gather(source, dim=2, index=index)
        keys = gathered[:, :, :, 0, :]
        values = gathered[:, :, :, 1, :]
        logits = torch.einsum("bthd,btkd->bthk", q, keys) * softmax_scale
        sink_logits = self._broadcast_sink(sink, logits)
        max_real = logits.max(dim=-1, keepdim=True).values
        max_all = torch.maximum(max_real, sink_logits)
        exp_logits = torch.exp(logits - max_all)
        sink_exp = torch.exp(sink_logits - max_all)
        weights = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + sink_exp)
        return torch.einsum("bthk,btkd->bthd", weights, values)

    @staticmethod
    def _broadcast_sink(sink: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        sink = sink.to(device=logits.device, dtype=logits.dtype)
        if sink.ndim == 0:
            return sink.view(1, 1, 1, 1)
        if sink.ndim == 1 and sink.size(0) == logits.size(2):
            return sink.view(1, 1, -1, 1)
        if sink.ndim == 4:
            return sink
        raise ValueError(
            "sink must be a scalar, a [num_heads] tensor, or a tensor broadcastable to [B, T, H, 1]"
        )


class CudaSparseAttentionBackend:
    """CUDA extension backend for sparse compressed-KV attention."""

    def name(self) -> str:
        return "cuda_extension"

    def available(self) -> bool:
        return cuda_extension_unavailable_reason(require_device=True) is None

    def availability_error(self) -> Optional[str]:
        return cuda_extension_unavailable_reason(require_device=True)

    def sparse_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        sink: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        return sparse_sink_attention(
            q.contiguous(),
            kv.contiguous(),
            topk_indices.contiguous(),
            sink.contiguous(),
            softmax_scale,
        )


class OnDiskPrefixKVStore:
    """Disk-backed prefix cache for compressed KV entries and optional SWA state.

    This matches the paper's serving design at the policy level:
    - store compressed CSA/HCA entries up to the last complete compression block
    - choose one of full / periodic / zero SWA caching
    - return a restore plan that tells the serving layer what must be recomputed
    """

    def __init__(
        self,
        root_dir: str,
        layout: HybridCacheLayout,
        swa_mode: SWACacheMode = SWACacheMode.PERIODIC,
        checkpoint_stride: int = 4096,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.layout = layout
        self.swa_mode = swa_mode
        self.checkpoint_stride = checkpoint_stride
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_dir / "index.json"
        if not self.index_path.exists():
            self._write_index({})

    def _read_index(self) -> Dict[str, Dict[str, object]]:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text())

    def _write_index(self, index: Dict[str, Dict[str, object]]) -> None:
        tmp_path = self.index_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(index, indent=2, sort_keys=True))
        tmp_path.replace(self.index_path)

    def _validate_metadata(self, metadata: PrefixCacheMetadata) -> None:
        if metadata.csa_compression != self.layout.csa_compression:
            raise ValueError("CSA compression in cache metadata does not match the active layout")
        if metadata.hca_compression != self.layout.hca_compression:
            raise ValueError("HCA compression in cache metadata does not match the active layout")
        if metadata.window_size != self.layout.window_size:
            raise ValueError("SWA window size in cache metadata does not match the active layout")
        if metadata.num_layers != self.layout.num_layers:
            raise ValueError("Layer count in cache metadata does not match the active layout")

    def _hash_tokens(self, token_ids: Iterable[int]) -> str:
        digest = hashlib.sha256()
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(4, byteorder="little", signed=False))
        return digest.hexdigest()

    def _tensor_file(self, prefix_hash: str, name: str) -> Path:
        return self.root_dir / f"{prefix_hash}.{name}.pt"

    def save_prefix(
        self,
        token_ids: List[int],
        compressed_cache: Dict[str, torch.Tensor],
        swa_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> PrefixCacheMetadata:
        reusable = self.layout.full_block_prefix(len(token_ids))
        if reusable <= 0:
            raise ValueError(
                f"Prefix of length {len(token_ids)} does not contain a full compression block of {self.layout.block_tokens} tokens"
            )
        prefix_hash = self._hash_tokens(token_ids[:reusable])
        metadata = PrefixCacheMetadata(
            prefix_hash=prefix_hash,
            prefix_token_count=len(token_ids),
            reusable_token_count=reusable,
            swa_mode=self.swa_mode.value,
            window_size=self.layout.window_size,
            num_layers=self.layout.num_layers,
            csa_compression=self.layout.csa_compression,
            hca_compression=self.layout.hca_compression,
            checkpoint_stride=self.checkpoint_stride,
        )
        compressed_path = self._tensor_file(prefix_hash, "compressed")
        torch.save(compressed_cache, compressed_path)
        metadata.files["compressed"] = compressed_path.name

        if swa_state and self.swa_mode != SWACacheMode.ZERO:
            if self.swa_mode == SWACacheMode.FULL:
                swa_path = self._tensor_file(prefix_hash, "swa_full")
                torch.save(swa_state, swa_path)
                metadata.files["swa_full"] = swa_path.name
            elif self.swa_mode == SWACacheMode.PERIODIC:
                checkpoint_token = self._periodic_checkpoint_token(reusable)
                swa_path = self._tensor_file(prefix_hash, f"swa_ckpt_{checkpoint_token}")
                torch.save({"token_offset": checkpoint_token, "state": swa_state}, swa_path)
                metadata.files["swa_periodic"] = swa_path.name

        index = self._read_index()
        index[prefix_hash] = asdict(metadata)
        self._write_index(index)
        return metadata

    def lookup_prefix(self, token_ids: List[int]) -> Optional[PrefixCacheMetadata]:
        reusable = self.layout.full_block_prefix(len(token_ids))
        if reusable <= 0:
            return None
        prefix_hash = self._hash_tokens(token_ids[:reusable])
        index = self._read_index()
        record = index.get(prefix_hash)
        if record is None:
            return None
        metadata = PrefixCacheMetadata(**record)
        self._validate_metadata(metadata)
        return metadata

    def load_compressed_cache(self, metadata: PrefixCacheMetadata) -> Dict[str, torch.Tensor]:
        self._validate_metadata(metadata)
        path = self.root_dir / metadata.files["compressed"]
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_swa_state(self, metadata: PrefixCacheMetadata) -> Optional[Dict[str, torch.Tensor]]:
        self._validate_metadata(metadata)
        mode = SWACacheMode(metadata.swa_mode)
        if mode == SWACacheMode.ZERO:
            return None
        if mode == SWACacheMode.FULL and "swa_full" in metadata.files:
            return torch.load(self.root_dir / metadata.files["swa_full"], map_location="cpu", weights_only=False)
        if mode == SWACacheMode.PERIODIC and "swa_periodic" in metadata.files:
            payload = torch.load(self.root_dir / metadata.files["swa_periodic"], map_location="cpu", weights_only=False)
            return payload["state"]
        return None

    def _periodic_checkpoint_token(self, reusable_tokens: int) -> int:
        if reusable_tokens <= self.checkpoint_stride:
            return reusable_tokens
        return reusable_tokens - (reusable_tokens % self.checkpoint_stride)

    def make_restore_plan(self, prefix_tokens: int) -> RestorePlan:
        reusable = self.layout.full_block_prefix(prefix_tokens)
        if self.swa_mode == SWACacheMode.FULL:
            return RestorePlan(
                reusable_prefix_tokens=reusable,
                compressed_prefix_tokens=reusable,
                recompute_start_token=reusable,
                needs_tail_recompute=(reusable != prefix_tokens),
                swa_mode=self.swa_mode.value,
                restore_last_window_tokens=self.layout.window_size,
            )
        if self.swa_mode == SWACacheMode.PERIODIC:
            checkpoint_token = self._periodic_checkpoint_token(reusable)
            return RestorePlan(
                reusable_prefix_tokens=reusable,
                compressed_prefix_tokens=reusable,
                recompute_start_token=checkpoint_token,
                needs_tail_recompute=(checkpoint_token != prefix_tokens),
                swa_mode=self.swa_mode.value,
                load_swa_checkpoint_from=checkpoint_token,
                restore_last_window_tokens=self.layout.window_size,
            )
        recompute_window = self.layout.window_size * self.layout.num_layers
        recompute_start = max(0, prefix_tokens - recompute_window)
        return RestorePlan(
            reusable_prefix_tokens=reusable,
            compressed_prefix_tokens=reusable,
            recompute_start_token=recompute_start,
            needs_tail_recompute=(recompute_start != prefix_tokens),
            swa_mode=self.swa_mode.value,
            restore_last_window_tokens=self.layout.window_size,
        )


@dataclass
class PrefixReuseResult:
    metadata: PrefixCacheMetadata
    compressed_cache: Dict[str, torch.Tensor]
    swa_state: Optional[Dict[str, torch.Tensor]]
    restore_plan: RestorePlan


class LongContextServingManager:
    """Coordinates on-disk prefix reuse and kernel backend selection."""

    def __init__(
        self,
        cache_store: OnDiskPrefixKVStore,
        backend: Optional[AttentionKernelBackend] = None,
    ) -> None:
        self.cache_store = cache_store
        self.backend = backend or PytorchAttentionBackend()

    def backend_name(self) -> str:
        return self.backend.name()

    def save_prefix(
        self,
        token_ids: List[int],
        compressed_cache: Dict[str, torch.Tensor],
        swa_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> PrefixCacheMetadata:
        return self.cache_store.save_prefix(token_ids, compressed_cache, swa_state=swa_state)

    def reuse_prefix(self, token_ids: List[int]) -> Optional[PrefixReuseResult]:
        metadata = self.cache_store.lookup_prefix(token_ids)
        if metadata is None:
            return None
        return PrefixReuseResult(
            metadata=metadata,
            compressed_cache=self.cache_store.load_compressed_cache(metadata),
            swa_state=self.cache_store.load_swa_state(metadata),
            restore_plan=self.cache_store.make_restore_plan(len(token_ids)),
        )
