"""Paged KV cache memory allocator for long-context serving.

Implements the paper's design where compressed KV blocks are evicted from device
memory to host RAM or NVMe when device memory is exhausted and paged back in on
demand.  Each "page" is one HybridCacheLayout.block_tokens-wide compressed slice
covering all layers.

This module is pure Python so it runs with or without the CUDA extension.

Usage
-----
    allocator = PagedKVAllocator(
        num_device_pages=512,
        num_host_pages=4096,
        page_shape={"hca_compressed": (num_hca_layers, head_dim),
                    "csa_compressed": (num_csa_layers, head_dim),
                    "csa_index":      (num_csa_layers, index_head_dim)},
        dtype=torch.float16,
        device="cuda",
    )

    handle = allocator.allocate(seq_id="req-0")
    allocator.append_page(handle, page_tensors)
    pages = allocator.load_pages(handle, page_range=(0, 32))
    allocator.free(handle)
"""

from __future__ import annotations
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class PageHandle:
    """Opaque handle returned by PagedKVAllocator.allocate()."""

    seq_id: str
    # Ordered list of (device_slot, host_slot) pairs.  Exactly one is not None.
    page_slots: List[Tuple[Optional[int], Optional[int]]] = field(default_factory=list)


class PagedKVAllocator:
    """Two-tier LRU page pool: device HBM (fast) and host RAM (slow).

    Thread-safe via a single lock.  Eviction from device -> host is triggered
    whenever a device-page allocation would exceed ``num_device_pages``.

    Parameters
    ----------
    num_device_pages : int
        Maximum number of pages kept in device (GPU) memory simultaneously.
    num_host_pages : int
        Maximum number of pages kept in host (CPU) memory simultaneously.
        Pages beyond this limit are evicted to disk (write_to_disk must be provided).
    page_shape : dict[str, tuple[int, ...]]
        Shape of the **per-page** tensor for each named KV stream.
        The allocator pre-allocates a pool of physical page buffers.
    dtype : torch.dtype
    device : str or torch.device
    write_to_disk : callable, optional
        write_to_disk(page_key: str, tensors: dict[str, torch.Tensor]) called when
        the host pool is full.  Must be provided if host overflow is expected.
    read_from_disk : callable, optional
        read_from_disk(page_key: str) -> dict[str, torch.Tensor]  Called to page
        data back in from disk.
    """

    def __init__(
        self,
        num_device_pages: int,
        num_host_pages: int,
        page_shape: Dict[str, Tuple[int, ...]],
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        write_to_disk=None,
        read_from_disk=None,
    ) -> None:
        self._lock = threading.Lock()
        self._dtype = dtype
        self._device = torch.device(device)
        self._host_device = torch.device("cpu")
        self._page_shape = page_shape
        self._write_to_disk = write_to_disk
        self._read_from_disk = read_from_disk

        # Physical device page pool: list of dicts of tensors.
        self._device_pool: List[Dict[str, torch.Tensor]] = [
            {name: torch.empty(shape, dtype=dtype, device=self._device)
             for name, shape in page_shape.items()}
            for _ in range(num_device_pages)
        ]
        self._device_free: List[int] = list(range(num_device_pages))
        # LRU tracking: slot -> page_key
        self._device_lru: OrderedDict[int, str] = OrderedDict()

        # Physical host page pool.
        self._host_pool: List[Dict[str, torch.Tensor]] = [
            {name: torch.empty(shape, dtype=dtype, device=self._host_device)
             for name, shape in page_shape.items()}
            for _ in range(num_host_pages)
        ]
        self._host_free: List[int] = list(range(num_host_pages))
        self._host_lru: OrderedDict[int, str] = OrderedDict()

        # page_key -> (device_slot | None, host_slot | None)
        self._page_map: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        self._page_owner: Dict[str, Tuple[PageHandle, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, seq_id: str) -> PageHandle:
        """Reserve a sequence handle.  No pages are allocated yet."""
        return PageHandle(seq_id=seq_id)

    def append_page(
        self,
        handle: PageHandle,
        page_tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Append one compressed KV page to ``handle``'s sequence."""
        page_index = len(handle.page_slots)
        page_key = f"{handle.seq_id}:{page_index}"
        with self._lock:
            slot = self._alloc_device_page(page_key)
            # Copy data into the physical slot.
            for name, tensor in page_tensors.items():
                self._device_pool[slot][name].copy_(tensor)
            self._device_lru[slot] = page_key
            self._page_map[page_key] = (slot, None)
            handle.page_slots.append((slot, None))
            self._page_owner[page_key] = (handle, page_index)

    def load_pages(
        self,
        handle: PageHandle,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Return device-resident copies of pages in ``page_range``.

        Pages that were evicted to host or disk are paged back in first.
        """
        start, stop = page_range if page_range is not None else (0, len(handle.page_slots))
        result = []
        with self._lock:
            for page_index in range(start, stop):
                page_key = f"{handle.seq_id}:{page_index}"
                page_slot, host_slot = self._page_map.get(page_key, (None, None))
                if page_slot is not None:
                    # Already on device; touch LRU.
                    self._device_lru.move_to_end(page_slot)
                    result.append({k: v.clone() for k, v in self._device_pool[page_slot].items()})
                elif host_slot is not None:
                    # On host; page in to device.
                    dev_slot = self._alloc_device_page(page_key)
                    for name in self._page_shape:
                        self._device_pool[dev_slot][name].copy_(self._host_pool[host_slot][name])
                    self._device_lru[dev_slot] = page_key
                    self._page_map[page_key] = (dev_slot, None)
                    self._host_free.append(host_slot)
                    if host_slot in self._host_lru:
                        del self._host_lru[host_slot]
                    # Update handle's slot record.
                    handle.page_slots[page_index] = (dev_slot, None)
                    result.append({k: v.clone() for k, v in self._device_pool[dev_slot].items()})
                else:
                    # On disk; read back.
                    if self._read_from_disk is None:
                        raise RuntimeError(f"Page {page_key} has been evicted to disk but no read_from_disk handler was provided.")
                    disk_tensors = self._read_from_disk(page_key)
                    dev_slot = self._alloc_device_page(page_key)
                    for name in self._page_shape:
                        self._device_pool[dev_slot][name].copy_(disk_tensors[name])
                    self._device_lru[dev_slot] = page_key
                    self._page_map[page_key] = (dev_slot, None)
                    handle.page_slots[page_index] = (dev_slot, None)
                    result.append({k: v.clone() for k, v in self._device_pool[dev_slot].items()})
        return result

    def free(self, handle: PageHandle) -> None:
        """Release all pages belonging to ``handle``."""
        with self._lock:
            for page_index, (dev_slot, host_slot) in enumerate(handle.page_slots):
                page_key = f"{handle.seq_id}:{page_index}"
                if dev_slot is not None:
                    self._device_free.append(dev_slot)
                    if dev_slot in self._device_lru:
                        del self._device_lru[dev_slot]
                if host_slot is not None:
                    self._host_free.append(host_slot)
                    if host_slot in self._host_lru:
                        del self._host_lru[host_slot]
                self._page_map.pop(page_key, None)
                self._page_owner.pop(page_key, None)
            handle.page_slots.clear()

    def device_pages_used(self) -> int:
        with self._lock:
            return len(self._device_lru)

    def host_pages_used(self) -> int:
        with self._lock:
            return len(self._host_lru)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _alloc_device_page(self, page_key: str) -> int:
        """Return a free device slot, evicting LRU if needed (lock held)."""
        if self._device_free:
            return self._device_free.pop()
        # Evict LRU device page to host.
        lru_slot, lru_key = next(iter(self._device_lru.items()))
        del self._device_lru[lru_slot]
        host_slot = self._alloc_host_page(lru_key)
        for name in self._page_shape:
            self._host_pool[host_slot][name].copy_(self._device_pool[lru_slot][name])
        self._host_lru[host_slot] = lru_key
        self._page_map[lru_key] = (None, host_slot)
        owner = self._page_owner.get(lru_key)
        if owner is not None:
            handle, page_index = owner
            handle.page_slots[page_index] = (None, host_slot)
        return lru_slot

    def _alloc_host_page(self, page_key: str) -> int:
        """Return a free host slot, evicting LRU host page to disk if needed (lock held)."""
        if self._host_free:
            return self._host_free.pop()
        if not self._host_lru:
            raise MemoryError("Both device and host KV page pools are full and no host page can be evicted.")
        lru_slot, lru_key = next(iter(self._host_lru.items()))
        del self._host_lru[lru_slot]
        if self._write_to_disk is None:
            raise MemoryError(
                f"Host KV page pool is full but no write_to_disk handler was provided; "
                f"increase num_host_pages or supply write_to_disk."
            )
        disk_tensors = {name: self._host_pool[lru_slot][name].clone() for name in self._page_shape}
        self._write_to_disk(lru_key, disk_tensors)
        self._page_map[lru_key] = (None, None)  # now on disk
        owner = self._page_owner.get(lru_key)
        if owner is not None:
            handle, page_index = owner
            handle.page_slots[page_index] = (None, None)
        return lru_slot
