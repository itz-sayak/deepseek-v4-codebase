import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_kernels.paged_kv_allocator import PagedKVAllocator


def test_paged_allocator_spills_to_disk_and_keeps_handle_slots_consistent(tmp_path):
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()

    def spill_path(page_key: str) -> Path:
        return spill_dir / f"{page_key.replace(':', '_')}.pt"

    def write_to_disk(page_key: str, tensors: dict[str, torch.Tensor]) -> None:
        torch.save({name: tensor.cpu() for name, tensor in tensors.items()}, spill_path(page_key))

    def read_from_disk(page_key: str) -> dict[str, torch.Tensor]:
        return torch.load(spill_path(page_key), map_location="cpu", weights_only=True)

    allocator = PagedKVAllocator(
        num_device_pages=1,
        num_host_pages=1,
        page_shape={"hca": (1, 4)},
        dtype=torch.float32,
        device="cpu",
        write_to_disk=write_to_disk,
        read_from_disk=read_from_disk,
    )
    handle = allocator.allocate("seq")

    for value in (1.0, 2.0, 3.0):
        allocator.append_page(handle, {"hca": torch.full((1, 4), value)})

    assert handle.page_slots[0] == (None, None)
    assert handle.page_slots[1][0] is None and handle.page_slots[1][1] is not None
    assert handle.page_slots[2][0] is not None and handle.page_slots[2][1] is None
    assert allocator.device_pages_used() == 1
    assert allocator.host_pages_used() == 1

    pages = allocator.load_pages(handle, (0, 3))
    restored = [page["hca"][0, 0].item() for page in pages]
    assert restored == [1.0, 2.0, 3.0]

    allocator.free(handle)
    assert allocator.device_pages_used() == 0
    assert allocator.host_pages_used() == 0
    assert handle.page_slots == []
