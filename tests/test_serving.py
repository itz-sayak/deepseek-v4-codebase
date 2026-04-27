import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_pipeline.serving import (
    HybridCacheLayout,
    OnDiskPrefixKVStore,
    PytorchAttentionBackend,
    SWACacheMode,
)


def test_layout_block_sizes():
    layout = HybridCacheLayout(csa_compression=8, hca_compression=32, window_size=64, num_layers=6)
    assert layout.block_tokens == 32
    assert layout.csa_tokens_per_block == 4
    assert layout.hca_tokens_per_block == 1
    assert layout.full_block_prefix(99) == 96


def test_short_prefix_is_not_cacheable(tmp_path):
    layout = HybridCacheLayout(csa_compression=8, hca_compression=16, window_size=32, num_layers=4)
    store = OnDiskPrefixKVStore(str(tmp_path), layout, swa_mode=SWACacheMode.FULL)
    with pytest.raises(ValueError, match="does not contain a full compression block"):
        store.save_prefix([1, 2, 3], {"csa": torch.randn(1, 2, 2, 8)})
    assert store.lookup_prefix([1, 2, 3]) is None


def test_periodic_prefix_cache_round_trip(tmp_path):
    layout = HybridCacheLayout(csa_compression=8, hca_compression=16, window_size=32, num_layers=4)
    store = OnDiskPrefixKVStore(str(tmp_path), layout, swa_mode=SWACacheMode.PERIODIC, checkpoint_stride=64)
    token_ids = list(range(174))
    compressed_cache = {
        "csa": torch.randn(2, 20, 2, 16),
        "hca": torch.randn(2, 10, 2, 16),
    }
    swa_state = {"layer0": torch.randn(2, 32, 16)}

    metadata = store.save_prefix(token_ids, compressed_cache, swa_state=swa_state)
    assert metadata.reusable_token_count == 160
    assert metadata.files["swa_periodic"].endswith("swa_ckpt_128.pt")

    loaded_metadata = store.lookup_prefix(token_ids)
    assert loaded_metadata is not None
    loaded_cache = store.load_compressed_cache(loaded_metadata)
    loaded_swa = store.load_swa_state(loaded_metadata)
    plan = store.make_restore_plan(len(token_ids))

    assert torch.equal(loaded_cache["csa"], compressed_cache["csa"])
    assert torch.equal(loaded_cache["hca"], compressed_cache["hca"])
    assert torch.equal(loaded_swa["layer0"], swa_state["layer0"])
    assert plan.reusable_prefix_tokens == 160
    assert plan.recompute_start_token == 128
    assert plan.needs_tail_recompute is True


def test_zero_swa_restore_plan(tmp_path):
    layout = HybridCacheLayout(csa_compression=8, hca_compression=16, window_size=32, num_layers=4)
    store = OnDiskPrefixKVStore(str(tmp_path), layout, swa_mode=SWACacheMode.ZERO)
    plan = store.make_restore_plan(190)
    assert plan.reusable_prefix_tokens == 176
    assert plan.recompute_start_token == 62
    assert plan.restore_last_window_tokens == 32


def test_pytorch_attention_backend_shapes_are_valid():
    torch.manual_seed(0)
    backend = PytorchAttentionBackend()
    q = torch.randn(2, 3, 4, 8)
    kv = torch.randn(2, 12, 2, 8)
    topk = torch.randint(0, 12, (2, 3, 5))
    sink = torch.zeros(4)
    out = backend.sparse_attention(q, kv, topk, sink, softmax_scale=0.5)
    assert out.shape == (2, 3, 4, 8)
    assert torch.isfinite(out).all()
