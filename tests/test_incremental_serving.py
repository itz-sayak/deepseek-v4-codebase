import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aether_kernels.paged_kv_allocator import PagedKVAllocator
from aether_pipeline.serving import HybridCacheLayout, LongContextServingManager, OnDiskPrefixKVStore, PytorchAttentionBackend, SWACacheMode
from aether_2b.configuration import Aether2BConfig
from aether_2b.modeling import Aether2BForCausalLM
from aether_2b.serving import Aether2BServingEngine


def tiny_config():
    return Aether2BConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        attention_head_dim=16,
        query_compression_dim=32,
        indexer_num_heads=2,
        indexer_head_dim=8,
        csa_compression=2,
        hca_compression=4,
        csa_top_k=3,
        sliding_window=4,
        rope_dim=8,
        output_groups=2,
        group_output_dim=32,
        num_routed_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        hash_routed_layers=1,
        mhc_expansion=2,
        mtp_depth=1,
    )


def make_engine(tmp_path=None, swa_mode=SWACacheMode.FULL, checkpoint_stride=8):
    model = Aether2BForCausalLM(tiny_config()).eval()
    backend = PytorchAttentionBackend()
    if tmp_path is None:
        return model, Aether2BServingEngine(model, backend=backend, device="cpu")
    layout = HybridCacheLayout(
        csa_compression=model.config.csa_compression,
        hca_compression=model.config.hca_compression,
        window_size=model.config.sliding_window,
        num_layers=model.config.num_hidden_layers,
    )
    store = OnDiskPrefixKVStore(str(tmp_path), layout, swa_mode=swa_mode, checkpoint_stride=checkpoint_stride)
    manager = LongContextServingManager(store, backend=backend)
    return model, Aether2BServingEngine(model, prefix_manager=manager, device="cpu")


def make_engine_with_allocator(tmp_path, swa_mode=SWACacheMode.FULL, checkpoint_stride=8):
    model = Aether2BForCausalLM(tiny_config()).eval()
    backend = PytorchAttentionBackend()
    layout = HybridCacheLayout(
        csa_compression=model.config.csa_compression,
        hca_compression=model.config.hca_compression,
        window_size=model.config.sliding_window,
        num_layers=model.config.num_hidden_layers,
    )
    store = OnDiskPrefixKVStore(str(tmp_path / "prefix"), layout, swa_mode=swa_mode, checkpoint_stride=checkpoint_stride)
    shape_probe = Aether2BServingEngine(model, backend=backend, device="cpu")
    allocator = PagedKVAllocator(
        num_device_pages=4,
        num_host_pages=8,
        page_shape=shape_probe.allocator_page_shape(),
        dtype=model.lm_head.weight.dtype,
        device="cpu",
    )
    manager = LongContextServingManager(store, backend=backend)
    return model, allocator, Aether2BServingEngine(
        model,
        prefix_manager=manager,
        paged_allocator=allocator,
        device="cpu",
    )


def test_incremental_logits_match_full_forward():
    model, engine = make_engine()
    tokens = [1, 7, 3, 9, 2, 11, 5, 4]
    full = model(torch.tensor([tokens], dtype=torch.long)).logits[0]
    state = None
    stepped = []
    for token in tokens:
        logits, state = engine.step_token(token, state)
        stepped.append(logits[0])
    got = torch.stack(stepped, dim=0)
    torch.testing.assert_close(got, full, atol=1e-5, rtol=1e-5)


def test_full_prefix_reuse_matches_fresh_prefill(tmp_path):
    _, engine = make_engine(tmp_path=tmp_path, swa_mode=SWACacheMode.FULL)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    engine.build_prefix_cache(tokens)
    fresh = engine.prefill(tokens)
    reused = engine.prefill_with_reuse(tokens)
    torch.testing.assert_close(reused.last_logits, fresh.last_logits, atol=1e-5, rtol=1e-5)
    assert engine.generate(tokens, max_new_tokens=3, temperature=0.0, use_prefix_cache=False) == engine.generate(
        tokens, max_new_tokens=3, temperature=0.0, use_prefix_cache=True
    )


def test_periodic_prefix_reuse_replays_exactly(tmp_path):
    _, engine = make_engine(tmp_path=tmp_path, swa_mode=SWACacheMode.PERIODIC, checkpoint_stride=8)
    tokens = [1, 7, 3, 9, 2, 11, 5, 4, 8, 6, 10, 12, 14, 15, 16, 17, 18]
    engine.build_prefix_cache(tokens)
    fresh = engine.prefill(tokens)
    reused = engine.prefill_with_reuse(tokens)
    torch.testing.assert_close(reused.last_logits, fresh.last_logits, atol=1e-5, rtol=1e-5)


def test_zero_swa_reuse_replays_exactly(tmp_path):
    _, engine = make_engine(tmp_path=tmp_path, swa_mode=SWACacheMode.ZERO, checkpoint_stride=8)
    tokens = [1, 7, 3, 9, 2, 11, 5, 4, 8, 6, 10, 12, 14, 15, 16, 17, 18]
    engine.build_prefix_cache(tokens)
    fresh = engine.prefill(tokens)
    reused = engine.prefill_with_reuse(tokens)
    torch.testing.assert_close(reused.last_logits, fresh.last_logits, atol=1e-5, rtol=1e-5)


def test_allocator_backed_prefix_reuse_matches_fresh_and_frees_pages(tmp_path):
    _, allocator, engine = make_engine_with_allocator(tmp_path=tmp_path, swa_mode=SWACacheMode.FULL)
    tokens = [1, 7, 3, 9, 2, 11, 5, 4, 8, 6, 10, 12, 14, 15, 16, 17]
    engine.build_prefix_cache(tokens)
    fresh = engine.prefill(tokens)
    reused = engine.prefill_with_reuse(tokens)
    torch.testing.assert_close(reused.last_logits, fresh.last_logits, atol=1e-5, rtol=1e-5)
    assert reused.paged_prefix is not None
    assert allocator.device_pages_used() > 0
    engine.free_state(reused)
    assert allocator.device_pages_used() == 0


def test_allocator_flushes_live_compressed_tail_pages(tmp_path):
    _, allocator, engine = make_engine_with_allocator(tmp_path=tmp_path, swa_mode=SWACacheMode.FULL)
    state = None
    tokens = list(range(1, 25))
    for token in tokens:
        _, state = engine.step_token(token, state)
    assert state.paged_prefix is not None
    assert state.paged_prefix.page_count > 0
    assert allocator.device_pages_used() > 0
    engine.free_state(state)
    assert allocator.device_pages_used() == 0
