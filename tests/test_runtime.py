import json
import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_pipeline.download import save_source
from deepseek_pipeline.manifest import SourceSpec
from deepseek_pipeline.preprocess import load_preprocessed_source
from deepseek_v4_pro_2b.configuration import DeepSeekV4Pro2BConfig
from deepseek_v4_pro_2b.modeling import DeepSeekV4Pro2BForCausalLM
from deepseek_v4_pro_2b.muon import Muon


def tiny_config():
    return DeepSeekV4Pro2BConfig(
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


def test_forward_with_padding_and_labels():
    model = DeepSeekV4Pro2BForCausalLM(tiny_config())
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0, 0, 0],
            [6, 7, 8, 9, 10, 11, 12, 13],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    out = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert out.logits.shape == (2, 8, 128)
    assert len(out.mtp_logits) == 1
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.balance_loss)


def test_all_ignored_and_fully_masked_batch_is_finite():
    cfg = tiny_config()
    cfg.hidden_size = 48
    cfg.attention_head_dim = 12
    cfg.group_output_dim = 24
    cfg.query_compression_dim = 24
    cfg.indexer_head_dim = 6
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    input_ids = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    out = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.balance_loss)


def test_muon_step_runs():
    weight = torch.nn.Parameter(torch.randn(8, 4))
    opt = Muon([weight], lr=1e-3)
    loss = (weight ** 2).sum()
    loss.backward()
    opt.step()
    assert torch.isfinite(weight).all()


def test_download_resume_recovers_from_partial_source(tmp_path, monkeypatch):
    spec = SourceSpec(
        name="dummy_source",
        stage="pretrain",
        hf_name="dummy/repo",
        split="train",
        text_kind="text",
    )
    raw_items = [
        {"text": "alpha"},
        {"text": "beta"},
        {"text": "gamma"},
    ]
    starts = []
    fail_once = {"enabled": True}

    def fake_iter_source_items(spec_arg, start_raw_index=0):
        assert spec_arg == spec
        starts.append(start_raw_index)
        for index in range(start_raw_index, len(raw_items)):
            if fail_once["enabled"] and index == 2:
                fail_once["enabled"] = False
                raise RuntimeError("stream interrupted")
            yield index, raw_items[index]

    monkeypatch.setattr("deepseek_pipeline.download._iter_source_items", fake_iter_source_items)

    output_root = tmp_path / "datasets"
    with pytest.raises(RuntimeError, match="stream interrupted"):
        save_source(spec, str(output_root), max_samples=None, force=False, shard_size=16)

    output_dir = output_root / spec.stage / spec.name
    interrupted_state = json.loads((output_dir / "_resume_state.json").read_text())
    assert interrupted_state["completed"] is False
    assert interrupted_state["raw_items_seen"] == 2
    assert interrupted_state["kept_records"] == 2

    path = save_source(spec, str(output_root), max_samples=None, force=False, shard_size=16)
    assert path == str(output_dir)
    resumed_state = json.loads((output_dir / "_resume_state.json").read_text())
    assert resumed_state["completed"] is True
    assert starts == [0, 2]

    dataset = load_preprocessed_source(str(output_dir))
    assert len(dataset) == 3
    assert dataset[0]["text"] == "alpha"
    assert dataset[2]["text"] == "gamma"


# ---------------------------------------------------------------------------
# Serving: fast_prefill correctness
# ---------------------------------------------------------------------------

def _make_serving_engine(config, seed=0):
    """Construct a tiny model + ServingEngine with a fixed seed."""
    from deepseek_v4_pro_2b.serving import ServingEngine
    torch.manual_seed(seed)
    model = DeepSeekV4Pro2BForCausalLM(config)
    model.eval()
    return ServingEngine(model)


def test_fast_prefill_matches_step_token_next_logits():
    """fast_prefill state must produce identical next-token logits to prefill()."""
    from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine
    from deepseek_pipeline.serving import PytorchAttentionBackend
    cfg = tiny_config()
    torch.manual_seed(42)
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    model.eval()

    token_ids = torch.randint(0, cfg.vocab_size, (12,))

    backend = PytorchAttentionBackend()
    engine_slow = DeepSeekV4Pro2BServingEngine(model, backend=backend)
    state_slow = engine_slow.prefill(token_ids)

    engine_fast = DeepSeekV4Pro2BServingEngine(model, backend=backend)
    state_fast = engine_fast.fast_prefill(token_ids)

    next_token = torch.tensor([7])
    logits_slow, _ = engine_slow.step_token(next_token, state_slow)
    logits_fast, _ = engine_fast.step_token(next_token, state_fast)

    assert logits_slow.shape == logits_fast.shape, "logit shape mismatch"
    assert torch.allclose(logits_slow, logits_fast, atol=1e-4), (
        f"fast_prefill diverged from prefill: max_diff={(logits_slow - logits_fast).abs().max().item():.6f}"
    )


def test_offload_restore_preserves_step_token():
    """offload_swa_to_host + restore_swa_to_device must preserve step_token output."""
    from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine
    from deepseek_pipeline.serving import PytorchAttentionBackend
    cfg = tiny_config()
    torch.manual_seed(7)
    model = DeepSeekV4Pro2BForCausalLM(cfg)
    model.eval()

    token_ids = torch.randint(0, cfg.vocab_size, (8,))
    next_token = torch.tensor([3])

    engine = DeepSeekV4Pro2BServingEngine(model, backend=PytorchAttentionBackend())
    state = engine.prefill(token_ids)

    import copy
    state_base = copy.deepcopy(state)
    logits_base, _ = engine.step_token(next_token, state_base)

    state_offload = copy.deepcopy(state)
    engine.offload_swa_to_host(state_offload)
    engine.restore_swa_to_device(state_offload)
    logits_restored, _ = engine.step_token(next_token, state_offload)

    assert torch.allclose(logits_base, logits_restored, atol=1e-5), (
        "offload/restore changed step_token output"
    )
