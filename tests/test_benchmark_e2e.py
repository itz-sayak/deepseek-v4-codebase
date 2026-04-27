import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from deepseek_v4_pro_2b.modeling import DeepSeekV4Pro2BForCausalLM
from scripts.benchmark_e2e_serving import _tiny_config, run_benchmark
from deepseek_pipeline.serving import SWACacheMode


def test_run_benchmark_uses_scheduler_batching(tmp_path):
    model = DeepSeekV4Pro2BForCausalLM(_tiny_config()).eval()
    result = run_benchmark(
        model=model,
        source_tokens=32,
        decode_tokens=4,
        batch_size=2,
        device=torch.device("cpu"),
        use_prefix_cache=True,
        prefix_cache_dir=str(tmp_path),
        swa_mode=SWACacheMode.PERIODIC,
        backend_name="pytorch",
        allocator_device_pages=0,
        allocator_host_pages=0,
        warmup_runs=0,
        bench_runs=1,
    )
    assert result["batch_size"] == 2
    assert result["backend"] == "pytorch"
    assert result["use_prefix_cache"] is True
    assert result["prefill_tokens_per_s"] > 0
    assert result["decode_tokens_per_s"] > 0


def test_run_benchmark_can_enable_allocator_backed_prefixes(tmp_path):
    model = DeepSeekV4Pro2BForCausalLM(_tiny_config()).eval()
    result = run_benchmark(
        model=model,
        source_tokens=32,
        decode_tokens=4,
        batch_size=2,
        device=torch.device("cpu"),
        use_prefix_cache=True,
        prefix_cache_dir=str(tmp_path / "alloc"),
        swa_mode=SWACacheMode.FULL,
        backend_name="pytorch",
        allocator_device_pages=4,
        allocator_host_pages=8,
        warmup_runs=0,
        bench_runs=1,
    )
    assert result["allocator_device_pages"] == 4
    assert result["allocator_host_pages"] == 8
    assert result["prefill_tokens_per_s"] > 0
