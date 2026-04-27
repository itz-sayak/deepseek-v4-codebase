"""End-to-end long-context throughput benchmark.

Drives the full serving stack — tokenise, prefill, decode — and reports
tokens/second at user-specified source token lengths.  Also tracks peak
HBM usage, prefill latency, and per-token decode latency.

Usage
-----
    # CPU reference run (no CUDA required):
    python scripts/benchmark_e2e_serving.py --preset tiny --source-tokens 512 1024

    # GPU run with tiled prefill kernel:
    conda run -n deepfill bash -lc '
      export CUDA_HOME=$CONDA_PREFIX
      export PYTHONPATH=/home/seema/deepseek-v4:$PYTHONPATH
      python scripts/benchmark_e2e_serving.py \\
        --preset 2b \\
        --source-tokens 8192 65536 131072 \\
        --decode-tokens 128 \\
        --batch-size 1 \\
        --dtype bf16 \\
        --use-prefix-cache
    '
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from deepseek_kernels.paged_kv_allocator import PagedKVAllocator
from deepseek_v4_pro_2b import DeepSeekV4Pro2BConfig, DeepSeekV4Pro2BForCausalLM
from deepseek_v4_pro_2b.scheduler import DecodeScheduler, GenerationRequest
from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine
from deepseek_pipeline.serving import (
    CudaSparseAttentionBackend,
    HybridCacheLayout,
    LongContextServingManager,
    OnDiskPrefixKVStore,
    PytorchAttentionBackend,
    SWACacheMode,
)


def _tiny_config() -> DeepSeekV4Pro2BConfig:
    cfg = DeepSeekV4Pro2BConfig(
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
        sliding_window=8,
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
    return cfg


def _bytes_to_mib(b: int) -> float:
    return b / (1024 ** 2)


def _peak_memory_mib(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return _bytes_to_mib(torch.cuda.max_memory_allocated(device))


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _make_tokens(n: int, vocab_size: int) -> List[int]:
    """Deterministic pseudo-random token sequence."""
    import random
    rng = random.Random(42)
    return [rng.randint(1, vocab_size - 1) for _ in range(n)]


def run_benchmark(
    model: DeepSeekV4Pro2BForCausalLM,
    source_tokens: int,
    decode_tokens: int,
    batch_size: int,
    device: torch.device,
    use_prefix_cache: bool,
    prefix_cache_dir: str,
    swa_mode: SWACacheMode,
    backend_name: str,
    allocator_device_pages: int,
    allocator_host_pages: int,
    warmup_runs: int,
    bench_runs: int,
    fast_prefill: bool = False,
    swa_offload: bool = False,
    num_gpus: int = 1,
) -> dict:
    vocab_size = model.config.vocab_size
    cfg = model.config

    layout = HybridCacheLayout(
        csa_compression=cfg.csa_compression,
        hca_compression=cfg.hca_compression,
        window_size=cfg.sliding_window,
        num_layers=cfg.num_hidden_layers,
    )

    if backend_name == "cuda":
        backend = CudaSparseAttentionBackend()
        if not backend.available():
            raise RuntimeError(backend.availability_error() or "CUDA backend is unavailable")
    elif backend_name == "auto":
        cuda_backend = CudaSparseAttentionBackend()
        backend = cuda_backend if device.type == "cuda" and cuda_backend.available() else PytorchAttentionBackend()
    else:
        backend = PytorchAttentionBackend()

    paged_allocator = None
    if allocator_device_pages > 0 and allocator_host_pages > 0:
        spill_dir = Path(prefix_cache_dir) / "allocator_spill"
        spill_dir.mkdir(parents=True, exist_ok=True)

        def _spill_path(page_key: str) -> Path:
            safe_name = page_key.replace(os.sep, "_").replace(":", "_")
            return spill_dir / f"{safe_name}.pt"

        def _write_to_disk(page_key: str, tensors: dict[str, torch.Tensor]) -> None:
            torch.save({name: tensor.cpu() for name, tensor in tensors.items()}, _spill_path(page_key))

        def _read_from_disk(page_key: str) -> dict[str, torch.Tensor]:
            payload = torch.load(_spill_path(page_key), map_location="cpu", weights_only=True)
            return {
                name: tensor.to(device=device, dtype=model.lm_head.weight.dtype)
                for name, tensor in payload.items()
            }

        probe_engine = DeepSeekV4Pro2BServingEngine(model, backend=backend, device=device)
        paged_allocator = PagedKVAllocator(
            num_device_pages=allocator_device_pages,
            num_host_pages=allocator_host_pages,
            page_shape=probe_engine.allocator_page_shape(),
            dtype=model.lm_head.weight.dtype,
            device=device,
            write_to_disk=_write_to_disk,
            read_from_disk=_read_from_disk,
        )

    if use_prefix_cache:
        store = OnDiskPrefixKVStore(
            root_dir=prefix_cache_dir,
            layout=layout,
            swa_mode=swa_mode,
            checkpoint_stride=max(layout.block_tokens, 4096),
        )
        manager = LongContextServingManager(
            cache_store=store,
            backend=backend,
        )
        engine = DeepSeekV4Pro2BServingEngine(model, prefix_manager=manager, paged_allocator=paged_allocator, device=device)
    else:
        engine = DeepSeekV4Pro2BServingEngine(model, backend=backend, paged_allocator=paged_allocator, device=device)

    if num_gpus > 1:
        engine.shard_across_gpus(num_gpus)

    prompt_ids = _make_tokens(source_tokens, vocab_size)
    batched_prompts = []
    for batch_idx in range(batch_size):
        if batch_idx == 0:
            batched_prompts.append(prompt_ids)
        else:
            # share almost all of the prefix so the scheduler can exercise prefix reuse
            shared = prompt_ids[:-min(32, len(prompt_ids))] if len(prompt_ids) > 32 else prompt_ids[:]
            suffix = _make_tokens(min(32, len(prompt_ids)), vocab_size)
            batched_prompts.append((shared + suffix)[:source_tokens])

    # Warm-up runs (not timed).
    for _ in range(warmup_runs):
        _ = engine.fast_prefill(batched_prompts[0][:min(16, source_tokens)]) if fast_prefill else engine.prefill(batched_prompts[0][:min(16, source_tokens)])
        gc.collect()

    prefill_times: List[float] = []
    decode_times: List[float] = []
    peak_hbm_mib: List[Optional[float]] = []

    for run_idx in range(bench_runs):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _reset_peak_memory(device)

        requests = [
            GenerationRequest(
                seq_id=f"bench-{run_idx}-{batch_idx}",
                prompt_ids=batched_prompts[batch_idx],
                max_new_tokens=decode_tokens,
                temperature=0.0,
                eos_token_id=None,
            )
            for batch_idx in range(batch_size)
        ]
        t0 = time.perf_counter()
        for request in requests:
            if fast_prefill:
                request._state = engine.fast_prefill(request.prompt_ids)
            elif use_prefix_cache:
                request._state = engine.prefill_with_reuse(request.prompt_ids)
            else:
                request._state = engine.prefill(request.prompt_ids)
            request._output_ids = list(request.prompt_ids)
            request._pending_logits = request._state.last_logits
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        prefill_times.append(t1 - t0)

        scheduler = DecodeScheduler(engine, max_batch_size=batch_size, greedy=True, swa_offload=swa_offload)
        for request in requests:
            scheduler.submit(request)
        t2 = time.perf_counter()
        finished: List[object] = []
        for result in scheduler.run():
            finished.extend(result.finished)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        decode_times.append(t3 - t2)
        if len(finished) != batch_size:
            raise RuntimeError(f"Expected {batch_size} finished requests, got {len(finished)}")
        peak_hbm_mib.append(_peak_memory_mib(device))

    avg_prefill_s = sum(prefill_times) / len(prefill_times)
    avg_decode_s  = sum(decode_times) / len(decode_times)

    total_prefill_tokens = source_tokens * batch_size
    total_decode_tokens = decode_tokens * batch_size
    prefill_tps = total_prefill_tokens / avg_prefill_s if avg_prefill_s > 0 else float("nan")
    decode_tps  = total_decode_tokens / avg_decode_s  if avg_decode_s  > 0 else float("nan")
    valid_hbm   = [v for v in peak_hbm_mib if v is not None]

    return {
        "source_tokens": source_tokens,
        "decode_tokens": decode_tokens,
        "batch_size": batch_size,
        "use_prefix_cache": use_prefix_cache,
        "backend": engine.backend.name(),
        "allocator_device_pages": allocator_device_pages,
        "allocator_host_pages": allocator_host_pages,
        "swa_mode": swa_mode.value,
        "fast_prefill": fast_prefill,
        "swa_offload": swa_offload,
        "num_gpus": num_gpus,
        "bench_runs": bench_runs,
        "avg_prefill_s": round(avg_prefill_s, 4),
        "avg_decode_s": round(avg_decode_s, 4),
        "prefill_tokens_per_s": round(prefill_tps, 1),
        "decode_tokens_per_s": round(decode_tps, 1),
        "ms_per_decode_token": round(avg_decode_s * 1000 / max(total_decode_tokens, 1), 3),
        "peak_hbm_mib": round(max(valid_hbm), 1) if valid_hbm else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end serving throughput benchmark.")
    parser.add_argument("--preset", choices=["2b", "tiny"], default="tiny",
                        help="Model size preset (tiny for CPU testing, 2b for GPU)")
    parser.add_argument("--source-tokens", type=int, nargs="+", default=[512, 1024],
                        help="Source sequence length(s) to benchmark")
    parser.add_argument("--decode-tokens", type=int, default=32,
                        help="Number of decode steps per run")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of concurrent requests to benchmark through the decode scheduler")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--backend", choices=["auto", "pytorch", "cuda"], default="auto")
    parser.add_argument("--use-prefix-cache", action="store_true",
                        help="Enable on-disk prefix KV cache reuse")
    parser.add_argument("--swa-mode", choices=["full", "periodic", "zero"], default="periodic")
    parser.add_argument("--prefix-cache-dir", default="/tmp/deepseek_prefix_cache")
    parser.add_argument("--allocator-device-pages", type=int, default=0)
    parser.add_argument("--allocator-host-pages", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--bench-runs", type=int, default=3)
    parser.add_argument("--fast-prefill", action="store_true",
                        help="Use fast_prefill (batch forward + hook-based state extraction) instead of step_token loop")
    parser.add_argument("--swa-offload", action="store_true",
                        help="Enable scheduler SWA offload: move window/buffer tensors to host RAM between decode steps")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to shard transformer layers across (uses cuda:0..N-1)")
    parser.add_argument("--output-json", help="Write results to this JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.preset == "tiny":
        cfg = _tiny_config()
    else:
        cfg = DeepSeekV4Pro2BConfig()

    print(f"device={device}  dtype={dtype}  preset={args.preset}", flush=True)
    model = DeepSeekV4Pro2BForCausalLM(cfg).to(device=device, dtype=dtype).eval()

    swa_mode = SWACacheMode(args.swa_mode)
    results = []

    for src_len in args.source_tokens:
        print(f"\n--- source_tokens={src_len} ---", flush=True)
        result = run_benchmark(
            model=model,
            source_tokens=src_len,
            decode_tokens=args.decode_tokens,
            batch_size=args.batch_size,
            device=device,
            use_prefix_cache=args.use_prefix_cache,
            prefix_cache_dir=args.prefix_cache_dir,
            swa_mode=swa_mode,
            backend_name=args.backend,
            allocator_device_pages=args.allocator_device_pages,
            allocator_host_pages=args.allocator_host_pages,
            warmup_runs=args.warmup_runs,
            bench_runs=args.bench_runs,
            fast_prefill=args.fast_prefill,
            swa_offload=args.swa_offload,
            num_gpus=args.num_gpus,
        )
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
