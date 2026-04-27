from __future__ import annotations

import argparse
import time

import torch

from deepseek_kernels.loader import load_deepseek_cuda_kernels
from deepseek_pipeline.serving import PytorchAttentionBackend


def run_once(batch: int, target_tokens: int, heads: int, head_dim: int, source_tokens: int, top_k: int, dtype: torch.dtype):
    q = torch.randn(batch, target_tokens, heads, head_dim, device="cuda", dtype=dtype)
    kv = torch.randn(batch, source_tokens, 2, head_dim, device="cuda", dtype=dtype)
    topk = torch.randint(0, source_tokens, (batch, target_tokens, top_k), device="cuda", dtype=torch.int32)
    sink = torch.zeros(heads, device="cuda", dtype=dtype)
    scale = head_dim ** -0.5
    ext = load_deepseek_cuda_kernels()
    ref = PytorchAttentionBackend()

    for _ in range(10):
        ext.sparse_sink_attention(q, kv, topk, sink, scale)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        ext.sparse_sink_attention(q, kv, topk, sink, scale)
    torch.cuda.synchronize()
    kernel_ms = (time.perf_counter() - t0) * 1000 / 50

    qf = q.float()
    kvf = kv.float()
    sinkf = sink.float()
    topk64 = topk.long()
    for _ in range(3):
        ref.sparse_attention(qf, kvf, topk64, sinkf, scale)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(10):
        ref.sparse_attention(qf, kvf, topk64, sinkf, scale)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t1) * 1000 / 10

    print(
        {
            "batch": batch,
            "target_tokens": target_tokens,
            "heads": heads,
            "head_dim": head_dim,
            "source_tokens": source_tokens,
            "top_k": top_k,
            "dtype": str(dtype),
            "cuda_kernel_ms": kernel_ms,
            "pytorch_ref_ms": ref_ms,
            "speedup_vs_ref": ref_ms / kernel_ms,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the CUDA sparse sink-attention kernel against the PyTorch reference.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--target-tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--source-tokens", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    run_once(args.batch, args.target_tokens, args.heads, args.head_dim, args.source_tokens, args.top_k, dtype)


if __name__ == "__main__":
    main()
