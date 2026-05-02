import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aether_kernels.loader import load_aether_cuda_kernels
from aether_pipeline.serving import PytorchAttentionBackend


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test requires a CUDA-capable PyTorch environment")


def _run_case(dtype: torch.dtype, index_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    ext = load_aether_cuda_kernels()
    ref = PytorchAttentionBackend()

    q = torch.randn(2, 5, 4, 16, device="cuda", dtype=dtype)
    kv = torch.randn(2, 23, 2, 16, device="cuda", dtype=dtype)
    topk = torch.randint(0, 23, (2, 5, 7), device="cuda", dtype=index_dtype)
    sink = torch.randn(4, device="cuda", dtype=dtype)
    scale = 0.125

    got = ext.sparse_sink_attention(q, kv, topk, sink, scale)
    expected = ref.sparse_attention(q.float(), kv.float(), topk.long(), sink.float(), scale).to(dtype)

    atol = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    rtol = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    torch.testing.assert_close(got.float(), expected.float(), atol=atol, rtol=rtol)


def test_cuda_extension_matches_reference_fp32():
    _run_case(torch.float32, torch.int64)


def test_cuda_extension_matches_reference_fp16():
    _run_case(torch.float16, torch.int32)


def test_cuda_extension_matches_reference_bf16():
    _run_case(torch.bfloat16, torch.int64)
