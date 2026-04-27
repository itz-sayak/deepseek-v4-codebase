import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_kernels.loader import cuda_build_environment, cuda_extension_unavailable_reason
from deepseek_pipeline.serving import CudaSparseAttentionBackend


def test_cuda_loader_reports_environment():
    env = cuda_build_environment()
    assert set(env) == {
        "torch_cuda_built",
        "torch_cuda_available",
        "cuda_home",
        "nvcc",
        "conda_target_include",
        "conda_target_lib",
        "cuda_runtime_lib",
    }


def test_cuda_backend_availability_is_consistent():
    backend = CudaSparseAttentionBackend()
    reason = cuda_extension_unavailable_reason()
    assert backend.available() is (reason is None)
    assert backend.availability_error() == reason
