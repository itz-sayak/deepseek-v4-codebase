from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
SOURCES = [
    str(CSRC / "bindings.cpp"),
    str(CSRC / "sparse_sink_attention_cuda.cu"),
    str(CSRC / "tiled_prefill_cuda.cu"),
    str(CSRC / "csa_indexer_cuda.cu"),
    str(CSRC / "hca_compress_cuda.cu"),
]


def cuda_build_environment() -> Dict[str, object]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    target_include = None
    target_lib = None
    runtime_lib = None
    if conda_prefix:
        candidate = Path(conda_prefix) / "targets" / "x86_64-linux" / "include"
        if candidate.exists():
            target_include = str(candidate)
        target_lib_candidate = Path(conda_prefix) / "targets" / "x86_64-linux" / "lib"
        if target_lib_candidate.exists():
            target_lib = str(target_lib_candidate)
        runtime_candidate = Path(conda_prefix) / "lib" / "python3.10" / "site-packages" / "nvidia" / "cuda_runtime" / "lib"
        if runtime_candidate.exists():
            runtime_lib = str(runtime_candidate)

    # Resolve nvcc: check PATH first, then fall back to CUDA_HOME/bin and
    # a set of well-known system locations so the tests work even when
    # /usr/local/cuda/bin is not on the shell PATH.
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        _candidates = []
        if CUDA_HOME:
            _candidates.append(Path(CUDA_HOME) / "bin" / "nvcc")
        for _root in ("/usr/local/cuda", "/usr/local/cuda-12.1", "/usr/local/cuda-12",
                      "/usr/local/cuda-11.8", "/usr/cuda"):
            _candidates.append(Path(_root) / "bin" / "nvcc")
        for _p in _candidates:
            if _p.is_file():
                nvcc = str(_p)
                break

    return {
        "torch_cuda_built": torch.backends.cuda.is_built(),
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_home": CUDA_HOME,
        "nvcc": nvcc,
        "conda_target_include": target_include,
        "conda_target_lib": target_lib,
        "cuda_runtime_lib": runtime_lib,
    }


def cuda_extension_unavailable_reason(require_device: bool = False) -> Optional[str]:
    env = cuda_build_environment()
    if not env["torch_cuda_built"]:
        return "PyTorch was built without CUDA support."
    if require_device and not env["torch_cuda_available"]:
        return "No CUDA device is available for executing the custom kernels."
    if env["cuda_home"] is None:
        return "CUDA_HOME is not set; install a CUDA toolkit that matches the PyTorch build."
    if env["nvcc"] is None:
        return "nvcc was not found on PATH; install the CUDA toolkit compiler."
    return None


def _repair_cudart_symlinks(env: Dict[str, object]) -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    runtime_lib = env.get("cuda_runtime_lib")
    if not conda_prefix or not runtime_lib:
        return
    env_lib = Path(conda_prefix) / "lib"
    target_file = Path(str(runtime_lib)) / "libcudart.so.12"
    if not target_file.exists():
        return
    pairs = [
        (env_lib / "libcudart.so.12", target_file),
        (env_lib / "libcudart.so", Path("libcudart.so.12")),
    ]
    for link_path, target in pairs:
        try:
            if link_path.exists() and not link_path.is_symlink():
                continue
            if link_path.is_symlink() and link_path.exists():
                continue
            if link_path.lexists():
                link_path.unlink()
            link_path.symlink_to(target)
        except OSError:
            continue


@lru_cache(maxsize=1)
def load_aether_cuda_kernels(
    verbose: bool = False,
    build_directory: Optional[str] = None,
):
    reason = cuda_extension_unavailable_reason(require_device=False)
    if reason is not None:
        raise RuntimeError(f"Aether CUDA kernels are unavailable: {reason}")
    build_root = build_directory or os.environ.get("AETHER_KERNEL_BUILD_DIR")
    env = cuda_build_environment()

    # Ensure nvcc's parent directory is on PATH so torch's JIT compiler can
    # invoke it even if the shell environment does not include it.
    nvcc_path = env.get("nvcc")
    if nvcc_path:
        nvcc_bin = str(Path(nvcc_path).parent)
        current_path = os.environ.get("PATH", "")
        if nvcc_bin not in current_path.split(os.pathsep):
            os.environ["PATH"] = nvcc_bin + os.pathsep + current_path
    _repair_cudart_symlinks(env)
    include_paths = [str(CSRC)]
    if env["conda_target_include"]:
        include_paths.append(env["conda_target_include"])
    extra_ldflags = []
    for lib_dir in [env["conda_target_lib"], env["cuda_runtime_lib"]]:
        if lib_dir:
            extra_ldflags.extend([f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"])
    extra_cflags = ["-O3"]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        extra_cuda_cflags.extend(["-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"])
    return load(
        name="aether_cuda_kernels",
        sources=SOURCES,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=include_paths,
        extra_ldflags=extra_ldflags,
        build_directory=build_root,
        verbose=verbose,
        with_cuda=True,
    )
