from __future__ import annotations

import argparse

from deepseek_kernels.loader import load_deepseek_cuda_kernels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DeepSeek CUDA serving kernels.")
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    ext = load_deepseek_cuda_kernels(verbose=args.verbose, build_directory=args.build_dir)
    print(f"BUILT {ext}")


if __name__ == "__main__":
    main()
