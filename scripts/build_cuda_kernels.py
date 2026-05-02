from __future__ import annotations

import argparse

from aether_kernels.loader import load_aether_cuda_kernels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Aether CUDA serving kernels.")
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    ext = load_aether_cuda_kernels(verbose=args.verbose, build_directory=args.build_dir)
    print(f"BUILT {ext}")


if __name__ == "__main__":
    main()
