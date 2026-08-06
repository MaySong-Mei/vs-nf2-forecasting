#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Report and validate the UCSD reproduction environment")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    required = ["numpy", "scipy", "pandas", "yaml", "nibabel", "SimpleITK"]
    missing = []
    for name in required:
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'installed')}")
        except ImportError:
            print(f"{name}: MISSING")
            missing.append(name)
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch CUDA runtime: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            print(f"GPU: {properties.name}")
            print(f"VRAM: {properties.total_memory / 2**30:.1f} GiB")
            print(f"Compute capability: {properties.major}.{properties.minor}")
            print("Training AMP dtype: float16 (BF16 is not requested)")
            try:
                driver = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip().splitlines()[0]
                print(f"NVIDIA driver: {driver}")
            except (OSError, subprocess.CalledProcessError, IndexError):
                print("NVIDIA driver: unavailable from nvidia-smi")
        elif args.require_cuda:
            missing.append("CUDA-enabled PyTorch/GPU")
    except ImportError:
        print("torch: MISSING (install a CUDA build separately)")
        missing.append("torch")
    if missing:
        print("\nInstall/enable required dependencies before preprocessing/training: " + ", ".join(missing))
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

