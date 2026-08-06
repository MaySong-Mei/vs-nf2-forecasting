#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import sys


def main() -> int:
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
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            print(f"GPU: {properties.name}")
            print(f"VRAM: {properties.total_memory / 2**30:.1f} GiB")
    except ImportError:
        print("torch: MISSING (install a CUDA build separately)")
        missing.append("torch")
    if missing:
        print("\nInstall missing dependencies before preprocessing/training: " + ", ".join(missing))
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

