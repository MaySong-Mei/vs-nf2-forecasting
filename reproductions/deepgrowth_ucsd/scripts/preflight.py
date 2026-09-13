#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucsd_repro.provenance import (
    collect_runtime_provenance,
    validation_problems,
    write_strict_json,
)


def print_human_report(report: dict) -> None:
    print(f"Python: {report['python']['version']} ({platform.platform()})")
    for name, version in report["packages"].items():
        print(f"{name}: {version if version is not None else 'MISSING'}")
    cuda = report["cuda"]
    print(f"torch CUDA runtime: {cuda['runtime_version']}")
    print(f"CUDA available: {cuda['available']}")
    if cuda["available"]:
        print(f"GPU: {cuda['device_name']}")
        print(f"VRAM: {cuda['total_vram_bytes'] / 2**30:.1f} GiB")
        print(f"Compute capability: {cuda['compute_capability']}")
        print("Training AMP dtype: float16 (BF16 is not requested)")
    print(f"NVIDIA driver: {cuda['driver_version'] or 'unavailable from nvidia-smi'}")
    print(f"Git commit: {report['git']['commit']}")
    print(f"Git dirty: {report['git']['dirty']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report and validate the UCSD reproduction environment")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit strict JSON only")
    parser.add_argument("--output-json", type=Path, help="also save strict JSON to this path")
    args = parser.parse_args()
    report = collect_runtime_provenance(Path(__file__).resolve().parents[1])
    problems = validation_problems(report, require_cuda=args.require_cuda)
    report["validation"] = {
        "require_cuda": args.require_cuda,
        "passed": not problems,
        "problems": problems,
    }
    if args.output_json:
        write_strict_json(args.output_json, report)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print_human_report(report)
    if problems:
        if not args.json:
            print("\nInstall/enable required dependencies before preprocessing/training: " + ", ".join(problems))
        return 1
    if not args.json:
        print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

