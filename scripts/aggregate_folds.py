#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate completed five-fold test predictions")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = sorted(args.run_root.glob("fold_*/test_per_case.csv"))
    if not files:
        raise SystemExit(f"no fold result files under {args.run_root}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["fold"] = int(path.parent.name.split("_")[-1])
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    duplicate_cases = results["triplet_id"].duplicated().sum()
    if duplicate_cases:
        raise SystemExit(f"found {duplicate_cases} duplicate test cases across folds")
    metrics = ["dice", "hd95_mm", "absolute_rvd", "signed_rvd", "volume_absolute_error_mm3"]
    summary = {
        "folds_found": sorted(results["fold"].unique().astype(int).tolist()),
        "cases": int(len(results)),
        "patients": int(results["patient_id"].astype(str).nunique()),
        "case_level": {},
        "patient_level": {},
    }
    patient_results = results.groupby("patient_id", as_index=False)[metrics].mean(numeric_only=True)
    for name, frame in (("case_level", results), ("patient_level", patient_results)):
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna()
            summary[name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(values.median()),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    results.to_csv(args.output.with_suffix(".per_case.csv"), index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

