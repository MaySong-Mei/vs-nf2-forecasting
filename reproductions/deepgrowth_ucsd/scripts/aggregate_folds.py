#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["dice", "hd95_mm", "absolute_rvd", "signed_rvd", "volume_absolute_error_mm3"]


def summarize_metrics(frame: pd.DataFrame) -> dict:
    summary = {"cases": int(len(frame)), "patients": int(frame["patient_id"].astype(str).nunique())}
    for metric in METRICS:
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        values = numeric[finite]
        summary[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "finite_cases": int(finite.sum()),
            "nonfinite_cases": int((~finite).sum()),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate completed five-fold test predictions")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-folds", type=int, default=5)
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
    folds_found = sorted(results["fold"].unique().astype(int).tolist())
    expected = list(range(args.expected_folds))
    if folds_found != expected:
        raise SystemExit(f"incomplete/unexpected folds: found {folds_found}, expected {expected}")
    duplicate_cases = results["triplet_id"].duplicated().sum()
    if duplicate_cases:
        raise SystemExit(f"found {duplicate_cases} duplicate test cases across folds")
    required = ["patient_id", "triplet_id", "target_relative_change", *METRICS]
    missing = [column for column in required if column not in results]
    if missing:
        raise SystemExit(f"fold results are missing columns: {missing}")
    absolute_change = pd.to_numeric(results["target_relative_change"], errors="coerce").abs()
    finite_change = np.isfinite(
        absolute_change.to_numpy(dtype=float, na_value=np.nan)
    )
    if not finite_change.any():
        raise SystemExit("target_relative_change has no finite values")
    cutoff = float(absolute_change[finite_change].quantile(0.8))
    top20 = results[finite_change & (absolute_change >= cutoff)].copy()
    summary = {
        "folds_found": folds_found,
        "cases": int(len(results)),
        "patients": int(results["patient_id"].astype(str).nunique()),
        "top20_absolute_change_cutoff": cutoff,
        "target_relative_change_nonfinite_cases": int((~finite_change).sum()),
        "case_level": summarize_metrics(results),
        "patient_level": summarize_metrics(
            results.groupby("patient_id", as_index=False)[METRICS].mean(numeric_only=True)
        ),
        "top20_change": {
            "case_level": summarize_metrics(top20),
            "patient_level": summarize_metrics(
                top20.groupby("patient_id", as_index=False)[METRICS].mean(numeric_only=True)
            ),
        },
        "per_fold": {},
        "fold_mean_distribution": {},
    }
    fold_means = []
    for fold, frame in results.groupby("fold", sort=True):
        summary["per_fold"][str(int(fold))] = summarize_metrics(frame)
        fold_means.append({"fold": int(fold), **frame[METRICS].mean(numeric_only=True).to_dict()})
    fold_mean_frame = pd.DataFrame(fold_means)
    for metric in METRICS:
        numeric = pd.to_numeric(fold_mean_frame[metric], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        values = numeric[finite]
        summary["fold_mean_distribution"][metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
            "finite_folds": int(finite.sum()),
            "nonfinite_folds": int((~finite).sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    results.to_csv(args.output.with_suffix(".per_case.csv"), index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

