#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucsd_repro.metrics import case_metrics


def finite_summary(frame: pd.DataFrame, columns: list[str]) -> dict:
    summary = {"cases": int(len(frame)), "patients": int(frame["patient_id"].nunique())}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        values = numeric[finite]
        summary[column] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
            "finite_cases": int(finite.sum()),
            "nonfinite_cases": int((~finite).sum()),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate no-training longitudinal baselines")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-qc-failed",
        action="store_true",
        help="diagnostic only: include prepared cases that failed crop/registration QC",
    )
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata)
    input_cases = len(metadata)
    if "preprocessing_qc_pass" in metadata and not args.include_qc_failed:
        passed = metadata["preprocessing_qc_pass"].astype(str).str.lower().eq("true")
        metadata = metadata[passed].copy()
    if metadata.empty:
        raise SystemExit("no prepared cases remain after preprocessing QC filtering")
    print(
        f"Preprocessing QC: using {len(metadata)}/{input_cases} prepared cases; "
        f"include_qc_failed={args.include_qc_failed}"
    )
    base = args.metadata.resolve().parent
    rows = []
    for _, item in metadata.iterrows():
        path = Path(str(item["prepared_path"]))
        path = path if path.is_absolute() else base / path
        with np.load(path, allow_pickle=False) as archive:
            masks = archive["masks"].astype(bool)
            sdfs = archive["sdfs"].astype(np.float32)
            days = archive["days"].astype(np.float32)
            spacing = archive["spacing_mm"].astype(float)
        interval_1 = float(days[1] - days[0])
        interval_2 = float(days[2] - days[1])
        ratio = interval_2 / interval_1
        predictions = {
            "stable_mask": masks[1],
            "linear_sdf": (sdfs[1] + ratio * (sdfs[1] - sdfs[0])) < 0,
        }
        voxel_volume = float(np.prod(spacing))
        previous_volume = float(masks[1].sum() * voxel_volume)
        target_volume = float(masks[2].sum() * voxel_volume)
        relative_change = (target_volume - previous_volume) / previous_volume if previous_volume else np.nan
        for method, prediction in predictions.items():
            metrics = case_metrics(prediction, masks[2], spacing)
            rows.append(
                {
                    "patient_id": str(item["patient_id"]),
                    "triplet_id": str(item["triplet_id"]),
                    "method": method,
                    "target_relative_change": relative_change,
                    **metrics,
                }
            )
    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "per_case.csv", index=False)
    change_by_case = results.drop_duplicates("triplet_id").set_index("triplet_id")["target_relative_change"].abs()
    cutoff = float(change_by_case.quantile(0.8))
    metric_columns = ["dice", "hd95_mm", "absolute_rvd", "signed_rvd", "volume_absolute_error_mm3"]
    summary = {
        "prepared_input_cases": input_cases,
        "evaluated_cases_after_qc": len(metadata),
        "included_qc_failed": args.include_qc_failed,
        "top20_absolute_change_cutoff": cutoff,
        "methods": {},
    }
    for method, group in results.groupby("method"):
        top_ids = change_by_case[change_by_case >= cutoff].index
        summary["methods"][method] = {
            "all": finite_summary(group, metric_columns),
            "top20_change": finite_summary(group[group["triplet_id"].isin(top_ids)], metric_columns),
        }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(results.groupby("method")[metric_columns[:3]].mean().round(4).to_string())
    print(f"Wrote per-case results and summary to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
