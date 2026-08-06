#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge UCSD timing/treatment columns into an indexed manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--patient-column", required=True)
    parser.add_argument("--timepoint-column", required=True)
    parser.add_argument("--days-column", required=True)
    parser.add_argument("--intervention-column")
    parser.add_argument("--growth-column")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str, "timepoint_id": str})
    clinical = pd.read_csv(
        args.clinical,
        dtype={args.patient_column: str, args.timepoint_column: str},
    )
    required = [args.patient_column, args.timepoint_column, args.days_column]
    missing = [column for column in required if column not in clinical]
    if missing:
        raise SystemExit(f"clinical table is missing columns: {missing}")
    selected = clinical[required].rename(
        columns={
            args.patient_column: "patient_id",
            args.timepoint_column: "timepoint_id",
            args.days_column: "study_days_clinical",
        }
    )
    if args.intervention_column:
        selected["intervened_since_previous_clinical"] = clinical[args.intervention_column]
    if args.growth_column:
        selected["growth_class_clinical"] = clinical[args.growth_column]
    if selected.duplicated(["patient_id", "timepoint_id"]).any():
        raise SystemExit("clinical table has duplicate patient/timepoint keys")
    merged = manifest.merge(selected, on=["patient_id", "timepoint_id"], how="left", validate="one_to_one")
    merged["study_days"] = merged["study_days_clinical"].combine_first(merged.get("study_days"))
    merged = merged.drop(columns=["study_days_clinical"])
    if "intervened_since_previous_clinical" in merged:
        merged["intervened_since_previous"] = merged[
            "intervened_since_previous_clinical"
        ].combine_first(merged.get("intervened_since_previous"))
        merged = merged.drop(columns=["intervened_since_previous_clinical"])
    if "growth_class_clinical" in merged:
        merged["growth_class"] = merged["growth_class_clinical"].combine_first(
            merged.get("growth_class")
        )
        merged = merged.drop(columns=["growth_class_clinical"])
    unmatched = int(merged["study_days"].isna().sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows to {args.output}; rows without study_days: {unmatched}")
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())

