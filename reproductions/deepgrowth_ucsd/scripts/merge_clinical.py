#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OFFICIAL_COLUMNS = {
    "id": "ID",
    "interval": "Days from prior scan",
    "status": "Pre- vs Post-Tx",
    "growth": "Longitudinal classification (respect to prior scan)",
    "no_residual": "No residual VS",
    "treatment_days": (
        "Days between scan and treatment #1",
        "Days between scan and treatment #2",
    ),
}


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator, dtype=str)


def official_clinical_rows(clinical: pd.DataFrame) -> pd.DataFrame:
    required = [
        OFFICIAL_COLUMNS["id"],
        OFFICIAL_COLUMNS["interval"],
        OFFICIAL_COLUMNS["status"],
        OFFICIAL_COLUMNS["growth"],
        OFFICIAL_COLUMNS["no_residual"],
        *OFFICIAL_COLUMNS["treatment_days"],
    ]
    missing = [column for column in required if column not in clinical]
    if missing:
        raise ValueError(f"official clinical table is missing columns: {missing}")
    selected = clinical.copy()
    parsed = selected[OFFICIAL_COLUMNS["id"]].astype(str).str.extract(
        r"^(?P<patient_id>VS_\d{4})_(?P<order>\d{2})$"
    )
    if parsed.isna().any().any():
        raise ValueError("official clinical table contains an unexpected ID")
    selected["patient_id"] = parsed["patient_id"]
    selected["timepoint_id"] = selected[OFFICIAL_COLUMNS["id"]].astype(str)
    selected["timepoint_order"] = pd.to_numeric(parsed["order"])
    selected = selected.sort_values(["patient_id", "timepoint_order"]).reset_index(drop=True)

    intervals = pd.to_numeric(selected[OFFICIAL_COLUMNS["interval"]], errors="coerce")
    first = selected.groupby("patient_id").cumcount().eq(0)
    if intervals[~first].isna().any() or (intervals[~first] <= 0).any():
        raise ValueError("non-baseline Days from prior scan must be positive and complete")
    selected["study_days"] = intervals.fillna(0).groupby(selected["patient_id"]).cumsum()

    intervention = pd.Series(False, index=selected.index)
    for column in OFFICIAL_COLUMNS["treatment_days"]:
        relative_days = pd.to_numeric(selected[column], errors="coerce")
        previous = relative_days.groupby(selected["patient_id"]).shift()
        intervention |= previous.lt(0) & relative_days.ge(0)
    status = selected[OFFICIAL_COLUMNS["status"]].astype(str).str.strip().str.lower()
    previous_status = status.groupby(selected["patient_id"]).shift()
    # The signed treatment dates contain two anomalies/missing cases in v1.
    # Union with the official Pre/Post transition so neither is silently missed.
    intervention |= previous_status.eq("pre") & status.eq("post")
    intervention[first] = False

    return pd.DataFrame(
        {
            "patient_id": selected["patient_id"],
            "timepoint_id": selected["timepoint_id"],
            "study_days_clinical": selected["study_days"],
            "intervened_since_previous_clinical": intervention,
            "no_residual_vs_clinical": selected[OFFICIAL_COLUMNS["no_residual"]]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes"),
            "growth_class_clinical": selected[OFFICIAL_COLUMNS["growth"]],
            "pre_post_treatment": selected[OFFICIAL_COLUMNS["status"]],
        }
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge timing/treatment data into an indexed manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ucsd-official", action="store_true")
    parser.add_argument("--patient-column")
    parser.add_argument("--timepoint-column")
    parser.add_argument("--days-column")
    parser.add_argument("--intervention-column")
    parser.add_argument("--growth-column")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str, "timepoint_id": str})
    clinical = read_table(args.clinical)
    if args.ucsd_official:
        selected = official_clinical_rows(clinical)
    else:
        required_args = (args.patient_column, args.timepoint_column, args.days_column)
        if not all(required_args):
            raise SystemExit(
                "generic mode requires --patient-column, --timepoint-column, and --days-column"
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
    for target in ("study_days", "intervened_since_previous", "no_residual_vs", "growth_class"):
        source = f"{target}_clinical"
        if source in merged:
            existing = merged[target] if target in merged else pd.Series(pd.NA, index=merged.index)
            merged[target] = merged[source].where(merged[source].notna(), existing)
            merged = merged.drop(columns=[source])
    unmatched = int(merged["study_days"].isna().sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows to {args.output}; rows without study_days: {unmatched}")
    if args.ucsd_official:
        print(f"Intervals crossing treatment: {int(merged['intervened_since_previous'].eq(True).sum())}")
        print(f"No-residual timepoints: {int(merged['no_residual_vs'].eq(True).sum())}")
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
