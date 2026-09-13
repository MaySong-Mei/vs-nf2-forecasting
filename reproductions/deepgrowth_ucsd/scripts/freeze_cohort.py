#!/usr/bin/env python3
"""Freeze the QC-approved modeling cohort and patient-level folds with hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qc_pass(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def freeze_cohort(
    metadata_path: Path,
    splits_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    review_index: Path | None = None,
    failures_path: Path | None = None,
    runtime_provenance: Path | None = None,
    expected_cases: int | None = None,
    expected_patients: int | None = None,
) -> dict[str, Any]:
    metadata = pd.read_csv(metadata_path, dtype={"patient_id": str, "triplet_id": str})
    required = {"patient_id", "triplet_id", "prepared_path", "preprocessing_qc_pass"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
    if metadata["triplet_id"].duplicated().any():
        raise ValueError("metadata contains duplicate triplet IDs")

    selected = metadata.loc[_qc_pass(metadata["preprocessing_qc_pass"])].copy()
    selected = selected.sort_values(["patient_id", "triplet_id"], kind="stable")
    patient_count = int(selected["patient_id"].nunique())
    if expected_cases is not None and len(selected) != expected_cases:
        raise ValueError(f"expected {expected_cases} cases, found {len(selected)}")
    if expected_patients is not None and patient_count != expected_patients:
        raise ValueError(f"expected {expected_patients} patients, found {patient_count}")

    metadata_root = metadata_path.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_prepared = []
    frozen_paths = []
    for value in selected["prepared_path"]:
        path = Path(str(value))
        resolved = path.resolve() if path.is_absolute() else (metadata_root / path).resolve()
        if not resolved.is_file():
            raise ValueError(f"prepared array does not exist: {value}")
        resolved_prepared.append(resolved)
        frozen_paths.append(Path(os.path.relpath(resolved, output_dir.resolve())).as_posix())
    selected["prepared_path"] = frozen_paths

    splits = pd.read_csv(splits_path, dtype={"patient_id": str})
    if splits["patient_id"].duplicated().any():
        raise ValueError("split table contains duplicate patients")
    frozen_splits = splits.loc[splits["patient_id"].isin(selected["patient_id"])].copy()
    frozen_splits = frozen_splits.sort_values("patient_id", kind="stable")
    if set(frozen_splits["patient_id"]) != set(selected["patient_id"]):
        raise ValueError("at least one modeling patient is missing a fold assignment")
    folds = sorted(pd.to_numeric(frozen_splits["fold"], errors="raise").astype(int).unique())
    if folds != list(range(5)):
        raise ValueError(f"frozen cohort must populate folds 0-4, found {folds}")

    metadata_output = output_dir / "modeling_metadata.csv"
    splits_output = output_dir / "patient_folds.csv"
    config_output = output_dir / "resolved_config.yaml"
    selected.to_csv(metadata_output, index=False, lineterminator="\n")
    frozen_splits.to_csv(splits_output, index=False, lineterminator="\n")
    shutil.copyfile(config_path, config_output)

    joined = selected[["patient_id", "triplet_id"]].merge(
        frozen_splits[["patient_id", "fold"]], on="patient_id", validate="many_to_one"
    )
    fold_counts = {
        str(int(fold)): {
            "triplets": int(len(group)),
            "patients": int(group["patient_id"].nunique()),
        }
        for fold, group in joined.groupby("fold", sort=True)
    }

    review_summary = None
    if review_index is not None:
        review = pd.read_csv(review_index)
        if "reviewer_decision" not in review:
            raise ValueError("review index lacks reviewer_decision")
        decisions = review["reviewer_decision"].fillna("").astype(str).str.strip().str.lower()
        if (decisions == "").any():
            raise ValueError("review index contains blank decisions")
        invalid = sorted(set(decisions) - {"pass", "fail", "uncertain"})
        if invalid:
            raise ValueError(f"invalid review decisions: {invalid}")
        review_summary = {
            "rows": int(len(review)),
            "decision_counts": {
                str(key): int(value) for key, value in decisions.value_counts().sort_index().items()
            },
            "rendering_config_hashes": sorted(
                review.get("rendering_config_hash", pd.Series(dtype=str)).dropna().astype(str).unique()
            ),
            "review_index_sha256": file_sha256(review_index),
        }

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    failure_count = int(len(pd.read_csv(failures_path))) if failures_path else None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "decision": "retain_reference_automatic_qc_cohort",
        "formal_model_evaluation_consulted": False,
        "source_prepared_triplets": int(len(metadata)),
        "source_prepared_patients": int(metadata["patient_id"].nunique()),
        "automatic_qc_excluded_triplets": int(len(metadata) - len(selected)),
        "hard_preprocessing_failures": failure_count,
        "modeling_triplets": int(len(selected)),
        "modeling_patients": patient_count,
        "seed": int(config["experiment"]["seed"]),
        "fold_counts": fold_counts,
        "hashes": {
            "modeling_metadata_sha256": file_sha256(metadata_output),
            "patient_folds_sha256": file_sha256(splits_output),
            "resolved_config_sha256": file_sha256(config_output),
            "runtime_provenance_sha256": (
                file_sha256(runtime_provenance) if runtime_provenance else None
            ),
        },
        "visual_review": review_summary,
        "privacy": "Patient-level files remain local and Git-ignored.",
    }
    with (output_dir / "freeze_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--review-index", type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--runtime-provenance", type=Path)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--expected-patients", type=int)
    args = parser.parse_args()
    summary = freeze_cohort(
        args.metadata,
        args.splits,
        args.config,
        args.output_dir,
        review_index=args.review_index,
        failures_path=args.failures,
        runtime_provenance=args.runtime_provenance,
        expected_cases=args.expected_cases,
        expected_patients=args.expected_patients,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
