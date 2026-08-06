#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucsd_repro.manifest import (
    iter_triplets,
    load_manifest,
    parse_optional_bool,
    resolve_path,
    triplet_exclusion_reasons,
    validate_files,
)


def validate_nifti_pairs(frame, manifest_path: Path, data_root: Path | None) -> tuple[list[str], dict]:
    import nibabel as nib

    errors: list[str] = []
    stats = {
        "pairs_checked": 0,
        "declared_no_residual": 0,
        "empty_masks": 0,
        "geometry_mismatches": 0,
        "nonfinite_images": 0,
        "nonfinite_masks": 0,
    }
    for row_number, row in frame.iterrows():
        image_path = resolve_path(str(row["image_path"]), manifest_path, data_root)
        mask_path = resolve_path(str(row["mask_path"]), manifest_path, data_root)
        try:
            image = nib.load(image_path)
            mask = nib.load(mask_path)
            if len(image.shape) != 3 or len(mask.shape) != 3:
                errors.append(
                    f"row {row_number}: expected 3D image/mask, got {image.shape}/{mask.shape}"
                )
                continue
            zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
            if not np.isfinite(zooms).all() or (zooms <= 0).any():
                errors.append(f"row {row_number}: invalid image voxel spacing {zooms.tolist()}")
            same_geometry = image.shape == mask.shape and np.allclose(
                image.affine, mask.affine, rtol=1e-5, atol=1e-4
            )
            if not same_geometry:
                stats["geometry_mismatches"] += 1
                errors.append(
                    f"row {row_number}: image/mask shape or affine mismatch: "
                    f"{image.shape}/{mask.shape}"
                )
            image_data = np.asanyarray(image.dataobj)
            mask_data = np.asanyarray(mask.dataobj)
            if not np.isfinite(image_data).all():
                stats["nonfinite_images"] += 1
                errors.append(f"row {row_number}: image contains NaN/Inf")
            if not np.isfinite(mask_data).all():
                stats["nonfinite_masks"] += 1
                errors.append(f"row {row_number}: mask contains NaN/Inf")
            mask_empty = not np.asarray(mask_data).astype(bool).any()
            declared_no_residual = (
                parse_optional_bool(row.get("no_residual_vs")) is True
            )
            stats["declared_no_residual"] += int(declared_no_residual)
            stats["empty_masks"] += int(mask_empty)
            if mask_empty and not declared_no_residual:
                errors.append(
                    f"row {row_number}: mask is empty but no_residual_vs is not true"
                )
            stats["pairs_checked"] += 1
        except Exception as error:
            errors.append(f"row {row_number}: {type(error).__name__}: {error}")
        if (row_number + 1) % 50 == 0:
            print(f"NIfTI QC: {row_number + 1}/{len(frame)}", flush=True)
    return errors, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a UCSD longitudinal manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument(
        "--check-nifti",
        action="store_true",
        help="read every image/mask and validate 3D geometry, finite voxels, and mask emptiness",
    )
    parser.add_argument("--keep-intervened", action="store_true")
    parser.add_argument("--keep-no-residual", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="write the complete machine-readable QC report (including errors)",
    )
    args = parser.parse_args()
    frame = load_manifest(args.manifest)
    errors = validate_files(frame, args.manifest, args.data_root) if args.check_files else []
    nifti_stats = None
    if args.check_nifti:
        nifti_errors, nifti_stats = validate_nifti_pairs(
            frame, args.manifest, args.data_root
        )
        errors.extend(nifti_errors)
        print(f"NIfTI QC summary: {nifti_stats}")
    candidates = list(iter_triplets(frame, exclude_intervened=False, exclude_no_residual=False))
    reason_counts: dict[str, int] = {}
    triplets = []
    for triplet in candidates:
        reasons = triplet_exclusion_reasons(
            triplet,
            exclude_intervened=not args.keep_intervened,
            exclude_no_residual=not args.keep_no_residual,
        )
        if reasons:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            triplets.append(triplet)
    print(f"Rows: {len(frame)}")
    print(f"Patients: {frame.patient_id.nunique()}")
    print(f"Candidate rolling triples: {len(candidates)}")
    print(f"Eligible rolling triples: {len(triplets)}")
    print(f"Excluded by reason (overlap allowed): {reason_counts}")
    print(f"Triples with unknown intervention status: {sum(item.intervention_unknown for item in triplets)}")
    if args.output_json:
        report = {
            "manifest_rows": len(frame),
            "patients": int(frame.patient_id.nunique()),
            "candidate_rolling_triplets": len(candidates),
            "eligible_rolling_triplets": len(triplets),
            "excluded_by_reason": reason_counts,
            "unknown_intervention_triplets": sum(
                item.intervention_unknown for item in triplets
            ),
            "nifti_qc": nifti_stats,
            "error_count": len(errors),
            "errors": errors,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print(f"QC report: {args.output_json}")
    if errors:
        print(f"File errors: {len(errors)}")
        for error in errors[:50]:
            print("  " + error)
        return 1
    if not triplets:
        print("ERROR: no eligible patient has at least three timepoints")
        return 1
    print("Manifest validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
