#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucsd_repro.geometry import (
    crop_or_pad,
    mask_centroid,
    normalize_intensity,
    signed_distance,
    touches_border,
)
from ucsd_repro.manifest import (
    iter_triplets,
    load_manifest,
    resolve_path,
    triplet_exclusion_reasons,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UCSD longitudinal triples")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-triplets",
        type=int,
        help="process only the first N eligible triplets (timing/smoke runs only)",
    )
    parser.add_argument(
        "--triplet-id",
        action="append",
        help="process only this eligible triplet ID; repeat for multiple debugging targets",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def failure_category(error: Exception) -> str:
    message = str(error).lower()
    if "source mask is empty" in message:
        return "source_mask_empty"
    if "registered mask" in message and "empty" in message:
        return "registered_mask_empty"
    if ("crop" in message and "empty" in message) or "sdf for an empty mask" in message:
        return "cropped_mask_empty"
    if "image has no finite nonzero voxels" in message:
        return "cropped_image_no_signal"
    if "nan" in message or "inf" in message or "finite" in message:
        return "nonfinite_array"
    return type(error).__name__


def count_values(values: list[str]) -> dict[str, int]:
    counts = pd.Series(values, dtype=object).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def isotropic_reference(image, spacing_mm: float):
    import SimpleITK as sitk

    size = [int(round(old_size * old_spacing / spacing_mm)) for old_size, old_spacing in zip(image.GetSize(), image.GetSpacing())]
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing((spacing_mm,) * 3)
    return reference


def rigid_transform(fixed, moving, config: dict):
    import SimpleITK as sitk

    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(
        float(config.get("registration_sampling_percentage", 0.2)), 100
    )
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=int(config["registration_iterations"]),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    shrink_factors = [
        int(value) for value in config.get("registration_shrink_factors", [1])
    ]
    smoothing_sigmas = [
        float(value) for value in config.get("registration_smoothing_sigmas_mm", [0])
    ]
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError("registration pyramid factors and sigmas must have equal lengths")
    registration.SetShrinkFactorsPerLevel(shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)
    return registration.Execute(fixed, moving)


def resample(image, reference, transform, is_mask: bool):
    import SimpleITK as sitk

    interpolation = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    pixel_type = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
    return sitk.Resample(image, reference, transform, interpolation, 0, pixel_type)


def geometry_matches(image, mask) -> bool:
    return bool(
        mask.GetSize() == image.GetSize()
        and np.allclose(mask.GetSpacing(), image.GetSpacing(), rtol=1e-5, atol=1e-4)
        and np.allclose(mask.GetOrigin(), image.GetOrigin(), rtol=1e-5, atol=1e-4)
        and np.allclose(mask.GetDirection(), image.GetDirection(), rtol=1e-5, atol=1e-4)
    )


def geometry_record(image, matched: bool) -> dict:
    return {
        "image_size_xyz": list(image.GetSize()),
        "image_spacing_xyz": list(image.GetSpacing()),
        "image_origin_xyz": list(image.GetOrigin()),
        "image_direction": list(image.GetDirection()),
        "mask_geometry_matched_image": matched,
    }


def read_header(path: str):
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return reader


def refresh_cached_geometry(metadata: dict) -> None:
    image_paths = json.loads(metadata.get("source_images", "[]"))
    mask_paths = json.loads(metadata.get("source_masks", "[]"))
    if len(image_paths) != 3 or len(mask_paths) != 3:
        return
    records = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image = read_header(image_path)
        mask = read_header(mask_path)
        records.append(geometry_record(image, geometry_matches(image, mask)))
    metadata["source_geometry"] = json.dumps(records, sort_keys=True)


def apply_case_qc(metadata: dict, overlap_threshold: float) -> None:
    reasons = []
    overlaps = json.loads(metadata.get("registration_overlap_dice_to_t2", "[]"))
    moving_overlaps = [float(value) for index, value in enumerate(overlaps) if index != 1]
    if any(value < overlap_threshold for value in moving_overlaps):
        reasons.append("registration_overlap_below_threshold")
    crop_flags = json.loads(metadata.get("crop_touches_border_by_timepoint", "[]"))
    if any(bool(value) for value in crop_flags):
        reasons.append("crop_touches_border")
    metadata["registration_overlap_review_threshold"] = overlap_threshold
    metadata["preprocessing_qc_reasons"] = json.dumps(reasons)
    metadata["preprocessing_qc_pass"] = not reasons


def read_pair(image_path: Path, mask_path: Path):
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("only 3D images/masks are supported")
    if not np.asarray(sitk.GetArrayViewFromImage(mask)).astype(bool).any():
        raise ValueError("source mask is empty")
    same_geometry = geometry_matches(image, mask)
    if not same_geometry:
        mask = sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    geometry = geometry_record(image, same_geometry)
    return image, mask, geometry


def process_triplet(triplet, manifest_path: Path, data_root: Path | None, config: dict) -> tuple[dict, dict]:
    import SimpleITK as sitk

    spacing = float(config["spacing_mm"])
    crop_shape = tuple(int(value) for value in config["crop_size"])
    pairs = []
    for row in triplet.rows:
        image_path = resolve_path(str(row["image_path"]), manifest_path, data_root)
        mask_path = resolve_path(str(row["mask_path"]), manifest_path, data_root)
        image, mask, geometry = read_pair(image_path, mask_path)
        pairs.append((image, mask, image_path, mask_path, geometry))

    fixed = pairs[1][0]
    reference = isotropic_reference(fixed, spacing)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    images = []
    masks = []
    registration_metrics = []
    for index, (image, mask, _, _, _) in enumerate(pairs):
        transform = identity if index == 1 else rigid_transform(fixed, image, config)
        registered_image = resample(image, reference, transform, is_mask=False)
        registered_mask = resample(mask, reference, transform, is_mask=True)
        image_array = sitk.GetArrayFromImage(registered_image)
        mask_array = sitk.GetArrayFromImage(registered_mask) > 0
        if not mask_array.any():
            raise ValueError(f"registered mask at timepoint {index + 1} is empty")
        images.append(image_array)
        masks.append(mask_array)
        registration_metrics.append(None if index == 1 else "rigid_mattes_mi")

    center = mask_centroid(masks[1])
    cropped_images = np.stack(
        [
            normalize_intensity(
                crop_or_pad(image, center, crop_shape), config["intensity_percentiles"]
            )
            for image in images
        ]
    )
    cropped_masks = np.stack([crop_or_pad(mask, center, crop_shape) for mask in masks]).astype(np.uint8)
    cropped_sdfs = np.stack(
        [signed_distance(mask, spacing, float(config["sdf_clip_mm"])) for mask in cropped_masks]
    )
    if not np.isfinite(cropped_images).all() or not np.isfinite(cropped_sdfs).all():
        raise ValueError("preprocessed image/SDF contains NaN/Inf")
    registration_overlap_dice = []
    for mask in masks:
        denominator = int(mask.sum() + masks[1].sum())
        overlap = 2 * int(np.logical_and(mask, masks[1]).sum()) / denominator if denominator else 1.0
        registration_overlap_dice.append(float(overlap))
    days = np.asarray([float(row["study_days"]) for row in triplet.rows], dtype=np.float32)
    arrays = {
        "images": cropped_images.astype(np.float32),
        "masks": cropped_masks,
        "sdfs": cropped_sdfs.astype(np.float32),
        "days": days,
        "spacing_mm": np.asarray([spacing] * 3, dtype=np.float32),
    }
    metadata = {
        "patient_id": triplet.patient_id,
        "triplet_id": triplet.triplet_id,
        "timepoint_ids": json.dumps([str(row["timepoint_id"]) for row in triplet.rows]),
        "interval_1_days": float(days[1] - days[0]),
        "interval_2_days": float(days[2] - days[1]),
        "intervention_unknown": triplet.intervention_unknown,
        "input_crop_touches_border": touches_border(cropped_masks[1]),
        "target_crop_touches_border": touches_border(cropped_masks[2]),
        "source_images": json.dumps([str(pair[2]) for pair in pairs]),
        "source_masks": json.dumps([str(pair[3]) for pair in pairs]),
        "registration": json.dumps(registration_metrics),
        "registration_parameters": json.dumps(
            {
                "iterations_per_level": int(config["registration_iterations"]),
                "sampling_percentage": float(
                    config.get("registration_sampling_percentage", 0.2)
                ),
                "shrink_factors": config.get("registration_shrink_factors", [1]),
                "smoothing_sigmas_mm": config.get(
                    "registration_smoothing_sigmas_mm", [0]
                ),
            },
            sort_keys=True,
        ),
        "source_geometry": json.dumps([pair[4] for pair in pairs], sort_keys=True),
        "registration_overlap_dice_to_t2": json.dumps(registration_overlap_dice),
        "crop_touches_border_by_timepoint": json.dumps(
            [bool(touches_border(mask)) for mask in cropped_masks]
        ),
        "output_shape_zyx": json.dumps(list(cropped_masks.shape[1:])),
        "preprocessed_finite": True,
    }
    return arrays, metadata


def main() -> int:
    args = arguments()
    if args.max_triplets is not None and args.max_triplets <= 0:
        raise SystemExit("--max-triplets must be positive")
    with args.config.open() as handle:
        config = yaml.safe_load(handle)["preprocessing"]
    frame = load_manifest(args.manifest)
    all_triplets = list(iter_triplets(frame, exclude_intervened=False, exclude_no_residual=False))
    exclusion_rows = []
    triplets = []
    for triplet in all_triplets:
        reasons = triplet_exclusion_reasons(
            triplet,
            exclude_intervened=bool(config["exclude_intervened"]),
            exclude_no_residual=bool(config.get("exclude_no_residual", False)),
        )
        if reasons:
            exclusion_rows.append(
                {
                    "patient_id": triplet.patient_id,
                    "triplet_id": triplet.triplet_id,
                    "reasons": ";".join(reasons),
                }
            )
        else:
            triplets.append(triplet)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exclusion_rows, columns=["patient_id", "triplet_id", "reasons"]).to_csv(
        args.output_dir / "exclusions.csv", index=False
    )
    if not triplets:
        raise SystemExit("no eligible triples; exclusions.csv records why")
    eligible_triplet_count = len(triplets)
    if args.triplet_id:
        requested = set(args.triplet_id)
        triplets = [triplet for triplet in triplets if triplet.triplet_id in requested]
        found = {triplet.triplet_id for triplet in triplets}
        if found != requested:
            raise SystemExit(
                f"requested triplets are missing or excluded: {sorted(requested - found)}"
            )
    elif args.max_triplets is not None:
        triplets = triplets[: args.max_triplets]
    triplet_dir = args.output_dir / "triplets"
    triplet_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    failures = []
    for number, triplet in enumerate(triplets, start=1):
        output_path = triplet_dir / f"{safe_name(triplet.triplet_id)}.npz"
        print(f"[{number}/{len(triplets)}] {triplet.triplet_id}")
        try:
            if output_path.exists() and not args.overwrite:
                with np.load(output_path, allow_pickle=False) as archive:
                    cached_images = archive["images"]
                    cached_masks = archive["masks"].astype(bool)
                    cached_sdfs = archive["sdfs"]
                    cached_days = archive["days"].astype(float)
                    cached_finite = bool(
                        np.isfinite(cached_images).all() and np.isfinite(cached_sdfs).all()
                    )
                    if not cached_finite:
                        raise ValueError("cached image/SDF contains NaN/Inf")
                    if "_metadata_json" in archive:
                        metadata = json.loads(str(archive["_metadata_json"].item()))
                        refresh_cached_geometry(metadata)
                    else:
                        metadata = {
                            "patient_id": triplet.patient_id,
                            "triplet_id": triplet.triplet_id,
                            "timepoint_ids": json.dumps([str(row["timepoint_id"]) for row in triplet.rows]),
                            "interval_1_days": float(cached_days[1] - cached_days[0]),
                            "interval_2_days": float(cached_days[2] - cached_days[1]),
                            "intervention_unknown": triplet.intervention_unknown,
                            "input_crop_touches_border": touches_border(cached_masks[1]),
                            "target_crop_touches_border": touches_border(cached_masks[2]),
                            "source_images": json.dumps(
                                [str(resolve_path(str(row["image_path"]), args.manifest, args.data_root)) for row in triplet.rows]
                            ),
                            "source_masks": json.dumps(
                                [str(resolve_path(str(row["mask_path"]), args.manifest, args.data_root)) for row in triplet.rows]
                            ),
                            "registration": json.dumps(["rigid_mattes_mi", None, "rigid_mattes_mi"]),
                            "cache_metadata": "legacy_incomplete",
                            "preprocessed_finite": cached_finite,
                        }
            else:
                arrays, metadata = process_triplet(triplet, args.manifest, args.data_root, config)
                arrays["_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
                np.savez_compressed(output_path, **arrays)
            metadata["prepared_path"] = output_path.relative_to(args.output_dir).as_posix()
            apply_case_qc(
                metadata,
                float(config.get("registration_overlap_review_threshold", 0.1)),
            )
            metadata_rows.append(metadata)
        except Exception as error:  # continue to produce an auditable failure table
            failures.append(
                {
                    "patient_id": triplet.patient_id,
                    "triplet_id": triplet.triplet_id,
                    "category": failure_category(error),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"  FAILED: {failures[-1]['error']}")
        # Persist progress after every case so an interrupted multi-hour run
        # never hides which triplets succeeded or failed.
        pd.DataFrame(metadata_rows).to_csv(args.output_dir / "metadata.csv", index=False)
        pd.DataFrame(
            failures, columns=["patient_id", "triplet_id", "category", "error"]
        ).to_csv(args.output_dir / "failures.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(args.output_dir / "metadata.csv", index=False)
    pd.DataFrame(
        failures, columns=["patient_id", "triplet_id", "category", "error"]
    ).to_csv(
        args.output_dir / "failures.csv", index=False
    )
    metadata_frame = pd.DataFrame(metadata_rows)
    overlap_values = []
    for value in metadata_frame.get("registration_overlap_dice_to_t2", []):
        overlap_values.extend(
            overlap for index, overlap in enumerate(json.loads(value)) if index != 1
        )
    geometry_mismatches = 0
    for value in metadata_frame.get("source_geometry", []):
        geometry_mismatches += sum(
            not item["mask_geometry_matched_image"] for item in json.loads(value)
        )
    all_finite = bool(
        len(metadata_frame)
        and metadata_frame.get("preprocessed_finite", pd.Series(dtype=bool)).fillna(False).all()
    )
    crop_review_count = int(
        metadata_frame.get("target_crop_touches_border", pd.Series(dtype=bool))
        .fillna(False)
        .sum()
    )
    overlap_review_threshold = float(config.get("registration_overlap_review_threshold", 0.1))
    low_overlap_count = sum(value < overlap_review_threshold for value in overlap_values)
    zero_overlap_count = sum(value == 0.0 for value in overlap_values)
    qc_pass = metadata_frame.get(
        "preprocessing_qc_pass", pd.Series(False, index=metadata_frame.index)
    ).fillna(False).astype(bool)
    qc_reason_counts = count_values(
        [
            reason
            for value in metadata_frame.get("preprocessing_qc_reasons", [])
            for reason in json.loads(value)
        ]
    )
    qc_summary = {
        "candidate_triplets": len(all_triplets),
        "eligible_triplets": eligible_triplet_count,
        "selected_triplets": len(triplets),
        "excluded_triplets": len(exclusion_rows),
        "prepared_triplets": len(metadata_rows),
        "prepared_patients": int(metadata_frame["patient_id"].nunique()) if len(metadata_frame) else 0,
        "preprocessing_qc_pass_triplets": int(qc_pass.sum()),
        "preprocessing_qc_pass_patients": int(
            metadata_frame.loc[qc_pass, "patient_id"].nunique()
        ) if len(metadata_frame) else 0,
        "preprocessing_qc_failure_reason_counts": qc_reason_counts,
        "failed_triplets": len(failures),
        "failure_reason_counts": count_values(
            [item["category"] for item in failures]
        ),
        "exclusion_reason_counts": count_values(
            [reason for item in exclusion_rows for reason in item["reasons"].split(";")]
        ),
        "input_crop_touches_border": int(metadata_frame.get("input_crop_touches_border", pd.Series(dtype=bool)).fillna(False).sum()),
        "target_crop_touches_border": crop_review_count,
        "source_image_mask_geometry_mismatches": geometry_mismatches,
        "moving_registration_overlap_dice_mean": float(np.mean(overlap_values)) if overlap_values else None,
        "registration_overlap_review_threshold": overlap_review_threshold,
        "moving_registration_overlap_below_review_threshold": low_overlap_count,
        "moving_registration_zero_overlap": zero_overlap_count,
        "all_preprocessed_finite": all_finite,
        "requires_visual_crop_review": crop_review_count > 0,
        "requires_visual_registration_review": low_overlap_count > 0,
        "automatic_qc_pass": (
            not failures
            and geometry_mismatches == 0
            and all_finite
            and crop_review_count == 0
            and low_overlap_count == 0
        ),
    }
    with (args.output_dir / "qc_summary.json").open("w") as handle:
        json.dump(qc_summary, handle, indent=2, allow_nan=False)
    print(f"Prepared {len(metadata_rows)}/{len(triplets)} triples in {args.output_dir}")
    print(f"Excluded {len(exclusion_rows)}/{len(all_triplets)} candidate triples; see exclusions.csv")
    if failures:
        print(f"WARNING: {len(failures)} failures; inspect failures.csv")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
