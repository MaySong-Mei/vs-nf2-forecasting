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
from ucsd_repro.manifest import iter_triplets, load_manifest, resolve_path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UCSD longitudinal triples")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def isotropic_reference(image, spacing_mm: float):
    import SimpleITK as sitk

    size = [int(round(old_size * old_spacing / spacing_mm)) for old_size, old_spacing in zip(image.GetSize(), image.GetSpacing())]
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing((spacing_mm,) * 3)
    return reference


def rigid_transform(fixed, moving, iterations: int):
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
    registration.SetMetricSamplingPercentage(0.2, 100)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)
    return registration.Execute(fixed, moving)


def resample(image, reference, transform, is_mask: bool):
    import SimpleITK as sitk

    interpolation = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    pixel_type = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
    return sitk.Resample(image, reference, transform, interpolation, 0, pixel_type)


def read_pair(image_path: Path, mask_path: Path):
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("only 3D images/masks are supported")
    same_geometry = (
        mask.GetSize() == image.GetSize()
        and np.allclose(mask.GetSpacing(), image.GetSpacing())
        and np.allclose(mask.GetOrigin(), image.GetOrigin())
        and np.allclose(mask.GetDirection(), image.GetDirection())
    )
    if not same_geometry:
        mask = sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return image, mask


def process_triplet(triplet, manifest_path: Path, data_root: Path | None, config: dict) -> tuple[dict, dict]:
    import SimpleITK as sitk

    spacing = float(config["spacing_mm"])
    crop_shape = tuple(int(value) for value in config["crop_size"])
    pairs = []
    for row in triplet.rows:
        image_path = resolve_path(str(row["image_path"]), manifest_path, data_root)
        mask_path = resolve_path(str(row["mask_path"]), manifest_path, data_root)
        image, mask = read_pair(image_path, mask_path)
        pairs.append((image, mask, image_path, mask_path))

    fixed = pairs[1][0]
    reference = isotropic_reference(fixed, spacing)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    images = []
    masks = []
    registration_metrics = []
    for index, (image, mask, _, _) in enumerate(pairs):
        transform = identity if index == 1 else rigid_transform(
            fixed, image, int(config["registration_iterations"])
        )
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
    }
    return arrays, metadata


def main() -> int:
    args = arguments()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)["preprocessing"]
    frame = load_manifest(args.manifest)
    triplets = list(iter_triplets(frame, exclude_intervened=bool(config["exclude_intervened"])))
    if not triplets:
        raise SystemExit("no eligible triples")
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
                    cached_masks = archive["masks"].astype(bool)
                    cached_days = archive["days"].astype(float)
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
                }
            else:
                arrays, metadata = process_triplet(triplet, args.manifest, args.data_root, config)
                np.savez_compressed(output_path, **arrays)
            metadata["prepared_path"] = output_path.relative_to(args.output_dir).as_posix()
            metadata_rows.append(metadata)
        except Exception as error:  # continue to produce an auditable failure table
            failures.append({"triplet_id": triplet.triplet_id, "error": f"{type(error).__name__}: {error}"})
            print(f"  FAILED: {failures[-1]['error']}")
    pd.DataFrame(metadata_rows).to_csv(args.output_dir / "metadata.csv", index=False)
    pd.DataFrame(failures, columns=["triplet_id", "error"]).to_csv(
        args.output_dir / "failures.csv", index=False
    )
    print(f"Prepared {len(metadata_rows)}/{len(triplets)} triples in {args.output_dir}")
    if failures:
        print(f"WARNING: {len(failures)} failures; inspect failures.csv")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
