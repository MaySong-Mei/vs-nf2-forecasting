#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


def _as_uint8(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    values = image[finite]
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.rint(255.0 * scaled).astype(np.uint8)


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = _as_uint8(image)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    boundary = mask.astype(bool) & ~ndimage.binary_erosion(mask.astype(bool))
    rgb[boundary] = np.asarray([255, 48, 48], dtype=np.uint8)
    return rgb


def render_montage(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if images.shape != masks.shape or images.shape != (3, 64, 64, 64):
        raise ValueError(f"expected matching (3, 64, 64, 64) arrays, got {images.shape}/{masks.shape}")
    center = np.rint(np.argwhere(masks[1]).mean(axis=0)).astype(int) if masks[1].any() else np.asarray([32, 32, 32])
    center = np.clip(center, 0, 63)
    rows = []
    for timepoint in range(3):
        panels = [
            _overlay(images[timepoint, center[0], :, :], masks[timepoint, center[0], :, :]),
            _overlay(images[timepoint, :, center[1], :], masks[timepoint, :, center[1], :]),
            _overlay(images[timepoint, :, :, center[2]], masks[timepoint, :, :, center[2]]),
        ]
        rows.append(np.concatenate(panels, axis=1))
    return np.concatenate(rows, axis=0)


def _overlap_min(value: object) -> float:
    try:
        overlaps = [float(item) for index, item in enumerate(json.loads(str(value))) if index != 1]
        return min(overlaps) if overlaps else float("inf")
    except (TypeError, ValueError, json.JSONDecodeError):
        return float("inf")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render de-identified local preprocessing QC montages")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=6)
    args = parser.parse_args()
    frame = pd.read_csv(args.metadata)
    if args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")
    frame["_lowest_overlap"] = frame.get(
        "registration_overlap_dice_to_t2", pd.Series(index=frame.index, dtype=object)
    ).map(_overlap_min)
    frame["_border_review"] = frame.get(
        "target_crop_touches_border", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    selected = frame.sort_values(
        ["_border_review", "_lowest_overlap", "triplet_id"],
        ascending=[False, True, True],
        kind="stable",
    ).head(args.max_cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for case_number, (_, row) in enumerate(selected.iterrows(), start=1):
        prepared_path = args.prepared_root / str(row["prepared_path"])
        with np.load(prepared_path, allow_pickle=False) as archive:
            montage = render_montage(archive["images"], archive["masks"].astype(bool))
        public_name = f"qc_case_{case_number:03d}.png"
        sitk.WriteImage(sitk.GetImageFromArray(montage, isVector=True), args.output_dir / public_name)
        index_rows.append(
            {
                "qc_file": public_name,
                "triplet_id": row["triplet_id"],
                "reason": "crop_border" if row["_border_review"] else "lowest_registration_overlap",
                "lowest_registration_overlap": row["_lowest_overlap"],
            }
        )
    pd.DataFrame(index_rows).to_csv(args.output_dir / "private_qc_index.csv", index=False)
    print(f"Rendered {len(index_rows)} de-identified montages in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
