#!/usr/bin/env python3
"""Render local, de-identified, auditable preprocessing QC panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucsd_repro.manifest import iter_triplets, load_manifest, resolve_path


RENDERER_VERSION = 2
TIMEPOINT_COLORS = (
    np.asarray([255, 64, 64], dtype=np.uint8),
    np.asarray([64, 255, 96], dtype=np.uint8),
    np.asarray([64, 128, 255], dtype=np.uint8),
)


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


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    return mask & ~ndimage.binary_erosion(mask)


def _overlay(image: np.ndarray, masks: Iterable[np.ndarray]) -> np.ndarray:
    gray = _as_uint8(image)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    for color, mask in zip(TIMEPOINT_COLORS, masks):
        rgb[_boundary(mask)] = color
    return rgb


def _center(masks: np.ndarray) -> np.ndarray:
    union = masks.astype(bool).any(axis=0)
    if union.any():
        center = np.rint(np.argwhere(union).mean(axis=0)).astype(int)
    else:
        center = np.asarray(union.shape) // 2
    return np.clip(center, 0, np.asarray(union.shape) - 1)


def _planes(volume: np.ndarray, center: np.ndarray) -> list[np.ndarray]:
    z, y, x = (int(value) for value in center)
    return [volume[z, :, :], volume[:, y, :], volume[:, :, x]]


def border_contacts(mask: np.ndarray) -> list[str]:
    if mask.ndim != 3:
        raise ValueError("border contact expects a 3D mask")
    names_and_values = (
        ("z_min", mask[0]),
        ("z_max", mask[-1]),
        ("y_min", mask[:, 0]),
        ("y_max", mask[:, -1]),
        ("x_min", mask[:, :, 0]),
        ("x_max", mask[:, :, -1]),
    )
    return [name for name, values in names_and_values if np.asarray(values).astype(bool).any()]


def _crop_coverage_panel(image: np.ndarray, masks: np.ndarray, center: np.ndarray) -> list[np.ndarray]:
    panels = []
    image_planes = _planes(image, center)
    mask_planes = [_planes(mask, center) for mask in masks]
    for plane_index, plane in enumerate(image_planes):
        panel = _overlay(plane, [items[plane_index] for items in mask_planes])
        for timepoint, mask in enumerate(masks):
            if border_contacts(mask):
                offset = timepoint
                color = TIMEPOINT_COLORS[timepoint]
                panel[offset, :, :] = color
                panel[-(offset + 1), :, :] = color
                panel[:, offset, :] = color
                panel[:, -(offset + 1), :] = color
        panels.append(panel)
    return panels


def sdf_heatmap(sdf: np.ndarray, mask: np.ndarray, clip_mm: float) -> np.ndarray:
    """Create a blue-inside/red-outside SDF panel with boundary overlays."""
    if clip_mm <= 0:
        raise ValueError("SDF clip must be positive")
    scaled = np.clip(sdf.astype(float) / clip_mm, -1.0, 1.0)
    rgb = np.empty((*sdf.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.rint(255.0 * np.clip(scaled, 0.0, 1.0)).astype(np.uint8)
    rgb[..., 2] = np.rint(255.0 * np.clip(-scaled, 0.0, 1.0)).astype(np.uint8)
    rgb[..., 1] = np.rint(255.0 * (1.0 - np.abs(scaled))).astype(np.uint8)
    sign_boundary = ndimage.binary_dilation(sdf < 0) ^ ndimage.binary_erosion(sdf < 0)
    rgb[sign_boundary] = np.asarray([255, 255, 0], dtype=np.uint8)
    rgb[_boundary(mask)] = np.asarray([255, 255, 255], dtype=np.uint8)
    return rgb


def _fit_panel(array: np.ndarray, target_shape: tuple[int, int], order: int) -> np.ndarray:
    zoom = (target_shape[0] / array.shape[0], target_shape[1] / array.shape[1])
    resized = ndimage.zoom(array, zoom, order=order)
    output = np.zeros(target_shape, dtype=resized.dtype)
    height = min(target_shape[0], resized.shape[0])
    width = min(target_shape[1], resized.shape[1])
    output[:height, :width] = resized[:height, :width]
    return output


def _source_rows(source_images: list[np.ndarray], source_masks: list[np.ndarray], panel_shape: tuple[int, int]) -> list[np.ndarray]:
    rows = []
    for timepoint, (image, mask) in enumerate(zip(source_images, source_masks)):
        center = _center(mask[None])
        panels = []
        for image_plane, mask_plane in zip(_planes(image, center), _planes(mask, center)):
            resized_image = _fit_panel(image_plane, panel_shape, order=1)
            resized_mask = _fit_panel(mask_plane.astype(np.uint8), panel_shape, order=0).astype(bool)
            rgb = _overlay(resized_image, [resized_mask])
            own_boundary = _boundary(resized_mask)
            rgb[own_boundary] = TIMEPOINT_COLORS[timepoint]
            panels.append(rgb)
        rows.append(np.concatenate(panels, axis=1))
    return rows


def render_case_montage(
    images: np.ndarray,
    masks: np.ndarray,
    sdfs: np.ndarray | None = None,
    *,
    sdf_clip_mm: float = 16.0,
    source_images: list[np.ndarray] | None = None,
    source_masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    if images.shape != masks.shape or images.ndim != 4 or images.shape[0] != 3:
        raise ValueError(f"expected matching (3, D, H, W) arrays, got {images.shape}/{masks.shape}")
    if sdfs is not None and sdfs.shape != images.shape:
        raise ValueError(f"SDF shape {sdfs.shape} does not match images {images.shape}")
    center = _center(masks)
    rows = []
    for timepoint in range(3):
        panels = [
            _overlay(image_plane, [mask_plane])
            for image_plane, mask_plane in zip(
                _planes(images[timepoint], center), _planes(masks[timepoint], center)
            )
        ]
        own_color = TIMEPOINT_COLORS[timepoint]
        for panel, mask_plane in zip(panels, _planes(masks[timepoint], center)):
            panel[_boundary(mask_plane)] = own_color
        rows.append(np.concatenate(panels, axis=1))

    mask_planes = [_planes(mask, center) for mask in masks]
    rows.append(
        np.concatenate(
            [
                _overlay(image_plane, [items[index] for items in mask_planes])
                for index, image_plane in enumerate(_planes(images[1], center))
            ],
            axis=1,
        )
    )
    rows.append(np.concatenate(_crop_coverage_panel(images[1], masks, center), axis=1))

    if source_images is not None and source_masks is not None:
        rows.extend(_source_rows(source_images, source_masks, images.shape[-2:]))
    if sdfs is not None:
        for timepoint in range(3):
            panels = [
                sdf_heatmap(sdf_plane, mask_plane, sdf_clip_mm)
                for sdf_plane, mask_plane in zip(
                    _planes(sdfs[timepoint] * sdf_clip_mm, center),
                    _planes(masks[timepoint], center),
                )
            ]
            rows.append(np.concatenate(panels, axis=1))

    montage = np.concatenate(rows, axis=0)
    finite = np.isfinite(images).all() and (sdfs is None or np.isfinite(sdfs).all())
    contrast_ok = all(np.nanpercentile(image, 99) > np.nanpercentile(image, 1) for image in images)
    if not finite or not contrast_ok:
        montage[:3, :, :] = np.asarray([255, 0, 255], dtype=np.uint8)
        montage[-3:, :, :] = np.asarray([255, 0, 255], dtype=np.uint8)
        montage[:, :3, :] = np.asarray([255, 0, 255], dtype=np.uint8)
        montage[:, -3:, :] = np.asarray([255, 0, 255], dtype=np.uint8)
    return montage


def render_montage(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Backward-compatible compact montage used by earlier tests."""
    if images.shape != masks.shape or images.shape != (3, 64, 64, 64):
        raise ValueError(f"expected matching (3, 64, 64, 64) arrays, got {images.shape}/{masks.shape}")
    center = _center(masks)
    rows = []
    for timepoint in range(3):
        panels = []
        for image_plane, mask_plane in zip(
            _planes(images[timepoint], center), _planes(masks[timepoint], center)
        ):
            rgb = _overlay(image_plane, [mask_plane])
            rgb[_boundary(mask_plane)] = TIMEPOINT_COLORS[timepoint]
            panels.append(rgb)
        rows.append(np.concatenate(panels, axis=1))
    return np.concatenate(rows, axis=0)


def _overlap_min(value: object) -> float:
    try:
        overlaps = [float(item) for index, item in enumerate(json.loads(str(value))) if index != 1]
        return min(overlaps) if overlaps else float("inf")
    except (TypeError, ValueError, json.JSONDecodeError):
        return float("inf")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _row_flags(row: pd.Series) -> list[str]:
    raw = row.get("preprocessing_qc_reasons", "[]")
    try:
        flags = [str(value) for value in json.loads(str(raw))]
    except (TypeError, json.JSONDecodeError):
        flags = []
    if not flags and _as_bool(row.get("target_crop_touches_border", False)):
        flags.append("crop_touches_border")
    threshold = float(row.get("registration_overlap_review_threshold", 0.1))
    if not flags and _overlap_min(row.get("registration_overlap_dice_to_t2", "[]")) < threshold:
        flags.append("registration_overlap_below_threshold")
    return sorted(set(flags))


def _sample_passed(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[0:0]
    ordered = frame.sort_values(["_lowest_overlap", "triplet_id"], kind="stable")
    if count >= len(ordered):
        return ordered
    chunks = np.array_split(np.arange(len(ordered)), count)
    positions = []
    for stratum, chunk in enumerate(chunks):
        digest = hashlib.sha256(f"{seed}:{stratum}".encode()).digest()
        positions.append(int(chunk[int.from_bytes(digest[:4], "big") % len(chunk)]))
    return ordered.iloc[positions]


def select_cases(
    frame: pd.DataFrame,
    *,
    all_flagged: bool,
    sample_passed: int,
    max_cases: int,
    seed: int,
) -> pd.DataFrame:
    work = frame.copy()
    work["_lowest_overlap"] = work.get(
        "registration_overlap_dice_to_t2", pd.Series(index=work.index, dtype=object)
    ).map(_overlap_min)
    work["_automatic_flags"] = work.apply(_row_flags, axis=1)
    if "preprocessing_qc_pass" in work:
        work["_qc_pass"] = work["preprocessing_qc_pass"].map(_as_bool)
    else:
        work["_qc_pass"] = work["_automatic_flags"].map(lambda value: not value)
    flagged = work.loc[~work["_qc_pass"]].sort_values(
        ["_lowest_overlap", "triplet_id"], kind="stable"
    )
    if not all_flagged:
        flagged = flagged.head(max_cases)
    passed = _sample_passed(work.loc[work["_qc_pass"]], sample_passed, seed)
    selected = pd.concat([flagged, passed], ignore_index=False)
    if not all_flagged and sample_passed == 0:
        selected = selected.head(max_cases)
    return selected.drop_duplicates("triplet_id").reset_index(drop=True)


def _load_sources(image_paths: list[str], mask_paths: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    import SimpleITK as sitk

    images = []
    masks = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
        mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        images.append(sitk.GetArrayFromImage(image).astype(np.float32))
        masks.append(sitk.GetArrayFromImage(mask).astype(bool))
    return images, masks


def _image_ranges(images: np.ndarray) -> list[dict[str, float | bool]]:
    result = []
    for image in images:
        finite = np.isfinite(image)
        values = image[finite]
        result.append(
            {
                "finite": bool(finite.all()),
                "minimum": float(values.min()) if len(values) else 0.0,
                "p01": float(np.percentile(values, 1)) if len(values) else 0.0,
                "p99": float(np.percentile(values, 99)) if len(values) else 0.0,
                "maximum": float(values.max()) if len(values) else 0.0,
            }
        )
    return result


def _render_config_hash(args: argparse.Namespace, selected_ids: list[str], failure_ids: list[str]) -> str:
    config = {
        "renderer_version": RENDERER_VERSION,
        "all_flagged": args.all_flagged,
        "sample_passed": args.sample_passed,
        "max_cases": args.max_cases,
        "seed": args.seed,
        "include_registration": args.include_registration,
        "include_sdf": args.include_sdf,
        "sdf_clip_mm": args.sdf_clip_mm,
        "selected_ids": selected_ids,
        "failure_ids": failure_ids,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def _write_png(path: Path, montage: np.ndarray) -> None:
    import SimpleITK as sitk

    sitk.WriteImage(sitk.GetImageFromArray(montage, isVector=True), str(path))


def _failure_triplets(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    return {
        triplet.triplet_id: triplet
        for triplet in iter_triplets(
            manifest, exclude_intervened=False, exclude_no_residual=False
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render de-identified local preprocessing QC panels")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--all-flagged", action="store_true")
    parser.add_argument("--sample-passed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--include-registration", action="store_true")
    parser.add_argument("--include-sdf", action="store_true")
    parser.add_argument("--sdf-clip-mm", type=float, default=16.0)
    args = parser.parse_args()
    if args.max_cases < 1 or args.sample_passed < 0:
        raise SystemExit("--max-cases must be positive and --sample-passed cannot be negative")
    if bool(args.failures) != bool(args.manifest):
        raise SystemExit("--failures and --manifest must be supplied together")
    if args.manifest and not args.manifest.is_file():
        raise SystemExit(f"manifest does not exist: {args.manifest}")
    if args.failures and not args.failures.is_file():
        raise SystemExit(f"failure table does not exist: {args.failures}")

    frame = pd.read_csv(args.metadata)
    selected = select_cases(
        frame,
        all_flagged=args.all_flagged,
        sample_passed=args.sample_passed,
        max_cases=args.max_cases,
        seed=args.seed,
    )
    failures = pd.read_csv(args.failures) if args.failures else pd.DataFrame()
    selected_ids = selected["triplet_id"].astype(str).tolist()
    failure_ids = failures.get("triplet_id", pd.Series(dtype=str)).astype(str).tolist()
    config_hash = _render_config_hash(args, selected_ids, failure_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    case_number = 0
    for _, row in selected.iterrows():
        case_number += 1
        opaque_id = f"qc-{config_hash}-{case_number:03d}"
        public_name = f"{opaque_id}.png"
        prepared_path = args.prepared_root / str(row["prepared_path"])
        source_images = source_masks = None
        source_status = "not_requested"
        with np.load(prepared_path, allow_pickle=False) as archive:
            images = archive["images"].astype(np.float32)
            masks = archive["masks"].astype(bool)
            sdfs = archive["sdfs"].astype(np.float32) if args.include_sdf else None
        if args.include_registration:
            try:
                image_paths = json.loads(str(row.get("source_images", "[]")))
                mask_paths = json.loads(str(row.get("source_masks", "[]")))
                source_images, source_masks = _load_sources(image_paths, mask_paths)
                source_status = "rendered"
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
                source_status = "unavailable"
        montage = render_case_montage(
            images,
            masks,
            sdfs,
            sdf_clip_mm=args.sdf_clip_mm,
            source_images=source_images,
            source_masks=source_masks,
        )
        _write_png(args.output_dir / public_name, montage)
        contacts = {
            f"t{index + 1}": border_contacts(mask) for index, mask in enumerate(masks)
        }
        review_rows.append(
            {
                "opaque_id": opaque_id,
                "qc_file": public_name,
                "case_kind": "prepared",
                "render_status": "rendered",
                "source_registration_panel": source_status,
                "automatic_flags": json.dumps(row["_automatic_flags"]),
                "lowest_registration_overlap": float(row["_lowest_overlap"]),
                "registration_review_threshold": float(
                    row.get("registration_overlap_review_threshold", 0.1)
                ),
                "border_contacts": json.dumps(contacts, sort_keys=True),
                "intensity_ranges": json.dumps(_image_ranges(images), sort_keys=True),
                "sdf_negative_inside": bool(
                    sdfs is not None
                    and all(float(np.median(sdf[mask])) < 0 for sdf, mask in zip(sdfs, masks) if mask.any())
                ),
                "sdf_storage_scale": "normalized_by_clip_mm" if sdfs is not None else "not_rendered",
                "sdf_clip_mm": args.sdf_clip_mm if sdfs is not None else None,
                "reviewer_decision": "",
                "review_reason_codes": "",
                "reviewer_notes": "",
                "reviewer_timestamp_utc": "",
                "rendering_config_hash": config_hash,
            }
        )
        mapping_rows.append(
            {
                "opaque_id": opaque_id,
                "patient_id": row.get("patient_id", ""),
                "triplet_id": row["triplet_id"],
                "prepared_path": str(row["prepared_path"]),
                "source_images": row.get("source_images", ""),
                "source_masks": row.get("source_masks", ""),
            }
        )

    failure_lookup = _failure_triplets(args.manifest) if args.manifest else {}
    for _, failure in failures.iterrows():
        case_number += 1
        opaque_id = f"qc-{config_hash}-{case_number:03d}"
        public_name = f"{opaque_id}.png"
        triplet_id = str(failure["triplet_id"])
        render_status = "source_debug_unavailable"
        triplet = failure_lookup.get(triplet_id)
        source_paths: list[str] = []
        mask_paths: list[str] = []
        if triplet is not None:
            try:
                source_paths = [
                    str(resolve_path(str(row["image_path"]), args.manifest, args.data_root))
                    for row in triplet.rows
                ]
                mask_paths = [
                    str(resolve_path(str(row["mask_path"]), args.manifest, args.data_root))
                    for row in triplet.rows
                ]
                source_images, source_masks = _load_sources(source_paths, mask_paths)
                rows = _source_rows(source_images, source_masks, (64, 64))
                _write_png(args.output_dir / public_name, np.concatenate(rows, axis=0))
                render_status = "source_debug_rendered"
            except (OSError, RuntimeError, ValueError):
                render_status = "source_debug_unavailable"
        review_rows.append(
            {
                "opaque_id": opaque_id,
                "qc_file": public_name if render_status.endswith("rendered") else "",
                "case_kind": "preprocessing_failure",
                "render_status": render_status,
                "source_registration_panel": "not_applicable",
                "automatic_flags": json.dumps([str(failure.get("category", "preprocessing_failure"))]),
                "lowest_registration_overlap": None,
                "registration_review_threshold": None,
                "border_contacts": "{}",
                "intensity_ranges": "[]",
                "sdf_negative_inside": None,
                "sdf_storage_scale": "not_available",
                "sdf_clip_mm": None,
                "reviewer_decision": "",
                "review_reason_codes": "",
                "reviewer_notes": "",
                "reviewer_timestamp_utc": "",
                "rendering_config_hash": config_hash,
            }
        )
        mapping_rows.append(
            {
                "opaque_id": opaque_id,
                "patient_id": failure.get("patient_id", ""),
                "triplet_id": triplet_id,
                "prepared_path": "",
                "source_images": json.dumps(source_paths),
                "source_masks": json.dumps(mask_paths),
            }
        )

    review_path = args.output_dir / f"review_index_{config_hash}.csv"
    mapping_path = args.output_dir / f"private_case_mapping_{config_hash}.csv"
    pd.DataFrame(review_rows).to_csv(review_path, index=False)
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)
    summary = {
        "rendering_config_hash": config_hash,
        "renderer_version": RENDERER_VERSION,
        "selected_prepared_cases": int(len(selected)),
        "selected_automatic_qc_failed": int((~selected["_qc_pass"]).sum()),
        "selected_automatic_qc_passed": int(selected["_qc_pass"].sum()),
        "preprocessing_failures_accounted_for": int(len(failures)),
        "preprocessing_failure_debug_views_rendered": int(
            sum(row["render_status"] == "source_debug_rendered" for row in review_rows)
        ),
        "review_index": review_path.name,
        "private_mapping": mapping_path.name,
        "privacy": "Local only; no patient/triplet identifiers appear in filenames or review index.",
    }
    with (args.output_dir / f"render_summary_{config_hash}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"Rendered {sum(bool(row['qc_file']) for row in review_rows)} de-identified panels; "
        f"review {review_path} and keep {mapping_path} private"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
