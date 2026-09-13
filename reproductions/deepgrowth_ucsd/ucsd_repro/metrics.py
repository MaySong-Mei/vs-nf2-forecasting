from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    denominator = pred.sum() + truth.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, truth).sum() / denominator)


def hd95(
    prediction: np.ndarray, target: np.ndarray, spacing_mm: float | Sequence[float]
) -> float:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if not pred.any() or not truth.any():
        return float("nan")
    spacing = (spacing_mm,) * pred.ndim if np.isscalar(spacing_mm) else tuple(spacing_mm)
    structure = ndimage.generate_binary_structure(pred.ndim, 1)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
    truth_surface = truth ^ ndimage.binary_erosion(truth, structure=structure, border_value=0)
    distance_to_truth = ndimage.distance_transform_edt(~truth_surface, sampling=spacing)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([distance_to_truth[pred_surface], distance_to_pred[truth_surface]])
    return float(np.percentile(distances, 95))


def case_metrics(
    prediction: np.ndarray, target: np.ndarray, spacing_mm: float | Sequence[float]
) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    spacing = (spacing_mm,) * pred.ndim if np.isscalar(spacing_mm) else tuple(spacing_mm)
    voxel_volume_mm3 = float(np.prod(spacing))
    pred_volume = float(pred.sum() * voxel_volume_mm3)
    target_volume = float(truth.sum() * voxel_volume_mm3)
    if target_volume == 0:
        signed_rvd = float("nan")
    else:
        signed_rvd = (pred_volume - target_volume) / target_volume
    return {
        "dice": dice(pred, truth),
        "hd95_mm": hd95(pred, truth, spacing),
        "signed_rvd": float(signed_rvd),
        "absolute_rvd": float(abs(signed_rvd)),
        "pred_volume_mm3": pred_volume,
        "target_volume_mm3": target_volume,
        "volume_absolute_error_mm3": abs(pred_volume - target_volume),
    }

