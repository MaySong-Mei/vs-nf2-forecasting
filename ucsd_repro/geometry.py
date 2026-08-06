from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage


def signed_distance(
    mask: np.ndarray, spacing_mm: float | Sequence[float], clip_mm: float | None = None
) -> np.ndarray:
    """Return an SDF that is negative inside, matching released DeepGrowth code."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        raise ValueError("cannot compute an SDF for an empty mask")
    spacing = (spacing_mm,) * binary.ndim if np.isscalar(spacing_mm) else tuple(spacing_mm)
    outside = ndimage.distance_transform_edt(~binary, sampling=spacing)
    inside = ndimage.distance_transform_edt(binary, sampling=spacing)
    sdf = outside - inside
    if clip_mm is not None:
        if clip_mm <= 0:
            raise ValueError("clip_mm must be positive")
        sdf = np.clip(sdf, -clip_mm, clip_mm) / clip_mm
    return sdf.astype(np.float32)


def crop_or_pad(
    array: np.ndarray, center_zyx: Sequence[float], output_shape: Sequence[int]
) -> np.ndarray:
    """Crop around a center and zero-pad without wrapping at image boundaries."""
    if array.ndim != len(output_shape):
        raise ValueError("array and output_shape dimensionality differ")
    output_shape = tuple(int(value) for value in output_shape)
    starts = [int(round(center - size / 2)) for center, size in zip(center_zyx, output_shape)]
    result = np.zeros(output_shape, dtype=array.dtype)
    source_slices = []
    target_slices = []
    for start, size, source_size in zip(starts, output_shape, array.shape):
        source_start = max(start, 0)
        source_stop = min(start + size, source_size)
        target_start = max(-start, 0)
        target_stop = target_start + max(source_stop - source_start, 0)
        source_slices.append(slice(source_start, source_stop))
        target_slices.append(slice(target_start, target_stop))
    result[tuple(target_slices)] = array[tuple(source_slices)]
    return result


def mask_centroid(mask: np.ndarray) -> tuple[float, ...]:
    points = np.argwhere(np.asarray(mask) > 0)
    if not len(points):
        raise ValueError("mask is empty")
    return tuple(points.mean(axis=0).tolist())


def touches_border(mask: np.ndarray) -> bool:
    binary = np.asarray(mask, dtype=bool)
    return any(np.any(np.take(binary, [0, -1], axis=axis)) for axis in range(binary.ndim))


def normalize_intensity(
    image: np.ndarray, percentiles: Sequence[float] = (0.5, 99.5)
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    foreground = image[np.isfinite(image) & (image != 0)]
    if not len(foreground):
        raise ValueError("image has no finite nonzero voxels")
    low, high = np.percentile(foreground, percentiles)
    if high <= low:
        raise ValueError("image intensity percentile range is degenerate")
    clipped = np.clip(image, low, high)
    normalized = 2.0 * (clipped - low) / (high - low) - 1.0
    normalized[image == 0] = -1.0
    return normalized.astype(np.float32)

