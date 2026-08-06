#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucsd_repro.geometry import signed_distance


def sphere(shape: tuple[int, int, int], center: tuple[float, float, float], radius: float) -> np.ndarray:
    z, y, x = np.indices(shape)
    return ((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2) <= radius**2


def main() -> int:
    parser = argparse.ArgumentParser(description="Create small synthetic longitudinal tumor triples")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--shape", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    triplet_dir = args.output_dir / "triplets"
    triplet_dir.mkdir(exist_ok=True)
    rows = []
    split_rows = []
    shape = (args.shape,) * 3
    z, y, x = np.indices(shape)
    rng = np.random.default_rng(100)
    for patient_number in range(args.patients):
        patient_id = f"SYN-{patient_number:03d}"
        center = tuple(np.asarray(shape) / 2 + rng.uniform(-2, 2, 3))
        base_radius = 4.0 + (patient_number % 3)
        rate = (-0.4, 0.0, 0.5)[patient_number % 3]
        radii = [base_radius + rate * time for time in range(3)]
        masks = np.stack([sphere(shape, center, radius) for radius in radii]).astype(np.uint8)
        images = []
        for mask in masks:
            distance = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
            image = np.exp(-distance / 8.0) + 0.7 * mask + rng.normal(0, 0.03, shape)
            image = 2 * (image - image.min()) / (image.max() - image.min()) - 1
            images.append(image.astype(np.float32))
        sdfs = np.stack([signed_distance(mask, 0.58, 16.0) for mask in masks])
        triplet_id = f"{patient_id}__t0__t1__t2"
        output = triplet_dir / f"{triplet_id}.npz"
        np.savez_compressed(
            output,
            images=np.stack(images),
            masks=masks,
            sdfs=sdfs,
            days=np.asarray([0, 365, 730], dtype=np.float32),
            spacing_mm=np.asarray([0.58] * 3, dtype=np.float32),
        )
        rows.append(
            {
                "patient_id": patient_id,
                "triplet_id": triplet_id,
                "prepared_path": output.relative_to(args.output_dir).as_posix(),
                "target_crop_touches_border": False,
            }
        )
        split_rows.append({"patient_id": patient_id, "fold": patient_number % 5})
    pd.DataFrame(rows).to_csv(args.output_dir / "metadata.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "splits.csv", index=False)
    print(f"Wrote {args.patients} synthetic patients to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
