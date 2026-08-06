#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucsd_repro.manifest import iter_triplets, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic patient-level folds")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--keep-intervened", action="store_true")
    parser.add_argument("--keep-no-residual", action="store_true")
    args = parser.parse_args()
    if args.folds < 3:
        raise SystemExit("at least three folds are required")
    frame = load_manifest(args.manifest)
    triplets = list(
        iter_triplets(
            frame,
            exclude_intervened=not args.keep_intervened,
            exclude_no_residual=not args.keep_no_residual,
        )
    )
    patients = sorted({triplet.patient_id for triplet in triplets})
    if len(patients) < args.folds:
        raise SystemExit(f"{len(patients)} eligible patients cannot populate {args.folds} folds")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)
    split = pd.DataFrame({"patient_id": patients, "fold": np.arange(len(patients)) % args.folds})
    split = split.sort_values("patient_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.output, index=False)
    counts = split.groupby("fold").size().to_dict()
    print(
        f"Wrote {len(split)} eligible patients ({len(triplets)} triplets) to "
        f"{args.output}; fold counts: {counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
