#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucsd_repro.manifest import iter_triplets, load_manifest, validate_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a UCSD longitudinal manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--keep-intervened", action="store_true")
    args = parser.parse_args()
    frame = load_manifest(args.manifest)
    errors = validate_files(frame, args.manifest, args.data_root) if args.check_files else []
    triplets = list(iter_triplets(frame, exclude_intervened=not args.keep_intervened))
    print(f"Rows: {len(frame)}")
    print(f"Patients: {frame.patient_id.nunique()}")
    print(f"Eligible rolling triples: {len(triplets)}")
    print(f"Triples with unknown intervention status: {sum(item.intervention_unknown for item in triplets)}")
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
