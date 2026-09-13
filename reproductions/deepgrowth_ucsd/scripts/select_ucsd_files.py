#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path, PurePosixPath

import pandas as pd


EXAM_ID = re.compile(r"^VS_\d{4}_\d{2}$")


def parse_required_bool(value: object, field: str, exam_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{field} is not boolean for {exam_id}: {value!r}")


def selected_relative_paths(
    clinical: pd.DataFrame, package_root: str = "UCSD-VS-Longitudinal"
) -> list[str]:
    required = ["ID", "Has T1post", "Has T1postIAC"]
    missing = [column for column in required if column not in clinical]
    if missing:
        raise ValueError(f"clinical table is missing columns: {missing}")
    if clinical["ID"].duplicated().any():
        raise ValueError("clinical table contains duplicate examination IDs")

    paths = []
    for _, row in clinical.sort_values("ID").iterrows():
        exam_id = str(row["ID"])
        if not EXAM_ID.fullmatch(exam_id):
            raise ValueError(f"unexpected examination ID: {exam_id!r}")
        has_iac = parse_required_bool(row["Has T1postIAC"], "Has T1postIAC", exam_id)
        has_standard = parse_required_bool(row["Has T1post"], "Has T1post", exam_id)
        if not has_iac and not has_standard:
            raise ValueError(f"no usable post-contrast T1 sequence for {exam_id}")
        reference = "t1postIAC" if has_iac else "t1post"
        directory = PurePosixPath(package_root) / exam_id
        paths.extend(
            [
                str(directory / f"{exam_id}_{reference}.nii.gz"),
                str(directory / f"{exam_id}_seg_{reference}.nii.gz"),
            ]
        )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the minimum image/mask path list for the official UCSD Faspex package"
    )
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package-root", default="UCSD-VS-Longitudinal")
    args = parser.parse_args()
    clinical = pd.read_csv(args.clinical, sep="\t", dtype=str)
    paths = selected_relative_paths(clinical, args.package_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(paths) + "\n", encoding="utf-8")
    print(f"Wrote {len(paths)} paths for {len(paths) // 2} examinations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
