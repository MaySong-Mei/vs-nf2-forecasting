#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_PATIENT = r"(?P<patient>sub-[^/]+)"
DEFAULT_SESSION = r"(?P<session>ses-[^/]+)"
DEFAULT_IMAGE = r"(?i)(t1.*(ce|post|gad)|(ce|post|gad).*t1).*\.nii(\.gz)?$"
DEFAULT_MASK = r"(?i)(seg|mask|label).*\.nii(\.gz)?$"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a BIDS-like UCSD NIfTI tree")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--patient-regex", default=DEFAULT_PATIENT)
    parser.add_argument("--session-regex", default=DEFAULT_SESSION)
    parser.add_argument("--image-regex", default=DEFAULT_IMAGE)
    parser.add_argument("--mask-regex", default=DEFAULT_MASK)
    return parser.parse_args()


def extract(pattern: re.Pattern[str], text: str, group: str) -> str | None:
    match = pattern.search(text)
    return match.groupdict().get(group) if match else None


def parse_session_date(session: str) -> datetime | None:
    digits = re.sub(r"\D", "", session)
    for fmt, length in (("%Y%m%d", 8), ("%Y-%m-%d", 10)):
        if len(digits) == length:
            try:
                return datetime.strptime(digits, fmt.replace("-", ""))
            except ValueError:
                pass
    return None


def main() -> int:
    args = arguments()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"data root does not exist: {root}")
    patient_pattern = re.compile(args.patient_regex)
    session_pattern = re.compile(args.session_regex)
    image_pattern = re.compile(args.image_regex)
    mask_pattern = re.compile(args.mask_regex)
    candidates: dict[tuple[str, str], dict[str, list[Path]]] = {}
    for path in sorted(root.rglob("*.nii*")):
        relative = path.relative_to(root).as_posix()
        patient = extract(patient_pattern, relative, "patient")
        session = extract(session_pattern, relative, "session")
        if not patient or not session:
            continue
        bucket = candidates.setdefault((patient, session), {"image": [], "mask": []})
        if mask_pattern.search(path.name):
            bucket["mask"].append(path)
        elif image_pattern.search(path.name):
            bucket["image"].append(path)

    rows = []
    ambiguous = []
    for (patient, session), bucket in sorted(candidates.items()):
        if len(bucket["image"]) != 1 or len(bucket["mask"]) != 1:
            ambiguous.append((patient, session, len(bucket["image"]), len(bucket["mask"])))
            continue
        rows.append(
            {
                "patient_id": patient,
                "timepoint_id": session,
                "session_date": parse_session_date(session),
                "image_path": bucket["image"][0].relative_to(root).as_posix(),
                "mask_path": bucket["mask"][0].relative_to(root).as_posix(),
                "intervened_since_previous": "",
                "growth_class": "",
            }
        )
    if not rows:
        raise SystemExit(
            "no unique image/mask pairs found; inspect the archive layout and override the regex arguments"
        )
    frame = pd.DataFrame(rows)
    frame["study_days"] = pd.NA
    for patient, indexes in frame.groupby("patient_id").groups.items():
        dates = frame.loc[indexes, "session_date"]
        if dates.notna().all():
            baseline = dates.min()
            frame.loc[indexes, "study_days"] = dates.map(lambda value: (value - baseline).days)
    frame = frame[
        [
            "patient_id",
            "timepoint_id",
            "study_days",
            "image_path",
            "mask_path",
            "intervened_since_previous",
            "growth_class",
        ]
    ].sort_values(["patient_id", "study_days", "timepoint_id"], na_position="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} paired timepoints from {frame.patient_id.nunique()} patients to {args.output}")
    if frame["study_days"].isna().any():
        print("WARNING: study_days could not be inferred for all rows; merge the clinical timing table before validation.")
    if ambiguous:
        print(f"WARNING: skipped {len(ambiguous)} ambiguous sessions (patient, session, images, masks):")
        for item in ambiguous[:20]:
            print("  ", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

