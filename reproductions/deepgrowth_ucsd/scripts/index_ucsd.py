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
UCSD_EXAM = re.compile(r"^(?P<patient>VS_\d{4})_(?P<timepoint>\d{2})$", re.IGNORECASE)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index the official UCSD NIfTI archive")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--layout",
        choices=("ucsd", "regex"),
        default="ucsd",
        help="official VS_XXXX_YY layout (default), or the legacy regex/BIDS-like layout",
    )
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
    if len(digits) == 8:
        try:
            return datetime.strptime(digits, "%Y%m%d")
        except ValueError:
            pass
    return None


def nifti_by_stem(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        lower = path.name.lower()
        if not path.is_file() or not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
            continue
        stem = lower[:-7] if lower.endswith(".nii.gz") else lower[:-4]
        files[stem] = path
    return files


def index_official(root: Path) -> tuple[list[dict], list[tuple]]:
    rows: list[dict] = []
    ambiguous: list[tuple] = []
    exam_directories = sorted(
        path for path in root.rglob("*") if path.is_dir() and UCSD_EXAM.fullmatch(path.name)
    )
    if UCSD_EXAM.fullmatch(root.name):
        exam_directories.insert(0, root)
    for directory in exam_directories:
        match = UCSD_EXAM.fullmatch(directory.name)
        assert match is not None
        exam_id = directory.name.upper()
        files = nifti_by_stem(directory)
        pairs = []
        # The segmentation suffix declares its exact reference image. Prefer no
        # modality heuristics: a t1postIAC mask must be paired with t1postIAC.
        for reference in ("t1postiac", "t1post"):
            mask = files.get(f"{exam_id.lower()}_seg_{reference}")
            image = files.get(f"{exam_id.lower()}_{reference}")
            if mask is not None and image is not None:
                pairs.append((reference, image, mask))
        if len(pairs) != 1:
            ambiguous.append((exam_id, len(pairs), str(directory)))
            continue
        reference, image, mask = pairs[0]
        rows.append(
            {
                "patient_id": match.group("patient").upper(),
                "timepoint_id": exam_id,
                "study_days": pd.NA,
                "image_path": image.relative_to(root).as_posix(),
                "mask_path": mask.relative_to(root).as_posix(),
                "intervened_since_previous": "",
                "no_residual_vs": "",
                "growth_class": "",
                "reference_sequence": reference,
            }
        )
    return rows, ambiguous


def index_regex(root: Path, args: argparse.Namespace) -> tuple[list[dict], list[tuple]]:
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

    rows: list[dict] = []
    ambiguous: list[tuple] = []
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
                "no_residual_vs": "",
                "growth_class": "",
                "reference_sequence": "regex-selected",
            }
        )
    return rows, ambiguous


def main() -> int:
    args = arguments()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"data root does not exist: {root}")
    rows, ambiguous = index_official(root) if args.layout == "ucsd" else index_regex(root, args)
    if not rows:
        raise SystemExit(
            "no unique image/mask pairs found; inspect the archive layout and use --layout regex only if needed"
        )
    frame = pd.DataFrame(rows)
    if "session_date" in frame:
        for _, indexes in frame.groupby("patient_id").groups.items():
            dates = frame.loc[indexes, "session_date"]
            if dates.notna().all():
                baseline = dates.min()
                frame.loc[indexes, "study_days"] = dates.map(lambda value: (value - baseline).days)
        frame = frame.drop(columns=["session_date"])
    frame = frame.sort_values(["patient_id", "timepoint_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} paired timepoints from {frame.patient_id.nunique()} patients to {args.output}")
    if frame["study_days"].isna().any():
        print("WARNING: merge the official clinical TSV to populate study_days and treatment fields.")
    if ambiguous:
        print(f"WARNING: skipped {len(ambiguous)} ambiguous/unpaired examinations:")
        for item in ambiguous[:20]:
            print("  ", item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
