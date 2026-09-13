#!/usr/bin/env python3
"""P1 simple baselines: next-follow-up tumor volume from prior volumes only.

Unit of analysis: a rolling triple (t1, t2, t3) inside one patient. Inputs are
the native-space mask volumes at t1 and t2, the two intervals, and optionally
age/sex. Target is the native-space mask volume at t3. No registration,
cropping, or imaging features are used, so the cohort is the clinically
eligible set rather than the registration-QC subset.

Evaluation reuses the frozen patient-level five-fold split (seed 100). Models
that need fitting (pooled rate, ridge, LMM) are fit on training folds only;
ridge regularization is tuned by inner patient-grouped CV inside the training
fold. Patient-level per-case outputs go to an ignored data directory; only
aggregate summaries go to the tracked results directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 100
LAMBDA_GRID = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
BOOTSTRAP = 1000
METRICS = ["abs_rvd", "abs_error_mm3", "abs_log_ratio", "signed_rvd"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opt_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"unexpected boolean {value!r}")


def git_state(root: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception as error:  # pragma: no cover
        return {"commit": None, "dirty": None, "error": str(error)}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_volumes(manifest: pd.DataFrame, data_root: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        cached = pd.read_csv(cache, dtype={"patient_id": str, "timepoint_id": str})
        if set(cached["timepoint_id"]) == set(manifest["timepoint_id"]):
            return cached
    import nibabel as nib

    rows = []
    for row in manifest.itertuples(index=False):
        image = nib.load(str(data_root / row.mask_path))
        data = np.asanyarray(image.dataobj)
        voxel_mm3 = float(abs(np.linalg.det(image.affine[:3, :3])))
        labels = np.unique(data)
        voxels = int((data > 0).sum())
        rows.append(
            {
                "patient_id": row.patient_id,
                "timepoint_id": row.timepoint_id,
                "mask_voxels": voxels,
                "voxel_mm3": voxel_mm3,
                "volume_mm3": voxels * voxel_mm3,
                "mask_labels": ";".join(str(v) for v in labels[:6]),
                "mask_shape": "x".join(str(int(v)) for v in data.shape),
            }
        )
    frame = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def load_clinical(path: Path) -> pd.DataFrame:
    clinical = pd.read_csv(path, sep="\t", dtype=str)
    return pd.DataFrame(
        {
            "timepoint_id": clinical["ID"].str.strip(),
            "age_years": pd.to_numeric(clinical["Age at scan"], errors="coerce"),
            "sex_male": clinical["Sex at birth"].str.strip().str.upper().eq("MALE").astype(float),
            "radiology_growth_class": clinical[
                "Longitudinal classification (respect to prior scan)"
            ]
            .fillna("")
            .str.strip(),
        }
    )


def build_triplets(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        records = group.sort_values("study_days").to_dict("records")
        for start in range(max(0, len(records) - 2)):
            r1, r2, r3 = records[start : start + 3]
            reasons = []
            if (
                opt_bool(r2["intervened_since_previous"]) is True
                or opt_bool(r3["intervened_since_previous"]) is True
            ):
                reasons.append("intervention_between_scans")
            if any(opt_bool(r["no_residual_vs"]) is True for r in (r1, r2, r3)):
                reasons.append("no_residual_tumor")
            if any(r["volume_mm3"] <= 0 for r in (r1, r2, r3)):
                reasons.append("empty_mask")
            unknown = any(opt_bool(r["intervened_since_previous"]) is None for r in (r2, r3))
            posts = sum(str(r.get("pre_post_treatment", "")).strip().lower() == "post" for r in (r1, r2, r3))
            rows.append(
                {
                    "patient_id": patient_id,
                    "triplet_id": (
                        f"{patient_id}__{r1['timepoint_id']}__{r2['timepoint_id']}__{r3['timepoint_id']}"
                    ),
                    "v1": r1["volume_mm3"],
                    "v2": r2["volume_mm3"],
                    "v3": r3["volume_mm3"],
                    "dt1_days": r2["study_days"] - r1["study_days"],
                    "dt2_days": r3["study_days"] - r2["study_days"],
                    "age_t2": r2.get("age_years", np.nan),
                    "sex_male": r2.get("sex_male", np.nan),
                    "radiology_growth_class_t3": r3.get("radiology_growth_class", ""),
                    "intervention_unknown": unknown,
                    "treatment_status": {0: "all_pre", 3: "all_post"}.get(posts, "mixed"),
                    "exclusion_reasons": ";".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def eligible_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Consecutive (t_k, t_k+1) pairs usable for fitting population growth."""
    rows = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        records = group.sort_values("study_days").to_dict("records")
        for a, b in zip(records[:-1], records[1:]):
            if opt_bool(b["intervened_since_previous"]) is True:
                continue
            if opt_bool(a["no_residual_vs"]) is True or opt_bool(b["no_residual_vs"]) is True:
                continue
            if a["volume_mm3"] <= 0 or b["volume_mm3"] <= 0:
                continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "t_start": a["study_days"],
                    "t_end": b["study_days"],
                    "v_start": a["volume_mm3"],
                    "v_end": b["volume_mm3"],
                    "log_rate_per_day": (np.log(b["volume_mm3"]) - np.log(a["volume_mm3"]))
                    / (b["study_days"] - a["study_days"]),
                }
            )
    return pd.DataFrame(rows)


def eligible_timepoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Longest pre-treatment, non-empty run from the first scan of each patient."""
    rows = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        records = group.sort_values("study_days").to_dict("records")
        kept = []
        for index, record in enumerate(records):
            if index > 0 and opt_bool(record["intervened_since_previous"]) is True:
                break
            if opt_bool(record["no_residual_vs"]) is True or record["volume_mm3"] <= 0:
                break
            kept.append(record)
        if len(kept) >= 2:
            origin = kept[0]["study_days"]
            for record in kept:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "t_years": (record["study_days"] - origin) / 365.25,
                        "log_v": np.log(record["volume_mm3"]),
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def design(frame: pd.DataFrame, clinical: bool) -> np.ndarray:
    log_v1 = np.log(frame["v1"].to_numpy())
    log_v2 = np.log(frame["v2"].to_numpy())
    dt1 = frame["dt1_days"].to_numpy() / 365.25
    dt2 = frame["dt2_days"].to_numpy() / 365.25
    rate = (log_v2 - log_v1) / dt1
    columns = [log_v2, log_v2 - log_v1, rate, dt2, rate * dt2, np.log(dt1)]
    if clinical:
        columns += [frame["age_t2"].to_numpy(), frame["sex_male"].to_numpy()]
    return np.column_stack(columns).astype(float)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    penalty = np.eye(Xc.shape[1]) * lam
    penalty[0, 0] = 0.0
    return np.linalg.solve(Xc.T @ Xc + penalty, Xc.T @ y)


def ridge_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def inner_lambda(
    train: pd.DataFrame, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> float:
    patients = train["patient_id"].unique().copy()
    rng.shuffle(patients)
    assignment = {p: i % 5 for i, p in enumerate(patients)}
    inner_fold = train["patient_id"].map(assignment).to_numpy()
    scores = {}
    for lam in LAMBDA_GRID:
        errors = []
        for k in range(5):
            fit, hold = inner_fold != k, inner_fold == k
            if hold.sum() == 0 or fit.sum() < X.shape[1] + 2:
                continue
            mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-12
            beta = ridge_fit((X[fit] - mu) / sd, y[fit], lam)
            errors.append(np.abs(ridge_predict(beta, (X[hold] - mu) / sd) - y[hold]).mean())
        scores[lam] = float(np.mean(errors)) if errors else np.inf
    return min(scores, key=scores.get)


def fit_lmm(points: pd.DataFrame) -> dict | None:
    import statsmodels.formula.api as smf

    result = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "log_v ~ t_years", points, groups=points["patient_id"], re_formula="~t_years"
        )
        for method in ("lbfgs", "powell", "nm"):
            try:
                candidate = model.fit(method=method, reml=True, maxiter=2000)
            except Exception:
                continue
            if np.isfinite(candidate.params).all():
                result = candidate
                break
    if result is None:
        return None
    beta = np.array([result.fe_params["Intercept"], result.fe_params["t_years"]])
    G = np.asarray(result.cov_re, dtype=float)
    return {
        "beta": beta,
        "G": G,
        "sigma2": float(result.scale),
        "converged": bool(result.converged),
    }


def lmm_predict(fit: dict, frame: pd.DataFrame) -> np.ndarray:
    preds = []
    for row in frame.itertuples(index=False):
        t1, t2 = 0.0, row.dt1_days / 365.25
        t3 = t2 + row.dt2_days / 365.25
        Z = np.array([[1.0, t1], [1.0, t2]])
        y = np.array([np.log(row.v1), np.log(row.v2)])
        V = Z @ fit["G"] @ Z.T + fit["sigma2"] * np.eye(2)
        b = fit["G"] @ Z.T @ np.linalg.solve(V, y - Z @ fit["beta"])
        x3 = np.array([1.0, t3])
        preds.append(x3 @ fit["beta"] + x3 @ b)
    return np.exp(np.array(preds))


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def run_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pairs: pd.DataFrame,
    points: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    v1, v2 = test["v1"].to_numpy(), test["v2"].to_numpy()
    dt1, dt2 = test["dt1_days"].to_numpy(), test["dt2_days"].to_numpy()
    log_rate = (np.log(v2) - np.log(v1)) / dt1
    preds = {
        "persistence": v2,
        "linear_volume": np.clip(v2 + (v2 - v1) / dt1 * dt2, 0.0, None),
        "log_linear_volume": v2 * np.exp(log_rate * dt2),
    }
    diagnostics: dict = {}

    train_patients = set(train["patient_id"])
    train_pairs = pairs[pairs["patient_id"].isin(train_patients)]
    pooled_rate = float(train_pairs["log_rate_per_day"].mean())
    preds["pooled_rate"] = v2 * np.exp(pooled_rate * dt2)
    diagnostics["pooled_log_rate_per_year"] = pooled_rate * 365.25
    diagnostics["pooled_rate_pairs"] = int(len(train_pairs))

    y_train = np.log(train["v3"].to_numpy())
    for name, clinical in (("ridge_history", False), ("ridge_history_clinical", True)):
        X_train, X_test = design(train, clinical), design(test, clinical)
        ok = np.isfinite(X_train).all(1)
        lam = inner_lambda(train[ok].reset_index(drop=True), X_train[ok], y_train[ok], rng)
        mu, sd = X_train[ok].mean(0), X_train[ok].std(0) + 1e-12
        beta = ridge_fit((X_train[ok] - mu) / sd, y_train[ok], lam)
        pred = np.exp(ridge_predict(beta, (X_test - mu) / sd))
        pred[~np.isfinite(X_test).all(1)] = np.nan
        preds[name] = pred
        diagnostics[f"{name}_lambda"] = lam
        diagnostics[f"{name}_train_rows"] = int(ok.sum())

    train_points = points[points["patient_id"].isin(train_patients)].reset_index(drop=True)
    fit = fit_lmm(train_points)
    if fit is None:
        preds["lmm_random_slope"] = np.full(len(test), np.nan)
        diagnostics["lmm"] = "fit_failed"
    else:
        preds["lmm_random_slope"] = lmm_predict(fit, test)
        diagnostics["lmm"] = {
            "converged": fit["converged"],
            "fixed_intercept": float(fit["beta"][0]),
            "fixed_slope_per_year": float(fit["beta"][1]),
            "random_sd_intercept": float(np.sqrt(max(fit["G"][0, 0], 0.0))),
            "random_sd_slope": float(np.sqrt(max(fit["G"][1, 1], 0.0))),
            "residual_sd": float(np.sqrt(fit["sigma2"])),
            "train_points": int(len(train_points)),
        }

    rows = []
    for method, pred in preds.items():
        for i, row in enumerate(test.itertuples(index=False)):
            p, t = float(pred[i]), float(row.v3)
            finite = np.isfinite(p)
            rows.append(
                {
                    "patient_id": row.patient_id,
                    "triplet_id": row.triplet_id,
                    "method": method,
                    "predicted_mm3": p,
                    "target_mm3": t,
                    "previous_mm3": float(row.v2),
                    "target_relative_change": (t - row.v2) / row.v2,
                    "predicted_relative_change": (p - row.v2) / row.v2 if finite else np.nan,
                    "abs_error_mm3": abs(p - t) if finite else np.nan,
                    "signed_rvd": (p - t) / t if finite else np.nan,
                    "abs_rvd": abs(p - t) / t if finite else np.nan,
                    "abs_log_ratio": abs(np.log(max(p, 1.0)) - np.log(t)) if finite else np.nan,
                    "horizon_days": float(row.dt2_days),
                }
            )
    return pd.DataFrame(rows), diagnostics


def summarize(group: pd.DataFrame, rng: np.random.Generator) -> dict:
    out = {"cases": int(len(group)), "patients": int(group["patient_id"].nunique())}
    if group.empty:
        return out
    for metric in METRICS:
        values = pd.to_numeric(group[metric], errors="coerce")
        finite = values[np.isfinite(values)]
        patient_means = group.assign(_v=values).groupby("patient_id")["_v"].mean()
        out[metric] = {
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "median": float(finite.median()) if len(finite) else None,
            "patient_level_mean": float(patient_means.mean()) if len(patient_means) else None,
            "finite_cases": int(len(finite)),
            "nonfinite_cases": int(len(values) - len(finite)),
        }
    patients = group["patient_id"].unique()
    by_patient = {p: g for p, g in group.groupby("patient_id")}
    for metric in ("abs_rvd", "abs_error_mm3"):
        stats = []
        for _ in range(BOOTSTRAP):
            sample = rng.choice(patients, size=len(patients), replace=True)
            values = pd.concat([by_patient[p][metric] for p in sample]).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                stats.append(values.mean())
        if stats:
            out[metric]["bootstrap_ci95"] = [
                float(np.percentile(stats, 2.5)),
                float(np.percentile(stats, 97.5)),
            ]
    if group["method"].iloc[0] != "persistence":
        pair = (
            group[["predicted_relative_change", "target_relative_change"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        out["spearman_predicted_vs_actual_change"] = (
            float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) > 2 else None
        )
        sign = np.sign(pair["predicted_relative_change"]) == np.sign(pair["target_relative_change"])
        out["direction_agreement"] = float(sign.mean()) if len(pair) else None
    return out


# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument(
        "--qc-metadata",
        type=Path,
        help="prepared metadata.csv used to tag the registration-QC subset",
    )
    parser.add_argument(
        "--private-dir", required=True, type=Path, help="ignored directory for patient-level outputs"
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path, help="tracked directory for aggregate outputs"
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str, "timepoint_id": str})
    volumes = load_volumes(manifest, args.data_root, args.private_dir / "timepoint_volumes.csv")
    clinical = load_clinical(args.clinical)
    frame = manifest.merge(volumes, on=["patient_id", "timepoint_id"], validate="one_to_one")
    frame = frame.merge(clinical, on="timepoint_id", how="left", validate="one_to_one")

    triplets = build_triplets(frame)
    eligible = triplets[triplets["exclusion_reasons"].eq("")].reset_index(drop=True)
    splits = pd.read_csv(args.splits, dtype={"patient_id": str})
    if splits["patient_id"].duplicated().any():
        raise SystemExit("duplicate patients in split file")
    eligible = eligible.merge(splits, on="patient_id", how="left", validate="many_to_one")
    if eligible["fold"].isna().any():
        missing = sorted(eligible.loc[eligible["fold"].isna(), "patient_id"].unique())
        raise SystemExit(f"eligible patients missing from frozen split: {missing}")
    eligible["fold"] = eligible["fold"].astype(int)
    if args.qc_metadata:
        qc = pd.read_csv(args.qc_metadata, dtype={"triplet_id": str})
        qc_pass = set(
            qc.loc[qc["preprocessing_qc_pass"].astype(str).str.lower().eq("true"), "triplet_id"]
        )
        eligible["registration_qc_pass"] = eligible["triplet_id"].isin(qc_pass)
    else:
        eligible["registration_qc_pass"] = False

    pairs = eligible_pairs(frame)
    points = eligible_timepoints(frame)
    rng = np.random.default_rng(SEED)
    per_fold, diagnostics = [], {}
    for fold in sorted(eligible["fold"].unique()):
        test = eligible[eligible["fold"] == fold].reset_index(drop=True)
        train = eligible[eligible["fold"] != fold].reset_index(drop=True)
        assert not set(train["patient_id"]) & set(test["patient_id"])
        result, diag = run_fold(train, test, pairs, points, rng)
        result["fold"] = int(fold)
        per_fold.append(result)
        diagnostics[f"fold_{fold}"] = {
            "train_cases": int(len(train)),
            "test_cases": int(len(test)),
            **diag,
        }
    per_case = pd.concat(per_fold, ignore_index=True)
    per_case = per_case.merge(
        eligible[["triplet_id", "registration_qc_pass", "treatment_status"]],
        on="triplet_id",
        how="left",
    )

    args.private_dir.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(args.private_dir / "per_case.csv", index=False)
    triplets.to_csv(args.private_dir / "all_triplets.csv", index=False)

    change = (
        per_case.drop_duplicates("triplet_id")
        .set_index("triplet_id")["target_relative_change"]
        .abs()
    )
    cutoff = float(change.quantile(0.8))
    top_ids = set(change[change >= cutoff].index)
    subsets = {
        "all": lambda g: g,
        "top20_abs_change": lambda g: g[g["triplet_id"].isin(top_ids)],
        "horizon_le_365d": lambda g: g[g["horizon_days"] <= 365],
        "horizon_gt_365d": lambda g: g[g["horizon_days"] > 365],
        "all_pre_treatment": lambda g: g[g["treatment_status"] == "all_pre"],
        "all_post_treatment": lambda g: g[g["treatment_status"] == "all_post"],
    }
    if args.qc_metadata:
        subsets["registration_qc_pass_subset"] = lambda g: g[g["registration_qc_pass"]]

    summary_rng = np.random.default_rng(SEED)
    methods = {}
    for method, group in per_case.groupby("method", sort=False):
        methods[method] = {name: summarize(select(group), summary_rng) for name, select in subsets.items()}

    reasons = triplets["exclusion_reasons"].str.split(";").explode()
    # Radiology report class at t3 versus mask-volume change t2 -> t3 (±20 %),
    # recorded as context for measurement noise; not used by any model.
    change_t3 = (eligible["v3"] - eligible["v2"]) / eligible["v2"]
    volume_class = pd.Series(
        np.where(change_t3 > 0.2, "Increased", np.where(change_t3 < -0.2, "Decreased", "Unchanged")),
        index=eligible.index,
    )
    radiology = eligible["radiology_growth_class_t3"].fillna("").replace("", "Missing")
    crosstab = pd.crosstab(radiology, volume_class)
    radiology_vs_volume = {
        str(r): {str(c): int(crosstab.loc[r, c]) for c in crosstab.columns} for r in crosstab.index
    }
    summary = {
        "experiment": "p1_volume_baselines",
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "seed": SEED,
        "cohort": {
            "timepoints": int(len(frame)),
            "patients": int(frame["patient_id"].nunique()),
            "candidate_triplets": int(len(triplets)),
            "candidate_patients": int(triplets["patient_id"].nunique()),
            "eligible_triplets": int(len(eligible)),
            "eligible_patients": int(eligible["patient_id"].nunique()),
            "exclusion_reason_counts": {
                k: int(v) for k, v in reasons[reasons != ""].value_counts().items()
            },
            "intervention_unknown_eligible": int(eligible["intervention_unknown"].sum()),
            "registration_qc_pass_subset": int(eligible["registration_qc_pass"].sum()),
            "treatment_status_counts": {
                str(k): int(v) for k, v in eligible["treatment_status"].value_counts().items()
            },
            "fold_test_cases": {
                int(f): int(n) for f, n in eligible["fold"].value_counts().sort_index().items()
            },
            "fold_test_patients": {
                int(f): int(n) for f, n in eligible.groupby("fold")["patient_id"].nunique().items()
            },
            "growth_pairs_for_pooled_rate": int(len(pairs)),
            "lmm_points": int(len(points)),
            "lmm_patients": int(points["patient_id"].nunique()),
            "top20_abs_change_cutoff": cutoff,
            "radiology_class_t3_vs_volume_change_20pct": radiology_vs_volume,
            "targets_below_50_mm3": int((eligible["v3"] < 50).sum()),
            "horizon_days": {
                "median": float(eligible["dt2_days"].median()),
                "min": float(eligible["dt2_days"].min()),
                "max": float(eligible["dt2_days"].max()),
            },
            "target_volume_mm3": {
                "median": float(eligible["v3"].median()),
                "min": float(eligible["v3"].min()),
                "max": float(eligible["v3"].max()),
            },
        },
        "methods": methods,
        "fold_diagnostics": diagnostics,
        "provenance": {
            "git": git_state(args.repo_root),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": __import__("statsmodels").__version__,
            "manifest_sha256": sha256(args.manifest),
            "splits_sha256": sha256(args.splits),
            "clinical_sha256": sha256(args.clinical),
            "command": " ".join(sys.argv),
        },
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    table = []
    for method, blocks in methods.items():
        for name in ("all", "top20_abs_change"):
            b = blocks[name]
            table.append(
                {
                    "method": method,
                    "subset": name,
                    "n": b["cases"],
                    "abs_rvd_mean": round(b["abs_rvd"]["mean"], 4),
                    "abs_rvd_median": round(b["abs_rvd"]["median"], 4),
                    "mae_mm3": round(b["abs_error_mm3"]["mean"], 1),
                    "abs_log_ratio_mean": round(b["abs_log_ratio"]["mean"], 4),
                    "signed_rvd_mean": round(b["signed_rvd"]["mean"], 4),
                }
            )
    print(json.dumps(summary["cohort"], indent=2))
    print(pd.DataFrame(table).to_string(index=False))
    print(f"aggregate summary -> {args.results_dir / 'summary.json'}")
    print(f"patient-level outputs -> {args.private_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
