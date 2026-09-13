#!/usr/bin/env python3
"""P1 regression: predict the log volume change to the next follow-up.

Unit: every interval (t_k -> t_k+1) whose target scan is not the first scan
after a treatment, and whose two masks are non-empty. History features use
every scan since the last treatment crossing up to t_k (variable length), so
the cohort is larger than the fixed two-point triplet cohort. Only information
available at t_k is used: prior volumes and their timing, the requested
horizon, age/sex, treatment status at t_k, and the acquisition parameters of
the t_k scan. Target-scan acquisition is never used.

Target y = log(V_k+1 / V_k). Persistence predicts y = 0. Models: shrunk
trend (one coefficient), ridge, and histogram gradient boosting, each with
inner patient-grouped CV on the training fold, across nested feature sets so
information and model capacity are compared separately.

Patient-level folds extend the frozen ucsd_splits.csv: patients already in it
keep their fold; new patients are balanced across folds with seed 100.
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
BOOTSTRAP = 1000
LAMBDA_GRID = [1e-2, 1e-1, 1.0, 10.0, 100.0]
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
GBM_GRID = [
    {"max_depth": 2, "learning_rate": 0.03, "max_iter": 150},
    {"max_depth": 2, "learning_rate": 0.1, "max_iter": 100},
    {"max_depth": 3, "learning_rate": 0.03, "max_iter": 150},
    {"max_depth": 3, "learning_rate": 0.1, "max_iter": 100},
]

FEATURE_SETS = {
    "history": ["log_v", "n_prior", "has_prev", "last_rate_yr", "run_rate_yr", "t_run_yr", "horizon_yr", "last_rate_x_horizon", "log_horizon"],
    "history_clinical": ["age", "sex_male"],
    "history_treatment": ["post", "tx_surgery", "tx_radiation"],
    "history_acquisition": ["slice_mm", "inplane_mm", "seq_iac", "tesla", "log_voxel"],
}
FEATURE_SETS["all"] = []
ORDER = ["history", "history_clinical", "history_treatment", "history_acquisition", "all"]


def features_for(name: str) -> list[str]:
    if name == "history":
        return FEATURE_SETS["history"]
    if name == "all":
        return sum((FEATURE_SETS[k] for k in ORDER[:-1]), [])
    return FEATURE_SETS["history"] + FEATURE_SETS[name]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opt_bool(v) -> bool | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return {"true": True, "false": False}.get(str(v).strip().lower())


def git_state(root: Path) -> dict:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception as error:  # pragma: no cover
        return {"commit": None, "dirty": None, "error": str(error)}


# ----------------------------------------------------------------------------
def build_instances(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("patient_id", sort=True):
        recs = g.sort_values("study_days").to_dict("records")
        run: list[dict] = []
        for i, r in enumerate(recs):
            crossed = i > 0 and opt_bool(r["intervened_since_previous"]) is True
            usable = (not r["no_residual"]) and r["volume_mm3"] > 0
            if crossed or not usable:
                run = [r] if usable else []
                continue
            if run:
                k = run[-1]
                hist = run
                logs = np.log([h["volume_mm3"] for h in hist])
                ts = np.array([h["study_days"] for h in hist]) / 365.25
                horizon = (r["study_days"] - k["study_days"]) / 365.25
                if len(hist) >= 2:
                    last_rate = (logs[-1] - logs[-2]) / (ts[-1] - ts[-2])
                    run_rate = float(np.polyfit(ts - ts[0], logs, 1)[0])
                else:
                    last_rate, run_rate = 0.0, 0.0
                rows.append(
                    {
                        "patient_id": pid,
                        "instance_id": f"{pid}__{k['timepoint_id']}__{r['timepoint_id']}",
                        "v_k": k["volume_mm3"],
                        "v_target": r["volume_mm3"],
                        "y": float(np.log(r["volume_mm3"] / k["volume_mm3"])),
                        "log_v": float(logs[-1]),
                        "n_prior": len(hist),
                        "has_prev": float(len(hist) >= 2),
                        "last_rate_yr": float(last_rate),
                        "run_rate_yr": run_rate,
                        "t_run_yr": float(ts[-1] - ts[0]),
                        "horizon_yr": float(horizon),
                        "last_rate_x_horizon": float(last_rate * horizon),
                        "log_horizon": float(np.log(horizon)),
                        "age": k["age"],
                        "sex_male": k["sex_male"],
                        "post": float(k["post"]),
                        "tx_surgery": float(k["post"] and str(k["tx1"]).strip().lower() == "surgery"),
                        "tx_radiation": float(k["post"] and str(k["tx1"]).strip().lower() == "radiation"),
                        "slice_mm": k["slice_mm"],
                        "inplane_mm": k["inplane_mm"],
                        "seq_iac": float(str(k["reference_sequence"]).lower() == "t1postiac"),
                        "tesla": k["tesla"],
                        "log_voxel": float(np.log(k["voxel_mm3"])),
                        "horizon_days": r["study_days"] - k["study_days"],
                        "small_tumor": k["volume_mm3"] < 100,
                    }
                )
            run.append(r)
    return pd.DataFrame(rows)


def extend_folds(patients: list[str], frozen: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    assigned = frozen.set_index("patient_id")["fold"].astype(int).to_dict()
    n_folds = max(assigned.values()) + 1
    extra = sorted(p for p in patients if p not in assigned)
    rng.shuffle(extra)
    counts = {f: sum(1 for p in patients if assigned.get(p) == f) for f in range(n_folds)}
    for p in extra:
        f = min(counts, key=counts.get)
        assigned[p] = f
        counts[f] += 1
    return pd.DataFrame({"patient_id": patients, "fold": [assigned[p] for p in patients]})


# ----------------------------------------------------------------------------
def inner_folds(train: pd.DataFrame, rng: np.random.Generator, k: int = 5) -> np.ndarray:
    pats = train["patient_id"].unique().copy()
    rng.shuffle(pats)
    m = {p: i % k for i, p in enumerate(pats)}
    return train["patient_id"].map(m).to_numpy()


def ridge_fit(X, y, lam):
    Xc = np.column_stack([np.ones(len(X)), X])
    P = np.eye(Xc.shape[1]) * lam
    P[0, 0] = 0
    return np.linalg.solve(Xc.T @ Xc + P, Xc.T @ y)


def ridge_pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def fit_ridge(train, Xtr, ytr, Xte, rng):
    inner = inner_folds(train, rng)
    best, best_err = None, np.inf
    for lam in LAMBDA_GRID:
        errs = []
        for f in range(5):
            a, b = inner != f, inner == f
            mu, sd = Xtr[a].mean(0), Xtr[a].std(0) + 1e-9
            beta = ridge_fit((Xtr[a] - mu) / sd, ytr[a], lam)
            errs.append(np.abs(ridge_pred(beta, (Xtr[b] - mu) / sd) - ytr[b]).mean())
        if np.mean(errs) < best_err:
            best, best_err = lam, np.mean(errs)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    beta = ridge_fit((Xtr - mu) / sd, ytr, best)
    return ridge_pred(beta, (Xte - mu) / sd), {"lambda": best}


def fit_gbm(train, Xtr, ytr, Xte, rng):
    from sklearn.ensemble import HistGradientBoostingRegressor

    inner = inner_folds(train, rng)
    best, best_err = None, np.inf
    for params in GBM_GRID:
        errs = []
        for f in range(5):
            a, b = inner != f, inner == f
            m = HistGradientBoostingRegressor(loss="absolute_error", min_samples_leaf=10, random_state=SEED, **params)
            m.fit(Xtr[a], ytr[a])
            errs.append(np.abs(m.predict(Xtr[b]) - ytr[b]).mean())
        if np.mean(errs) < best_err:
            best, best_err = params, np.mean(errs)
    m = HistGradientBoostingRegressor(loss="absolute_error", min_samples_leaf=10, random_state=SEED, **best)
    m.fit(Xtr, ytr)
    return m.predict(Xte), {"params": best}


def fit_shrunk_trend(train, test, rng):
    inner = inner_folds(train, rng)
    tr = train["last_rate_x_horizon"].to_numpy()
    ytr = train["y"].to_numpy()
    best, best_err = 0.0, np.inf
    for alpha in ALPHA_GRID:
        errs = [np.abs(alpha * tr[inner == f] - ytr[inner == f]).mean() for f in range(5)]
        if np.mean(errs) < best_err:
            best, best_err = alpha, np.mean(errs)
    return best * test["last_rate_x_horizon"].to_numpy(), {"alpha": best}


# ----------------------------------------------------------------------------
def case_rows(test: pd.DataFrame, model: str, feature_set: str, yhat: np.ndarray) -> list[dict]:
    out = []
    for i, r in enumerate(test.itertuples(index=False)):
        pred_v = r.v_k * np.exp(yhat[i])
        out.append(
            {
                "patient_id": r.patient_id,
                "instance_id": r.instance_id,
                "model": model,
                "feature_set": feature_set,
                "y": r.y,
                "yhat": float(yhat[i]),
                "abs_log_ratio": abs(float(yhat[i]) - r.y),
                "abs_rvd": abs(pred_v - r.v_target) / r.v_target,
                "signed_rvd": (pred_v - r.v_target) / r.v_target,
                "abs_error_mm3": abs(pred_v - r.v_target),
                "post": bool(r.post),
                "n_prior": r.n_prior,
                "horizon_days": r.horizon_days,
                "small_tumor": bool(r.small_tumor),
            }
        )
    return out


def summarize(g: pd.DataFrame, base: pd.DataFrame, rng: np.random.Generator) -> dict:
    out = {"cases": int(len(g)), "patients": int(g["patient_id"].nunique())}
    if g.empty:
        return out
    for m in ("abs_log_ratio", "abs_rvd", "abs_error_mm3", "signed_rvd"):
        v = g[m].to_numpy(dtype=float)
        out[m] = {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std(ddof=1)) if len(v) > 1 else None}
    y, yhat = g["y"].to_numpy(), g["yhat"].to_numpy()
    ss_res, ss_tot = ((y - yhat) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    out["r2_vs_zero"] = float(1 - ss_res / (y**2).sum()) if (y**2).sum() > 0 else None
    out["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    out["spearman"] = float(pd.Series(y).corr(pd.Series(yhat), method="spearman")) if yhat.std() > 0 else None
    # paired against persistence on the same instances
    b = base.set_index("instance_id").loc[g["instance_id"], "abs_log_ratio"].to_numpy()
    d = g["abs_log_ratio"].to_numpy() - b
    out["paired_vs_persistence"] = {"mean_diff_abs_log_ratio": float(d.mean()), "fraction_better": float((d < 0).mean())}
    pats = g["patient_id"].to_numpy()
    uniq = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in uniq}
    stats = []
    for _ in range(BOOTSTRAP):
        s = rng.choice(uniq, size=len(uniq), replace=True)
        stats.append(d[np.concatenate([idx[p] for p in s])].mean())
    out["paired_vs_persistence"]["ci95"] = [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]
    return out


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--clinical", required=True, type=Path)
    ap.add_argument("--volumes", required=True, type=Path)
    ap.add_argument("--spacing", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--private-dir", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = ap.parse_args()

    man = pd.read_csv(args.manifest, dtype={"patient_id": str, "timepoint_id": str})
    vol = pd.read_csv(args.volumes, dtype={"patient_id": str, "timepoint_id": str})
    sp = pd.read_csv(args.spacing, dtype={"timepoint_id": str})
    sp["inplane_mm"] = sp[["sx", "sy"]].max(axis=1)
    sp["slice_mm"] = sp[["sx", "sy", "sz"]].max(axis=1)
    clin = pd.read_csv(args.clinical, sep="\t", dtype=str)
    clin = pd.DataFrame(
        {
            "timepoint_id": clin["ID"].str.strip(),
            "age": pd.to_numeric(clin["Age at scan"], errors="coerce"),
            "sex_male": clin["Sex at birth"].str.strip().str.upper().eq("MALE").astype(float),
            "tx1": clin["Treatment #1"].fillna("No"),
            "tesla": pd.to_numeric(clin["Tesla"], errors="coerce"),
        }
    )
    df = man.merge(vol, on=["patient_id", "timepoint_id"]).merge(sp, on="timepoint_id").merge(clin, on="timepoint_id", how="left")
    df["post"] = df["pre_post_treatment"].astype(str).str.strip().str.lower().eq("post")
    df["no_residual"] = df["no_residual_vs"].map(opt_bool).fillna(False).astype(bool)

    inst = build_instances(df)
    inst = inst.dropna(subset=features_for("all")).reset_index(drop=True)
    frozen = pd.read_csv(args.splits, dtype={"patient_id": str})
    rng = np.random.default_rng(SEED)
    patients = sorted(inst["patient_id"].unique())
    folds = extend_folds(patients, frozen, rng)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.private_dir / "extended_patient_folds.csv", index=False)
    inst = inst.merge(folds, on="patient_id", validate="many_to_one")
    inst.to_csv(args.private_dir / "instances.csv", index=False)

    rows: list[dict] = []
    diag: dict = {}
    for fold in sorted(inst["fold"].unique()):
        train = inst[inst["fold"] != fold].reset_index(drop=True)
        test = inst[inst["fold"] == fold].reset_index(drop=True)
        assert not set(train["patient_id"]) & set(test["patient_id"])
        d: dict = {"train": int(len(train)), "test": int(len(test))}
        rows += case_rows(test, "persistence", "none", np.zeros(len(test)))
        rows += case_rows(test, "train_median_change", "none", np.full(len(test), float(train["y"].median())))
        yhat, info = fit_shrunk_trend(train, test, rng)
        rows += case_rows(test, "shrunk_trend", "history", yhat)
        d["shrunk_trend"] = info
        for fs in ORDER:
            cols = features_for(fs)
            Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
            ytr = train["y"].to_numpy(float)
            yhat, info = fit_ridge(train, Xtr, ytr, Xte, rng)
            rows += case_rows(test, "ridge", fs, yhat)
            d[f"ridge_{fs}"] = info
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yhat, info = fit_gbm(train, Xtr, ytr, Xte, rng)
            rows += case_rows(test, "gbm", fs, yhat)
            d[f"gbm_{fs}"] = info
        diag[f"fold_{fold}"] = d
    per_case = pd.DataFrame(rows)
    per_case.to_csv(args.private_dir / "per_case.csv", index=False)

    base = per_case[per_case["model"] == "persistence"]
    subsets = {
        "all": lambda g: g,
        "pre_treatment": lambda g: g[~g["post"]],
        "post_treatment": lambda g: g[g["post"]],
        "one_prior_scan": lambda g: g[g["n_prior"] == 1],
        "two_or_more_prior": lambda g: g[g["n_prior"] >= 2],
        "horizon_le_365d": lambda g: g[g["horizon_days"] <= 365],
        "horizon_gt_365d": lambda g: g[g["horizon_days"] > 365],
        "small_tumor_lt_100mm3": lambda g: g[g["small_tumor"]],
        "tumor_ge_100mm3": lambda g: g[~g["small_tumor"]],
    }
    srng = np.random.default_rng(SEED)
    results: dict = {}
    for (model, fs), g in per_case.groupby(["model", "feature_set"], sort=False):
        results[f"{model}|{fs}"] = {name: summarize(sel(g), base, srng) for name, sel in subsets.items()}

    summary = {
        "experiment": "p1_regression_log_change",
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "seed": SEED,
        "cohort": {
            "instances": int(len(inst)),
            "patients": int(inst["patient_id"].nunique()),
            "pre_treatment_instances": int((inst["post"] == 0).sum()),
            "post_treatment_instances": int((inst["post"] == 1).sum()),
            "n_prior_counts": {int(k): int(v) for k, v in inst["n_prior"].value_counts().sort_index().items()},
            "fold_instances": {int(k): int(v) for k, v in inst["fold"].value_counts().sort_index().items()},
            "fold_patients": {int(k): int(v) for k, v in inst.groupby("fold")["patient_id"].nunique().items()},
            "patients_from_frozen_split": int(inst["patient_id"].isin(frozen["patient_id"]).sum() and inst.drop_duplicates("patient_id")["patient_id"].isin(frozen["patient_id"]).sum()),
            "target_y": {"mean": float(inst["y"].mean()), "sd": float(inst["y"].std()), "median": float(inst["y"].median())},
            "horizon_days_median": float(inst["horizon_days"].median()),
        },
        "feature_sets": {k: features_for(k) for k in ORDER},
        "results": results,
        "fold_diagnostics": diag,
        "provenance": {
            "git": git_state(args.repo_root),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "manifest_sha256": sha256(args.manifest),
            "splits_sha256": sha256(args.splits),
            "extended_folds_sha256": sha256(args.private_dir / "extended_patient_folds.csv"),
            "command": " ".join(sys.argv),
        },
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    lines = [f"{'model|features':34s} {'n':>4s} {'|logΔ| mean':>11s} {'median':>7s} {'MAE mm3':>8s} {'R2 vs 0':>8s} {'better%':>8s} {'diff CI95':>20s}"]
    for key, blocks in results.items():
        b = blocks["all"]
        p = b["paired_vs_persistence"]
        lines.append(
            f"{key:34s} {b['cases']:4d} {b['abs_log_ratio']['mean']:11.3f} {b['abs_log_ratio']['median']:7.3f} {b['abs_error_mm3']['mean']:8.1f} {b['r2_vs_zero'] if b['r2_vs_zero'] is None else round(b['r2_vs_zero'],3)!s:>8s} {100*p['fraction_better']:7.0f}% [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
        )
    (args.results_dir / "table.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["cohort"], indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
