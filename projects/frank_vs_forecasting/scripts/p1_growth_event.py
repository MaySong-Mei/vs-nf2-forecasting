#!/usr/bin/env python3
"""P1 growth-event baselines on pre-treatment UCSD follow-up.

Task A (next-visit classification): for every pre-treatment interval
(t_k -> t_k+1), predict whether the next scan shows growth >= threshold
relative to t_k, from information available at t_k. Models: prevalence,
logistic on log volume only, logistic on history + clinical. Metrics: AUC,
Brier, log-loss, calibration slope/intercept, ECE. Patient-level five-fold
evaluation with inner patient-grouped tuning of the L2 strength.

Task B (time to growth): per patient, from the first scan, time until the
first pre-treatment scan whose volume exceeds baseline by >= threshold;
otherwise censored at the last pre-treatment scan. Kaplan-Meier growth-free
fractions and a Cox model on baseline covariates with cross-validated C-index.

Both tasks are repeated at a 20 % (primary) and 50 % (noise-robust) threshold.
Aggregate outputs only; per-instance tables stay in the ignored data dir.
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
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
THRESHOLDS = {"growth_20pct": 0.20, "growth_50pct": 0.50}

FEATURES = {
    "volume_only": ["log_v"],
    "volume_horizon": ["log_v", "log_horizon"],
    "history": ["log_v", "log_horizon", "has_prev", "last_rate_yr", "prior_growth", "t_run_yr"],
    "history_clinical": ["log_v", "log_horizon", "has_prev", "last_rate_yr", "prior_growth", "t_run_yr", "age", "sex_male"],
    "history_clinical_acq": ["log_v", "log_horizon", "has_prev", "last_rate_yr", "prior_growth", "t_run_yr", "age", "sex_male", "slice_mm", "seq_iac"],
}
COX_FEATURES = {
    "volume_only": ["log_v0"],
    "volume_clinical": ["log_v0", "age0", "sex_male"],
    "volume_clinical_acq": ["log_v0", "age0", "sex_male", "slice_mm0"],
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def opt_bool(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return {"true": True, "false": False}.get(str(v).strip().lower())


def git_state(root: Path) -> dict:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception as e:  # pragma: no cover
        return {"commit": None, "dirty": None, "error": str(e)}


# ----------------------------------------------------------------------------
def pre_treatment_runs(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Initial pre-treatment run of each patient (first scan until treatment / empty mask)."""
    runs = {}
    for pid, g in df.groupby("patient_id", sort=True):
        recs = g.sort_values("study_days").to_dict("records")
        kept = []
        for i, r in enumerate(recs):
            if i > 0 and opt_bool(r["intervened_since_previous"]) is True:
                break
            if r["post"] or r["no_residual"] or r["volume_mm3"] <= 0:
                break
            kept.append(r)
        if len(kept) >= 2:
            runs[pid] = kept
    return runs


def is_event(v_from: float, v_to: float, thr, abs_floor: float) -> bool:
    """Growth event: relative increase >= thr AND absolute increase >= abs_floor mm3.

    `thr` may be a float or a dict {"size_cut_mm3", "below", "at_or_above"} giving a
    size-stratified relative threshold (noise-calibrated per stratum).
    """
    if isinstance(thr, dict):
        thr = thr["below"] if v_from < thr["size_cut_mm3"] else thr["at_or_above"]
    return (v_to / v_from - 1 >= thr) and (v_to - v_from >= abs_floor)


def build_intervals(runs: dict[str, list[dict]], thr: float, abs_floor: float = 0.0) -> pd.DataFrame:
    rows = []
    for pid, recs in runs.items():
        for k in range(1, len(recs)):
            cur, nxt = recs[k - 1], recs[k]
            hist = recs[:k]
            logs = np.log([h["volume_mm3"] for h in hist])
            ts = np.array([h["study_days"] for h in hist]) / 365.25
            horizon = (nxt["study_days"] - cur["study_days"]) / 365.25
            if len(hist) >= 2:
                last_rate = (logs[-1] - logs[-2]) / (ts[-1] - ts[-2])
                prior_growth = float(is_event(hist[-2]["volume_mm3"], hist[-1]["volume_mm3"], thr, abs_floor))
            else:
                last_rate, prior_growth = 0.0, 0.0
            rel = nxt["volume_mm3"] / cur["volume_mm3"] - 1
            rows.append(
                {
                    "patient_id": pid,
                    "instance_id": f"{pid}__{cur['timepoint_id']}__{nxt['timepoint_id']}",
                    "event": int(is_event(cur["volume_mm3"], nxt["volume_mm3"], thr, abs_floor)),
                    "mirrored_shrink": int(is_event(nxt["volume_mm3"], cur["volume_mm3"], thr, abs_floor)),
                    "rel_change": rel,
                    "radiology_increased": str(nxt.get("growth_class", "")).strip().lower() == "increased",
                    "log_v": float(logs[-1]),
                    "log_horizon": float(np.log(horizon)),
                    "horizon_days": nxt["study_days"] - cur["study_days"],
                    "has_prev": float(len(hist) >= 2),
                    "last_rate_yr": float(last_rate),
                    "prior_growth": prior_growth,
                    "t_run_yr": float(ts[-1] - ts[0]),
                    "age": cur["age"],
                    "sex_male": cur["sex_male"],
                    "slice_mm": cur["slice_mm"],
                    "seq_iac": float(str(cur["reference_sequence"]).lower() == "t1postiac"),
                    "small_tumor": cur["volume_mm3"] < 100,
                }
            )
    return pd.DataFrame(rows)


def build_survival(runs: dict[str, list[dict]], thr: float, abs_floor: float = 0.0) -> pd.DataFrame:
    rows = []
    for pid, recs in runs.items():
        v0, t0 = recs[0]["volume_mm3"], recs[0]["study_days"]
        event, time = 0, (recs[-1]["study_days"] - t0) / 365.25
        for r in recs[1:]:
            if is_event(v0, r["volume_mm3"], thr, abs_floor):
                event, time = 1, (r["study_days"] - t0) / 365.25
                break
        rows.append(
            {
                "patient_id": pid,
                "time_yr": max(time, 1e-3),
                "event": event,
                "log_v0": float(np.log(v0)),
                "age0": recs[0]["age"],
                "sex_male": recs[0]["sex_male"],
                "slice_mm0": recs[0]["slice_mm"],
                "n_scans": len(recs),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def inner_folds(train: pd.DataFrame, rng: np.random.Generator, k: int = 5) -> np.ndarray:
    pats = train["patient_id"].unique().copy()
    rng.shuffle(pats)
    m = {p: i % k for i, p in enumerate(pats)}
    return train["patient_id"].map(m).to_numpy()


def fit_logistic(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], rng: np.random.Generator):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    Xtr, ytr, Xte = train[cols].to_numpy(float), train["event"].to_numpy(int), test[cols].to_numpy(float)
    inner = inner_folds(train, rng)
    best, best_ll = None, np.inf
    for C in C_GRID:
        lls = []
        for f in range(5):
            a, b = inner != f, inner == f
            if ytr[a].min() == ytr[a].max() or b.sum() == 0:
                continue
            mu, sd = Xtr[a].mean(0), Xtr[a].std(0) + 1e-9
            m = LogisticRegression(C=C, max_iter=2000).fit((Xtr[a] - mu) / sd, ytr[a])
            lls.append(log_loss(ytr[b], m.predict_proba((Xtr[b] - mu) / sd)[:, 1], labels=[0, 1]))
        if lls and np.mean(lls) < best_ll:
            best, best_ll = C, np.mean(lls)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = LogisticRegression(C=best, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    coef = dict(zip(cols, (m.coef_[0]).round(4).tolist()))
    return m.predict_proba((Xte - mu) / sd)[:, 1], {"C": best, "std_coef": coef}


def classification_metrics(g: pd.DataFrame, rng: np.random.Generator) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y, p = g["event"].to_numpy(int), np.clip(g["p"].to_numpy(float), 1e-6, 1 - 1e-6)
    out = {"cases": int(len(g)), "patients": int(g["patient_id"].nunique()), "events": int(y.sum()), "event_rate": float(y.mean())}
    if y.min() == y.max():
        return out
    out["auc"] = float(roc_auc_score(y, p)) if p.std() > 0 else 0.5
    out["brier"] = float(brier_score_loss(y, p))
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    # calibration slope/intercept: logit(y) ~ a + b * logit(p)
    if p.std() > 0:
        from sklearn.linear_model import LogisticRegression

        lp = np.log(p / (1 - p)).reshape(-1, 1)
        cal = LogisticRegression(C=1e6, max_iter=2000).fit(lp, y)
        out["calibration_slope"] = float(cal.coef_[0][0])
        out["calibration_intercept"] = float(cal.intercept_[0])
    bins = np.quantile(p, [0.2, 0.4, 0.6, 0.8]) if p.std() > 0 else []
    idx = np.digitize(p, bins) if len(bins) else np.zeros(len(p), int)
    ece = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx))
    out["ece_5bins"] = float(ece)
    # Pooled AUC is biased for fold-wise constant predictors (per-fold
    # prevalence) and by fold-to-fold shifts; report the fold-mean AUC too.
    fold_aucs = []
    for _, gf in g.groupby("fold"):
        yf, pf = gf["event"].to_numpy(int), gf["p"].to_numpy(float)
        if yf.min() != yf.max():
            fold_aucs.append(0.5 if pf.std() == 0 else roc_auc_score(yf, pf))
    out["auc_fold_mean"] = float(np.mean(fold_aucs)) if fold_aucs else None
    out["auc_fold_sd"] = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else None
    pats = g["patient_id"].to_numpy()
    uniq = np.unique(pats)
    where = {u: np.where(pats == u)[0] for u in uniq}
    aucs = []
    for _ in range(BOOTSTRAP):
        s = np.concatenate([where[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        if y[s].min() != y[s].max() and p[s].std() > 0:
            aucs.append(roc_auc_score(y[s], p[s]))
    if aucs:
        out["auc_ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    return out


# ----------------------------------------------------------------------------
def run_classification(inst: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    rows, diag = [], {}
    for fold in sorted(inst["fold"].unique()):
        train, test = inst[inst["fold"] != fold].reset_index(drop=True), inst[inst["fold"] == fold].reset_index(drop=True)
        assert not set(train["patient_id"]) & set(test["patient_id"])
        d = {}
        rows += [dict(patient_id=r.patient_id, instance_id=r.instance_id, model="prevalence", event=r.event, p=float(train["event"].mean()), small_tumor=r.small_tumor, horizon_days=r.horizon_days, has_prev=r.has_prev, fold=int(fold)) for r in test.itertuples(index=False)]
        for name, cols in FEATURES.items():
            p, info = fit_logistic(train, test, cols, rng)
            d[name] = info
            rows += [dict(patient_id=r.patient_id, instance_id=r.instance_id, model=name, event=r.event, p=float(p[i]), small_tumor=r.small_tumor, horizon_days=r.horizon_days, has_prev=r.has_prev, fold=int(fold)) for i, r in enumerate(test.itertuples(index=False))]
        diag[f"fold_{fold}"] = d
    pc = pd.DataFrame(rows)
    srng = np.random.default_rng(SEED)
    subsets = {
        "all": lambda g: g,
        "two_or_more_prior": lambda g: g[g["has_prev"] == 1],
        "tumor_ge_100mm3": lambda g: g[~g["small_tumor"]],
        "small_tumor_lt_100mm3": lambda g: g[g["small_tumor"]],
        "horizon_le_365d": lambda g: g[g["horizon_days"] <= 365],
    }
    res = {m: {s: classification_metrics(sel(g), srng) for s, sel in subsets.items()} for m, g in pc.groupby("model", sort=False)}
    return {"results": res, "fold_diagnostics": diag}, pc


def run_survival(surv: pd.DataFrame, rng: np.random.Generator) -> dict:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index

    km = KaplanMeierFitter().fit(surv["time_yr"], surv["event"])
    out = {
        "patients": int(len(surv)),
        "events": int(surv["event"].sum()),
        "followup_years_median": float(surv["time_yr"].median()),
        "growth_free_fraction": {f"{t}y": float(km.predict(t)) for t in (1, 2, 3, 5)},
        "median_time_to_growth_years": None if not np.isfinite(km.median_survival_time_) else float(km.median_survival_time_),
        "cox": {},
    }
    for name, cols in COX_FEATURES.items():
        cidx, coefs = [], []
        for fold in sorted(surv["fold"].unique()):
            tr, te = surv[surv["fold"] != fold], surv[surv["fold"] == fold]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph = CoxPHFitter(penalizer=0.05).fit(tr[cols + ["time_yr", "event"]], "time_yr", "event")
            risk = cph.predict_partial_hazard(te[cols])
            if te["event"].sum() > 0:
                cidx.append(concordance_index(te["time_yr"], -risk, te["event"]))
            coefs.append(cph.params_.to_dict())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = CoxPHFitter(penalizer=0.05).fit(surv[cols + ["time_yr", "event"]], "time_yr", "event")
        out["cox"][name] = {
            "cv_c_index_mean": float(np.mean(cidx)),
            "cv_c_index_folds": [round(c, 3) for c in cidx],
            "full_fit_hazard_ratios": {k: round(float(np.exp(v)), 3) for k, v in full.params_.items()},
            "full_fit_p_values": {k: round(float(v), 4) for k, v in full.summary["p"].items()},
        }
    out["events_by_baseline_volume_tertile"] = None
    # KM by baseline volume tertile
    tert = pd.qcut(surv["log_v0"], 3, labels=["small", "medium", "large"])
    out["growth_free_2y_by_baseline_volume_tertile"] = {}
    for lab in ["small", "medium", "large"]:
        s = surv[tert == lab]
        k = KaplanMeierFitter().fit(s["time_yr"], s["event"])
        out["growth_free_2y_by_baseline_volume_tertile"][lab] = {"patients": int(len(s)), "events": int(s["event"].sum()), "growth_free_2y": float(k.predict(2))}
    return out


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--clinical", required=True, type=Path)
    ap.add_argument("--volumes", required=True, type=Path)
    ap.add_argument("--spacing", required=True, type=Path)
    ap.add_argument("--folds", required=True, type=Path, help="extended_patient_folds.csv from p1_regression.py")
    ap.add_argument("--private-dir", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument(
        "--thresholds",
        default=",".join(f"{k}={v}" for k, v in THRESHOLDS.items()),
        help=(
            "comma-separated name=relative[:absolute_mm3], e.g. growth_20pct=0.2,rel20_abs100=0.2:100; "
            "or name=stratified/<rel_below>/<rel_at_or_above>@<size_cut_mm3> for a size-stratified relative threshold"
        ),
    )
    args = ap.parse_args()
    thresholds = {}
    for item in args.thresholds.split(","):
        name, value = item.split("=")
        if value.startswith("stratified/"):
            body, _, cut = value[len("stratified/"):].partition("@")
            below, above = body.split("/")
            thresholds[name.strip()] = {
                "relative": {"size_cut_mm3": float(cut or 100), "below": float(below), "at_or_above": float(above)},
                "absolute_mm3": 0.0,
            }
        else:
            rel, _, absolute = value.partition(":")
            thresholds[name.strip()] = {"relative": float(rel), "absolute_mm3": float(absolute) if absolute else 0.0}

    man = pd.read_csv(args.manifest, dtype={"patient_id": str, "timepoint_id": str})
    vol = pd.read_csv(args.volumes, dtype={"patient_id": str, "timepoint_id": str})
    sp = pd.read_csv(args.spacing, dtype={"timepoint_id": str})
    sp["slice_mm"] = sp[["sx", "sy", "sz"]].max(axis=1)
    clin = pd.read_csv(args.clinical, sep="\t", dtype=str)
    clin = pd.DataFrame({"timepoint_id": clin["ID"].str.strip(), "age": pd.to_numeric(clin["Age at scan"], errors="coerce"), "sex_male": clin["Sex at birth"].str.strip().str.upper().eq("MALE").astype(float)})
    df = man.merge(vol, on=["patient_id", "timepoint_id"]).merge(sp, on="timepoint_id").merge(clin, on="timepoint_id", how="left")
    df["post"] = df["pre_post_treatment"].astype(str).str.strip().str.lower().eq("post")
    df["no_residual"] = df["no_residual_vs"].map(opt_bool).fillna(False).astype(bool)
    folds = pd.read_csv(args.folds, dtype={"patient_id": str})

    runs = pre_treatment_runs(df)
    rng = np.random.default_rng(SEED)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "experiment": "p1_growth_event",
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "seed": SEED,
        "cohort": {"patients_with_ge2_pre_treatment_scans": len(runs), "scans_in_runs": int(sum(len(r) for r in runs.values()))},
        "thresholds": thresholds,
        "classification": {},
        "time_to_growth": {},
        "feature_sets": FEATURES,
        "cox_feature_sets": COX_FEATURES,
    }
    for name, spec in thresholds.items():
        thr, abs_floor = spec["relative"], spec["absolute_mm3"]
        inst = build_intervals(runs, thr, abs_floor).dropna(subset=FEATURES["history_clinical_acq"]).merge(folds, on="patient_id", validate="many_to_one")
        inst.to_csv(args.private_dir / f"intervals_{name}.csv", index=False)
        cls, pc = run_classification(inst, rng)
        pc.to_csv(args.private_dir / f"per_case_{name}.csv", index=False)
        agree = pd.crosstab(inst["event"], inst["radiology_increased"])
        cls["cohort"] = {
            "intervals": int(len(inst)),
            "patients": int(inst["patient_id"].nunique()),
            "events": int(inst["event"].sum()),
            "event_rate": float(inst["event"].mean()),
            "shrink_ge_threshold": int((inst["rel_change"] <= -thr).sum()) if not isinstance(thr, dict) else None,
            "mirrored_shrink_events": int(inst["mirrored_shrink"].sum()),
            "mirrored_shrink_to_growth_ratio": float(inst["mirrored_shrink"].sum() / max(inst["event"].sum(), 1)),
            "event_vs_radiology_increased": {f"event={e}": {f"radiology_increased={c}": int(agree.loc[e, c]) for c in agree.columns} for e in agree.index},
        }
        summary["classification"][name] = cls
        surv = build_survival(runs, thr, abs_floor).dropna(subset=["age0"]).merge(folds, on="patient_id", validate="one_to_one")
        surv.to_csv(args.private_dir / f"survival_{name}.csv", index=False)
        summary["time_to_growth"][name] = run_survival(surv, rng)

    summary["provenance"] = {
        "git": git_state(args.repo_root),
        "python": platform.python_version(),
        "sklearn": __import__("sklearn").__version__,
        "lifelines": __import__("lifelines").__version__,
        "manifest_sha256": sha256(args.manifest),
        "folds_sha256": sha256(args.folds),
        "command": " ".join(sys.argv),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    lines = []
    for name in thresholds:
        c = summary["classification"][name]
        lines.append(f"== Task A next-visit {name} {thresholds[name]}: {c['cohort']['intervals']} intervals / {c['cohort']['patients']} patients, events {c['cohort']['events']} ({c['cohort']['event_rate']:.2f}), mirrored shrink {c['cohort']['mirrored_shrink_events']} (ratio {c['cohort']['mirrored_shrink_to_growth_ratio']:.2f}) ==")
        lines.append(f"{'model':22s} {'AUC':>6s} {'AUC CI95':>16s} {'foldAUC':>8s} {'Brier':>6s} {'logloss':>8s} {'cal.slope':>9s} {'ECE':>6s} | {'AUC >=100mm3':>12s} {'AUC <100mm3':>12s}")
        for m, b in c["results"].items():
            a = b["all"]
            ci = a.get("auc_ci95", [np.nan, np.nan])
            big, small = b["tumor_ge_100mm3"], b["small_tumor_lt_100mm3"]
            lines.append(
                f"{m:22s} {a.get('auc', np.nan):6.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {a.get('auc_fold_mean') or np.nan:8.3f} {a.get('brier', np.nan):6.3f} {a.get('log_loss', np.nan):8.3f} {a.get('calibration_slope', np.nan):9.2f} {a.get('ece_5bins', np.nan):6.3f} | {big.get('auc_fold_mean') or np.nan:12.3f} {small.get('auc_fold_mean') or np.nan:12.3f}"
            )
        s = summary["time_to_growth"][name]
        lines.append(f"-- Task B time-to-growth {name}: {s['patients']} patients, {s['events']} events; growth-free 1/2/3/5y = " + "/".join(f"{s['growth_free_fraction'][k]:.2f}" for k in ('1y', '2y', '3y', '5y')))
        for m, b in s["cox"].items():
            lines.append(f"   cox {m:22s} CV C-index {b['cv_c_index_mean']:.3f} folds {b['cv_c_index_folds']} HR {b['full_fit_hazard_ratios']} p {b['full_fit_p_values']}")
        lines.append("")
    (args.results_dir / "table.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
