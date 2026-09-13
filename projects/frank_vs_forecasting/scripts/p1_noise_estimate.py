#!/usr/bin/env python3
"""Estimate single-measurement volume noise from longitudinal pairs only.

No new segmentation is needed. Two independent estimators of the SD of
y = log(V_b / V_a) between two scans of the same untreated tumor under the
null of no true change:

1. Zero-interval extrapolation. With y = g*dt + e_b - e_a and g ~ (mu, tau^2),
   E[y^2] = 2*sigma_m^2 + (mu^2 + tau^2)*dt^2, so regressing a robust scale^2
   of y on dt^2 within interval bins gives 2*sigma_m^2 at the intercept.
2. Shrinkage side. Untreated vestibular schwannomas rarely shrink, so the
   negative side of y is dominated by noise; a half-normal fit and a
   nonparametric mirrored quantile both give the noise scale, with the
   caveat that true growth thins the negative side (a mild underestimate).

Everything is stratified by acquisition covariates and tumor size, with
patient-cluster bootstrap CIs, and converted to growth thresholds that a
noise-only pair exceeds with 5 % (one-sided) probability.

A cheap extra component: native-space volume versus the 0.58 mm resampled
crop volume of the same scan (from the DeepGrowth preprocessing) isolates the
interpolation / partial-volume contribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 100
BOOTSTRAP = 1000
ANOMALY_LOG = np.log(8.0)  # |y| beyond an 8-fold change is a data anomaly, not noise
Z95_ONE_SIDED = 1.6449
Z975 = 1.96


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


def q(series, digits=3):
    s = pd.Series(series).dropna()
    if s.empty:
        return "n/a"
    p = s.quantile([0.25, 0.5, 0.75])
    return f"{p[0.5]:.{digits}f} [{p[0.25]:.{digits}f}, {p[0.75]:.{digits}f}]"


# ----------------------------------------------------------------------------
def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("patient_id", sort=True):
        recs = g.sort_values("study_days").to_dict("records")
        for a, b in zip(recs[:-1], recs[1:]):
            if opt_bool(b["intervened_since_previous"]) is True:
                continue
            if a["no_residual"] or b["no_residual"] or a["volume_mm3"] <= 0 or b["volume_mm3"] <= 0:
                continue
            if a["post"] != b["post"]:
                continue
            rows.append(
                {
                    "patient_id": pid,
                    "pair_id": f"{pid}__{a['timepoint_id']}__{b['timepoint_id']}",
                    "post": bool(a["post"]),
                    "dt_yr": (b["study_days"] - a["study_days"]) / 365.25,
                    "v_a": a["volume_mm3"],
                    "v_b": b["volume_mm3"],
                    "y": float(np.log(b["volume_mm3"] / a["volume_mm3"])),
                    "gm_volume": float(np.sqrt(a["volume_mm3"] * b["volume_mm3"])),
                    "thick_a": a["slice_mm"],
                    "thick_b": b["slice_mm"],
                    "seq_switch": a["reference_sequence"] != b["reference_sequence"],
                    "scanner_switch": (a["Manufacturer"], a["Model"]) != (b["Manufacturer"], b["Model"]),
                    "tesla_switch": a["tesla"] != b["tesla"],
                }
            )
    P = pd.DataFrame(rows)
    P["thick_class"] = np.select(
        [(P["thick_a"] <= 1.5) & (P["thick_b"] <= 1.5), (P["thick_a"] > 2.5) | (P["thick_b"] > 2.5)],
        ["both_le_1.5mm", "any_gt_2.5mm"],
        default="mixed_1.5_2.5mm",
    )
    P["size_class"] = pd.cut(P["gm_volume"], [0, 100, 1000, np.inf], labels=["lt_100mm3", "100_1000mm3", "ge_1000mm3"]).astype(str)
    P["anomaly"] = P["y"].abs() > ANOMALY_LOG
    return P


# ----------------------------------------------------------------------------
def robust_sd(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def zero_interval_extrapolation(P: pd.DataFrame, n_bins: int = 5) -> dict:
    """Bin by dt, robust scale^2 per bin, weighted LS on mean dt^2 -> intercept."""
    P = P[~P["anomaly"]]
    if len(P) < 3 * n_bins:
        n_bins = max(2, len(P) // 15)
    edges = np.quantile(P["dt_yr"], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(P["dt_yr"], edges) - 1, 0, n_bins - 1)
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        g = P[idx == b]
        if len(g) < 5:
            continue
        xs.append(float((g["dt_yr"] ** 2).mean()))
        ys.append(robust_sd(g["y"].to_numpy()) ** 2)
        ws.append(len(g))
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws, float)
    X = np.column_stack([np.ones(len(xs)), xs])
    W = np.diag(ws)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ ys)
    intercept = max(beta[0], 0.0)
    return {
        "bins": [{"mean_dt2": float(x), "robust_var": float(v), "n": int(w)} for x, v, w in zip(xs, ys, ws)],
        "intercept_2sigma_m2": float(intercept),
        "slope_growth_var_per_yr2": float(beta[1]),
        "sigma_pair": float(np.sqrt(intercept)),
        "sigma_single": float(np.sqrt(intercept / 2)),
    }


def shrinkage_side(P: pd.DataFrame, max_dt: float | None = None) -> dict:
    P = P[~P["anomaly"]]
    if max_dt is not None:
        P = P[P["dt_yr"] <= max_dt]
    neg = -P.loc[P["y"] < 0, "y"].to_numpy()
    n_neg = len(neg)
    out = {"pairs": int(len(P)), "negative_pairs": n_neg, "negative_fraction": float(n_neg / len(P)) if len(P) else None}
    if n_neg < 5:
        return out
    hn = float(np.sqrt(np.mean(neg**2)))  # half-normal MLE of sigma
    qn = float(np.median(neg) / 0.6745)  # quantile-based normal scale
    out.update(
        {
            "sigma_pair_halfnormal": hn,
            "sigma_pair_quantile": qn,
            "sigma_single_halfnormal": hn / np.sqrt(2),
            "empirical_neg_quantiles": {"p50": float(np.quantile(neg, 0.5)), "p90": float(np.quantile(neg, 0.9)), "p95": float(np.quantile(neg, 0.95))},
        }
    )
    return out


def thresholds_from_sigma(sigma_pair: float) -> dict:
    return {
        "one_sided_95_log": float(Z95_ONE_SIDED * sigma_pair),
        "one_sided_95_relative": float(np.exp(Z95_ONE_SIDED * sigma_pair) - 1),
        "two_sided_95_relative": float(np.exp(Z975 * sigma_pair) - 1),
        "false_growth_rate_at_20pct": float(1 - _ncdf(np.log(1.2) / sigma_pair)),
        "false_growth_rate_at_50pct": float(1 - _ncdf(np.log(1.5) / sigma_pair)),
    }


def _ncdf(z: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1 + erf(z / sqrt(2)))


def bootstrap(P: pd.DataFrame, fn, rng: np.random.Generator, key: str) -> list[float] | None:
    pats = P["patient_id"].unique()
    groups = {p: g for p, g in P.groupby("patient_id")}
    vals = []
    for _ in range(BOOTSTRAP):
        s = rng.choice(pats, size=len(pats), replace=True)
        try:
            v = fn(pd.concat([groups[p] for p in s], ignore_index=True)).get(key)
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 50 else None


# ----------------------------------------------------------------------------
def resampling_component(prepared_metadata: Path, volumes: pd.DataFrame) -> dict | None:
    if not prepared_metadata or not prepared_metadata.exists():
        return None
    meta = pd.read_csv(prepared_metadata)
    ok = meta["preprocessing_qc_pass"].astype(str).str.lower().eq("true") & ~meta["crop_touches_border_by_timepoint"].astype(str).str.contains("true", case=False)
    meta = meta[ok]
    base = prepared_metadata.parent
    native = volumes.set_index("timepoint_id")["volume_mm3"]
    ys = []
    for row in meta.itertuples(index=False):
        tps = json.loads(row.timepoint_ids)
        path = Path(row.prepared_path)
        path = path if path.is_absolute() else base / path
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as a:
            masks, spacing = a["masks"].astype(bool), a["spacing_mm"].astype(float)
        v_res = masks[1].sum() * float(np.prod(spacing))  # timepoint 2: resampled only, not registered
        v_nat = float(native.get(tps[1], np.nan))
        if v_res > 0 and np.isfinite(v_nat) and v_nat > 0:
            ys.append(np.log(v_res / v_nat))
    ys = np.array(ys)
    if len(ys) < 10:
        return {"cases": int(len(ys))}
    return {
        "cases": int(len(ys)),
        "log_ratio_resampled_over_native": {"mean": float(ys.mean()), "sd": float(ys.std(ddof=1)), "robust_sd": robust_sd(ys), "p05": float(np.quantile(ys, 0.05)), "p95": float(np.quantile(ys, 0.95))},
        "note": "0.58 mm isotropic linear resampling of the timepoint-2 mask versus native voxel counting; interpolation/partial-volume component only",
    }


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--clinical", required=True, type=Path)
    ap.add_argument("--volumes", required=True, type=Path)
    ap.add_argument("--spacing", required=True, type=Path)
    ap.add_argument("--prepared-metadata", type=Path)
    ap.add_argument("--private-dir", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = ap.parse_args()

    man = pd.read_csv(args.manifest, dtype={"patient_id": str, "timepoint_id": str})
    vol = pd.read_csv(args.volumes, dtype={"patient_id": str, "timepoint_id": str})
    sp = pd.read_csv(args.spacing, dtype={"timepoint_id": str})
    sp["slice_mm"] = sp[["sx", "sy", "sz"]].max(axis=1)
    clin = pd.read_csv(args.clinical, sep="\t", dtype=str)
    clin = pd.DataFrame(
        {
            "timepoint_id": clin["ID"].str.strip(),
            "tesla": pd.to_numeric(clin["Tesla"], errors="coerce"),
            "Manufacturer": clin["Manufacturer"],
            "Model": clin["Model"],
            "has_t1post": clin["Has T1post"].astype(str).str.lower().eq("true"),
            "has_t1postiac": clin["Has T1postIAC"].astype(str).str.lower().eq("true"),
        }
    )
    df = man.merge(vol, on=["patient_id", "timepoint_id"]).merge(sp, on="timepoint_id").merge(clin, on="timepoint_id", how="left")
    df["post"] = df["pre_post_treatment"].astype(str).str.strip().str.lower().eq("post")
    df["no_residual"] = df["no_residual_vs"].map(opt_bool).fillna(False).astype(bool)

    P = build_pairs(df)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    P.to_csv(args.private_dir / "pairs.csv", index=False)
    pre, post = P[~P["post"]].reset_index(drop=True), P[P["post"]].reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    R: dict = {
        "experiment": "p1_noise_estimate",
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "seed": SEED,
        "anomaly_rule": "|log ratio| > log(8) excluded from scale estimates",
        "cohort": {
            "pre_treatment_pairs": int(len(pre)),
            "pre_treatment_patients": int(pre["patient_id"].nunique()),
            "pre_anomalies_excluded": int(pre["anomaly"].sum()),
            "post_treatment_pairs": int(len(post)),
            "post_treatment_patients": int(post["patient_id"].nunique()),
            "post_anomalies_excluded": int(post["anomaly"].sum()),
            "interval_years_pre": q(pre["dt_yr"], 2),
            "exams_with_both_t1post_and_t1postiac": int((df["has_t1post"] & df["has_t1postiac"]).sum()),
            "patients_with_ge2_dual_sequence_exams": int(df[df["has_t1post"] & df["has_t1postiac"]].groupby("patient_id").size().ge(2).sum()),
        },
    }

    # --- estimator 1: zero-interval extrapolation (pre-treatment)
    e1 = zero_interval_extrapolation(pre)
    e1["sigma_pair_ci95"] = bootstrap(pre, zero_interval_extrapolation, rng, "sigma_pair")
    R["zero_interval_extrapolation_pre"] = e1
    R["zero_interval_extrapolation_post"] = zero_interval_extrapolation(post) if len(post) >= 30 else None

    # --- estimator 2: shrinkage side (pre-treatment), all intervals and <= 1 y
    e2 = shrinkage_side(pre)
    e2["sigma_pair_halfnormal_ci95"] = bootstrap(pre, shrinkage_side, rng, "sigma_pair_halfnormal")
    e2["sigma_pair_quantile_ci95"] = bootstrap(pre, shrinkage_side, rng, "sigma_pair_quantile")
    R["shrinkage_side_pre_all"] = e2
    R["shrinkage_side_pre_le_1yr"] = shrinkage_side(pre, max_dt=1.0)
    R["shrinkage_side_pre_le_1yr"]["sigma_pair_halfnormal_ci95"] = bootstrap(pre[pre["dt_yr"] <= 1.0], shrinkage_side, rng, "sigma_pair_halfnormal")

    # --- consensus sigma and thresholds
    candidates = [e1["sigma_pair"], e2["sigma_pair_halfnormal"], e2["sigma_pair_quantile"]]
    sigma_pair = float(np.median(candidates))
    R["consensus"] = {
        "sigma_pair_candidates": {"zero_interval": e1["sigma_pair"], "halfnormal": e2["sigma_pair_halfnormal"], "quantile": e2["sigma_pair_quantile"]},
        "sigma_pair": sigma_pair,
        "sigma_single_measurement": sigma_pair / np.sqrt(2),
        "single_measurement_cv_percent": float(100 * np.sqrt(np.exp(sigma_pair**2 / 2) - 1)),
        "thresholds_normal": thresholds_from_sigma(sigma_pair),
        "thresholds_empirical_mirrored": {
            "one_sided_95_log": e2["empirical_neg_quantiles"]["p95"],
            "one_sided_95_relative": float(np.exp(e2["empirical_neg_quantiles"]["p95"]) - 1),
            "one_sided_90_relative": float(np.exp(e2["empirical_neg_quantiles"]["p90"]) - 1),
        },
        "observed_shrink_fraction_ge_20pct": float((pre["y"] <= np.log(0.8)).mean()),
        "observed_shrink_fraction_ge_50pct": float((pre["y"] <= np.log(0.5)).mean()),
    }

    # --- strata (shrinkage-side half-normal, pre-treatment)
    strata = {}
    for col, label in [("thick_class", "slice_thickness"), ("seq_switch", "reference_sequence_switch"), ("scanner_switch", "scanner_model_switch"), ("size_class", "tumor_volume")]:
        strata[label] = {}
        for val, g in pre.groupby(col):
            g = g.reset_index(drop=True)
            s = shrinkage_side(g)
            if "sigma_pair_halfnormal" in s:
                s["ci95"] = bootstrap(g, shrinkage_side, rng, "sigma_pair_halfnormal") if g["patient_id"].nunique() >= 10 else None
                s["thresholds"] = thresholds_from_sigma(s["sigma_pair_halfnormal"])
                s["abs_rel_change_median"] = float(np.median(np.abs(np.exp(g["y"]) - 1)))
            strata[label][str(val)] = s
    R["strata_pre_treatment"] = strata

    # --- resampling component
    R["resampling_component"] = resampling_component(args.prepared_metadata, vol) if args.prepared_metadata else None

    R["provenance"] = {
        "git": git_state(args.repo_root),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "manifest_sha256": sha256(args.manifest),
        "command": " ".join(sys.argv),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(json.dumps(R, indent=2, allow_nan=False, default=lambda o: None), encoding="utf-8")

    # --- markdown report
    c = R["consensus"]
    md = ["# 体积测量噪声估计（UCSD 治疗前区间，聚合）\n"]
    md.append(f"治疗前配对 {len(pre)}（{pre['patient_id'].nunique()} 患者），剔除 |log 比| > log 8 的异常 {int(pre['anomaly'].sum())} 对；治疗后配对 {len(post)} 作对照。\n")
    md.append("## 两种估计量（σ 为两次测量 log 比在无真实变化下的 SD）\n")
    md.append("| 估计量 | σ_pair | 95% CI | 单次测量 σ |\n|---|---:|---:|---:|")
    md.append(f"| 零间隔外推（截距） | {e1['sigma_pair']:.3f} | {e1['sigma_pair_ci95']} | {e1['sigma_single']:.3f} |")
    md.append(f"| 缩小侧半正态 | {e2['sigma_pair_halfnormal']:.3f} | {e2['sigma_pair_halfnormal_ci95']} | {e2['sigma_single_halfnormal']:.3f} |")
    md.append(f"| 缩小侧分位数 | {e2['sigma_pair_quantile']:.3f} | {e2['sigma_pair_quantile_ci95']} | {e2['sigma_pair_quantile']/np.sqrt(2):.3f} |")
    md.append(f"| 缩小侧半正态，仅 ≤1 年 | {R['shrinkage_side_pre_le_1yr'].get('sigma_pair_halfnormal', float('nan')):.3f} | {R['shrinkage_side_pre_le_1yr'].get('sigma_pair_halfnormal_ci95')} | |")
    if R["zero_interval_extrapolation_post"]:
        md.append(f"| 零间隔外推，治疗后对照 | {R['zero_interval_extrapolation_post']['sigma_pair']:.3f} | | |")
    md.append(f"\n共识 σ_pair = **{c['sigma_pair']:.3f}**（单次测量 σ ≈ {c['sigma_single_measurement']:.3f}，约 {c['single_measurement_cv_percent']:.0f}% 变异系数）。零间隔外推的斜率 {e1['slope_growth_var_per_yr2']:.3f}/年² 对应真实增长率的离散度。\n")
    md.append("## 由噪声推出的生长阈值\n")
    t = c["thresholds_normal"]
    md.append("| 规则 | 相对体积阈值 | 说明 |\n|---|---:|---|")
    md.append(f"| 正态假设，单侧 95% | {t['one_sided_95_relative']:.0%} | 噪声对儿被误判为增长的概率 5% |")
    md.append(f"| 正态假设，双侧 95% | {t['two_sided_95_relative']:.0%} | |")
    md.append(f"| 经验镜像分位数，单侧 95% | {c['thresholds_empirical_mirrored']['one_sided_95_relative']:.0%} | 不假设正态，抗重尾 |")
    md.append(f"| 经验镜像分位数，单侧 90% | {c['thresholds_empirical_mirrored']['one_sided_90_relative']:.0%} | |")
    md.append(f"\n正态假设下 20% 阈值的误判率 {t['false_growth_rate_at_20pct']:.0%}，50% 阈值 {t['false_growth_rate_at_50pct']:.1%}；实际观察到的缩小 ≥20% 比例 {c['observed_shrink_fraction_ge_20pct']:.0%}、≥50% 比例 {c['observed_shrink_fraction_ge_50pct']:.1%}。两者若不一致说明噪声重尾或异方差，应采用分层阈值。\n")
    md.append("## 分层（缩小侧半正态，治疗前）\n")
    md.append("| 分层 | 取值 | 配对 | 负侧数 | σ_pair | 95% CI | 单侧 95% 阈值 | \\|相对变化\\| 中位数 |\n|---|---|---:|---:|---:|---|---:|---:|")
    for label, d in strata.items():
        for val, s in d.items():
            if "sigma_pair_halfnormal" in s:
                md.append(f"| {label} | {val} | {s['pairs']} | {s['negative_pairs']} | {s['sigma_pair_halfnormal']:.3f} | {s.get('ci95')} | {s['thresholds']['one_sided_95_relative']:.0%} | {s['abs_rel_change_median']:.3f} |")
    rc = R["resampling_component"]
    if rc and rc.get("cases", 0) >= 10:
        md.append(f"\n## 重采样分量\n\n同一扫描 0.58 mm 各向同性重采样后的 mask 体积与原生体素计数之比：n={rc['cases']}，log 比均值 {rc['log_ratio_resampled_over_native']['mean']:+.3f}，SD {rc['log_ratio_resampled_over_native']['sd']:.3f}。这是插值/部分容积单独贡献的量级，远小于总噪声即说明噪声主要来自分割本身和采集差异。\n")
    md.append("## 跨序列 test-retest 的可行性\n")
    md.append(f"临床表中同时含 t1post 与 t1postIAC 的检查 {R['cohort']['exams_with_both_t1post_and_t1postiac']} 次；本地只有参考序列的 mask。做真正的 test-retest 需要：(1) 用 Aspera 口令流程补下另一序列图像（需人工授权）；(2) 训练或获取 VS 自动分割模型并在两序列上分别分割；(3) 用同一分割器的跨序列体积差估 σ。当前估计不依赖这些步骤。\n")
    (args.results_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
