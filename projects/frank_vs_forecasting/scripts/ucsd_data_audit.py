#!/usr/bin/env python3
"""Descriptive audit of UCSD-VS-Longitudinal for the Frank project.

Reads the paired manifest, the official clinical TSV, and the cached native
mask volumes (from p1_volume_baselines.py), plus NIfTI headers for voxel
spacing. Writes an aggregate-only Markdown report and JSON; no patient
identifiers are emitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def q(series: pd.Series, digits: int = 1) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    p = s.quantile([0.25, 0.5, 0.75])
    return f"{p[0.5]:.{digits}f} [{p[0.25]:.{digits}f}, {p[0.75]:.{digits}f}] (range {s.min():.{digits}f}–{s.max():.{digits}f})"


def counts(series: pd.Series, top: int | None = None) -> str:
    vc = series.fillna("Missing").replace("", "Missing").value_counts()
    if top:
        vc = vc.head(top)
    return "; ".join(f"{k}: {v}" for k, v in vc.items())


def opt_bool(v) -> bool | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    t = str(v).strip().lower()
    return {"true": True, "false": False}.get(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--clinical", required=True, type=Path)
    ap.add_argument("--volumes", required=True, type=Path, help="timepoint_volumes.csv cache")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--spacing-cache", required=True, type=Path)
    ap.add_argument("--output-md", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest, dtype={"patient_id": str, "timepoint_id": str})
    clin = pd.read_csv(args.clinical, sep="\t", dtype=str)
    clin["timepoint_id"] = clin["ID"].str.strip()
    vol = pd.read_csv(args.volumes, dtype={"patient_id": str, "timepoint_id": str})

    if args.spacing_cache.exists():
        sp = pd.read_csv(args.spacing_cache, dtype={"timepoint_id": str})
    else:
        import nibabel as nib

        rows = []
        for r in man.itertuples(index=False):
            z = nib.load(str(args.data_root / r.mask_path)).header.get_zooms()[:3]
            rows.append({"timepoint_id": r.timepoint_id, "sx": z[0], "sy": z[1], "sz": z[2]})
        sp = pd.DataFrame(rows)
        sp.to_csv(args.spacing_cache, index=False)
    sp["inplane_mm"] = sp[["sx", "sy"]].max(axis=1)
    sp["slice_mm"] = sp[["sx", "sy", "sz"]].max(axis=1)

    df = man.merge(vol, on=["patient_id", "timepoint_id"]).merge(sp, on="timepoint_id").merge(
        clin, on="timepoint_id", how="left"
    )
    df["age"] = pd.to_numeric(df["Age at scan"], errors="coerce")
    df["tesla"] = pd.to_numeric(df["Tesla"], errors="coerce")
    df["post"] = df["pre_post_treatment"].astype(str).str.strip().str.lower().eq("post")
    df["no_residual"] = df["no_residual_vs"].map(opt_bool).fillna(False).astype(bool)
    df["intervened"] = df["intervened_since_previous"].map(opt_bool)
    df["empty_mask"] = df["volume_mm3"] <= 0
    df = df.sort_values(["patient_id", "study_days"]).reset_index(drop=True)
    df["order"] = df.groupby("patient_id").cumcount()

    R: dict = {}
    md: list[str] = []
    md.append("# UCSD-VS-Longitudinal 数据审计（聚合，2026-09-11）\n")
    md.append("来源：官方 v1 临床 TSV、配对 manifest（570 时点）、原生空间 mask 体积、NIfTI 头文件。只报告聚合统计，不含患者 ID。\n")

    # ---------------- patients ----------------
    pat = df.groupby("patient_id").agg(
        n_tp=("timepoint_id", "size"),
        span_days=("study_days", "max"),
        sex=("Sex at birth", "first"),
        race=("Race", "first"),
        eth=("Ethnicity", "first"),
        age0=("age", "first"),
        tx1=("Treatment #1", "first"),
        tx2=("Treatment #2", "first"),
        any_post=("post", "any"),
        any_intervened=("intervened", lambda s: bool((s == True).any())),
        any_no_residual=("no_residual", "any"),
        any_empty=("empty_mask", "any"),
    )
    R["patients"] = {
        "n": int(len(pat)),
        "timepoints_per_patient": {int(k): int(v) for k, v in pat["n_tp"].value_counts().sort_index().items()},
        "sex": pat["sex"].fillna("Missing").value_counts().to_dict(),
        "race": pat["race"].fillna("Missing").value_counts().to_dict(),
        "ethnicity": pat["eth"].fillna("Missing").value_counts().to_dict(),
        "age_first_scan": q(pat["age0"]),
        "followup_span_days_patients_with_ge2": q(pat.loc[pat["n_tp"] >= 2, "span_days"], 0),
        "treatment1": pat["tx1"].fillna("Missing").value_counts().to_dict(),
        "treatment2": pat["tx2"].fillna("Missing").value_counts().to_dict(),
        "patients_with_any_post_treatment_scan": int(pat["any_post"].sum()),
        "patients_with_intervention_between_scans": int(pat["any_intervened"].sum()),
        "patients_with_no_residual_timepoint": int(pat["any_no_residual"].sum()),
    }
    md.append("## 1. 患者层\n")
    md.append(f"- 患者数 {len(pat)}；每人时点数 " + counts(pat["n_tp"].astype(str)) + "（时点总数 570）")
    md.append(f"- 性别：{counts(pat['sex'])}")
    md.append(f"- 种族：{counts(pat['race'])}；民族：{counts(pat['eth'])}")
    md.append(f"- 首次扫描年龄（岁）中位数 [IQR]：{q(pat['age0'])}")
    md.append(f"- 随访跨度（天，≥2 时点者）：{q(pat.loc[pat['n_tp'] >= 2, 'span_days'], 0)}")
    md.append(f"- 治疗 #1：{counts(pat['tx1'])}")
    md.append(f"- 治疗 #2：{counts(pat['tx2'])}")
    md.append(
        f"- 有治疗后扫描的患者 {int(pat['any_post'].sum())}；扫描间隔跨治疗的患者 {int(pat['any_intervened'].sum())}；有无残留时点的患者 {int(pat['any_no_residual'].sum())}\n"
    )

    # ---------------- timepoints ----------------
    md.append("## 2. 时点层\n")
    md.append(f"- Pre/Post 治疗：{counts(df['pre_post_treatment'])}")
    md.append(f"- 无残留（临床表）：{int(df['no_residual'].sum())}；空 mask：{int(df['empty_mask'].sum())}（其中未申报无残留 {int((df['empty_mask'] & ~df['no_residual']).sum())}）")
    md.append(f"- 放射科纵向分类（相对上一次）：{counts(df['growth_class'])}")
    md.append(f"- 分割参考序列：{counts(df['reference_sequence'])}")
    md.append(f"- 厂商：{counts(df['Manufacturer'])}")
    md.append(f"- 型号：{counts(df['Model'], top=8)}")
    md.append(f"- 场强：{counts(df['tesla'].astype(str))}")
    R["timepoints"] = {
        "pre_post": df["pre_post_treatment"].fillna("Missing").value_counts().to_dict(),
        "no_residual": int(df["no_residual"].sum()),
        "empty_mask": int(df["empty_mask"].sum()),
        "growth_class": df["growth_class"].fillna("Missing").value_counts().to_dict(),
        "reference_sequence": df["reference_sequence"].value_counts().to_dict(),
        "manufacturer": df["Manufacturer"].fillna("Missing").value_counts().to_dict(),
        "tesla": df["tesla"].fillna(-1).value_counts().to_dict(),
    }

    # ---------------- resolution ----------------
    md.append("\n## 3. 分辨率（分割参考图像）\n")
    md.append(f"- 面内像素（mm）：{q(sp['inplane_mm'], 3)}")
    md.append(f"- 层厚/最大轴向（mm）：{q(sp['slice_mm'], 2)}")
    thick_bins = pd.cut(sp["slice_mm"], [0, 1.0, 1.5, 2.5, 3.5, 10], labels=["≤1.0", "1.0–1.5", "1.5–2.5", "2.5–3.5", ">3.5"])
    md.append(f"- 层厚分布：{counts(thick_bins.astype(str))}")
    md.append(f"- 体素体积（mm³）：{q(vol['voxel_mm3'], 3)}")
    by_seq = df.groupby("reference_sequence")["slice_mm"].median()
    md.append("- 按参考序列的层厚中位数：" + "; ".join(f"{k}: {v:.2f} mm" for k, v in by_seq.items()))
    R["resolution"] = {
        "inplane_mm": q(sp["inplane_mm"], 3),
        "slice_mm": q(sp["slice_mm"], 2),
        "slice_bins": thick_bins.astype(str).value_counts().to_dict(),
        "slice_by_sequence": {k: float(v) for k, v in by_seq.items()},
    }

    # ---------------- volumes ----------------
    md.append("\n## 4. 肿瘤体积（原生空间，mm³）\n")
    nonempty = df[~df["empty_mask"]]
    pre = nonempty[~nonempty["post"]]
    first_pre = pre.groupby("patient_id").head(1)
    md.append(f"- 非空 mask 时点 {len(nonempty)}；其中治疗前 {len(pre)}，治疗后 {len(nonempty) - len(pre)}")
    md.append(f"- 首次治疗前扫描体积：{q(first_pre['volume_mm3'])}（n={len(first_pre)}）")
    size_bins = pd.cut(first_pre["volume_mm3"], [0, 100, 500, 1000, 4000, 1e9], labels=["<100", "100–500", "500–1000", "1000–4000", ">4000"])
    md.append(f"- 首次体积分级（mm³）：{counts(size_bins.astype(str))}")
    md.append(f"- 所有治疗前时点体积：{q(pre['volume_mm3'])}")
    md.append(f"- 治疗后非空 mask 体积：{q(nonempty.loc[nonempty['post'], 'volume_mm3'])}")
    small = int((pre["volume_mm3"] < 50).sum())
    md.append(f"- 治疗前时点体积 <50 mm³ 的数量 {small}（约 {small / len(pre):.0%}）。在 3 mm 层厚下 50 mm³ 只有约 1–2 层，相对误差极大")
    R["volumes"] = {
        "first_pre_treatment": q(first_pre["volume_mm3"]),
        "first_size_bins": size_bins.astype(str).value_counts().to_dict(),
        "pre_all": q(pre["volume_mm3"]),
        "pre_below_50mm3": small,
    }

    # ---------------- intervals / growth ----------------
    md.append("\n## 5. 随访间隔与治疗前生长\n")
    pairs = []
    for pid, g in df.groupby("patient_id"):
        recs = g.to_dict("records")
        for a, b in zip(recs[:-1], recs[1:]):
            pairs.append(
                {
                    "patient_id": pid,
                    "dt": b["study_days"] - a["study_days"],
                    "crosses_tx": b["intervened"] is True,
                    "post_pair": a["post"] or b["post"],
                    "no_res": a["no_residual"] or b["no_residual"],
                    "v_a": a["volume_mm3"],
                    "v_b": b["volume_mm3"],
                    "seq_switch": a["reference_sequence"] != b["reference_sequence"],
                    "thick_a": a["slice_mm"],
                    "thick_b": b["slice_mm"],
                    "scanner_switch": (a["Manufacturer"], a["Model"]) != (b["Manufacturer"], b["Model"]),
                    "growth_class_b": b["growth_class"] if isinstance(b["growth_class"], str) else "Missing",
                }
            )
    P = pd.DataFrame(pairs)
    md.append(f"- 相邻扫描间隔（天，全部 {len(P)} 对）：{q(P['dt'], 0)}")
    nat = P[~P["crosses_tx"] & ~P["post_pair"] & ~P["no_res"] & (P["v_a"] > 0) & (P["v_b"] > 0)].copy()
    nat["log_ratio"] = np.log(nat["v_b"] / nat["v_a"])
    nat["rel"] = nat["v_b"] / nat["v_a"] - 1
    nat["rate_yr"] = nat["log_ratio"] / (nat["dt"] / 365.25)
    nat["abs_rel"] = nat["rel"].abs()
    md.append(f"- 治疗前自然史区间：{len(nat)} 对 / {nat['patient_id'].nunique()} 患者；间隔 {q(nat['dt'], 0)} 天")
    md.append(f"- 相对体积变化：{q(nat['rel'], 3)}；年化对数增长率：{q(nat['rate_yr'], 3)}")
    md.append(
        f"- |变化| >20% 的区间 {int((nat['abs_rel'] > 0.2).sum())}（{(nat['abs_rel'] > 0.2).mean():.0%}）；增 >20% {int((nat['rel'] > 0.2).sum())}，减 >20% {int((nat['rel'] < -0.2).sum())}；>50% 增 {int((nat['rel'] > 0.5).sum())}，>50% 减 {int((nat['rel'] < -0.5).sum())}"
    )
    ct = pd.crosstab(nat["growth_class_b"], pd.cut(nat["rel"], [-2, -0.2, 0.2, 100], labels=["减>20%", "±20%内", "增>20%"]))
    md.append("- 放射科分类 × mask 体积变化（治疗前区间）：\n")
    md.append("| 报告分类 | 减>20% | ±20%内 | 增>20% |\n|---|---:|---:|---:|")
    for r in ct.index:
        md.append(f"| {r} | " + " | ".join(str(int(ct.loc[r, c])) if c in ct.columns else "0" for c in ["减>20%", "±20%内", "增>20%"]) + " |")
    R["natural_history_pairs"] = {
        "n_pairs": int(len(nat)),
        "n_patients": int(nat["patient_id"].nunique()),
        "interval_days": q(nat["dt"], 0),
        "relative_change": q(nat["rel"], 3),
        "annual_log_rate": q(nat["rate_yr"], 3),
        "abs_change_gt_20pct": int((nat["abs_rel"] > 0.2).sum()),
        "radiology_vs_volume": {str(r): {str(c): int(ct.loc[r, c]) for c in ct.columns} for r in ct.index},
    }

    # ---------------- noise proxies ----------------
    md.append("\n## 6. 测量噪声代理（治疗前区间的 |log 体积比|）\n")
    md.append("同一患者相邻两次扫描，按采集条件是否改变分层。若噪声主导，条件改变的区间波动应更大。\n")
    md.append("| 分层 | n | \\|log 比\\| 中位数 | \\|相对变化\\| 中位数 | \\|变化\\|>20% 比例 |\n|---|---:|---:|---:|---:|")
    strata = {
        "参考序列相同": nat[~nat["seq_switch"]],
        "参考序列切换 (t1post↔IAC)": nat[nat["seq_switch"]],
        "扫描仪型号相同": nat[~nat["scanner_switch"]],
        "扫描仪型号切换": nat[nat["scanner_switch"]],
        "两次层厚均 ≤1.5 mm": nat[(nat["thick_a"] <= 1.5) & (nat["thick_b"] <= 1.5)],
        "至少一次层厚 >2.5 mm": nat[(nat["thick_a"] > 2.5) | (nat["thick_b"] > 2.5)],
        "起始体积 <100 mm³": nat[nat["v_a"] < 100],
        "起始体积 100–1000 mm³": nat[(nat["v_a"] >= 100) & (nat["v_a"] < 1000)],
        "起始体积 ≥1000 mm³": nat[nat["v_a"] >= 1000],
        "间隔 ≤200 天": nat[nat["dt"] <= 200],
        "间隔 200–500 天": nat[(nat["dt"] > 200) & (nat["dt"] <= 500)],
        "间隔 >500 天": nat[nat["dt"] > 500],
    }
    R["noise_proxies"] = {}
    for name, g in strata.items():
        if len(g) == 0:
            continue
        a = g["log_ratio"].abs()
        md.append(f"| {name} | {len(g)} | {a.median():.3f} | {g['abs_rel'].median():.3f} | {(g['abs_rel'] > 0.2).mean():.0%} |")
        R["noise_proxies"][name] = {"n": int(len(g)), "abs_log_ratio_median": float(a.median()), "abs_rel_median": float(g["abs_rel"].median()), "gt20pct": float((g["abs_rel"] > 0.2).mean())}
    short = nat[nat["dt"] <= 200]
    md.append(
        f"\n短间隔（≤200 天）区间的体积波动可作为噪声下界：|相对变化| 中位数 {short['abs_rel'].median():.3f}，第 90 百分位 {short['abs_rel'].quantile(0.9):.3f}（n={len(short)}）。"
    )

    # ---------------- usable cohorts ----------------
    md.append("\n## 7. 不同任务定义下的可用样本\n")
    runs = []
    for pid, g in df.groupby("patient_id"):
        recs = g.to_dict("records")
        kept = 0
        for i, r in enumerate(recs):
            if i > 0 and r["intervened"] is True:
                break
            if r["no_residual"] or r["empty_mask"] or r["post"]:
                break
            kept += 1
        runs.append({"patient_id": pid, "pre_run": kept, "n_tp": len(recs)})
    RU = pd.DataFrame(runs)
    md.append("| 任务单位 | 患者 | 样本 |\n|---|---:|---:|")
    for k, label in [(2, "≥2 次治疗前扫描（persistence / 单区间）"), (3, "≥3 次（两点历史 → 第三次，DeepGrowth 三元组）"), (4, "≥4 次（三点历史）")]:
        sub = RU[RU["pre_run"] >= k]
        samples = int((sub["pre_run"] - (k - 1)).sum())
        md.append(f"| {label} | {len(sub)} | {samples} |")
        R.setdefault("usable", {})[f"pre_run_ge_{k}"] = {"patients": int(len(sub)), "samples": samples}
    md.append(f"| 治疗后随访（有治疗后非空 mask 的患者） | {int(df[df['post'] & ~df['empty_mask']]['patient_id'].nunique())} | {int((df['post'] & ~df['empty_mask']).sum())} 时点 |")
    md.append("\n注：治疗前连续段从首次扫描起算，遇到跨治疗、治疗后或空 mask 即截断。")

    # Composition of the DeepGrowth-style rolling-triplet cohort (151/98):
    # the rule only forbids an interval crossing treatment, so all-post
    # residual-tumor follow-up is retained. Quantify it explicitly.
    trip = []
    for pid, g in df.groupby("patient_id"):
        recs = g.to_dict("records")
        for s in range(max(0, len(recs) - 2)):
            r1, r2, r3 = recs[s : s + 3]
            if r2["intervened"] is True or r3["intervened"] is True:
                continue
            if any(r["no_residual"] or r["empty_mask"] for r in (r1, r2, r3)):
                continue
            posts = sum(r["post"] for r in (r1, r2, r3))
            trip.append({"patient_id": pid, "posts": posts, "tx1": r1["Treatment #1"]})
    T = pd.DataFrame(trip)
    comp = T["posts"].map({0: "全部治疗前", 3: "全部治疗后（残留肿瘤随访）"}).fillna("混合")
    md.append(f"\nDeepGrowth 滚动三元组规则（只要求 t2、t3 区间不跨治疗）得到 {len(T)} 三元组 / {T['patient_id'].nunique()} 患者，其构成：")
    md.append("\n| 构成 | 三元组 | 患者 |\n|---|---:|---:|")
    for k in ["全部治疗前", "全部治疗后（残留肿瘤随访）", "混合"]:
        sub = T[comp == k]
        if len(sub):
            md.append(f"| {k} | {len(sub)} | {sub['patient_id'].nunique()} |")
    post_t = T[comp == "全部治疗后（残留肿瘤随访）"]
    if len(post_t):
        md.append(f"\n治疗后三元组按首次治疗类型：{counts(post_t['tx1'])}。这部分是术后/放疗后残留肿瘤的动态，与未治疗自然史不同，P1 基线记录的 151 例队列包含了它们，后续分析应分层或剔除。")
    R["rolling_triplet_composition"] = {k: {"triplets": int((comp == k).sum()), "patients": int(T.loc[comp == k, "patient_id"].nunique())} for k in comp.unique()}

    # ---------------- anomalies ----------------
    md.append("\n## 7b. 数据异常清单（聚合）\n")
    ext = nat[(nat["rel"] > 5) | (nat["rel"] < -0.9)]
    md.append(f"- 治疗前区间中体积变化 >5 倍或减少 >90% 的极端区间 {len(ext)} 个（{ext['patient_id'].nunique()} 患者），疑似未记录治疗、分割失败或极小肿瘤；其中起始体积 <20 mm³ 的 {int((ext['v_a'] < 20).sum())} 个")
    md.append("- 临床表 `Treatment #1` 有大小写不一致的取值（No/no），`Treatment #2` 有尾随空格；解析时需归一化")
    md.append("- 未申报无残留但 mask 为空的时点 2 个（均属于只有 2 个时点的患者）")
    md.append(f"- 首次扫描即为治疗后的患者 {int(len(pat) - len(first_pre))} 人，没有任何治疗前体积")
    md.append(f"- 场强 0.35 T 的时点 {int((df['tesla'] == 0.35).sum())} 个（Time Medical PICA），分辨率与其余不可比")

    # ---------------- risks ----------------
    md.append("\n## 8. 对建模的直接含义\n")
    md.append("- 队列以观察随访为主，治疗前平均增长接近零，persistence 是强基线。")
    md.append("- 单次分割体积噪声与年增长同量级，两点斜率不可靠；任何声称超过 persistence 的模型都必须在噪声下界之上证明。")
    md.append("- 分割参考图像以层厚约 3 mm 的 IAC 专用序列为主（85%），少数为约 1 mm 各向同性的 t1post；序列在随访中切换、扫描仪型号切换、层厚 >2.5 mm 都对应明显更大的体积波动，是可识别的噪声来源，建模时应把参考序列、层厚、扫描仪作为协变量或分层因素。")
    md.append("- 滚动三元组队列混入了治疗后残留肿瘤的随访，其生长动态与自然史不同；Frank 项目的任务定义需要明确是否纳入。")
    md.append("- 小肿瘤（<100 mm³）占比高，相对误差指标在这些病例失效，评估要同时报告绝对误差和 log 误差。")
    md.append("- 数据集不含 NF2 标签，也无双侧肿瘤标注；作为 Frank 项目的散发性预训练/对照源可用，但不能替代 NF2 队列。")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    args.output_json.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.output_md} and {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
