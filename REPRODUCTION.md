# Reproduction protocol

## Claim boundary

The original DeepGrowth experiments used 131 private patients, each with three
consecutive contrast-enhanced T1 scans. The public code contains hard-coded local
paths and no model weights. Therefore this repository targets:

1. **method reproduction:** implement the same input/output task, SDF shape
   representation, local neural-field decoder, time conditioning, and recurrent
   latent prediction;
2. **external validation:** estimate performance on UCSD-VS-Longitudinal with
   patient-level five-fold cross-validation;
3. **engineering reproduction:** make data preparation, baselines, training, and
   evaluation runnable on commodity 8 GB hardware.

It does not claim to recover Table 1 or Table 2 of the paper.

## Frozen primary experiment

| Item | Decision |
|---|---|
| Cohort | UCSD subjects with at least three annotated T1CE timepoints |
| Unit | Rolling consecutive triple within a subject |
| Input | T1CE image + mask at timepoints 1 and 2; timing to timepoint 3 |
| Target | Mask/SDF at timepoint 3 |
| Interventions | Exclude a triple if intervention occurred between either adjacent pair when this field is known |
| Split | Five-fold, patient-level, seed 100 |
| Registration | Rigid, timepoints 1 and 3 to timepoint 2 |
| Sampling | 0.58 mm isotropic, 64³ crop centered only on timepoint-2 mask |
| Intensity | Nonzero 0.5–99.5 percentile clip, scaled to [-1, 1] |
| SDF sign | Negative inside, matching the released code's `< 0` inference rule |
| Primary metrics | Dice, HD95 (mm), absolute RVD |
| Secondary metrics | Signed RVD, volume MAE, crop-truncation rate |
| Stress subset | Top 20% by absolute target volume change from timepoint 2 to 3 |
| Optimizer | AdamW, learning rate 1e-4 |
| Repeats | One frozen split first; add seeds only after the full pipeline is stable |

### Treatment fields

The manifest field `intervened_since_previous` refers to the interval ending at
that row. For a triple `(t1, t2, t3)`, the default filter excludes it when either
the `t2` or `t3` row is true. Blank values mean unknown and are retained but
reported. This prevents silent mixing of natural history and treatment response.

## Differences from the paper

- The cohort and annotation process differ.
- The crop center is derived from timepoint 2, not the target, to avoid target
  leakage. Target-based crop placement can make a forecasting task artificially
  easy.
- The 8 GB profile uses fewer SDF samples and a smaller latent width than the
  authors' 4.9 M-parameter configuration. The original source is preserved for a
  later paper-profile run on larger hardware.
- The paper's SDF equation and released thresholding code use opposite-looking
  sign descriptions. This repository follows the executable code: tumor is
  `SDF < 0`.

## Required quality-control gates

Do not start the five-fold model run until all gates pass:

- no patient appears in more than one fold;
- every selected triple has strictly increasing `study_days`;
- image and mask geometry agree within each timepoint;
- masks are nonempty before and after registration;
- crop truncation is quantified; visually inspect every flagged target;
- stable-mask and linear-SDF baselines complete for all evaluable cases;
- the synthetic overfit/smoke run produces a checkpoint and predictions.

The preprocessing metadata records registration status, crop contact, source
paths, and time intervals. Keep this file with every result table.

## Reporting template

Use language such as:

> We performed a public-cohort reimplementation and external validation of the
> DeepGrowth task on UCSD-VS-Longitudinal. Exact reproduction of the published
> numerical results was not possible because the original training cohort and
> pretrained weights were unavailable.

Report per-case results and mean ± standard deviation across cases, with a second
patient-level aggregation if subjects contribute multiple rolling triples. Always
report the number of patients, triples, excluded intervention intervals, unknown
intervention intervals, and crop-truncated cases.

## Provenance

- DeepGrowth paper: <https://arxiv.org/abs/2404.02614>
- Official implementation: <https://github.com/cyjdswx/DeepGrowth>
- UCSD-VS-Longitudinal DOI: <https://doi.org/10.7937/bq0z-xa62>
- Dataset description paper: <https://arxiv.org/abs/2511.00472>

