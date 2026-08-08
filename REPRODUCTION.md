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
| No residual tumor | Exclude a triple containing a `No residual VS = Yes` timepoint because the negative-inside SDF is undefined for an empty mask; report every exclusion |
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

For the official v1 TSV, `study_days` is the per-patient cumulative sum of
`Days from prior scan`. An interval is marked as crossing treatment when either
a signed treatment-relative day changes from negative to nonnegative or the
official status changes from `Pre` to `Post`. The union is intentional: the v1
table contains two disagreements/missing signed values that would otherwise
silently miss an intervention.

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
- locally render every automatically flagged prepared case, an
  overlap-stratified deterministic sample of at least 12 passed cases, and all
  recoverable preprocessing failures;
- review registration/source, crop-face contact, mask contour, normalized
  intensity, and negative-inside/positive-outside SDF panels while retaining
  the private case mapping outside version control;
- record human pass/fail/uncertain separately from the frozen automatic gate;
- stable-mask and linear-SDF baselines complete for all evaluable cases;
- the synthetic overfit/smoke run produces a checkpoint and predictions.

The preprocessing metadata records registration status, crop contact, source
paths, and time intervals. Keep this file with every result table.

Before preprocessing or training, save machine-readable runtime provenance:

```bash
python scripts/preflight.py --require-cuda --json \
  --output-json outputs/runtime-provenance/environment.json
```

The strict JSON schema records the exact CUDA device, VRAM, driver, PyTorch/CUDA,
OS/Python, and git state without persisting hostnames, usernames, or paths.
Training embeds the same schema in each fold's `run_metadata.json`.

`make_splits.py` assigns only patients contributing at least one eligible
triplet; this avoids empty/non-evaluable patients distorting fold sizes. The
fold aggregator requires exactly folds 0–4 and reports pooled case-level,
patient-level, per-fold, across-fold, and top-20%-change summaries.

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

## Official UCSD provenance (verified 2026-08-06)

- DeepGrowth paper: <https://arxiv.org/abs/2404.02614>
- Official implementation: <https://github.com/cyjdswx/DeepGrowth>
- UCSD-VS-Longitudinal collection: <https://www.cancerimagingarchive.net/collection/ucsd-vs-longitudinal/>
- UCSD-VS-Longitudinal DOI: <https://doi.org/10.7937/WEFA-CP23>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Format/access: 40 GB NIfTI images and segmentations via IBM Aspera Connect;
  clinical data and data dictionary are separate TSV downloads.

Required dataset citation:

> Nguyen, U., Correia de Verdier, M., Saluja, R., Schulte, J., Brandel, M.,
> Suresh, K., Farid, N., & Rudie, J. (2026). *The University of California San
> Diego Longitudinal Vestibular Schwannoma Dataset (UCSD-VS-Longitudinal)*
> (Version 1) [Data set]. The Cancer Imaging Archive.
> <https://doi.org/10.7937/WEFA-CP23>

Do not confuse these three sources:

1. DeepGrowth used a private in-house cohort of 131 patients; it is the method
   source, not the public validation cohort.
2. `VESTIBULAR-SCHWANNOMA-MC-RC2` is a separate UK multi-center collection of
   190 subjects. Its DOI is `10.7937/BQ0Z-XA62`, and arXiv:2511.00472 describes
   that collection. Neither is a UCSD citation.
3. This protocol uses the distinct 191-subject UCSD cohort with DOI
   `10.7937/WEFA-CP23`.

## Metadata-only audit (official v1 TSV)

These are source-table statistics, not imaging QC or reproduction results:

| Item | Observed |
|---|---:|
| Clinical rows / patients | 570 / 191 |
| Patients with 2 / 3 / 4 timepoints | 74 / 46 / 71 |
| Patients with at least 3 timepoints | 117 |
| Candidate rolling triples before exclusions | 188 |
| Intervals marked as crossing treatment | 45 |
| No-residual timepoints | 44 |
| Projected eligible triples / patients after both rules | 151 / 98 |

At the rolling-triplet level, 33 candidates intersect a treatment interval and
26 contain a no-residual timepoint; reasons overlap, leaving 37 unique excluded
triplets. The projected five-fold distribution is 30 / 28 / 31 / 32 / 30
triplets across 20 / 20 / 20 / 19 / 19 patients, with each patient assigned to
exactly one fold. The downloaded minimum subset confirms the same 151 / 98
eligible triplet/patient counts after pairing all 570 image/mask timepoints.
Preprocessing QC remains a separate gate.

The collection web page currently states 75 / 44 / 72 patients with 2 / 3 / 4
timepoints, while the downloadable v1 TSV yields 74 / 46 / 71. The TSV-derived
counts above are recorded rather than silently forcing agreement. The
discrepancy is retained in the paired manifest rather than forcing agreement
with the web-page summary.

## Local environment and smoke audit (updated 2026-08-08)

- Windows, Python 3.10.15
- NVIDIA driver 596.08; driver-reported CUDA capability 13.2
- PyTorch 2.12.1+cu130; CUDA available
- NVIDIA GeForce RTX 3080 Laptop GPU, compute capability 8.6, 16.0 GiB
- NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, NiBabel 5.4.2,
  SimpleITK 2.5.6, PyYAML 6.0.3
- 30 unit tests passed, including file-level NIfTI/SimpleITK preprocessing,
  selective-download validation, and strict finite-metric reporting tests
- Synthetic GPU smoke: forward/backward, FP16 AMP, checkpoint evaluation and
  resume passed. Loss changed from 0.9353 (two batches) to 0.9014 after resume
  (one batch); epoch wall time was 4.3–4.6 s and peak reserved GPU memory was
  0.094 GiB.

Synthetic metrics and resource figures must not be reported as real UCSD
performance or as a 64³ full-profile memory estimate.

## Imaging access audit (2026-08-06)

- IBM Aspera Connect 4.2.19 was installed from IBM's signed Windows installer.
- The public Faspex package has 570 examination folders, and the observed
  filenames follow the
  `VS_XXXX_YY_{t1post,t1postIAC,seg_t1post,seg_t1postIAC}.nii.gz` convention.
- The clinical sequence flag permits a minimum requested set of 1,140 files
  (one declared T1-post reference image and one mask for each timepoint), rather
  than downloading unrelated modalities.
- All 1,140 selected files downloaded through the official public-link
  authorization and Connect local API. The local start request must include the
  official Faspex `Origin`/`Referer`; omitting them makes Connect wait without
  creating a transfer. The repository downloader applies these headers and
  never logs the short-lived credentials. The selective subset is 6,140,716,589
  bytes (5.719 GiB), with no zero-length or partial files.
- The paired manifest contains 570 timepoints from 191 patients. It yields 188
  rolling candidates and 151 eligible triples from 98 patients after explicit
  treatment/no-residual exclusions, matching the metadata projection.
- Full voxel-level QC read all 570 image/mask pairs. All pairs are 3D, all
  image and mask voxels are finite, and every mask has the same shape and
  affine as its paired image. It found 46 empty masks versus 44 timepoints
  declared as no residual tumor in the clinical table. The two undeclared
  empty masks are retained as explicit dataset-level QC errors; both belong to
  patients with only two timepoints and therefore do not enter any rolling
  triple or alter the 151 / 98 projected training cohort.
- A real-file preprocessing smoke retained the reference single-scale 20% MI
  sampling and 100 iterations. It completed in 26.6 s for one triplet, produced
  finite arrays, did not touch the crop border, and yielded moving-to-middle
  mask-overlap Dice values of 0.726 and 0.776. These are registration QC values,
  not prediction performance.

## Real public-cohort run status (2026-08-06)

The complete reference preprocessing pass took 6,733 s (1 h 52 min) on the
local 16-core Windows system. The auditable cohort flow is:

| Gate | Triples | Patients |
|---|---:|---:|
| Rolling candidates before clinical exclusions | 188 | 117 |
| Eligible after treatment/no-residual rules | 151 | 98 |
| Successfully prepared | 135 | 91 |
| Passed crop and registration QC for modeling | 103 | 73 |

The 16 preparation failures are retained in `failures.csv`: 10 registered-mask
empty, 5 crop/SDF-mask empty, and 1 cropped image with no usable signal. Among
the 135 prepared cases, 32 fail the modeling QC gate: 25 have at least one
moving-to-middle mask overlap below 0.1 and 10 have a mask touching the crop
border (reasons overlap). Full arrays are finite and source image/mask geometry
matches within the frozen `1e-4` header tolerance. Twelve de-identified local
three-plane montages were inspected, covering all six target-crop-border cases
and six lowest-overlap cases. The low-overlap views confirmed unreliable
longitudinal alignment, so these cases are explicitly filtered rather than
silently included. The raw and QC images remain outside version control.

On 2026-08-08, the version-2 systematic renderer expanded this to 60 local
opaque panels: all 32 automatically flagged prepared triples, 12 deterministic
QC-passed controls, and source debug views for all 16 preprocessing failures.
Its identifier-free review index and separate private mapping reconcile 60/60;
all rendered SDFs passed the negative-inside check. Review retained the 12
passed controls and accepted exclusion of the 32 automatically flagged plus 16
failed cases. The 103 / 73 reference cohort is therefore frozen for modeling;
formal model results were not consulted for this decision.

The frozen local artifacts record SHA-256 values for the 103-case modeling
metadata, 73-patient fold table, resolved configuration, runtime provenance, and
visual review index. The five test-fold case counts are 17 / 23 / 20 / 24 / 19;
patient counts are 12 / 17 / 15 / 16 / 13. Patient-level artifacts remain under
Git-ignored `data/processed/frozen_reference_103_73/`.

Real no-training baselines on the 103 QC-passed cases were:

| Method / group | Dice | HD95 (mm) | absolute RVD | volume abs. error (mm3) |
|---|---:|---:|---:|---:|
| stable mask / all | 0.5964 | 2.3616 | 0.3015 | 147.15 |
| linear SDF / all | 0.4578 | 4.1685 | 3.0378 | 576.91 |
| stable mask / top-20% absolute change | 0.4203 | 3.8718 | 0.8728 | 277.28 |
| linear SDF / top-20% absolute change | 0.2949 | 5.9055 | 10.1905 | 738.21 |

All baseline metrics are finite. Stable-mask is the stronger baseline on this
cohort; any trained model result must be compared against it.

The real fold-0 FP16 CUDA smoke used 103 modeling cases split into 63 train,
23 validation, and 17 test cases. A two-batch smoke plus one validation/test
case completed without OOM, then checkpoint resume successfully ran one full
63-batch epoch and full validation/test. The full epoch took 17.04 s; the
resumed invocation including test inference took 19.89 s. Peak allocated and
reserved GPU memory were 0.127 and 0.139 GiB. A 200-epoch, five-fold linear
budget is approximately 4.7 h; reserve approximately 6 h for I/O and variance.
The pilot test Dice of 0.0304 is a deliberately underfit execution check (and
the selected checkpoint came from the earlier smoke validation), not a model
performance claim. Formal five-fold training has not started.

