# DeepGrowth × UCSD-VS-Longitudinal

> **Experiment label:** public-cohort reproduction / external validation
>
> **Status:** minimum public subset downloaded; real preprocessing, QC, baselines, and fold-0 CUDA pilot completed; five-fold training not started
>
> **Upstream snapshot:** [`cyjdswx/DeepGrowth@ad57033`](https://github.com/cyjdswx/DeepGrowth/commit/ad57033f630ffba522decd61051943a5a1289e2d)

This fork preserves the official DeepGrowth implementation and adds a reproducible
pipeline for the public
[UCSD-VS-Longitudinal](https://doi.org/10.7937/WEFA-CP23) cohort. The scientific
question is prospective vestibular-schwannoma morphology prediction: use two
earlier contrast-enhanced T1 scans and their masks, plus the requested future
date, to predict the third tumor mask.

This is **not an exact numerical reproduction** of the paper. DeepGrowth was
trained on a private in-house cohort of 131 patients, and the public repository
does not provide its prepared data or pretrained checkpoints. Results produced
here must be reported as a public-cohort reimplementation/external validation.

## What is included

- manifest validation and patient-level five-fold splitting;
- rigid longitudinal registration, 0.58 mm isotropic resampling, intensity
  normalization, automatic 64³ tumor-centered crops, and signed-distance fields;
- stable-mask and linear-SDF extrapolation baselines;
- a memory-bounded DeepGrowth-style neural-field trainer for an 8 GB RTX 2080;
- Dice, HD95, signed/absolute RVD, volume error, and top-20%-change analysis;
- synthetic CPU smoke tests, so the pipeline can be checked before downloading
  roughly tens of gigabytes of imaging data.

The original author scripts remain at repository root. The maintained public
reproduction lives in `ucsd_repro/`, `scripts/`, `configs/`, and `tests/`.

## Quick start

Python 3.10 is recommended. The following Windows/PowerShell environment was
validated on 2026-08-06 (PyTorch's CUDA wheel includes its runtime; the local
driver must be new enough):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-ucsd-lock.txt
python -m pip install -e .
python scripts/preflight.py --require-cuda
```

For Linux/macOS activation use `source .venv/bin/activate`. The looser
`requirements-ucsd.txt` remains available for compatible future environments.

Run the data-independent checks:

```bash
python -m unittest discover -s tests -v
python scripts/make_synthetic.py --output-dir data/synthetic
python scripts/evaluate_baselines.py \
  --metadata data/synthetic/metadata.csv \
  --output-dir outputs/synthetic-baselines
python scripts/train_ucsd.py \
  --config configs/ucsd_2080.yaml \
  --metadata data/synthetic/metadata.csv \
  --splits data/synthetic/splits.csv \
  --fold 0 --epochs 1 --max-train-batches 2 \
  --max-val-cases 1 --max-test-cases 1
```

## UCSD data workflow

1. Download the 40 GB NIfTI imaging/annotation archive and the clinical TSV
   from the [official TCIA collection](https://www.cancerimagingarchive.net/collection/ucsd-vs-longitudinal/).
   The imaging package requires IBM Aspera Connect. Data are CC BY 4.0 and are
   intentionally excluded from git.

   To request only the segmentation reference image and mask for each
   examination, first generate the audited 1,140-path selection list:

   ```bash
   python scripts/select_ucsd_files.py \
     --clinical /path/to/UCSD-VS-Longitudinal_clinical_information.tsv \
     --output data/raw/ucsd_minimum_paths.txt
   ```

   With Connect running, submit the exact list through the official public
   Faspex authorization flow. The script does not print or persist the package
   passcode, Faspex access token, or Connect authorization key:

   ```bash
   python scripts/download_ucsd_aspera.py \
     --paths data/raw/ucsd_minimum_paths.txt \
     --destination data/raw/ucsd_images \
     --batch-size 20 --wait
   ```

   The downloader preserves the official examination layout, skips existing
   non-empty files on rerun, and can safely resume missing files with a smaller
   batch size if a large transfer stalls.
2. Build a candidate manifest from the official examination folders
   (`VS_XXXX_YY`). The segmentation suffix selects the exact `t1post` or
   `t1postIAC` reference image:

   ```bash
   python scripts/index_ucsd.py \
     --root /path/to/UCSD-VS-Longitudinal \
     --output manifests/ucsd_manifest.csv
   ```

   Use `--layout regex` plus the regex options only for a nonstandard/repacked
   archive. No manual cropping is required.
3. Merge the official clinical TSV. The adapter cumulatively reconstructs
   `study_days`, identifies treatment crossings, and retains no-residual and
   growth labels:

   ```bash
   python scripts/merge_clinical.py \
     --manifest manifests/ucsd_manifest.csv \
     --clinical /path/to/UCSD-VS-Longitudinal_clinical_information.tsv \
     --ucsd-official \
     --output manifests/ucsd_manifest_with_clinical.csv
   ```

   Then validate:

   ```bash
   python scripts/validate_manifest.py \
     --manifest manifests/ucsd_manifest_with_clinical.csv \
     --data-root /path/to/UCSD-VS-Longitudinal \
     --check-files --check-nifti \
     --output-json data/processed/ucsd_nifti_qc.json
   ```

   `--check-nifti` reads every image and mask, checks finite voxel data, 3D
   shape/affine consistency, valid spacing, and permits an empty mask only when
   the official `no_residual_vs` field is true.

4. Create leakage-safe folds and preprocess rolling triples:

   ```bash
   python scripts/make_splits.py \
     --manifest manifests/ucsd_manifest_with_clinical.csv \
     --output manifests/ucsd_splits.csv --folds 5 --seed 100
   python scripts/preprocess_ucsd.py \
     --manifest manifests/ucsd_manifest_with_clinical.csv \
     --output-dir data/ucsd_prepared \
     --config configs/ucsd_2080.yaml \
     --data-root /path/to/UCSD-VS-Longitudinal
   python scripts/render_qc.py \
     --metadata data/ucsd_prepared/metadata.csv \
     --prepared-root data/ucsd_prepared \
     --output-dir outputs/preprocessing-qc --max-cases 6
   ```

   The QC renderer prioritizes crop-border flags and the lowest registration
   overlap. PNG filenames are de-identified; its private local index retains
   the triplet mapping. Keep both outputs outside version control.

5. Establish baselines before training, then run each fold:

   ```bash
   python scripts/evaluate_baselines.py \
     --metadata data/ucsd_prepared/metadata.csv \
     --output-dir outputs/baselines
   python scripts/train_ucsd.py \
     --config configs/ucsd_2080.yaml \
     --metadata data/ucsd_prepared/metadata.csv \
     --splits manifests/ucsd_splits.csv --fold 0
   ```

   When `preprocessing_qc_pass` is present, baseline and training commands
   default to QC-passed cases and print the retained/input denominator. Use
   `--include-qc-failed` only for an explicitly labeled diagnostic analysis.

Repeat fold 0–4. See [REPRODUCTION.md](REPRODUCTION.md) for the frozen protocol,
assumptions, reporting language, and deviations from the original study.

After all folds finish:

```bash
python scripts/aggregate_folds.py \
  --run-root outputs/checkpoints/ucsd_deepgrowth_public_reproduction \
  --output outputs/five_fold_summary.json
```

## Hardware profile and validation boundary

`configs/ucsd_2080.yaml` uses batch size 1, 16,384 sampled SDF points, automatic
mixed precision, two data workers, and gradient accumulation of 8. It is meant
for an RTX 2080 with 8 GB VRAM. Lower `training.points_per_volume` to 8,192 if
memory is still tight. Preprocessing is CPU/RAM intensive and runs one case at a
time by default.

The 2026-08-06 local audit found an **RTX 3080 Laptop GPU with 16 GiB**, not an
RTX 2080. CUDA execution, FP16 AMP, checkpoint save/load, and resume were
validated on that device with synthetic data. The 1,140-file real subset and a
full real preprocessing pass were also completed. Of 151 clinically eligible
rolling triples, 135 were prepared and 103 (73 patients) passed the explicit
crop/registration QC gate used by baseline and training scripts. A complete
fold-0 training epoch took 17.04 s; the full invocation including 17 test cases
took 19.89 s and reserved 0.139 GiB at peak. A linear five-fold/200-epoch
budget is about 4.7 h, or about 6 h with I/O and runtime margin. The completed
pilot is intentionally underfit and is not a final model result. The
configuration never requests BF16.

## Data and licensing

The UCSD collection is not redistributed. It is licensed CC BY 4.0 and requires
the dataset citation specified in [REPRODUCTION.md](REPRODUCTION.md). The
upstream repository did not contain a `LICENSE` file at the pinned commit; this
fork does not imply additional rights over the upstream source.

## Original work

Y. Chen et al., “Vestibular schwannoma growth prediction from longitudinal MRI
by time-conditioned neural fields,” 2024, [arXiv:2404.02614](https://arxiv.org/abs/2404.02614).
