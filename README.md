# DeepGrowth × UCSD-VS-Longitudinal

> **Experiment label:** public-cohort reproduction / external validation
>
> **Status:** pipeline ready; UCSD data download and five-fold runs are pending
>
> **Upstream snapshot:** [`cyjdswx/DeepGrowth@ad57033`](https://github.com/cyjdswx/DeepGrowth/commit/ad57033f630ffba522decd61051943a5a1289e2d)

This fork preserves the official DeepGrowth implementation and adds a reproducible
pipeline for the public
[UCSD-VS-Longitudinal](https://doi.org/10.7937/bq0z-xa62) cohort. The scientific
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

Python 3.10 is recommended. Install a CUDA build of PyTorch that matches the
local NVIDIA driver, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch from https://pytorch.org/get-started/locally/ first.
python -m pip install -r requirements-ucsd.txt
python -m pip install -e .
python scripts/preflight.py
```

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
  --fold 0 --epochs 1 --max-train-batches 2 --max-val-cases 1
```

## UCSD data workflow

1. Download the NIfTI imaging/annotation archive and clinical table from the
   [TCIA collection](https://www.cancerimagingarchive.net/collection/ucsd-vs-longitudinal/).
   Data are intentionally excluded from git.
2. Build a candidate manifest from a BIDS-like `sub-*/ses-*` layout:

   ```bash
   python scripts/index_ucsd.py \
     --root /path/to/UCSD-VS-Longitudinal \
     --output manifests/ucsd_manifest.csv
   ```

   If the archive layout differs, supply `--patient-regex`, `--session-regex`,
   `--image-regex`, and `--mask-regex`. No manual cropping is required.
3. Add or merge true study timing into `study_days`. A generic keyed merge is
   included because TCIA supporting-table column names can change between
   releases:

   ```bash
   python scripts/merge_clinical.py \
     --manifest manifests/ucsd_manifest.csv \
     --clinical /path/to/clinical.csv \
     --patient-column PATIENT_COLUMN \
     --timepoint-column TIMEPOINT_COLUMN \
     --days-column DAYS_FROM_BASELINE_COLUMN \
     --output manifests/ucsd_manifest_with_clinical.csv
   ```

   Then validate:

   ```bash
   python scripts/validate_manifest.py \
     --manifest manifests/ucsd_manifest_with_clinical.csv \
     --data-root /path/to/UCSD-VS-Longitudinal --check-files
   ```

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
   ```

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

Repeat fold 0–4. See [REPRODUCTION.md](REPRODUCTION.md) for the frozen protocol,
assumptions, reporting language, and deviations from the original study.

After all folds finish:

```bash
python scripts/aggregate_folds.py \
  --run-root outputs/checkpoints/ucsd_deepgrowth_public_reproduction \
  --output outputs/five_fold_summary.json
```

## Hardware profile

`configs/ucsd_2080.yaml` uses batch size 1, 16,384 sampled SDF points, automatic
mixed precision, two data workers, and gradient accumulation of 8. It is meant
for an RTX 2080 with 8 GB VRAM. Lower `training.points_per_volume` to 8,192 if
memory is still tight. Preprocessing is CPU/RAM intensive and runs one case at a
time by default.

## Data and licensing

The UCSD collection is not redistributed. Follow its TCIA data-usage and citation
terms. The upstream repository did not contain a `LICENSE` file at the pinned
commit; this fork does not imply additional rights over the upstream source.

## Original work

Y. Chen et al., “Vestibular schwannoma growth prediction from longitudinal MRI
by time-conditioned neural fields,” 2024, [arXiv:2404.02614](https://arxiv.org/abs/2404.02614).
