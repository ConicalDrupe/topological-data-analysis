# Experiment 1 Log

- **Current experiment:** cohort construction + train/test split (Experiment 1 pipeline
  step 1). Normalization/filtration/vectorization/classification (steps 2-6) not started.
- **Preprocessing pipeline:** `build_cohort.py` (filter + dedup) -> `split_cohort.py`
  (train/test split). No image-level preprocessing yet.
- **Dataset version:** `data/exp1/pneumothorax_cohort_{train,valid}.csv` +
  `pneumothorax_{train,test}_split.csv`, EDA'd at `results/exp1/eda/v1/`.
- **Model version:** none yet.
- **Random seed:** 42 (train/test split only, so far).
- **Feature extraction method:** not yet implemented.
- **Parameters:** see Experiment 001 below.
- **Evaluation metric:** AUROC (planned primary), accuracy/F1 (planned secondary) — not
  yet computed, no model trained.
- **Current status:** cohort + split built and EDA'd; awaiting normalization/filtration/
  vectorization work.

---

# Experiment 001

## Goal

Build the Experiment 1 Pneumothorax cohort (AP-only, confirmed no support devices,
earliest jointly-qualifying study per patient) and a project-owned train/test split,
since CheXpert's `valid.csv` is too small/imbalanced to use as a held-out set once these
filters are applied.

## Dataset

- Source: `kaggle/train.csv` (223,414 rows) / `kaggle/valid.csv` (234 rows).
- Filters (applied jointly, before per-patient dedup): `Pneumothorax` in `{0.0, 1.0}`,
  `AP/PA == "AP"`, `Support Devices == 0.0`.
- Dedup: one row per patient — their earliest study satisfying all three filters at once
  (`select_studies(mode="first_qualifying")`).
- See `experiments.md`, Experiment 1 pipeline step 1, for the full rationale.

## Parameters

- `build_cohort(df, label="Pneumothorax", mode="first_qualifying", ap_only=True, require_no_support_devices=True)`
- `stratified_split(df, label="Pneumothorax", test_frac=0.2, random_state=42)`, split on
  unique `patient_id` (see Observations).

## Results

- Cohort: train 2,299 rows / 2,266 patients (1,430 negative / 869 positive); valid 77
  rows / 77 patients (76 negative / 1 positive).
- Split: train_split 1,842 rows / 1,812 patients (1,144 / 698); test_split 457 rows / 454
  patients (286 / 171).
- Full Dataset Summary (all 14 pathology class frequencies, missing values, duplicate
  paths, AP/PA, Sex, Age, etc.) for all four CSVs: `results/exp1/eda/v1/<dataset_name>/summary.json`
  + `age_histogram.png`.

## Observations

- The strict "no support devices" filter (`Support Devices == 0.0` only) produces a
  notably smaller cohort than an earlier ~6,000-row estimate, which likely assumed a
  looser reading (e.g. excluding only confirmed-positive `1.0`, which yields ~14,200
  rows instead). See `experiments.md`'s Open Questions for the discrepancy note.
- 33 patients in the cohort have 2 rows (their qualifying study has 2 frontal AP images).
  A first, naive row-level train/test split leaked 8 patients across both sides before
  this was caught — fixed by splitting on unique `patient_id` first, then assigning every
  row of a chosen patient to that patient's side. Verified 0 patient overlap after the fix.
- `valid.csv`'s filtered cohort (77 rows, 1 positive case) confirms it's unusable as this
  experiment's primary evaluation set; the project-owned split is used instead.
- `duplicate_path_count` is 0 across all four CSVs — no accidental row duplication.
- Image Statistics EDA (pixel intensity, contrast, brightness, dimensions) is not yet
  applicable — no image-loading/preprocessing pipeline exists yet (see Next Steps).

## Next Steps

- Experiment 1 pipeline step 2: normalization variants (Histogram Equalization, CLAHE —
  AGC explicitly skipped per `experiments.md`). Once images are loaded/preprocessed,
  add Image Statistics EDA (dimensions, aspect ratio, pixel intensity, contrast,
  brightness, before/after comparison) per `CLAUDE.md`.
