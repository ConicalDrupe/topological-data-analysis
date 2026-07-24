# Experiment 1 Log

- **Current experiment:** cohort construction + comorbidity-confound cleanup + train/test
  split (Experiment 1 pipeline step 1). Normalization/filtration/vectorization/
  classification (steps 2-6) not started.
- **Preprocessing pipeline:** `build_cohort.py` (filter + dedup + comorbidity_count/
  is_clean_negative + clean-negatives variant) -> `split_cohort.py` (train/test split,
  sourced from the clean-negatives cohort). No image-level preprocessing yet.
- **Dataset version:** primary/active: `data/exp1/pneumothorax_cohort_{train,valid}_clean_negatives.csv`
  -> `pneumothorax_{train,test}_split.csv`. Full (unfiltered) cohort
  `pneumothorax_cohort_{train,valid}.csv` retained on disk for a future clean-vs-full
  comparison, not currently used by the split. EDA'd at `results/exp1/eda/v1/` (all six
  CSVs).
- **Model version:** none yet.
- **Random seed:** 42 (train/test split only, so far).
- **Feature extraction method:** not yet implemented.
- **Parameters:** see Experiment 001/002 below.
- **Evaluation metric:** AUROC (planned primary), accuracy/F1 (planned secondary) — not
  yet computed, no model trained.
- **Current status:** clean-negatives cohort + split built and EDA'd; awaiting
  normalization/filtration/vectorization work.

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

---

# Experiment 002

## Goal

Address a comorbidity confound in the Experiment 1 cohort: many Pneumothorax-negative
rows still have other confirmed pathologies, which could produce topological features
that look abnormal for the wrong reason. Add visibility into this (comorbidity count,
clean-negative flag) and switch Experiment 1 to a cohort where the negative class is
restricted to truly clean (`No Finding == 1.0`) films.

## Dataset

- Source: `data/exp1/pneumothorax_cohort_{train,valid}.csv` (Experiment 001's output).
- New columns added to both: `comorbidity_count` (count of the 11 other pathology
  columns confirmed-positive, `== 1.0`, excluding `Pneumothorax`/`No Finding`/
  `Support Devices`; uncertain `-1.0` not counted) and `is_clean_negative`
  (`No Finding == 1.0`).
- New derived cohort: `pneumothorax_cohort_{train,valid}_clean_negatives.csv` — keeps
  every `Pneumothorax == 1.0` row, restricts `Pneumothorax == 0.0` rows to
  `is_clean_negative == True`.
- See `experiments.md`, Experiment 1 pipeline step 1 ("Comorbidity confound check and
  'clean negatives' cohort"), for the full rationale and the two definitions considered.

## Parameters

- `add_comorbidity_count(cohort, target_label="Pneumothorax")`
- `add_clean_negative_flag(cohort)`
- `filter_clean_negatives(cohort, target_label="Pneumothorax")`
- `split_cohort.py` now reads `pneumothorax_cohort_train_clean_negatives.csv` (same
  `stratified_split(..., test_frac=0.2, random_state=42)` as Experiment 001).

## Results

- Clean-negatives cohort: train 1,189 rows / 1,166 patients (869 positive / 320 clean
  negative); valid 13 rows / 13 patients (1 positive / 12 clean negative).
- Train/test split (from the clean-negatives train cohort): train_split 953 rows / 932
  patients (698 / 255); test_split 236 rows / 234 patients (171 / 65). Zero patient
  overlap between the two, verified.
- Full Dataset Summary for all six current CSVs (base + clean-negatives + split, each
  now including `comorbidity_count`/`is_clean_negative` breakdowns):
  `results/exp1/eda/v1/<dataset_name>/summary.json` (overwritten in place, per explicit
  instruction, rather than incrementing to `v2/`).

## Observations

- Of 1,430 Pneumothorax-negative rows in the train cohort, only 320 (22%) are truly
  clean (`No Finding == 1.0`); 411 have `comorbidity_count == 0` but aren't explicitly
  marked "No Finding" (unmentioned rather than ruled out) — the stricter `No Finding`
  definition was chosen over this looser one.
- Positive rows are *not* filtered by comorbidity — a pneumothorax case with a comorbid
  finding is still a real pneumothorax case; only the negative side of the contrast
  needed cleaning up.
- This shrinks Experiment 1's active train/test data considerably (2,299 -> 1,189 rows
  pre-split). Whether this actually improves measured AUROC over the full cohort is an
  open question, deferred until classification exists (see `experiments.md` Open
  Questions).

## Next Steps

- Experiment 1 pipeline step 2 (normalization variants), now against the
  clean-negatives train/test split.
- Once classification exists, run both the clean-negatives and full-cohort splits
  through the same pipeline to check whether the comorbidity confound was material in
  practice.
