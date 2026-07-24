# Experiment 1 Log

- **Current experiment:** normalization variants (HE, CLAHE) — Experiment 1 pipeline
  step 2 — implemented and compared on a representative image sample. Filtration/
  vectorization/classification (steps 3-6) not started.
- **Preprocessing pipeline:** `build_cohort.py` (filter + dedup + comorbidity_count/
  is_clean_negative + clean-negatives variant) -> `split_cohort.py` (train/test split,
  sourced from the clean-negatives cohort) -> `select_preprocessing_sample.py` +
  `compare_preprocessing.py` (image loading + HE/CLAHE normalization, compared on a
  representative sample). `tda_chexpr.preprocessing.apply_normalization(image, method,
  **params)` is the reusable normalization function (methods: `none`, `he`, `clahe`);
  no full-dataset batch normalization run yet — that lands with the filtration step.
- **Dataset version:** primary/active: `data/exp1/pneumothorax_cohort_{train,valid}_clean_negatives.csv`
  -> `pneumothorax_{train,test}_split.csv`. Full (unfiltered) cohort
  `pneumothorax_cohort_{train,valid}.csv` retained on disk for a future clean-vs-full
  comparison, not currently used by the split. EDA'd at `results/exp1/eda/v1/` (all six
  CSVs). Preprocessing comparison sample: `data/exp1/preprocessing_sample.csv` (8 rows,
  4 positive / 4 negative, drawn from `pneumothorax_train_split.csv`).
- **Model version:** none yet.
- **Random seed:** 42 (train/test split, and preprocessing sample selection).
- **Feature extraction method:** not yet implemented (filtration/vectorization pending).
- **Parameters:** see Experiment 001/002 below.
- **Evaluation metric:** AUROC (planned primary), accuracy/F1 (planned secondary) — not
  yet computed, no model trained.
- **Current status:** clean-negatives cohort + split built and EDA'd; HE/CLAHE
  normalization implemented and compared on a sample; awaiting filtration/vectorization
  work.

---

# Experiment 001

## Goal

Build the Experiment 1 Pneumothorax cohort (AP-only, confirmed no support devices,
earliest jointly-qualifying study per patient), address a comorbidity confound in it
(many Pneumothorax-negative rows still have other confirmed pathologies, which could
produce topological features that look abnormal for the wrong reason), and carve out a
project-owned train/test split — since CheXpert's `valid.csv` is too small/imbalanced to
use as a held-out set once these filters are applied.

## Dataset

- Source: `kaggle/train.csv` (223,414 rows) / `kaggle/valid.csv` (234 rows).
- Filters (applied jointly, before per-patient dedup): `Pneumothorax` in `{0.0, 1.0}`,
  `AP/PA == "AP"`, `Support Devices == 0.0`.
- Dedup: one row per patient — their earliest study satisfying all three filters at once
  (`select_studies(mode="first_qualifying")`). Written to
  `pneumothorax_cohort_{train,valid}.csv`.
- Two columns added to that cohort: `comorbidity_count` (count of the 11 other pathology
  columns confirmed-positive, `== 1.0`, excluding `Pneumothorax`/`No Finding`/
  `Support Devices`; uncertain `-1.0` not counted) and `is_clean_negative`
  (`No Finding == 1.0`).
- Derived clean-negatives cohort: `pneumothorax_cohort_{train,valid}_clean_negatives.csv`
  — keeps every `Pneumothorax == 1.0` row, restricts `Pneumothorax == 0.0` rows to
  `is_clean_negative == True`. This is Experiment 1's primary/active dataset; the full
  (unfiltered) cohort is kept on disk for a future clean-vs-full comparison.
- See `experiments.md`, Experiment 1 pipeline step 1, for the full rationale and the two
  clean-negative definitions considered.

## Parameters

- `build_cohort(df, label="Pneumothorax", mode="first_qualifying", ap_only=True, require_no_support_devices=True)`
- `add_comorbidity_count(cohort, target_label="Pneumothorax")`
- `add_clean_negative_flag(cohort)`
- `filter_clean_negatives(cohort, target_label="Pneumothorax")`
- `stratified_split(df, label="Pneumothorax", test_frac=0.2, random_state=42)`, sourced
  from the clean-negatives train cohort, split on unique `patient_id` (see Observations).

## Results

- Full cohort: train 2,299 rows / 2,266 patients (1,430 negative / 869 positive); valid
  77 rows / 77 patients (76 negative / 1 positive).
- Clean-negatives cohort: train 1,189 rows / 1,166 patients (869 positive / 320 clean
  negative); valid 13 rows / 13 patients (1 positive / 12 clean negative).
- Train/test split (from the clean-negatives train cohort): train_split 953 rows / 932
  patients (698 / 255); test_split 236 rows / 234 patients (171 / 65). Zero patient
  overlap between the two, verified.
- Full Dataset Summary (all 14 pathology class frequencies, missing values, duplicate
  paths, AP/PA, Sex, Age, `comorbidity_count`/`is_clean_negative` breakdowns, etc.) for
  all six current CSVs: `results/exp1/eda/v1/<dataset_name>/summary.json` +
  `age_histogram.png`.

## Observations

- The strict "no support devices" filter (`Support Devices == 0.0` only) produces a
  notably smaller cohort than an earlier ~6,000-row estimate, which likely assumed a
  looser reading (e.g. excluding only confirmed-positive `1.0`, which yields ~14,200
  rows instead). See `experiments.md`'s Open Questions for the discrepancy note.
- 33 patients in the full cohort have 2 rows (their qualifying study has 2 frontal AP
  images). A first, naive row-level train/test split leaked 8 patients across both sides
  before this was caught — fixed by splitting on unique `patient_id` first, then
  assigning every row of a chosen patient to that patient's side. Verified 0 patient
  overlap after the fix.
- `valid.csv`'s filtered cohort (77 rows, 1 positive case) confirms it's unusable as this
  experiment's primary evaluation set; the project-owned split is used instead.
- `duplicate_path_count` is 0 across all six CSVs — no accidental row duplication.
- Of 1,430 Pneumothorax-negative rows in the full train cohort, only 320 (22%) are truly
  clean (`No Finding == 1.0`); 411 have `comorbidity_count == 0` but aren't explicitly
  marked "No Finding" (unmentioned rather than ruled out) — the stricter `No Finding`
  definition was chosen over this looser one. Positive rows are *not* filtered by
  comorbidity — a pneumothorax case with a comorbid finding is still a real pneumothorax
  case; only the negative side of the contrast needed cleaning up.
- Switching to the clean-negatives cohort shrinks Experiment 1's active train/test data
  considerably (2,299 -> 1,189 rows pre-split). Whether this actually improves measured
  AUROC over the full cohort is an open question, deferred until classification exists
  (see `experiments.md` Open Questions).
- Image Statistics EDA (pixel intensity, contrast, brightness, dimensions) is not yet
  applicable — no image-loading/preprocessing pipeline exists yet (see Next Steps).

## Next Steps

- Once classification exists, run both the clean-negatives and full-cohort splits
  through the same pipeline to check whether the comorbidity confound was material in
  practice.
- (Superseded by Experiment 002 below for the normalization step.)

---

# Experiment 002

## Goal

Implement image loading and the two normalization variants from `experiments.md`
pipeline step 2 (Histogram Equalization, CLAHE — AGC explicitly skipped), and compare
them — against each other and against no normalization — on a small representative
image sample, including a parameter sweep for each method. Designed as a reusable,
parameterized function (not a one-off script) so it seeds the future normalization axis
of the normalization × filtration × vectorization experiment matrix and is directly
reusable by Experiment 3.

## Dataset

- Source: `data/exp1/pneumothorax_train_split.csv` (953 rows).
- Representative sample: `data/exp1/preprocessing_sample.csv` — 4 `Pneumothorax==1.0` +
  4 `Pneumothorax==0.0` rows, `random_state=42`
  (`src/preprocessing/exp1/select_preprocessing_sample.py`).
- Confirmed by inspection: CheXpert-small images are single-channel grayscale (PIL mode
  `L`, `uint8`) — no color-channel handling needed.

## Parameters

- `tda_chexpr.preprocessing.load_image_grayscale(path)` — loads to a 2D `float64` array
  in `[0, 1]` (`skimage.util.img_as_float`).
- `tda_chexpr.preprocessing.apply_normalization(image, method, **params)`:
  - `he`: `skimage.exposure.equalize_hist`, param `nbins`.
  - `clahe`: `skimage.exposure.equalize_adapthist`, params `clip_limit`, `kernel_size`.
- `NORMALIZATION_VARIANTS` (default comparison set): `none`; `he` (`nbins=256`); `clahe`
  (`clip_limit=0.01`, `kernel_size=32`).
- Parameter sweeps (`src/preprocessing/exp1/compare_preprocessing.py`), run on one
  representative positive-case image:
  - CLAHE: `clip_limit` in `{0.005, 0.01, 0.02, 0.05}` (fixed `kernel_size=32`); and
    `kernel_size` in `{8, 16, 32, 64}` (fixed `clip_limit=0.01`).
  - HE: `nbins` in `{32, 64, 128, 256}`.

## Results

Artifacts written to `results/exp1/eda/v2/preprocessing/`:
- `method_comparison.png` — Original/HE/CLAHE grid, one row per sampled image (8 rows).
- `clahe_parameter_grid.png`, `he_parameter_grid.png` — parameter sweeps.
- `image_stats.json` — Image Statistics EDA (dimensions, aspect ratio, min/max/mean/std,
  histogram) for all 8 sampled images x 3 methods (24 records).
- `intensity_histogram_comparison_{positive,negative}.png` — before/after pixel
  intensity distribution overlays.

Sample image dimensions: height fixed at 320px; width 389-390px within the 8-image
sample. A wider check across 100 random `pneumothorax_train_split.csv` rows: height
always 320px, width ranges 320-415px (mean ~385px) — confirms images are **not** a
fixed size across the dataset.

## Observations

- HE (global histogram equalization) produces a subtle, mostly-imperceptible contrast
  change on these images — visually and in the intensity histogram, the "before" and
  "after HE" curves nearly overlap. Its only real parameter (`nbins`) has minimal visual
  effect across the swept range. This is expected for global HE on images that already
  span most of the intensity range, not a bug.
- CLAHE produces a much stronger, spatially local contrast change — anatomical detail
  (rib structure, lung markings) becomes visibly sharper. The intensity histogram shows
  CLAHE redistributing pixel mass into two separated modes rather than HE's near-uniform
  flattening.
- CLAHE parameter sweep: low `clip_limit` (0.005) and small `kernel_size` (8) both
  produce visible over-amplification artifacts (blown-out patches, blocky noise) on
  large flat regions — expected CLAHE behavior when contextual regions are small
  relative to local structure, not a defect in the implementation. `clip_limit=0.01`,
  `kernel_size=32` (the chosen default) sits in a visually reasonable middle ground.
- **Image size varies across the dataset** (width 320-415px at fixed height 320px, per
  the 100-row check above). Not resolved in this iteration — no resizing/padding was
  applied. `CubicalPersistence` (the planned baseline filtration) and any
  `kernel_size`-style spatial parameter downstream may need a resizing/padding decision
  before the full experiment matrix runs; deferred to be revisited once this iteration
  is reviewed.
- Both `scikit-image` and `opencv-python` are now project dependencies; this phase uses
  only `scikit-image` (`exposure.equalize_hist`/`equalize_adapthist`) for its
  float-array, keyword-parameterized API. `opencv-python` remains available, unused so
  far, for a possible later filtration candidate (edge maps / distance transforms).

## Next Steps

- Decide the image-size handling policy (resize to a common shape vs. pad vs. leave
  per-image and handle inside the filtration step) before running normalization across
  the full train/test split.
- Experiment 1 pipeline step 3: filtration methods, starting with the baseline
  `gtda.homology.CubicalPersistence` sublevel-set filtration on (optionally normalized)
  grayscale intensities.
- Once filtration exists, wire `NORMALIZATION_VARIANTS` into the full
  normalization x filtration matrix and run it across the whole train/test split
  (not just the 8-image sample used for this comparison).
