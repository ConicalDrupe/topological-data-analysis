# Experiment 1 Log

- **Current experiment:** Experiment 002 (image preprocessing: normalization + ROI
  crop) is done pending your review — PSPNet lung-mask cropping is adopted as the
  ROI-crop method (substantially better than center-crop at removing border
  text/markers, 0/8 segmentation failures). Filtration/vectorization/classification
  not started (Experiment 003 next).
- **Preprocessing pipeline:** `build_cohort.py` (filter + dedup + comorbidity_count/
  is_clean_negative + clean-negatives variant) -> `split_cohort.py` (train/test split
  from the clean-negatives cohort) -> `select_preprocessing_sample.py` (sample
  selection) -> ROI crop (`tda_chexpr.roi.apply_roi_crop`, methods `none`/
  `center_crop`/`lung_mask`) -> normalization (`tda_chexpr.preprocessing.
  apply_normalization`, methods `none`/`he`/`clahe`). No full-dataset batch run of
  any stage yet — that lands with the filtration step.
- **Dataset version:** primary/active: `data/exp1/pneumothorax_cohort_{train,valid}_clean_negatives.csv`
  -> `pneumothorax_{train,test}_split.csv`. Full (unfiltered) cohort
  `pneumothorax_cohort_{train,valid}.csv` retained on disk for a future clean-vs-full
  comparison, not currently used by the split. EDA'd at `results/exp1/eda/v1/` (all six
  CSVs). Preprocessing/ROI-crop comparison sample: `data/exp1/preprocessing_sample.csv`
  (8 rows, 4 positive / 4 negative, drawn from `pneumothorax_train_split.csv`; reused
  unchanged across Experiment 002).
- **Model version:** `torchxrayvision` PSPNet (`pspnet_chestxray_best_model_4.pth`,
  ChestX-Det-trained, cached at `~/.torchxrayvision/models_data/`) for lung
  segmentation. No classification model yet.
- **Random seed:** 42 (train/test split, and preprocessing sample selection).
- **Feature extraction method:** not yet implemented (filtration/vectorization pending).
- **Parameters:** see Experiment 001/002 below.
- **Evaluation metric:** AUROC (planned primary), accuracy/F1 (planned secondary) — not
  yet computed, no model trained.
- **Current status:** clean-negatives cohort + split built and EDA'd; ROI crop and
  normalization pipeline implemented, PSPNet lung-mask crop adopted pending your
  review; filtration/vectorization not started.

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
- (Followed by Experiment 002 below: normalization + ROI crop.)

---

# Experiment 002

## Goal

Build the preprocessing pipeline before filtration: normalization (HE, CLAHE — AGC
skipped) and a ROI-crop stage to strip border text/lateral-view markers/other
burned-in annotations that could corrupt downstream topological features. Tried
progressively smarter ROI methods — none, then a deterministic center-crop baseline,
then content-aware PSPNet lung segmentation — to find one that actually works, and
compared HE vs. CLAHE independently along the way.

## Dataset

Reused `data/exp1/preprocessing_sample.csv` throughout (8 rows, 4 `Pneumothorax==1.0`
/ 4 `==0.0`, `random_state=42`, drawn from `pneumothorax_train_split.csv` via
`src/preprocessing/exp1/select_preprocessing_sample.py`) — no new sampling per stage.
Confirmed CheXpert-small images are single-channel grayscale (PIL mode `L`, `uint8`).

## Parameters

- `tda_chexpr.preprocessing.load_image_grayscale(path)` -> 2D `float64` array in
  `[0, 1]`.
- `tda_chexpr.preprocessing.apply_normalization(image, method, **params)`: `he`
  (`skimage.exposure.equalize_hist`, param `nbins`, default 256); `clahe`
  (`skimage.exposure.equalize_adapthist`, params `clip_limit`=0.01, `kernel_size`=32).
- `tda_chexpr.roi.apply_roi_crop(image, method, **params)`: `none`; `center_crop`
  (`torchxrayvision.datasets.XRayCenterCrop` + `XRayResizer`, `size=224`); `lung_mask`
  (PSPNet bbox, `margin_frac=0.05`) — bbox-only, no hard pixel masking outside the
  silhouette, to avoid introducing an artificial intensity edge that sublevel-set
  cubical persistence would pick up as a false topological feature.
- `tda_chexpr.segmentation.predict_lung_mask(image, threshold=0.5)` —
  `torchxrayvision` PSPNet (`xrv.baseline_models.chestx_det.PSPNet`,
  `pspnet_chestxray_best_model_4.pth`, cached at `~/.torchxrayvision/models_data/`):
  center-crop -> normalize to `[-1024,1024]` -> PSPNet forward (auto-resize 512x512)
  -> sigmoid -> union of `'Left Lung'`/`'Right Lung'` channels -> resize back down ->
  threshold -> paste into the original (non-square) image's coordinate frame.

## Results / decision trail

1. **Center-crop baseline** (`results/exp1/eda/v3/roi_crop/`: `roi_crop_comparison.png`,
   `method_comparison_cropped.png`, `roi_crop_stats.json`) — did **not** reliably
   remove border text: it only trims the width axis (height, 320px, is already the
   smaller dimension), so markers like "PORT"/"AP"/"ERECT"/"L" survived in 6/8 sampled
   images. Useful side effect: forcing a 224x224 resize resolved the separate
   "images aren't a fixed size" problem (raw widths vary 320-415px at fixed 320px
   height, confirmed across 100 rows).
2. **PSPNet lung-mask crop** (`results/exp1/eda/v4/lung_segmentation/`:
   `lung_mask_pipeline.png`, `roi_method_comparison.png`,
   `method_comparison_lung_cropped.png`, `lung_mask_stats.json`) — substantially
   better at removing border text since the crop follows actual lung content instead
   of a fixed geometric rule. **0/8 segmentation failures**, mask fraction 0.21-0.39
   across the sample, bboxes all reasonably sized (~195-300px/side), masks
   anatomically correct on visual inspection. **Adopted** as Experiment 1's ROI-crop
   method, pending final review.
3. **HE vs. CLAHE normalization** (`results/exp1/eda/v2/preprocessing/`:
   `method_comparison.png`, `clahe_parameter_grid.png`, `he_parameter_grid.png`,
   `image_stats.json`, `intensity_histogram_comparison_{positive,negative}.png`),
   tested independently of ROI method and re-confirmed on both cropped variants — HE
   gives a subtle, near-imperceptible global contrast shift; CLAHE gives a much
   stronger local contrast boost (sharper rib/lung detail) but over-amplifies flat
   regions at low `clip_limit`/small `kernel_size`. Defaults `clip_limit=0.01`,
   `kernel_size=32` chosen as a reasonable middle ground; holds consistently across
   raw, center-cropped, and lung-mask-cropped inputs.

## Observations

- Bbox-only cropping (no hard masking) intentionally keeps some non-lung tissue
  (rib cage, shoulder) — a deliberate tradeoff to avoid a false topological edge
  feature (see Parameters), not a defect.
- One bbox touched the raw image's top edge (`patient36302/study5`, lung apex near
  the boundary) — not a failure, but worth watching if it recurs at full-dataset
  scale (the 5% margin may not always be enough headroom).
- PSPNet performs well despite the domain shift (trained on ChestX-Det, applied to
  this cohort's AP portable films) — likely helped by the "no support devices"
  cohort filter excluding images with heavy tube/line clutter.
- Runtime: PSPNet model load ~9s once per run + sub-second inference/image on CPU at
  this 8-image scale; full-dataset timing (~1,189 images) is untested.
- `scikit-image` and `opencv-python` are both project dependencies; only
  `scikit-image` (`equalize_hist`/`equalize_adapthist`) is used so far.
  `opencv-python` remains available, unused, for a possible later filtration
  candidate (edge maps / distance transforms).

## Next Steps

- **Review the plots and confirm adoption of `"lung_mask"`** as Experiment 1's
  ROI-crop method — recommended based on the results above, final call pending.
- If adopted: decide whether to resize lung-mask crops to a fixed size (like
  `center_crop`'s 224x224) for full-dataset consistency — currently left at natural
  bbox size (varies per image), which was more diagnostic for this comparison.
  `apply_roi_crop(..., "lung_mask", size=...)` already supports this via the
  existing `resize()` helper.
- Check full-dataset PSPNet inference time before committing to a batch run over all
  1,189 clean-negatives-cohort images.
- Experiment 1 pipeline step 3 (filtration) not started — the next major pipeline
  stage once ROI cropping is finalized. (Followed by Experiment 003.)
