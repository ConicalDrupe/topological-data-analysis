# Experiment 1 Log

- **Current experiment:** PSPNet-based lung segmentation for ROI cropping — Experiment
  1 pipeline step 2 — implemented and compared against the center-crop baseline
  (Experiment 003) on the representative sample. Result: lung-mask bounding-box
  cropping is **substantially better** at removing border text/markers than
  center-crop alone, with 0/8 segmentation failures on the sample (see Experiment
  004). Recommended as the adopted ROI-crop method going forward, pending your
  review of the plots. Filtration/vectorization/classification not started.
- **Preprocessing pipeline:** `build_cohort.py` (filter + dedup + comorbidity_count/
  is_clean_negative + clean-negatives variant) -> `split_cohort.py` (train/test split,
  sourced from the clean-negatives cohort) -> `select_preprocessing_sample.py` (sample
  selection) -> `compare_roi_crop.py` (ROI crop: center-crop + resize to 224x224) /
  `compare_lung_segmentation.py` (ROI crop: PSPNet lung-mask bbox) ->
  `compare_preprocessing.py` (HE/CLAHE normalization). Reusable pipeline functions:
  `tda_chexpr.roi.apply_roi_crop(image, method, **params)` (methods: `none`,
  `center_crop`, `lung_mask`), `tda_chexpr.segmentation.predict_lung_mask(image)`
  (PSPNet), and `tda_chexpr.preprocessing.apply_normalization(image, method,
  **params)` (methods: `none`, `he`, `clahe`). No full-dataset batch run of any stage
  yet — that lands with the filtration step.
- **Dataset version:** primary/active: `data/exp1/pneumothorax_cohort_{train,valid}_clean_negatives.csv`
  -> `pneumothorax_{train,test}_split.csv`. Full (unfiltered) cohort
  `pneumothorax_cohort_{train,valid}.csv` retained on disk for a future clean-vs-full
  comparison, not currently used by the split. EDA'd at `results/exp1/eda/v1/` (all six
  CSVs). Preprocessing/ROI-crop comparison sample: `data/exp1/preprocessing_sample.csv`
  (8 rows, 4 positive / 4 negative, drawn from `pneumothorax_train_split.csv`; reused
  unchanged across Experiments 002-004).
- **Model version:** `torchxrayvision` PSPNet (`pspnet_chestxray_best_model_4.pth`,
  ChestX-Det-trained, cached at `~/.torchxrayvision/models_data/`) for lung
  segmentation. No classification model yet.
- **Random seed:** 42 (train/test split, and preprocessing sample selection).
- **Feature extraction method:** not yet implemented (filtration/vectorization pending).
- **Parameters:** see Experiment 001/002/003/004 below.
- **Evaluation metric:** AUROC (planned primary), accuracy/F1 (planned secondary) — not
  yet computed, no model trained.
- **Current status:** clean-negatives cohort + split built and EDA'd; HE/CLAHE
  normalization implemented; center-crop ROI baseline tested and found insufficient;
  PSPNet lung-mask ROI cropping implemented and found substantially better — final
  adoption decision pending your review; filtration/vectorization not started.

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
- (Superseded/reordered by Experiment 003 below: a ROI-crop stage is now inserted
  before normalization in the pipeline.)

---

# Experiment 003

## Goal

Insert a region-of-interest (ROI) crop stage **before** normalization in the pipeline
(`Raw -> ROI crop -> Normalize`), to remove border text, lateral-view markers, and
other burned-in annotations that could corrupt the topological features extracted
downstream. Start with a deterministic, no-inference baseline
(`torchxrayvision`'s `XRayCenterCrop` + `XRayResizer`) rather than committing to
PSPNet-based lung segmentation up front, and use the result to decide whether PSPNet
is actually needed.

## Dataset

- Reused `data/exp1/preprocessing_sample.csv` unchanged (same 8-image, 4 positive / 4
  negative sample from Experiment 002, seed 42) — no new sampling step.

## Parameters

- `tda_chexpr.roi.center_crop(image)` — `torchxrayvision.datasets.XRayCenterCrop`,
  crops to a centered square using `min(height, width)`.
- `tda_chexpr.roi.resize(image, size=224)` — `torchxrayvision.datasets.XRayResizer`.
- `tda_chexpr.roi.apply_roi_crop(image, method, **params)` — dispatcher mirroring
  `apply_normalization`'s shape (`none`, `center_crop`). `ROI_VARIANTS = [("none",
  {}), ("center_crop", {"size": 224})]`.
- Decision recorded for later (not yet implemented): if/when a lung-mask-based crop
  variant (PSPNet) is added, crop to the mask's bounding box only and keep all pixel
  values inside it — no hard pixel masking outside the lung silhouette, to avoid
  introducing an artificial intensity edge that sublevel-set cubical persistence would
  pick up as a false topological feature.

## Results

Artifacts written to `results/exp1/eda/v3/roi_crop/`:
- `roi_crop_comparison.png` — `[Raw, Center-cropped, Resized]` grid, one row per
  sampled image.
- `method_comparison_cropped.png` — the Experiment 002 HE/CLAHE comparison, re-run on
  the cropped images (reusing `tda_chexpr.preprocessing.plot_method_comparison`, moved
  there from `compare_preprocessing.py` in this iteration so both scripts share it).
- `roi_crop_stats.json` — Image Statistics EDA (dimensions, mean/std, histogram) for
  all 8 images x 3 stages (raw / center_cropped / resized; 24 records).

Confirmed: every sampled image's raw shape (height fixed 320px, width 389-390px in
this sample) becomes exactly 320x320 after center-crop and exactly 224x224 after
resize — 100% consistent across the sample.

## Observations

- **Center-crop + resize does not reliably remove border text/markers.** It only
  trims the width axis (since height, at 320px, is already the smaller dimension for
  every image in this cohort); most burned-in text (e.g. "PORT", "AP", "ERECT",
  "UPRIGHT", "L" orientation markers) sits within the vertical extent that the crop
  leaves untouched, so it's still clearly visible in `roi_crop_comparison.png`'s
  "Center-cropped"/"Resized" columns for 6 of the 8 sampled images. Only one image
  (row 1, positive) had its markers positioned near a trimmed corner and lost them.
  This is the key finding this test was designed to surface.
  This is expected once you look at *why* it happens, not a bug: `XRayCenterCrop` is
  a fixed geometric crop with no awareness of image content (lungs vs. text) — it
  can't distinguish "text near the lung field" from "text in the margin."
- The center-crop also isn't guaranteed to remove the raw image's black letterbox
  padding either, when that padding isn't symmetric around the image's own center
  (visible as a residual dark bar on one side in several cropped outputs) — same root
  cause: no content-awareness.
- Re-running the HE/CLAHE comparison on cropped images (`method_comparison_cropped.png`)
  shows the same qualitative HE/CLAHE behavior documented in Experiment 002 (HE:
  subtle global shift; CLAHE: strong local contrast, still picking up whatever
  border text survived the crop) — confirms the pipeline composes correctly stage to
  stage, independent of the crop-quality finding above.
- Useful side effect, unrelated to the border-text goal: resizing to a fixed 224x224
  resolves Experiment 002's deferred "images aren't a fixed size" observation — every
  output is now uniformly sized, which the eventual full-dataset normalization x
  filtration matrix will need anyway.
- Fixed a plot-rendering bug while building this: `fig.suptitle()` combined with
  `fig.tight_layout()` on multi-row 3-column grids caused the title to visually
  overlap the column headers (`Raw`/`Center-cropped`/`Resized`, and
  `Original`/`HE`/`CLAHE`); fixed via `tight_layout(rect=(0, 0, 1, 0.97))` in both
  `plot_method_comparison` and `plot_roi_crop_comparison`. Re-ran
  `compare_preprocessing.py` afterward and confirmed its output is otherwise
  unchanged (identical `image_stats.json`, identical PNG file sizes) — the plotting-
  helper refactor didn't alter Experiment 002's results, only fixed this rendering
  issue.

## Next Steps

- **Decide on PSPNet-based lung segmentation** (`xrv.baseline_models.chestx_det.PSPNet`,
  already available via the installed `torchxrayvision` package) given the finding
  above that center-crop+resize alone leaves most border text intact. A lung mask's
  bounding box would be content-aware and should remove markers that a fixed
  geometric crop can't reach — worth testing on the same 8-image sample before
  committing to it for the full dataset.
- If PSPNet is adopted: add a `"lung_mask"` entry to `ROI_VARIANTS`
  (`tda_chexpr/roi.py`), implementing the bbox-only/no-hard-masking decision recorded
  above.
- Still open from Experiment 002: the image-size-varies issue is now moot for any
  pipeline that includes the `center_crop` ROI stage (output is always 224x224), but
  would still need a decision if ROI cropping is ever skipped (`method="none"`).
- Experiment 1 pipeline step 3 (filtration) remains not started.
- (Superseded by Experiment 004 below: PSPNet-based lung segmentation implemented.)

---

# Experiment 004

## Goal

Implement content-aware ROI cropping via PSPNet-based lung segmentation
(`torchxrayvision`'s `xrv.baseline_models.chestx_det.PSPNet`), producing
`Raw -> Lung mask -> Bounding box -> Crop`, and compare it directly against the
Experiment 003 center-crop baseline on the same sample to decide which method to
adopt.

## Dataset

- Reused `data/exp1/preprocessing_sample.csv` unchanged (same 8-image, 4 positive / 4
  negative sample, seed 42) — no new sampling step.

## Parameters

- `tda_chexpr.segmentation.get_pspnet_model()` — lazy singleton; `PSPNet()` auto-
  downloads its checkpoint (`pspnet_chestxray_best_model_4.pth`, 273MB) to
  `~/.torchxrayvision/models_data/` on first use, then loads from cache (~9s either
  way, dominated by model construction, not the download once cached).
- `tda_chexpr.segmentation.predict_lung_mask(image, threshold=0.5)` — center-crops
  internally (`tda_chexpr.roi.center_crop`, to satisfy PSPNet's square-input
  requirement) -> `xrv.utils.normalize(img, 255)` (maps `[0,255]` to
  `[-1024,1024]`) -> `PSPNet` forward (auto-resizes to its native 512x512) ->
  `sigmoid` -> per-pixel max of the `'Left Lung'`/`'Right Lung'` channels (union) ->
  bilinear-resize the probability map back down to the crop's native resolution ->
  threshold -> paste back into the original (non-square) image's coordinate frame.
- `tda_chexpr.roi.mask_to_bbox(mask, margin_frac=0.05)` — bounding box of the mask,
  expanded 5% of its own height/width on each side, clipped to image bounds.
- `tda_chexpr.roi.crop_to_bbox(image, bbox)` — crops the **raw** image (not the
  center-cropped one) to that box, keeping all pixel values inside it (no hard pixel
  masking) — per the decision recorded in Experiment 003's context.
- `ROI_VARIANTS` now: `[("none", {}), ("center_crop", {"size": 224}), ("lung_mask",
  {"margin_frac": 0.05, "threshold": 0.5})]`.

## Results

Artifacts written to `results/exp1/eda/v4/lung_segmentation/`:
- `lung_mask_pipeline.png` — `[Raw, Mask+bbox overlay, Cropped]`, one row per image.
- `roi_method_comparison.png` — `[Raw, Center-crop, Lung-mask crop]` side by side.
- `method_comparison_lung_cropped.png` — the HE/CLAHE comparison re-run on the
  lung-mask-cropped images.
- `lung_mask_stats.json` — per-image bbox, mask fraction, and dimensions.

**0/8 segmentation failures** (no empty masks). Mask fraction (share of raw image
that's lung) ranged **0.21-0.39** across the 8 images — consistent, no outliers.
Bounding boxes were all reasonably sized rectangles (roughly 195-300px per side out of
the 320x390 raw images), none degenerate.

## Observations

- **Lung-mask cropping is substantially better than center-crop at removing border
  text/markers**, visible directly in `roi_method_comparison.png`: most burned-in
  annotations ("PORT", "AP", "ERECT", "L" markers, etc.) that survived center-crop in
  Experiment 003 are absent or much reduced in the lung-mask crop column, because the
  crop is now driven by actual lung content rather than a fixed geometric rule.
- Predicted masks look anatomically correct on visual inspection for all 8 images —
  two clearly lung-shaped regions, no scattered noise or spurious blobs — despite this
  cohort being a different domain (AP portable films) than PSPNet's ChestX-Det
  training set. The Experiment 1 cohort filter (confirmed no support devices) likely
  helps here by excluding images with heavy tube/line clutter that could otherwise
  confuse the segmentation.
- One image's bbox touched the raw image's top edge (`top=0`,
  `patient36302/study5`) — the lung apex sits right at/near the image boundary in that
  film. Not a failure, but worth watching: if this recurs often across the full
  dataset, the 5% margin may occasionally not be enough headroom at the image edge.
- Bounding boxes still include a modest amount of non-lung tissue (rib cage, some
  shoulder/soft tissue) by design (bbox-only, not a hard mask) — this is intentional
  per the earlier decision, not a defect.
- Re-running the HE/CLAHE comparison on the lung-mask-cropped images
  (`method_comparison_lung_cropped.png`) shows the same qualitative HE/CLAHE behavior
  documented in Experiments 002-003 — confirms the pipeline composes correctly with
  this new ROI method too.
- Runtime: model load ~9s (once per script run, not per image) + sub-second inference
  per image on CPU for this 8-image sample — fine at this scale; full-dataset timing
  (~1,189 images) is untested and would need checking before a full batch run.

## Next Steps

- **Review the plots and confirm adoption of `"lung_mask"` as Experiment 1's ROI-crop
  method.** Based on the results above this is the recommendation, but final call is
  yours before it's wired into the full pipeline.
- If adopted: decide whether to also resize lung-mask crops to a fixed size (like
  `center_crop`'s 224x224) for full-dataset consistency — currently left at natural
  bbox size (varies per image) since that was more diagnostic for this comparison.
  `apply_roi_crop(..., "lung_mask", size=...)` already supports this via the existing
  `resize()` helper, just needs a size decision.
- Check full-dataset PSPNet inference time before committing to a batch run over all
  1,189 clean-negatives-cohort images (currently only tested on 8).
- Experiment 1 pipeline step 3 (filtration) remains not started — the next major
  pipeline stage once ROI cropping is finalized.
