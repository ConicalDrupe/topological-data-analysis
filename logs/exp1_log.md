# Experiment 1 Log

- **Current experiment:** Experiment 007 (full-dataset conservative-CLAHE processed
  image export — see below) is done. Experiment 1 remains otherwise **halted** pending
  regeneration of the preprocessing sample and downstream artifacts against the
  corrected cohort; Experiments 002–005 (ROI crop, normalization, filtration,
  denoising) were tuned against the now-superseded cohort and still need to be
  re-validated at the Experiment 003 *default* CLAHE settings before Experiment 1
  proper resumes — Experiment 007 is a separate, deliberately more conservative export
  requested independently of that pipeline, not a substitute for it.
- **Preprocessing pipeline:** `build_cohort.py` (filter + comorbidity_count/
  is_clean_negative + `filter_no_comorbidity`, applied *before* per-patient dedup, then
  `select_studies` — see Experiment 006) -> `split_cohort.py` (train/test split from the
  corrected clean cohort) -> `select_preprocessing_sample.py` (sample selection, **stale
  — built from the pre-correction split, needs rerun**) -> ROI crop
  (`tda_chexpr.roi.apply_roi_crop`, methods `none`/`center_crop`/`lung_mask`;
  `lung_mask` now defaults to `size=224`, direct resize, no padding) -> normalization
  (`tda_chexpr.preprocessing.apply_normalization`, methods `none`/`he`/`clahe`, `clahe`
  default `kernel_size=16`/`clip_limit=0.01`) -> filtration
  (`tda_chexpr.filtration.compute_persistence_diagram`, applied directly to the CLAHE
  output with no denoising step by default; `apply_direction` is the only preprocessing
  knob) -> optional denoising (`tda_chexpr.denoising`: Anscombe+wavelet image-space
  denoise, `tda_chexpr.filtration.threshold_diagram` fixed-cutoff thresholding,
  `tda_chexpr.denoising.bottleneck_confidence_cutoff` data-driven cutoff), none yet
  selected as the pipeline default. First full-dataset batch run is now done for the
  ROI-crop + CLAHE stages only (Experiment 007, 575 images, conservative
  `clip_limit=0.002` variant) — filtration/denoising/vectorization/classification are
  still 10-image-sample-only.
- **Dataset version:** primary/active: `data/exp1/v2_corrected_cohort/pneumothorax_cohort_{train,valid}_clean.csv`
  -> `pneumothorax_{train,test}_split.csv` (corrected, Experiment 006). Reference
  (unfiltered-by-comorbidity) cohort `pneumothorax_cohort_{train,valid}.csv` retained
  alongside it for a future clean-vs-reference comparison. The old
  `data/exp1/pneumothorax_*` files (pre-Experiment-006) are left on disk but superseded
  — see `cohort_validation.md`. EDA'd at `results/exp1/eda/v9/` (all six corrected
  CSVs). Preprocessing/ROI-crop comparison sample: `data/exp1/preprocessing_sample.csv`
  (10 rows, 5 positive / 5 negative, drawn from the **pre-correction**
  `pneumothorax_train_split.csv`; bumped from 8 rows in Experiment 003) — stale, needs
  regenerating from the corrected split before Experiment 1 resumes.
- **Model version:** `torchxrayvision` PSPNet (`pspnet_chestxray_best_model_4.pth`,
  ChestX-Det-trained, cached at `~/.torchxrayvision/models_data/`) for lung
  segmentation. No classification model yet.
- **Random seed:** 42 (train/test split, preprocessing sample selection, and
  Experiment 005's block-bootstrap RNG).
- **Feature extraction method:** not yet implemented (filtration/vectorization pending).
- **Parameters:** see Experiment 001-007 below.
- **Evaluation metric:** AUROC (planned primary), accuracy/F1 (planned secondary) — not
  yet computed, no model trained.
- **Current status:** cohort-selection bug fixed and corrected cohort/split built + EDA'd
  (Experiment 006); ROI-crop + normalization pipeline previously finalized (lung-mask
  crop, direct resize to 224x224, CLAHE `kernel_size=16`, `clip_limit=0.01`) but built
  against the stale cohort; cubical-persistence filtration and three denoising
  strategies previously explored, also against the stale cohort; vectorization not
  started. Full-dataset processed-image export (lung-mask crop + conservative CLAHE,
  `clip_limit=0.002`) now done for both corrected splits (575/575 images, 0 failures) —
  Experiment 007, `kaggle/processed/`. Next: regenerate the preprocessing sample from
  the corrected split and re-validate Experiments 002–005 (at the `clip_limit=0.01`
  default) before resuming.

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

---

# Experiment 003

## Goal

Finalize the preprocessing pipeline ahead of filtration: bring the lung-mask crop to a
fixed size (open question from Experiment 002), and grid search CLAHE's `clip_limit`
at smaller, more locally-aggressive `kernel_size` values (8 and 16; Experiment 002 used
`kernel_size=32`).

## Dataset

`data/exp1/preprocessing_sample.csv`, bumped from 8 to 10 rows (5 `Pneumothorax==1.0`
/ 5 `==0.0`, `random_state=42`, same `select_preprocessing_sample.py`, drawn from
`pneumothorax_train_split.csv`) — the extra 2 rows (1 pos, 1 neg) added for more
visual evidence in the clip_limit grid.

## Parameters

- `tda_chexpr.roi.apply_roi_crop(image, "lung_mask", margin_frac=0.05, threshold=0.5,
  size=224)` — PSPNet bbox crop, then **direct `resize()` to 224x224** (aspect ratio
  warped, no padding/cropping to square).
- `tda_chexpr.preprocessing.apply_normalization(image, "clahe", kernel_size=ks,
  clip_limit=clip)` for `ks in [8, 16]` x `clip in [0.002, 0.004, 0.006, 0.008, 0.01,
  1.0]` (final grid — narrowed from two earlier, wider sweeps at your request: first
  `[0.01..0.05]`, then `[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]` which showed anything ≥0.01
  is too aggressive at `kernel_size=8`; `1.0` kept as the fixed saturation reference
  point). Produces one grid plot per `kernel_size`
  (`clahe_grid_comparison_kernel{8,16}.png`). HE at defaults (`nbins=256`) for
  reference.

## Results

**Squaring method (course correction):** first attempt padded the non-square lung-mask
bbox crop to square with edge-replication (`roi.pad_to_square(mode="edge")`) before
resizing, to avoid warping lung proportions. Bboxes need up to ~30% padding on their
short axis in this sample (aspect ratios 0.68-1.06, per Experiment 002's stats), and
edge-replicating a single row/column of real rib texture across that much padding
produced visible fake vertical "stripe" artifacts in 3/10 images
(`results/exp1/eda/v5/clahe_grid_search/clahe_grid_comparison.png`). Rejected in favor
of a **direct resize (aspect-ratio warp, no pad/crop)** — reasoning, given these images
also feed a separate image-features/CNN baseline for comparison:
  - No fabricated pixels (unlike edge- or constant-pad) and no lost content (unlike
    cropping the longer axis down to match, which would cut into the lung's
    lateral/apex margins — exactly where pneumothorax findings appear).
  - An affine (anisotropic) resize is a homeomorphism of the image domain, so it
    provably can't create or destroy sublevel-set topological features, up to ordinary
    discretization noise present in every option.
  - Plain non-aspect-preserving resize is standard/unremarkable for CNN pipelines
    (equivalent to `torchvision.transforms.Resize((H, W))`).
  - `roi.pad_to_square()` is kept in the codebase (supports `mode="edge"` or
    `mode="constant"`) as a documented, currently-unused alternative — not deleted.
    Cropping the longer axis to match was discussed and rejected (loses
    pneumothorax-relevant peripheral lung content) but is trivial to reconstruct later.

**`clip_limit` range/saturation finding:** `skimage.exposure.equalize_adapthist`'s
`clip_limit` is only meaningful in `[0, 1]` — internally it computes
`clim = int(clip(clip_limit * kernel_elements, 1, None))` (`kernel_elements =
kernel_size**2`), and per the library's own docstring "a clip limit of 0 or larger
than or equal to 1 results in standard (non-contrast-limited) AHE." Verified directly:
`clip_limit=1.0` and `clip_limit=2.0` produce byte-identical output (`np.array_equal`
on a test array) — this is why `1.0` alone is kept as the grid's saturation reference
point instead of also re-testing `2.0`.

**`clim` quantization finding (explains the final grid's shape):** because `clim` is
cast to `int`, its value is quantized in steps of `1/kernel_elements`. Computed
directly for this experiment's grid:

  | `clip_limit` | `clim` @ `kernel_size=8` (64 elements) | `clim` @ `kernel_size=16` (256 elements) |
  |---|---|---|
  | 0.002 | 1 | 1 |
  | 0.004 | 1 | 1 |
  | 0.006 | 1 | 1 |
  | 0.008 | 1 | 2 |
  | 0.01  | 1 | 2 |
  | 1.0   | 64 (unclipped) | 256 (unclipped) |

  At `kernel_size=8`, every `clip_limit` from `0.002` to `0.01` rounds down to the same
  `clim=1` — the whole range you asked to sweep is a **single degenerate plateau**;
  `clahe_grid_comparison_kernel8.png` confirms this visually (those 5 columns are
  indistinguishable from each other), then jumps straight to the fully-saturated
  `clip=1.0` column. The next `clim` step up (`clim=2`) only starts at
  `clip_limit >= 1/64 ≈ 0.0156` — outside this grid, but consistent with the earlier
  wider sweep, where `clim` was 1/1/1/2/3 at `clip=0.01/0.02/0.03/0.04/0.05` and visible
  grain onset tracked that `clim=1 -> 2` transition around `clip=0.04`.
  At `kernel_size=16`, the larger `kernel_elements` gives finer-grained `clim` steps —
  this grid actually resolves a real transition (`clim` 1 -> 2 between `clip=0.006` and
  `0.008`), and `clahe_grid_comparison_kernel16.png` shows a correspondingly mild but
  visible grain increase starting at `clip=0.008`.

**Final result** (`results/exp1/eda/v6/clahe_grid_search/`:
`clahe_grid_comparison_kernel{8,16}.png`, `image_stats.json`,
`intensity_histogram_comparison_{positive,negative}_kernel{8,16}.png`): 0/10 empty-mask
failures (no `center_crop` fallbacks triggered).
- **`kernel_size=8`:** `clip_limit` anywhere in `[0.002, 0.01]` gives the same result
  (see `clim` finding above) — a clean, modest local-contrast boost with no visible
  grain. Since the whole tested range is equivalent, **`clip_limit=0.01`** is the
  natural default (simplest to reason about, matches the "at most 0.01" ceiling you
  set), not because it's better than `0.002` — it isn't, measurably.
- **`kernel_size=16`:** `clip_limit=0.002`-`0.006` are clean; `0.008`-`0.01` show the
  beginning of mild grain. **`clip_limit=0.006`** is the recommended default — the
  largest value still on the `clim=1` plateau.
Global intensity `std` across the grid was roughly flat regardless of `clip_limit`
(consistent with Experiment 003's earlier finding that it doesn't discriminate grain) —
visual inspection plus the `clim` quantization math were the deciding factors, not
`image_stats.json`.
**Confirmed default: `kernel_size=16`, `clip_limit=0.01`** — your final call, taken over
both the `kernel_size=8`/`clip_limit=0.01` and `kernel_size=16`/`clip_limit=0.006`
candidates above. Note this sits at `clim=2` (not the more conservative `clim=1`
plateau capped at `clip_limit=0.006`), i.e. it's just past the point where mild grain
starts appearing in `clahe_grid_comparison_kernel16.png` — a deliberate choice of
slightly more local contrast over the strictly-cleanest option. `tda_chexpr.
preprocessing.DEFAULT_CLAHE_PARAMS` / `NORMALIZATION_VARIANTS` updated accordingly.

## Observations

- Border text/markers still occasionally survive at the very edge of the lung-mask
  bbox (e.g. an "L" or "PORT/AP" marker in a corner) — expected, not a regression:
  Experiment 002 already established bbox-only cropping (no hard masking) as a
  deliberate tradeoff, and the 5% margin can include a sliver of a corner marker.
- The pad-to-square attempt is a useful cautionary example for this project generally:
  an intuition borrowed from general image-processing practice ("don't warp aspect
  ratio") turned out to be the wrong default for a sublevel-set persistent-homology
  pipeline, where fabricating pixels is a strictly worse failure mode than a
  topology-preserving geometric transform.
- `clip_limit` grid points at/above `1.0` are not distinct experimental conditions —
  they're all the same "unclipped AHE" result by construction (see Results).
- `clip_limit`'s effect is quantized by `kernel_size` (`clim = int(clip_limit *
  kernel_size**2)`, clipped to a minimum of 1) — a fine sweep can silently be a no-op
  across its whole range if every point rounds to the same integer `clim`. Any future
  `clip_limit` sweep should either pick points that straddle a `clim` integer boundary
  for the `kernel_size` in use, or compute `clim` upfront to check the sweep will
  actually differentiate.

## Next Steps

- Check full-dataset PSPNet inference time before committing to a batch run over all
  1,189 clean-negatives-cohort images (still untested beyond this 10-image sample).
- Filtration (Pipeline step 4) is next — see Experiment 004.

---

# Experiment 004

## Goal

Implement the filtration pipeline stage (Pipeline step 4), starting with the baseline
method named in `experiments.md`: classical cubical persistence
(`gtda.homology.CubicalPersistence`, sublevel-set filtration on grayscale pixel
intensities), applied directly to the postprocessing pipeline's output with no
denoising step in between.

**Course correction**: an earlier pass of this experiment swept a Gaussian-smoothing
`sigma` parameter before filtration, as a candidate noise-reduction step. You asked to
remove that entirely — no Gaussian blur anywhere in this experiment or its code — so
persistence is now computed directly on the CLAHE output, and the results below
reflect that.

## Dataset

`data/exp1/preprocessing_sample.csv` (10 rows, unchanged from Experiment 003), run
through the finalized pipeline: `Raw -> lung-mask crop -> direct resize to 224x224 ->
CLAHE (kernel_size=16, clip_limit=0.01)`.

## Parameters

- `CubicalPersistence` itself has almost no filtration-shaping parameters
  (`homology_dimensions=(0, 1)` fixed for this experiment, `coeff=2` default) — the
  filtration function is just the image's own pixel intensities, with no
  pre-filtration denoising applied.
- `tda_chexpr.filtration.apply_direction(image, direction)` — `"sublevel"`
  (passthrough, dark structures born first) or `"superlevel"` (`1.0 - image`, bright
  structures born first) — the only swept parameter.
- Sweep: 10 images x 2 directions = 20 combinations for the quantitative stats;
  persistence-diagram plots limited to 2 representative images (1 positive, 1
  negative), per your earlier direction, to keep the figures readable. A new
  `postprocessing_before_after.png` shows all 10 sample images, Raw vs. Postprocessed.

## Results

**Cubical persistence on the current pipeline output is extremely noisy, and this
experiment does not attempt to reduce that noise.** One representative image produced
8,607 birth-death points (3,239 H0 + 5,368 H1); mean across all 10 images is
8,511.8 (sublevel) / 8,451.7 (superlevel) points (`filtration_stats.json`). Most of
these are short-persistence points hugging the diagonal in
`persistence_diagram_direction_comparison.png` — pixel-level noise, not real
anatomical structure (partly inherited from CLAHE's own local-contrast grain, see
Experiment 003).

**Filtration direction (sublevel vs superlevel) changes the diagram modestly**:
sublevel gives more points than superlevel for both classes (mean 8,511.8 vs 8,451.7
across the full 10-image sweep) — expected, since dark structures (lung fields, air)
and bright structures (ribs, mediastinum, any pleural line) are genuinely different
image content, not just a relabeling of the same features.

## Observations

- No denoising is applied before filtration in this experiment — the full,
  noisy point cloud is what feeds into any downstream vectorization step. If noise
  reduction turns out to be needed, it should be revisited as a deliberate, separate
  decision rather than folded into the filtration step.
- Direction is kept as a live experimental axis (not resolved to one choice here) —
  both sublevel and superlevel may carry independent diagnostic signal for
  pneumothorax (air lucency vs pleural line), consistent with `experiments.md`
  treating filtration choice as an experimental variable rather than a fixed step.
- This experiment stops at the persistence diagram — no vectorization
  (`gtda.diagrams`: Persistence Images/Landscapes/Betti Curves/etc., Pipeline step 5)
  yet, and no other filtration methods (height/radial/distance-transform) yet.

## Next Steps

- Decide whether to carry both filtration directions forward as separate feature
  sets, or pick one, before moving to vectorization.
- Vectorization of persistence diagrams (Pipeline step 5) is next — compare
  Persistence Images, Landscapes, Betti Curves, Persistence Entropy, Silhouettes via
  `gtda.diagrams`. The very high, un-denoised point counts here make vectorization
  method choice (and whether it's noise-robust) especially relevant.
- Other filtration methods (height/eccentricity, Vietoris-Rips, distance-transform)
  remain unexplored candidates per `experiments.md`.
- Still no full-dataset batch run of any pipeline stage — everything so far is the
  10-image sample.

---

# Experiment 005

## Goal

Explore denoising strategies for the noisy cubical persistence diagrams found in
Experiment 004 (~8,600 points/image, un-denoised), per
`human_notes/DeNoisingForPersitenceDiagrams.md`. Compare three methods against the raw
baseline: Anscombe transform + denoise + inverse (image-space), persistence
thresholding (diagram-space, fixed cutoff), and confidence sets / bottleneck bootstrap
(diagram-space, data-driven cutoff). **DTM filtration was dropped from scope** per your
instruction — verified hands-on that `gtda` only ships DTM via
`gtda.homology.WeightedRipsPersistence(weights="DTM")`, a point-cloud/Vietoris-Rips
method requiring the image to be converted to a point cloud first, not directly
comparable to our cubical pipeline. No cubical-grid DTM exists out of the box in
`gtda`.

## Dataset

`data/exp1/preprocessing_sample.csv` (same 10 rows), through
`Raw -> lung-mask crop -> resize to 224x224`, then two branches:
- `baseline = CLAHE(cropped)` (identical to Experiment 004 — the shared reference for
  thresholding and confidence-set methods).
- `anscombe_image = CLAHE(denoise_anscombe(cropped))` — denoising applied **before**
  CLAHE, since CLAHE's local contrast reshaping breaks the Poisson-noise assumption
  Anscombe relies on.

Sublevel filtration only (scope decision, not asked as a separate question — dark
air-lucency topology is the more pneumothorax-relevant of the two directions, and
doubling all three methods across both directions added limited insight for this
pass).

## Parameters

- **Anscombe + denoise + inverse** (`tda_chexpr.denoising.denoise_anscombe`): forward
  `2*sqrt(x + 3/8)`, `skimage.restoration.denoise_wavelet(rescale_sigma=True)`, inverse
  `(y/2)**2 - 3/8` (algebraic inverse, not the exact-unbiased Makitalo-Foi inverse — a
  documented simplification). Initially implemented with `denoise_nl_means` since
  `pywt` (required by `denoise_wavelet` and by skimage's own `estimate_sigma`) was not
  installed; switched to `denoise_wavelet` after you added `pywavelets` as a project
  dependency (`pyproject.toml`).
- **Persistence thresholding** (`tda_chexpr.filtration.threshold_diagram`, wrapping
  `gtda.diagrams.Filtering`): fixed `epsilon in [0.02, 0.05, 0.1]`, applied to the raw
  baseline diagram.
- **Confidence sets / bottleneck bootstrap**
  (`tda_chexpr.denoising.bottleneck_confidence_cutoff`): since a single image is not an
  i.i.d. point-cloud sample (unlike Fasy et al.'s original setting), bootstrap
  replicates are generated by a **spatial block bootstrap**
  (`tda_chexpr.denoising.block_bootstrap_image`) — the image is partitioned into
  non-overlapping 16x16 patches (matching CLAHE's kernel size) and reassembled by
  sampling patches with replacement. `n_bootstrap=20` replicates per image, bottleneck
  distance from each replicate's diagram to the original via
  `gtda.diagrams.PairwiseDistance(metric="bottleneck")` (default `delta=0.01`,
  approximate algorithm), `c_n` = the 0.95 empirical quantile of those distances,
  final cutoff `epsilon = 2 * c_n` per the Fasy et al. convention. `random_state=42`.

**Important caveat, logged explicitly**: CheXpert-small images are pre-processed 8-bit
JPEGs, not raw detector counts — true Poisson statistics are already only an
approximation in this dataset (JPEG compression and prior windowing/leveling reshape
the real sensor noise). The Anscombe transform here is applied directly to the
[0, 1]-normalized intensity as a heuristic VST, not a calibrated photon-count
transform.

## Results

Mean values across all 10 images (sublevel, `denoising_stats.json`):

| Method | mean `n_points` | reduction vs. raw |
|---|---|---|
| Raw baseline | 8,511.8 | — |
| Anscombe + wavelet denoise | 6,937.1 | 18.5% |
| Threshold (eps=0.02) | 4,140.9 | 51.4% |
| Threshold (eps=0.05) | 1,621.1 | 81.0% |
| Threshold (eps=0.1) | 449.1 | 94.7% |
| Confidence-set (bottleneck bootstrap) | 4.1 | ~99.95% |

`c_n` ranged 0.231-0.382 across the 10 images (mean 0.291), giving an effective
threshold `epsilon = 2*c_n` in roughly [0.46, 0.76] — far more aggressive than any of
the fixed thresholds tested, since it exceeds most images' `max_persistence` from
Experiment 004 (~0.5-0.6). Visually (`persistence_diagram_method_comparison.png`), the
raw and Anscombe-denoised columns look similar in density (the wavelet step trims the
near-diagonal cloud only modestly); the fixed threshold at eps=0.05 visibly thins the
cloud while keeping a substantial mid-persistence band; the confidence-set column
retains only 2-7 of the very highest-persistence points per image.

**Anscombe+denoise visibly smooths pixel-level speckle before CLAHE**
(`anscombe_denoise_before_after.png`) without erasing rib/vessel edges, but the
resulting ~18.5% point-count reduction is modest compared to the diagram-space
methods — most of the raw diagram's noise apparently survives CLAHE's own local
contrast amplification even after image-space denoising.

## Observations

- **The confidence-set bootstrap, as specified here, is extremely aggressive** — it
  collapses each diagram to a handful of points (2-7), which is likely too sparse to
  be useful for downstream vectorization on its own, though it may be a legitimate way
  to identify the small number of "statistically certain" topological features per
  image. The block-bootstrap's patch-shuffling (which destroys global anatomical
  layout while preserving local texture) may itself inflate estimated bottleneck
  distances beyond what a true noise-only perturbation would produce — this is a
  candidate explanation worth revisiting if the cutoff seems too strict for practical
  use.
- **Fixed persistence thresholding is the most controllable of the three**: a
  continuous dial (`epsilon`) trading off noise removal against retained
  low-persistence structure, with `eps=0.05` retaining a visually reasonable amount of
  detail while cutting ~81% of points.
- **Anscombe+wavelet denoising is the mildest intervention** and the only one that
  changes the input image itself rather than post-processing the diagram — useful if
  the denoised image also needs to be visually/perceptually reasonable (e.g. for
  comparison against a separate CNN baseline), but it leaves far more diagram noise
  than either diagram-space method.
- No single "best" method is picked here — each targets a different point in the
  noise/detail trade-off, and the right choice likely depends on the downstream
  vectorization method (Pipeline step 5), which is still unexplored.
- DTM filtration remains unexplored (dropped this pass, see Goal).

## Next Steps

- Decide whether any of these three methods (or a combination, e.g. Anscombe-denoise
  the image *and* threshold the resulting diagram) should become the pipeline default
  before vectorization, or whether to carry multiple forward as parallel candidates.
- If the confidence-set cutoff continues to look too aggressive, consider tuning
  `alpha`, `n_bootstrap`, or `block_size`, or trying a resampling scheme that
  preserves more global structure than block-shuffling.
- Vectorization of persistence diagrams (Pipeline step 5) is next — compare
  Persistence Images, Landscapes, Betti Curves, Persistence Entropy, Silhouettes via
  `gtda.diagrams`, now with denoised/thresholded diagrams as candidate inputs.
- Still no full-dataset batch run of any pipeline stage — everything so far is the
  10-image sample.

---

# Experiment 006

## Goal

Halt Experiment 1 and correct a cohort-selection error found against the four target
criteria (target Pneumothorax; positives have no comorbidity; negatives are a pure
false; one row per patient, earliest qualifying study). Full writeup in
`cohort_validation.md` (repo root).

## Dataset

Rebuilt from `kaggle/train.csv` / `kaggle/valid.csv` via corrected
`src/preprocessing/exp1/build_cohort.py` + `split_cohort.py` + `run_eda.py`. Outputs
under `data/exp1/v2_corrected_cohort/` (old `data/exp1/*.csv` left in place,
superseded, not deleted).

## Parameters

Same view/device filters as Experiment 001 (`AP/PA == "AP"`, `Support Devices == 0.0`),
plus corrected comorbidity filtering (`tda_chexpr.cohort.filter_no_comorbidity`) applied
*before* per-patient dedup, and dedup (`select_studies`) now tie-broken by lowest `view`
number in addition to lowest `study_number`. Split: `stratified_split`, 80/20,
`random_state=42`, unchanged.

## Results

Three bugs found and fixed in `tda_chexpr/cohort.py` (see `cohort_validation.md` for
full detail):

1. Positive rows were never filtered for comorbidity (`filter_clean_negatives` kept all
   `Pneumothorax == 1.0` unconditionally) — violated the "no comorbidity" criterion for
   positives. Fixed with a new `filter_no_comorbidity()`, applied to both labels
   (asymmetric rule: positives require `comorbidity_count == 0`, negatives require
   `No Finding == 1.0`).
2. Comorbidity filtering ran *after* per-patient dedup, so "earliest qualifying study"
   didn't account for it. Fixed by moving comorbidity computation/filtering before
   `select_studies()`.
3. `select_studies(mode="first_qualifying")` deduped by `study_number` only, so a study
   with 2+ qualifying views (e.g. `view1` + `view2`, both AP frontal) left 2 rows for one
   patient — present in both the old full cohort (33 patients) and old clean-negatives
   cohort (23 patients). Fixed by tie-breaking on lowest `view` number too; verified
   `n_records == n_patients` in every output now.

Corrected counts (`data/exp1/v2_corrected_cohort/`):

| Dataset | Rows / patients | Pos / neg |
|---|---|---|
| Reference train | 2,266 / 2,266 | 848 / 1,418 |
| Reference valid | 77 / 77 | 1 / 76 |
| Clean train (primary) | 575 / 575 | 253 / 322 |
| Clean valid | 12 / 12 | 0 / 12 |
| train_split | 460 / 460 | 202 / 258 |
| test_split | 115 / 115 | 51 / 64 |

EDA (all 6 datasets) at `results/exp1/eda/v9/`.

## Observations

- Enforcing "no comorbidity" on positives removes 71% of the previously-kept positive
  class (869 → 253 in the train cohort) — most Pneumothorax-positive studies in this
  dataset also have at least one other confirmed finding. The corrected primary cohort
  is roughly half the size of the old one but much closer to balanced (was 73%/27%
  pos/neg, now 44%/56%).
- The clean valid cohort now has 0 positive cases (its single prior positive case had a
  comorbidity) — doesn't block anything since Experiment 1 already carves its own split
  from train, but the valid clean cohort is negative-only now.
- Verified: zero duplicate patients in any output file, zero patient overlap between
  train_split/test_split, `len(train_split) + len(test_split) == len(clean train
  cohort)`.

## Next Steps

- Regenerate `preprocessing_sample.csv` from the corrected `train_split.csv` (the
  current one was drawn from the pre-correction split and is stale).
- Re-validate Experiments 002–005 (ROI crop, CLAHE params, filtration, denoising) against
  the corrected, smaller, more-balanced cohort before resuming Experiment 1 — parameter
  choices made against the old sample may not transfer unchanged.
- Update `experiments.md` §Experiment matrix / downstream sections if the corrected
  cohort size changes any planned batch-run sizing assumptions.

---

# Experiment 007

## Goal

Generate a physical, versioned processed-image dataset (segmentation → crop → CLAHE →
resize) from the corrected cohort's full train and test splits, written to
`kaggle/processed/` mirroring `kaggle/train/`'s own layout for easy lookup — requested
independently of, and prior to, the still-pending Experiment 002–005 re-validation
against the corrected cohort (see Experiment 006's Next Steps). CLAHE here uses a
deliberately more conservative `clip_limit` than Experiment 003's confirmed default, at
your explicit request. This is also the first full-dataset (575 images) run of the
PSPNet segmentation step — Experiments 002/003 both explicitly flagged full-dataset
timing as untested beyond an 8–10 image sample.

## Dataset

Both corrected-cohort splits in full: `data/exp1/v2_corrected_cohort/
pneumothorax_train_split.csv` (460 rows) and `pneumothorax_test_split.csv` (115 rows) —
575 rows total. Verified directly before running: zero path overlap between the two
files, every `Path` value in both is prefixed `train/` (no `valid/` rows), every row is
`Frontal/Lateral == "Frontal"`, and all 575 source JPEGs exist on disk.

## Parameters

- `tda_chexpr.roi.apply_roi_crop(image, "lung_mask", margin_frac=0.05, threshold=0.5,
  size=224)` — same PSPNet lung-mask bbox crop + direct resize to 224x224 (aspect-ratio
  warped, no padding) finalized in Experiment 003. Unchanged.
- `tda_chexpr.preprocessing.apply_normalization(image, "clahe", clip_limit=0.002,
  kernel_size=16, nbins=256)` — **deliberate deviation from Experiment 003's confirmed
  default (`clip_limit=0.01`)**, at your request for a "very conservative" CLAHE for
  this export. Verified directly against the installed skimage 0.26.0 source
  (`skimage/exposure/_adapthist.py`): `clim = int(clip(clip_limit * kernel_size**2, 1,
  None))`. At `kernel_size=16` (256 cells), both `clip_limit=0.001` and `0.002` floor to
  `clim=1` — byte-identical output either way, and the most conservative non-zero
  `clim` this `kernel_size` supports (consistent with Experiment 003's own `clim`
  quantization table, which already noted `clip_limit` in `[0.002, 0.006]` all sit on
  this same `clim=1` plateau at `kernel_size=16`). `0.002` was recorded as the nominal
  value since it's further from the `clim=1`/`clim=2` boundary at `0.0078` than `0.001`.
- Output: `skimage.util.img_as_ubyte` on the CLAHE `[0,1]` float64 output → 8-bit
  grayscale PNG (lossless), written to `kaggle/processed/<source Path, .jpg -> .png>`
  (e.g. `kaggle/processed/train/patient18455/study3/view1_frontal.png`), directory
  structure otherwise identical to `kaggle/train/`.
- New scripts: `src/preprocessing/exp1/preprocess_full_dataset.py` (batch pipeline +
  manifest, ~9s-class PSPNet singleton load, per-row try/except so one failure can't
  abort the batch) and `src/preprocessing/exp1/eda_processed_dataset.py` (EDA/report
  over the manifest + saved images only — no PSPNet/torch import, freely re-runnable).

## Results

- **575/575 rows processed successfully, 0 failures** (`results/exp1/preprocessing/v1/
  manifest.csv`) — no empty lung masks, no non-grayscale loads, no save errors, across
  both splits.
- **First full-dataset PSPNet timing measurement**: model load 0.6s (weights already
  cached locally from a prior run this session — Experiment 002/003's ~9s figure was a
  cold load), batch 721.1s for 575 images (**1.25s/image average** on CPU, no GPU used
  despite one being available in this environment). Resolves the "check full-dataset
  PSPNet timing" item open since Experiment 002.
- All 575 output images confirmed exactly 224x224, 8-bit grayscale (manifest's
  `processed_height`/`processed_width` columns, and a direct spot-check via PIL).
- EDA (`results/exp1/eda/v10/`): per-split `summary.json` (dataset summary + added
  preprocessing status/duration stats), `image_stats.json` and intensity-histogram
  comparison plots (raw vs. processed) over a 15-positive/15-negative stratified sample
  (`random_state=42`) of successfully-processed rows.
- Visual spot-check (one positive example, full raw vs. processed side-by-side):
  lung-mask crop correctly isolates the lung field and removes most border
  annotations; CLAHE's effect is visually subtle, consistent with the conservative
  `clim=1` setting, not a strong local-contrast boost.

## Observations

- Because the `clim=1` plateau (Experiment 003's own finding) already covers
  `clip_limit` values well below `0.01`, requesting an even more conservative value
  than the Experiment 003 default did not require any new code — only a different
  `clip_limit` argument to the already-existing `apply_normalization` call.
- 0/575 segmentation failures at full-dataset scale is a stronger result than the
  8-image sample in Experiment 002 could show on its own — the "no support devices"
  cohort filter (excluding heavy tube/line clutter) likely continues to help PSPNet
  generalize here, as hypothesized in Experiment 002.
- The manifest's `duration_sec` (mean ~1.25s/image, both splits) is measured with the
  PSPNet model already resident in memory — it does not include the one-time model
  load, which is reported separately and was fast here only because the weights file
  was already cached from an earlier step in this session (a cold environment would
  see the ~9s load Experiment 002 measured, once, plus this same per-image rate).
- This export uses a different CLAHE `clip_limit` (0.002) than Experiment 003's
  confirmed pipeline default (0.01) — anything built on top of `kaggle/processed/`
  going forward should treat it as the conservative-CLAHE variant, not the Experiment
  1 pipeline's eventual default output, until/unless the two are explicitly reconciled.

## Next Steps

- This does not resolve Experiment 006's still-open item: `preprocessing_sample.csv`
  regeneration and re-validation of Experiments 002–005 against the corrected cohort at
  the Experiment 003 *default* `clip_limit=0.01` remain pending before Experiment 1
  proper resumes.
- If GPU-accelerated PSPNet inference is ever needed (e.g. for a much larger future
  batch), `tda_chexpr.segmentation.predict_lung_mask` currently runs on CPU
  unconditionally — moving the model/tensor to `cuda` is a candidate optimization, not
  done here since 1.25s/image was fast enough for 575 images.
- Neither new script supports resuming a partially-completed run (unlike
  `GenerateEmbeddings`'s `--resume` flag) — acceptable at this dataset size, but worth
  adding if `kaggle/processed/` is ever regenerated over a much larger cohort.
- Companion vision-model embeddings (SigLIP/RAD-DINO/MedGemma) for this processed
  dataset now live at `data/embeddings/processed/` — see `logs/exp2_embeddings_log.md`
  Experiment 008.
