# Experiment 1 Log

- **Current experiment:** Experiment 004 (cubical persistence, no smoothing/denoising,
  filtration direction sweep, persistence diagrams) is done — both sublevel/superlevel
  directions kept as live axes; raw point counts are noisy (~8,600/image) and
  un-denoised by design. Vectorization not started.
- **Preprocessing pipeline:** `build_cohort.py` (filter + dedup + comorbidity_count/
  is_clean_negative + clean-negatives variant) -> `split_cohort.py` (train/test split
  from the clean-negatives cohort) -> `select_preprocessing_sample.py` (sample
  selection) -> ROI crop (`tda_chexpr.roi.apply_roi_crop`, methods `none`/
  `center_crop`/`lung_mask`; `lung_mask` now defaults to `size=224`, direct resize, no
  padding) -> normalization (`tda_chexpr.preprocessing.apply_normalization`, methods
  `none`/`he`/`clahe`, `clahe` default `kernel_size=16`/`clip_limit=0.01`) ->
  filtration (`tda_chexpr.filtration.compute_persistence_diagram`, applied directly to
  the CLAHE output with no denoising step; `apply_direction` is the only preprocessing
  knob). No full-dataset batch run of any stage yet — that lands with
  vectorization/classification.
- **Dataset version:** primary/active: `data/exp1/pneumothorax_cohort_{train,valid}_clean_negatives.csv`
  -> `pneumothorax_{train,test}_split.csv`. Full (unfiltered) cohort
  `pneumothorax_cohort_{train,valid}.csv` retained on disk for a future clean-vs-full
  comparison, not currently used by the split. EDA'd at `results/exp1/eda/v1/` (all six
  CSVs). Preprocessing/ROI-crop comparison sample: `data/exp1/preprocessing_sample.csv`
  (10 rows, 5 positive / 5 negative, drawn from `pneumothorax_train_split.csv`; bumped
  from 8 rows in Experiment 003).
- **Model version:** `torchxrayvision` PSPNet (`pspnet_chestxray_best_model_4.pth`,
  ChestX-Det-trained, cached at `~/.torchxrayvision/models_data/`) for lung
  segmentation. No classification model yet.
- **Random seed:** 42 (train/test split, and preprocessing sample selection).
- **Feature extraction method:** not yet implemented (filtration/vectorization pending).
- **Parameters:** see Experiment 001-004 below.
- **Evaluation metric:** AUROC (planned primary), accuracy/F1 (planned secondary) — not
  yet computed, no model trained.
- **Current status:** clean-negatives cohort + split built and EDA'd; ROI-crop +
  normalization pipeline finalized (lung-mask crop, direct resize to 224x224, CLAHE
  `kernel_size=16`, `clip_limit=0.01`); cubical-persistence filtration implemented and
  explored (no smoothing/denoising, direction kept as a live axis); vectorization not
  started.

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
