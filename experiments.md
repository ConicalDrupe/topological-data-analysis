# Experiments

This project runs three tracks of topological data analysis (TDA) experiments against the
CheXpert-v1.0-small chest x-ray dataset (see [`CLAUDE.md`](./CLAUDE.md) for how the data is
laid out and accessed). [`giotto-tda`](https://giotto-ai.github.io/gtda-docs/) is the
primary TDA library for all new work described here.

1. **Feature extraction & classification** — use persistent homology as a feature
   extractor for a binary classification task (starting with Pneumothorax), comparing
   image normalization, filtration, and vectorization choices.
2. **Embedding shape analysis** — investigate the topology of a pre-computed MedGemma
   image-embedding point cloud, and try to assign each point a representative/cluster
   label.
3. **Disease progression** — use patients with multiple studies to see whether
   topological change between sequential x-rays can flag progression a human read might
   miss.

Each section below lists a pipeline, not a fixed recipe — the intent is to run a matrix
of variants and compare, not to settle on one method up front.

---

## Experiment 1 — TDA Feature Extraction for Pneumothorax Classification

**Goal:** build persistence-diagram-derived features from chest x-rays and use them
(alone, or alongside other features) to classify Pneumothorax presence/absence, comparing
different normalization, filtration, and vectorization choices.

**Label handling (decided):** CheXpert labels Pneumothorax as `1.0` (positive), `0.0`
(negative), `-1.0` (uncertain), or blank (unmentioned). This experiment uses only rows
with a definite `0.0`/`1.0` label — `-1.0` and blank rows are dropped (effectively the
U-Ignore policy). No uncertainty-mapping (U-Ones/U-Zeros) is used for this baseline.

### Pipeline

1. **Cohort/label selection** — build the Experiment 1 baseline cohort as follows:
   1. Filter `train.csv`/`valid.csv` to rows where `Pneumothorax` is exactly `0.0` or
      `1.0` (drop `-1.0` and blank rows).
   2. **View narrowing — AP only**: keep `AP/PA == "AP"` rows only; drop `PA` and
      lateral. AP is the preferred view for diagnosing pneumothorax (clinical
      rationale), which is a stricter cut than "frontal only" — it also drops PA, not
      just lateral. (`AP/PA` is only ever set on frontal rows — lateral rows always have
      it blank — so this implies frontal-only as a side effect; there's no separate
      frontal-only step.)
   3. **No support devices**: keep only rows with `Support Devices == 0.0` (confirmed
      absence). `-1.0` (uncertain) and blank (unmentioned) are excluded too, not treated
      as "no device" — support devices (chest tubes, lines, etc.) visible on the film
      could confound the topological features this experiment extracts, so the
      strictest available label is used.
   4. **Per-patient dedup**: for patients with multiple studies, keep only the patient's
      **earliest (first) qualifying study** — the first study, in study-number order,
      that satisfies **all three** filters above *jointly* (0/1 Pneumothorax label, AP
      view, and confirmed no support devices, all in the same study). This yields at
      most one study per patient for this cohort. (Note: this is an
      Experiment-1-specific rule — Experiment 3 deliberately keeps *all* qualifying
      studies per patient; see Shared Infrastructure below.)
   5. Remap `Path` values to on-disk paths (strip the `CheXpert-v1.0-small/` prefix — see
      `CLAUDE.md`).

   **Resulting cohort size (verified against the real CSVs):** train 2,299 rows / 2,266
   patients (Pneumothorax balance: 1,430 negative / 869 positive); valid 77 rows / 77
   patients (76 negative / 1 positive).

   **Comorbidity confound check and "clean negatives" cohort (decided):** many
   Pneumothorax-negative rows still have *other* confirmed pathologies (Cardiomegaly,
   Lung Opacity, Effusion, etc.). Since cubical persistence is sensitive to any
   structural change in the lung field, a "negative" with a different disease could
   produce a topological signature that looks similarly abnormal to a true pneumothorax
   case — muddying the target-vs-healthy contrast. Two columns are added to every cohort
   CSV to make this visible and actionable:
   - **`comorbidity_count`** — how many of the 11 *other* pathology columns
     (all `PATHOLOGY_COLUMNS` minus `Pneumothorax`, `No Finding`, `Support Devices`) are
     confirmed-positive (`== 1.0`) for that row. Uncertain (`-1.0`) values are not
     counted — the strict/confirmed-only convention used throughout this cohort.
   - **`is_clean_negative`** — `No Finding == 1.0`: the labeler's own explicit "nothing
     found at all" signal, chosen over the looser `comorbidity_count == 0` (which still
     allows unmentioned/uncertain findings on other columns) as the strictest available
     "truly healthy" indicator.

   A **clean-negatives cohort variant** is then derived: keep every `Pneumothorax == 1.0`
   row (positives are kept regardless of their own comorbidities — only the negative
   side of the contrast is being cleaned up), but restrict `Pneumothorax == 0.0` rows to
   `is_clean_negative == True` only. Verified counts: train 869 positive + 320 clean
   negative = **1,189 rows** (`pneumothorax_cohort_train_clean_negatives.csv`); valid 1
   positive + 12 clean negative = **13 rows**
   (`pneumothorax_cohort_valid_clean_negatives.csv`).

   **This clean-negatives cohort is Experiment 1's primary/active dataset** — see
   Train/test split below. The original, unfiltered cohort (with both new columns) is
   still written and kept on disk as "the larger dataset," to revisit for comparison
   once the classification step exists (does restricting to clean negatives actually
   change measured AUROC, or was the comorbidity confound not material in practice?).

2. **ROI cropping** — crop each image to its region of interest *before*
   normalization, to remove border text, lateral-view markers, and other burned-in
   annotations that could otherwise be picked up as spurious topological features:
   - **Baseline (implemented):** `torchxrayvision`'s own deterministic, no-inference
     `XRayCenterCrop` (center-crop to a square using `min(height, width)`) +
     `XRayResizer` (resize to a fixed 224×224). Tested on the Experiment 1
     representative sample (`logs/exp1_log.md`, Experiment 002) — found **not** to
     reliably remove border text/markers, since it only trims the width axis and most
     annotations sit within the preserved height.
   - **Finalized:** PSPNet-based lung segmentation (`torchxrayvision`'s
     `xrv.baseline_models.chestx_det.PSPNet`), using the union of the Left Lung/Right
     Lung mask channels to compute a bounding box, cropped from the raw (not
     center-cropped) image, keeping all pixel values inside the box (heart/
     mediastinum included) — no pixel masking outside the lung silhouette, since a
     hard mask edge would introduce an artificial intensity discontinuity that
     sublevel-set cubical persistence (see step 4 below) would pick up as a false
     topological feature. Adopted after Experiment 002 (0/8 segmentation failures,
     substantially better at removing border text/markers than the center-crop
     baseline). The crop is then resized **directly** to 224×224 (aspect ratio
     warped, no pad-to-square/crop-to-square step) — a pad-to-square attempt
     (edge-replication) was tried and rejected in Experiment 003 after it fabricated
     repeated-texture stripe artifacts on high-aspect-ratio crops; a direct affine
     resize is provably topology-safe (a homeomorphism of the image domain) and
     doesn't lose the lung-periphery content a crop-to-square would. See
     `logs/exp1_log.md`, Experiment 003, and `tda_chexpr.roi.apply_roi_crop`.
3. **Normalization variants** (compare against each other and against no normalization):
   - Histogram Equalization (HE)
   - Adaptive Gamma Correction (AGC) - We will skip this for now.
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) — grid-searched
     `clip_limit` at `kernel_size=8` in `logs/exp1_log.md` Experiment 003;
     `clip_limit=0.02` recommended (visible local-contrast gain without the grain/
     speckle noise seen from `clip_limit=0.03` upward), pending your confirmation.
4. **Filtration methods:**
   - **Baseline:** classical cubical persistence (sublevel-set filtration directly on
     grayscale pixel intensities) — `gtda.homology.CubicalPersistence`.
   - **Candidates to experiment with:** height/eccentricity filtration, Vietoris-Rips on
     downsampled pixel coordinates or superpixel/keypoint coordinates, lower-star
     filtration on a distance transform (e.g. from a lung mask or edge map). Treat the
     filtration choice itself as an experimental variable, not a fixed step.
5. **Vectorization of persistence diagrams** — compare multiple representations via
   `gtda.diagrams`: Persistence Images, Persistence Landscapes, Betti Curves, Persistence
   Entropy, Silhouettes. No single method is assumed best; this is itself an experimental
   axis.
6. **Classification** — feed vectorized features into classical ML models (logistic
   regression, random forest, SVM, gradient boosting) rather than a deep model, since the
   point is to evaluate the TDA features themselves.
7. **Evaluation** — AUROC (the standard CheXpert benchmark metric), plus accuracy/F1 as
   secondary metrics, computed on **our own test split** (see Train/test split below),
   which is the primary evaluation set for this experiment. The filtered `valid.csv`
   cohort (77 rows, 1 positive case) is reported only as a secondary/supplementary check,
   never as the primary metric — it's too small and too imbalanced to trust alone.

### Train/test split (decided)

CheXpert's own `valid.csv` is the standard held-out split for this dataset, but after the
AP-only + no-support-devices filters above it shrinks to 77 rows with only **1** positive
Pneumothorax case — unusable as a reliable evaluation set on its own. Instead, Experiment
1 carves its own split out of the filtered *train* cohort:

- **Source (updated)**: `pneumothorax_cohort_train_clean_negatives.csv` (1,189 rows),
  not the full unfiltered cohort — Experiment 1 uses the clean-negatives cohort as its
  primary/active dataset (see Pipeline step 1). The full cohort remains on disk, unused
  by the split, for the future clean-vs-full comparison.
- **Method**: stratified 80/20 split by `Pneumothorax`, implemented as a plain per-class
  `pandas` sample (`tda_chexpr.split.stratified_split`) rather than a `scikit-learn`
  dependency.
- **Split by patient, not by row**: `select_studies(mode="first_qualifying")` guarantees
  one *study* per patient, but that study can still contribute more than one frontal AP
  image (33 patients in the original cohort have 2 rows). The split is therefore done on
  unique `patient_id` first, then every row belonging to a chosen patient follows that
  patient into train or test — a naive row-level split was tried first and produced 8
  patients leaking across both splits, which is why this two-step approach is needed.
- **Output** (verified against the real CSVs): `data/exp1/pneumothorax_train_split.csv`
  (953 rows / 932 patients: 698 positive / 255 clean negative) and
  `data/exp1/pneumothorax_test_split.csv` (236 rows / 234 patients: 171 positive / 65
  clean negative), written by `src/preprocessing/exp1/split_cohort.py`.
- The filtered `valid.csv` cohorts (`pneumothorax_cohort_valid.csv` and its
  `_clean_negatives` variant) are still produced and kept as a secondary/supplementary
  check, not the primary evaluation set.

### Experiment matrix

Treat this as a grid: **ROI crop × normalization × filtration × vectorization**, each
combination trained and evaluated independently, with results logged so combinations
can be compared directly (see Open Questions re: how results should be tracked).

---

## Experiment 2 — Topology of the MedGemma Embedding Space

**Goal:** investigate the shape of a point cloud of chest x-ray embeddings (pooled output
of a MedGemma/SigLIP-based vision encoder, supplied as a pre-computed artifact — no
generation pipeline is in scope here), and attempt to label each point with a
representative set/cluster derived from the embedding's topology.

This experiment is explicitly open-ended. The steps below are a starting point, not a
fixed procedure — deviating when something interesting turns up is expected.

### Pipeline

1. **Load embeddings** and join them back to image path / patient ID / CheXpert labels,
   so downstream topological structure can be compared against known pathology.
2. **Dimensionality handling** — embeddings are likely high-dimensional (e.g. 768/1152-d
   depending on the encoder). Distinguish between computing persistence directly from a
   full-dimensional distance matrix versus using PCA/UMAP/t-SNE purely for 2D/3D
   visualization (don't conflate the two).
3. **Persistent homology on the point cloud** — Vietoris-Rips as the default; if the point
   cloud is too large for full VR, consider landmark/witness-complex subsampling.
4. **Representative-set labeling** — the core open question of this experiment: can each
   point be assigned a topologically-derived cluster/representative label? Candidate
   approaches: connected-component structure (H0) at a chosen filtration threshold, H1
   cycle representatives, or a Mapper graph (`kmapper`) built on the embedding point cloud.
   Compare any derived clusters against the true pathology labels to see whether they
   align with known disease categories or reveal something else.
5. **Visualization** — 2D/3D projections colored by derived cluster and by true label.
   TTK (Topology ToolKit) is noted in the README as a possible tool for richer pointcloud
   visualization, but is left as future/optional work — no setup is documented yet.

---

## Experiment 3 — Disease Progression Across Sequential Studies

**Goal:** for patients with multiple studies over time, use TDA-derived features from
each study's x-ray to see whether topological change between studies correlates with
(or precedes) a recorded change in disease label — and whether that signal is something a
human read would otherwise miss.

### Important caveats (read before treating results as meaningful)

- CheXpert patients are organized as `patientXXXXX/study1/`, `study2/`, ... — this really
  does provide multiple images per patient over time, but **`study N` is not guaranteed to
  be evenly spaced or even confirmed chronological**; only per-study `Age` is available,
  no absolute dates.
- CheXpert's 14 pathology labels come from an **automated NLP labeler** run over
  radiology reports, not direct clinical adjudication — expect label noise, especially
  around uncertain (`-1.0`) values.
- Given the above, **success criteria should be modest**: e.g., "does a topological
  change metric between consecutive studies correlate at all with a recorded label
  change?" rather than "this detects clinical progression reliably." Treat this track as
  the highest-risk/most-exploratory of the three.

### Pipeline

1. **Cohort construction** — identify patients with ≥2 studies; start with Pneumothorax
   as the tracked label (reusing Experiment 1's label-policy decision), but keep the
   cohort-extraction logic general enough to key on any label.
2. **Order studies per patient** — by study number (and/or `Age`) to form a pseudo-
   timeline per patient.
3. **Per-image TDA features** — reuse Experiment 1's ROI crop → normalization →
   filtration → vectorization pipeline to produce a feature vector (or full diagram)
   per study image.
4. **Track feature trajectories** — look at how scalar summaries (e.g. total persistence,
   count of features above a threshold, persistence entropy) change study-to-study for
   each patient.
5. **Flag notable change** — define a metric to flag patients with a notable topological
   shift between consecutive studies. Candidates: bottleneck or Wasserstein distance
   between consecutive diagrams, or change-point/trend detection on a scalar summary
   series.
6. **Loose validation** — compare flagged study-pairs against recorded label changes
   (e.g. label flips from `0.0`/blank to `1.0`) as a rough signal, not ground truth,
   given the caveats above.

---

## Shared infrastructure

Reusable across Experiments 1 and 3:

- **Label loading + path remapping** — a utility that reads `train.csv`/`valid.csv`,
  applies the chosen uncertain/blank label policy, and remaps `Path` to the actual
  on-disk location (see `CLAUDE.md` for the exact mismatch).
- **Cohort construction** — a single utility for turning the filtered (0/1-only) label
  table into a per-patient cohort, parameterized by mode rather than hardcoding one rule,
  since Experiments 1 and 3 need opposite dedup behavior:
  - `first_qualifying` (Experiment 1): one row per patient — their earliest study with a
    0/1 Pneumothorax label, frontal view(s) only.
  - `all_ordered` (Experiment 3): all qualifying studies per patient, ordered by study
    number/`Age`, view filtering left as-is (Experiment 3 doesn't narrow to frontal-only
    by default).
  Keeping this as one parameterized utility (rather than two separate ad hoc filters)
  avoids the two experiments silently drifting onto inconsistent cohort logic.
- **View/device filters** — `filter_ap_only` (AP view only) and
  `filter_no_support_devices` (confirmed `Support Devices == 0.0` only) are optional
  flags on the same cohort-construction utility (`ap_only`,
  `require_no_support_devices`), applied before per-patient study selection so
  "qualifying" is a joint condition across every active filter. Experiment 1 uses both;
  Experiment 3 can opt in or out independently.
- **Stratified train/test split** — `tda_chexpr.split.stratified_split` is a generic,
  dependency-free (plain `pandas`) per-class split, usable by any experiment whose
  cohort is too small, or too imbalanced relative to `valid.csv`, to evaluate reliably
  against CheXpert's own held-out split.
- **Basic EDA / count analysis** — run immediately after cohort/split construction,
  before any normalization/filtration/vectorization work, to sanity-check size and
  composition. The full Dataset Summary schema (record/patient/study counts, class
  frequencies for *all* pathology columns, AP/PA, Frontal/Lateral, Sex, Age, missing
  values, duplicate paths) now lives in `CLAUDE.md`'s "Automatic Exploratory Data
  Analysis (EDA)" section, not duplicated here. Each run's artifacts (`summary.json` +
  plots) are written to a versioned `results/<experiment>/eda/vN/` directory
  (never overwritten), and the corresponding experiment log
  (`logs/<experiment>_log.md`) records what was found — see `logs/exp1_log.md` for the
  first entry.
- **ROI crop → normalization → filtration → vectorization pipeline** — built once,
  parameterized by which variant of each stage to use, so the Experiment 1 matrix and
  Experiment 3's per-study feature extraction share the same code path. The ROI crop
  and normalization stages are finalized as of `logs/exp1_log.md` Experiments 002-003
  (`tda_chexpr.roi.apply_roi_crop`, `tda_chexpr.preprocessing.apply_normalization`);
  filtration and vectorization are not yet implemented.

## Open questions / decisions needed

- ~~Uncertain/blank label policy~~ — **resolved for the Experiment 1 baseline**: U-Ignore
  (drop `-1.0`/blank, keep only `0.0`/`1.0`). Revisit if a later target label or
  experiment needs U-Ones/U-Zeros instead.
- ~~Frontal vs. lateral views~~ — **resolved for the Experiment 1 baseline**: superseded
  by the AP-only decision below (AP implies frontal). Lateral (and PA) could be explored
  as separate variants later, but aren't part of the current cohort.
- ~~AP vs. PA view, and support-devices handling~~ — **resolved for the Experiment 1
  baseline**: AP only, and `Support Devices == 0.0` (confirmed absence) only — see
  Pipeline step 1. Note: this strict reading yields a much smaller cohort (train 2,299
  rows) than an earlier ~6,000-row estimate based on a looser "no devices" reading;
  revisit the strictness choice if 2,299 rows proves too limiting for the experiment
  matrix.
- ~~Subsampling/compute strategy~~ — **resolved for Experiment 1**: no subsampling
  needed. The AP-only + no-support-devices cohort is already small (~2.3k rows), so the
  full normalization × filtration × vectorization matrix can run against all of it. This
  may still apply to Experiment 3, which doesn't share this filter set.
- **Experiment 2 embedding format** — exact shape/dimensionality and file format of the
  supplied MedGemma embeddings isn't pinned down yet; confirm when the artifact arrives.
- **Results tracking** — no mechanism is chosen yet for logging experiment-matrix runs
  (e.g. a results CSV/Parquet per run, or a lightweight tracking tool). Decide before
  Experiment 1's matrix grows large enough to lose track of manually.
- **Clean-negatives vs. full cohort comparison** — Experiment 1 now trains/evaluates on
  the clean-negatives cohort only (see Pipeline step 1). Whether restricting negatives to
  `is_clean_negative == True` actually changes measured AUROC (i.e., whether the
  comorbidity confound was material in practice) is deferred until the classification
  step exists and both cohorts can be run side by side.
- ~~ROI cropping method~~ — **resolved for Experiment 1**: the center-crop + resize
  baseline (Pipeline step 2) was tested on the Experiment 1 sample and found not to
  reliably remove border text/markers (`logs/exp1_log.md`, Experiment 002). PSPNet-
  based lung segmentation (bbox-only, no hard masking) was adopted instead (0/8
  failures, substantially better text/marker removal), and its final squaring/resize
  step was pinned down in Experiment 003 (direct resize to 224×224, no padding — see
  Pipeline step 2 above). Still pending: a full-dataset PSPNet timing check before
  committing to a batch run over all 1,189 clean-negatives-cohort images, and final
  confirmation of the recommended `clip_limit=0.02` CLAHE default.
