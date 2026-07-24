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
   2. **Per-patient dedup**: for patients with multiple studies, keep only the patient's
      **earliest (first) qualifying study** — the first study, in study-number order,
      that has a 0/1 Pneumothorax label under the filter above. This yields at most one
      study per patient for this cohort. (Note: this is an Experiment-1-specific rule —
      Experiment 3 deliberately keeps *all* qualifying studies per patient; see Shared
      Infrastructure below.)
   3. **View narrowing**: within the chosen study, keep frontal view(s) only; drop
      lateral. (Lateral could be revisited as a separate variant later — see Open
      Questions.)
   4. Remap `Path` values to on-disk paths (strip the `CheXpert-v1.0-small/` prefix — see
      `CLAUDE.md`).
2. **Normalization variants** (compare against each other and against no normalization):
   - Histogram Equalization (HE)
   - Adaptive Gamma Correction (AGC)
   - Contrast Limited Adaptive Histogram Equalization (CLAHE)
3. **Filtration methods:**
   - **Baseline:** classical cubical persistence (sublevel-set filtration directly on
     grayscale pixel intensities) — `gtda.homology.CubicalPersistence`.
   - **Candidates to experiment with:** height/eccentricity filtration, Vietoris-Rips on
     downsampled pixel coordinates or superpixel/keypoint coordinates, lower-star
     filtration on a distance transform (e.g. from a lung mask or edge map). Treat the
     filtration choice itself as an experimental variable, not a fixed step.
4. **Vectorization of persistence diagrams** — compare multiple representations via
   `gtda.diagrams`: Persistence Images, Persistence Landscapes, Betti Curves, Persistence
   Entropy, Silhouettes. No single method is assumed best; this is itself an experimental
   axis.
5. **Classification** — feed vectorized features into classical ML models (logistic
   regression, random forest, SVM, gradient boosting) rather than a deep model, since the
   point is to evaluate the TDA features themselves.
6. **Evaluation** — AUROC (the standard CheXpert benchmark metric) on the held-out
   `valid.csv` split, plus accuracy/F1 as secondary metrics.

### Experiment matrix

Treat this as a grid: **normalization × filtration × vectorization**, each combination
trained and evaluated independently, with results logged so combinations can be compared
directly (see Open Questions re: how results should be tracked).

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
3. **Per-image TDA features** — reuse Experiment 1's normalization → filtration →
   vectorization pipeline to produce a feature vector (or full diagram) per study image.
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
- **Basic EDA / count analysis** — run immediately after cohort construction, before any
  normalization/filtration/vectorization work, to sanity-check cohort size and
  composition. For the Experiment 1 baseline cohort (train and valid computed
  separately), report:
  - Total record count and unique-patient count.
  - Pneumothorax class balance (count/percentage of `0.0` vs `1.0`).
  - Breakdown by `AP/PA`.
  - Breakdown by `Frontal/Lateral` (sanity check — should be ~100% frontal after the
    view-narrowing step above; anything else indicates a bug in the filter).
  - Breakdown by `Sex`.
  - `Age` distribution (summary stats and a histogram).
  This is meant to catch cohort-construction bugs and surface class imbalance before any
  compute is spent on the TDA pipeline itself.
- **Normalization → filtration → vectorization pipeline** — built once, parameterized by
  which variant of each stage to use, so the Experiment 1 matrix and Experiment 3's
  per-study feature extraction share the same code path.

## Open questions / decisions needed

- ~~Uncertain/blank label policy~~ — **resolved for the Experiment 1 baseline**: U-Ignore
  (drop `-1.0`/blank, keep only `0.0`/`1.0`). Revisit if a later target label or
  experiment needs U-Ones/U-Zeros instead.
- ~~Frontal vs. lateral views~~ — **resolved for the Experiment 1 baseline**: frontal
  only. Lateral could be explored as a separate variant later, but isn't part of the
  current cohort.
- **Subsampling/compute strategy** — the full train set is ~223k images; decide whether
  to subsample for early iterations of the experiment matrix before scaling up.
- **Experiment 2 embedding format** — exact shape/dimensionality and file format of the
  supplied MedGemma embeddings isn't pinned down yet; confirm when the artifact arrives.
- **Results tracking** — no mechanism is chosen yet for logging experiment-matrix runs
  (e.g. a results CSV/Parquet per run, or a lightweight tracking tool). Decide before
  Experiment 1's matrix grows large enough to lose track of manually.
