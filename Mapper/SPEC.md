# Mapper POC — Agent Spec (v1)

Spec for an agent implementing the initial Mapper proof-of-concept described in
`Mapper/README.md`. This document is self-contained: it does not assume the executing
agent has access to prior conversation history that produced it.

See root `experiments.md` and `CLAUDE.md` for the broader research plan and repo
conventions (data access, `uv` workspace layout, experiment logging).

## 1. Objective

`experiments.md` Experiment 2 ("Topology of the MedGemma Embedding Space") investigates
the shape of a point cloud of chest x-ray embeddings and asks whether each point can be
assigned a topologically-derived cluster/representative label. One candidate approach it
names explicitly is **a Mapper graph (`kmapper`) built on the embedding point cloud**,
compared against the true Pneumothorax label to see whether derived clusters align with
known disease categories or reveal something else.

This spec is scaffolding toward that comparison — not a standalone visualization
exercise. The output should let a human look at the graph and ask "do same-colored
(same-disease) points cluster together, or are they scattered?"

**v1 scope is intentionally narrow**: an embedding-loading utility, and one basic
Mapper script producing a static interactive HTML graph. The enhanced cluster-detail
viewer, G-Mapper cover optimization, and cross-graph comparison tooling from
`Mapper/README.md` are **out of scope for implementation** in this pass — see Section 6
for what's deferred and why.

## 2. Data contract

**Input file**: `data/embeddings/processed/medgemma_train_embeddings.csv`
(path relative to repo root — do not use an absolute `/data/...` path; no such path
exists on disk. This mirrors the `kaggle/` dataset convention documented in root
`CLAUDE.md`.)

- 460 rows, 1161 columns:
  - 9 metadata columns: `output_path, raw_path, patient_id, Pneumothorax, cohort_split,
    Sex, Age, comorbidity_count, is_clean_negative`
  - 1152 embedding columns: `emb_0000` .. `emb_1151`, stored as CSV strings — must be
    cast to `float` on load.
- A sidecar `data/embeddings/processed/medgemma_train_embeddings.csv.meta.json` exists.
  **Read it rather than hardcoding assumptions** — it documents `backend`, `model_id`,
  `pooling`, `embedding_dim`, and `key_column`, and these could change if embeddings are
  regenerated.
- `Pneumothorax` is a float label: `1.0` (positive) or `0.0` (negative) — already
  filtered to definite labels only for this cohort.

**Image path resolution**:
- `output_path` (e.g. `train/patient18455/study3/view1_frontal.png`) resolves under
  `kaggle/processed/` → `kaggle/processed/train/patient18455/study3/view1_frontal.png`.
  This is the CLAHE/lung-crop postprocessed image.
- `raw_path` (e.g. `train/patient18455/study3/view1_frontal.jpg`) resolves under
  `kaggle/` → `kaggle/train/patient18455/study3/view1_frontal.jpg`. This is the original,
  unprocessed image.
- Both should be resolvable via the loader utility in Section 3, since even v1's basic
  graph benefits from tooltips referencing at least the path, and the deferred viewer
  (Section 6) will need the processed-image path directly.

**Explicit non-inputs** — do not use these, even though they may look similar:
- `data/embeddings/processed/_raw/*_embeddings.csv` — pre-label-join staging copies
  (bare `output_path` + `emb_*` only, no metadata/label columns).
- `results/exp2/embeddings/v1/*_embeddings.csv` — an older, separate embedding
  generation on **raw** (not postprocessed) images, keyed on `Path` (not `output_path`),
  without the denormalized label columns. Different schema, different cohort join.

## 3. Deliverable A — embedding loader utility

**Location**: `Mapper/src/mapper/data.py`, as an importable package (mirrors the
existing `GenerateEmbeddings/src/gemma_embeddings` convention used elsewhere in this
repo for backend-specific code).

`Mapper/pyproject.toml` currently declares only:
```toml
[project]
name = "mapper"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
    "kmapper>=2.1.0",
]
```
It has no `[build-system]` section, unlike the root `pyproject.toml` (which lists
`GenerateEmbeddings/src/gemma_embeddings` under
`[tool.hatch.build.targets.wheel] packages`). Add an equivalent block to
`Mapper/pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mapper"]
```
so the package is importable within the shared `uv` workspace virtualenv (`Mapper` is
already listed under `[tool.uv.workspace] members` in the root `pyproject.toml`). Run
`uv sync` from the repo root after adding this.

**Required functions** (exact signatures — pick one convention and use it consistently,
don't mix wide `emb_*` columns and a packed `embedding` column across functions):

```python
def load_embeddings(backend: str = "medgemma", split: str = "train") -> pd.DataFrame:
    """
    Loads data/embeddings/processed/{backend}_{split}_embeddings.csv.
    Casts emb_* columns to float and packs them into a single `embedding` column
    of np.ndarray (shape (embedding_dim,) per row). Returns the 9 metadata columns
    unchanged plus this `embedding` column — drop the individual emb_* columns
    from the returned frame to avoid an unwieldy 1152-wide DataFrame.
    """

def resolve_image_path(row: pd.Series, kind: str = "processed") -> Path:
    """
    kind="processed" -> repo_root / "kaggle" / "processed" / row["output_path"]
    kind="raw"        -> repo_root / "kaggle" / row["raw_path"]
    Must check the resolved path exists on disk and raise if it doesn't —
    do not silently return a dangling path.
    """
```

This satisfies the "Script to load embeddings. Will also include utility to reference
appropriate patient image" item from `Mapper/README.md`'s Initial Code Setup list.

## 4. Deliverable B — basic Mapper script

**Location**: `Mapper/scripts/build_mapper_graph.py`, using `Mapper/src/mapper/data.py`
from Section 3.

Pin these defaults explicitly — don't leave them to agent judgment at implementation
time:

- **Lens**: PCA to 2 components (`sklearn.decomposition.PCA`, already installed via
  `scikit-learn` in the root `uv.lock`) as the default "basic, out-of-box" lens.
  `umap-learn` is **not** currently installed (absent from `uv.lock`) — use PCA or
  `sklearn.manifold.TSNE` only for v1. `Mapper/README.md` says the lens "can use basic
  UMAP or t-SNE," not that it must — adding `umap-learn` to `Mapper/pyproject.toml` is an
  easy, explicitly-optional follow-up, not required here.
- **Cover**: `kmapper.Cover(n_cubes=10, perc_overlap=0.5)` — kmapper's own canonical
  defaults.
- **Clustering**: `sklearn.cluster.DBSCAN()` (default params), run on the **original
  1152-d embedding vectors**, not the 2D lens projection. State this explicitly in the
  script/docstring — clustering in lens-space vs. original-space is a common Mapper
  implementation mistake, and `Mapper/README.md` doesn't specify which; the correct
  Mapper algorithm clusters the pullback in the original feature space.
- **Node coloring**: color nodes by mean `Pneumothorax` value among member points
  (0–1 scale), via `KeplerMapper().visualize(..., color_values=..., color_function_name=...)`.
  This directly operationalizes Experiment 2's comparison of derived clusters vs. true
  pathology label and is a one-line addition, so it stays "basic."
- **Output**: static interactive HTML via `KeplerMapper().visualize()`, written to
  `Mapper/results/v1/graphs/medgemma_train_mapper.html`. This is a new versioned
  directory — do not overwrite prior experiment outputs, per root `CLAUDE.md`.
- **Reproducibility**: fix `random_state` on the PCA/TSNE lens. DBSCAN has no seed, but
  its `eps`/`min_samples` should be exposed as CLI flags rather than hardcoded — in
  1152-d space, DBSCAN's defaults often degenerate to either one giant cluster or nearly
  all noise. After running, compute and report the fraction of unclustered (`label == -1`)
  points; if it exceeds ~90% (or the inverse — one cluster holding ~90%+ of points),
  treat this as a degenerate result to call out in the log (Section 5), not something to
  silently accept.
- **No EDA step is required** before this script runs. The user will personally review
  the rendered HTML output directly as the acceptance check, rather than via a separate
  EDA artifact/report.

## 5. Experiment logging

Create `/Mapper/logs/mapper_log.md` (new file — follows the existing
`logs/exp2_embeddings_log.md` naming convention for Experiment 2 sub-stages) using the
standard Experiment Log template from root `CLAUDE.md`:

```markdown
# Experiment <N>

## Goal
## Dataset
## Parameters
## Results
## Observations
## Next Steps
```

Record: dataset version (medgemma train, 460 rows), the lens/cover/cluster parameters
actually used, the DBSCAN degeneracy check result from Section 4, and whether the
node-color-vs-label visualization showed any meaningful separation (or didn't — a null
result here is still a real, loggable finding for Experiment 2's open question).

## 6. Deferred for v2/v3 — enhanced cluster-detail viewer (detailed design, not built in v1)

Not implemented in this pass, but described here in full so a follow-up agent/spec
doesn't need a second design conversation to recover intent.

This extends **KeplerMapper's own Jinja2/D3 HTML template** (not a from-scratch
Plotly/networkx viewer) — staying close to the "altered KeplerMapper" framing in
`Mapper/README.md`, and reusing kmapper's existing graph layout rather than
reimplementing it.

**Required interaction**: clicking a node/cluster in the rendered graph expands a
"cluster details" panel containing:

1. **Distribution summaries, generalized to a configurable list of fields** — not
   hardcoded to just disease and age. E.g. a `detail_fields: list[str]` parameter
   defaulting to `["Pneumothorax", "Age"]`, but accepting any column from the metadata
   schema in Section 2 (`Sex`, `comorbidity_count`, `is_clean_negative`, etc.):
   - For a categorical/binary field (e.g. `Pneumothorax`, `Sex`): render counts/
     proportions among the cluster's members.
   - For a continuous field (e.g. `Age`): render summary stats (mean/median/std) or a
     small histogram.
   - The rendering choice should be driven by the column's dtype, not per-field
     special-cased logic — this keeps it generalizable to any future metadata column
     without code changes.
2. **Member inspection** — for the patients belonging to a cluster, either:
   - (a) inline-preview their postprocessed images (`kaggle/processed/...`, via
     `resolve_image_path(kind="processed")` from Section 3), or
   - (b) provide a link that opens a separate view showing the image plus that
     patient's `patient_id` and disease label (This could be a carousell or a grid with tooltips).

   **This choice (a vs. b) is intentionally left open** — decide once v1 (Sections 3–4)
   is built and validated, based on what's actually scalable and practical (e.g.
   inlining up to 460 patients' worth of images into one static HTML file may not
   scale; a linked per-patient view might hold up better as the cohort grows). Do not
   resolve this in the v1 implementation — flag it as the first decision for whoever
   picks up this section.

**v3, lower detail** (unchanged from `Mapper/README.md`, no further design
work done here):
- G-Mapper-based cover optimization: https://github.com/MRC-Mapper/G-Mapper
- The three "Future Enhancements" listed in `Mapper/README.md` (visual comparison of two
  Mapper graphs with shared nodes, global network metrics, adjacency matrix subtraction)
  — these are about comparing *two* graphs, which fits root `experiments.md`
  Experiment 3's disease-progression-across-studies framing better than this POC; scope
  them there when that experiment starts.

## 7. Verification

Confirm v1 works end to end before considering this spec complete:

1. `uv run python Mapper/scripts/build_mapper_graph.py` (or equivalent invocation)
   completes without error against the real 460-row medgemma train CSV.
2. The resulting HTML file opens and renders a non-trivial graph (more than one node,
   not fully disconnected). If DBSCAN degenerates (per Section 4's check), the script
   should still produce output, but this must be called out in the log (Section 5) —
   don't treat a degenerate clustering as a silent success.
3. `logs/exp2_mapper_log.md` exists and follows the template in Section 5.
4. `resolve_image_path` resolves to files that actually exist on disk for a sample of
   rows (spot-check a handful, not exhaustive) — catches path-prefix bugs early.
5. The user will personally review the rendered HTML output as the final acceptance
   check — this substitutes for a separate EDA/validation report.

## Notes

- `Mapper/` is a registered `uv` workspace member sharing the root `.venv`/`uv.lock`.
  Any new dependency (e.g. `umap-learn`, if adopted later) goes in
  `Mapper/pyproject.toml`, followed by `uv sync` from the repo root.
- Check `git status`/`git diff` before starting — there may be pre-existing uncommitted
  changes to root `pyproject.toml`/`uv.lock` unrelated to this work; don't assume a
  clean baseline.
