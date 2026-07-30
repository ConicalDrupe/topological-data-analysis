# Experiment 2 — Mapper Graph (v1, v2, v2.5)

## Goal

Build a basic Mapper graph (`kmapper`) on the MedGemma chest x-ray embedding point
cloud and compare derived clusters/nodes against the true `Pneumothorax` label, per
`experiments.md` Experiment 2's open question about whether topologically-derived
cluster labels align with known disease categories. This is v1 scaffolding
(`Mapper/SPEC.md`): embedding loader utility + one basic Mapper script producing a
static interactive HTML graph — no enhanced cluster-detail viewer or G-Mapper cover
optimization yet (deferred to v2/v3).

## Dataset

`data/embeddings/processed/medgemma_train_embeddings.csv` — 460 rows, 1152-d
embeddings (backend=medgemma, split=train). Sidecar meta: model_id=
google/medgemma-4b-it, pooling=mean_patch, embedding_dim=1152, key_column=
output_path. Overall Pneumothorax positive rate in this cohort: 43.9% (202/460).

## Parameters

- Lens: `sklearn.decomposition.PCA(n_components=2, random_state=42)`, fit on raw
  1152-d embeddings.
- Cover: `kmapper.Cover(n_cubes=10, perc_overlap=0.5)` (kmapper's canonical
  defaults).
- Clustering: `sklearn.cluster.DBSCAN(eps, min_samples)`, fit on the **original
  1152-d embedding vectors**, not the 2D lens projection — clustering in lens-space
  is a common Mapper implementation mistake; the correct algorithm clusters the
  pullback cover in the original feature space.
- Node coloring: mean `Pneumothorax` value among each node's member points
  (0-1 scale).
- Two runs performed, `--eps` swept:
  1. `eps=0.5, min_samples=5` — sklearn's own DBSCAN defaults, used as the script's
     CLI defaults per SPEC.md Section 4's literal "DBSCAN() (default params)" text.
  2. `eps=3.5, min_samples=5` — `eps` chosen from the 5th-nearest-neighbor distance
     distribution over the embedding matrix (median ≈3.32, IQR ≈3.05-3.58), as a
     quick diagnostic to find a workable scale in this 1152-d space (not a full
     EDA pass — SPEC.md explicitly waives that requirement for this script).

## Results

**Run 1 (`eps=0.5, min_samples=5`, sklearn defaults)** — FULLY DEGENERATE. 100.0%
of points were DBSCAN noise (label == -1), 0 clusters formed at all. This is more
extreme than the spec's anticipated ≥90%-noise degeneracy case: kmapper's
`visualize()` raises rather than rendering a 0-node graph, so the script detects
this, skips HTML generation for this run, and logs the diagnostic instead of
crashing. No HTML was written for this run. This confirms SPEC.md Section 4's
prediction that DBSCAN's sklearn defaults "often degenerate to ... nearly all
noise" in 1152-d space — here it degenerated completely.

**Run 2 (`eps=3.5, min_samples=5`)** — NOT degenerate. 16.3% noise, largest single
cluster holds 83.7% of points (both under the 90% threshold). Mapper graph: **54
nodes, 162 edges**. HTML written to
`Mapper/results/v1/graphs/medgemma_train_mapper.html` (156 KB).

Node-level Pneumothorax separation (computed directly from the 54 nodes, run 2):
mean Pneumothorax rate per node ranges from 0.20 to 0.875 (std 0.155) against an
overall cohort rate of 0.439. No node is fully pure (all-positive or
all-negative) — every node is a mix of both labels.

## Observations

- The DBSCAN default-params run degenerating to 100% noise (not just ≥90%) shows
  that "default params" is not a usable starting point at all for this embedding
  space/scale, at least for `eps` — `min_samples=5` was left untouched and didn't
  need tuning in the second run.
- The node-color-vs-label comparison shows a real but modest signal, not a clean
  split: node-level Pneumothorax rates range fairly widely (0.20-0.875) around the
  overall 0.439 rate, meaning some regions of the graph are enriched for
  Pneumothorax-positive patients and others depleted — but no node is a pure
  single-label cluster. This reads as a partial/null result for the "do same-
  disease points cluster together" question: there's topological structure that
  correlates with the label, but it doesn't cleanly separate it. A visual review
  of the rendered HTML (color gradient across the graph) is needed to judge
  whether this signal is spatially coherent (e.g. one dense sub-region of the
  graph runs consistently high) or just noisy per-node variance.
- 83.7% of points falling into a single dominant DBSCAN cluster at `eps=3.5`
  suggests the embedding space may not have strong natural density separation at
  this scale — the Mapper graph's node/edge structure is doing more of the
  differentiating work here than DBSCAN's cluster boundaries.

## Next Steps

- User to visually review `Mapper/results/v1/graphs/medgemma_train_mapper.html` as
  the acceptance check (per SPEC.md Section 7 point 5) — does the color gradient
  show any spatially coherent Pneumothorax enrichment across the graph, or is it
  scattered?
- If the single dominant cluster (83.7% of points) is judged too coarse, consider
  a narrower `eps` sweep between 0.5 and 3.5 to find a point with more balanced
  cluster sizes, before concluding on the "does topology align with disease
  label" question.
- Consider `sklearn.manifold.TSNE` as an alternative lens to PCA for comparison,
  now that the loader/script scaffolding is validated end-to-end.
- Section 6 of SPEC.md (enhanced cluster-detail viewer: per-node distribution
  panels, image previews) is the natural next build once this base graph is
  validated as informative.

## v2 — Enhanced Cluster-Detail Viewer

### Goal

SPEC.md Section 6: extend the existing Mapper graph (same lens/cover/clustering as v1,
now factored into `mapper.graph.build_graph`) so clicking a node's "Cluster Details"
panel also shows per-node field-distribution summaries and a way to inspect member
images, without replacing kmapper's own Jinja2/D3 template.

### Design decision (SPEC.md Section 6.2's open choice)

Resolved as: **inline batched thumbnail grid + in-page lightbox**, not a separate
linked page. Rationale: the stated scaling concern (a cluster could hold "hundreds of
thousands" of images in a future dataset) is a DOM-size/render-time problem, not an
image-byte-size problem — `kaggle/processed/*.png` are already small (224x224, ~30 KB)
and images are referenced by relative file path, never embedded as base64. So the grid
renders thumbnails in fixed batches (default 48) with a "Load N more" button, while the
full path/tooltip list stays in memory as lightweight JSON, letting the lightbox's
prev/next navigation jump instantly across the *entire* member list regardless of how
many thumbnails have been rendered. A linked separate page was rejected because a page
reload per "closer look" click directly fights the stated speed requirement.

### Implementation

- `mapper.cluster_details.classify_field`: dtype-driven categorical/continuous
  classification (SPEC.md Section 6.1), computed once from the *full* column so a
  field's type stays consistent across nodes (see Bugs below).
- Per-node payload (distributions + image paths/tooltips) is computed in Python and
  injected as a JS global into the HTML `visualize()` already returns
  (`save_file=False`), by splicing a `<style>`/`<script>` block before `</body>` —
  no fork of the installed kmapper package.
- The injected JS wraps kmapper's own `window.set_focus_node` (calls the original, then
  renders the new sections) rather than replacing kmapper's click/hover interaction
  model.

### Bugs found and fixed

1. `classify_field` was initially called on each node's small member *subset* rather
   than the full column — a continuous field like `Age` (72 unique values overall) was
   misclassified as categorical whenever a small cluster happened to have ≤10 distinct
   ages. Fixed by computing field types once from `df[field]` (the full column) and
   threading that through to every node.
2. An incidental "fix" to remove kmapper's CDN dependency for `d3.js` (swapping in
   kmapper's own bundled `static/d3.min.js`) broke the graph entirely — that bundled
   file is an old, incompatible d3 version (no `d3.scalePow`, pre-v4 style), not the
   `d3@6.1.1` the CDN serves and `kmapper.js` actually requires. Reverted; the v1/v2
   graphs still load `d3`/`file-saver` from CDN, same pre-existing behavior as before
   this work (unrelated to this change — flagged as a known limitation, not fixed here).

### Results

Ran against the same `eps=3.5, min_samples=5` graph as v1 (54 nodes, 162 edges) —
see the `v2` entry in the Run Log below for exact parameters. 0 images skipped
(all `resolve_image_path` lookups succeeded).

Verified end-to-end with a headless-Chromium interaction test (Playwright, driving the
real generated HTML file): node click expands the panel and renders correct
distributions (categorical bar/count/percent for `Pneumothorax`, mean/median/std/range +
10-bin histogram for `Age`); the gallery renders an initial batch, the "Load N more"
button correctly appends the remainder for a 61-image node (batch size 48 → "Load 13
more (13 remaining)" → all 61 rendered, button removed); clicking a thumbnail opens the
lightbox with the correct image and tooltip, next/prev navigation and Escape-to-close
both work. No console or page errors during any of this.

### Observations

- The dtype-driven classification, once fixed to use the global column, behaves as
  intended for this schema: `Pneumothorax` (2-valued float) and `Sex`/`is_clean_negative`
  would classify as categorical, `Age` as continuous, `comorbidity_count` as categorical
  (degenerate, single value in this cohort — harmless, just a one-bar chart).
- The real scaling risk for very large clusters, as anticipated, is DOM node count, not
  image bytes — the batched-render + full-list-in-memory design keeps the lightbox
  instant regardless of unrendered thumbnail count.

### Next Steps

- User to visually review `Mapper/results/v2/graphs/medgemma_train_mapper.html`
  interactively (click a few nodes, browse a gallery, open the lightbox) as the
  acceptance check.
- If a future dataset actually approaches "hundreds of thousands of images per node,"
  revisit whether the per-node image *path list* itself (not the images) becomes large
  enough in the embedded JSON payload to matter — not a concern at this cohort's scale
  (460 rows).
- The CDN dependency for `d3`/`file-saver` (pre-existing, not introduced here) means
  both v1 and v2 graphs need network access to render at all — worth a real fix
  (vendoring a correct d3@6.1.1 build) if offline use ever becomes a requirement.

## v2.5 — Pluggable Lens/Cover + UMAP vs. t-SNE Comparison

### Goal

Make the lens and cover pluggable (in preparation for a future G-Mapper-optimized cover,
SPEC.md's deferred v3 item) rather than hardcoding PCA/`kmapper.Cover`, then generate two
more v2-viewer graphs on the same medgemma/train embeddings and tuned `eps=3.5` — one
with a UMAP lens, one with t-SNE — as a direct comparison against the existing PCA graph.

### Implementation

- `mapper.graph.build_lens(name, random_state)`: factory returning a `fit_transform`-able
  projection object for kmapper's `projection=` argument — `"pca"` (default,
  unchanged), `"tsne"` (`sklearn.manifold.TSNE`), `"umap"` (`umap.UMAP`, new dependency
  `umap-learn` added to `Mapper/pyproject.toml`).
- `mapper.graph.build_cover(kind, n_cubes, perc_overlap)`: factory currently only
  implementing `"uniform"` (kmapper's own `Cover`) but kept as a separate function
  specifically so a `"gmapper"` branch can be added later without changing
  `build_graph`'s signature or either script's call sites.
- `--lens` (`pca`/`tsne`/`umap`) and `--cover-kind` (`uniform`) CLI flags added to both
  `build_mapper_graph.py` and `build_mapper_graph_v2.py`. Default behavior for both
  scripts is unchanged (`pca`, `uniform`) — verified via a regression run of v1 with no
  new flags, reproducing the same 54 nodes/162 edges as before this change.

### Results

Same dataset (medgemma/train, 460 rows), same `eps=3.5, min_samples=5`,
`n_cubes=10, perc_overlap=0.5`, same `--detail-fields Pneumothorax,Age` — only the lens
differs:

| Lens | Nodes | Edges | DBSCAN noise / largest cluster |
|------|-------|-------|--------------------------------|
| PCA (v2 baseline) | 54 | 162 | 16.3% / 83.7% |
| UMAP | 64 | 181 | 16.3% / 83.7% |
| t-SNE | 69 | 189 | 16.3% / 83.7% |

DBSCAN diagnostics are identical across all three (expected — clustering runs on the
original 1152-d embeddings, not the lens projection, so the lens choice cannot affect
it). The node/edge counts differ because the lens changes which points share a cover
cube, which changes the pullback partition kmapper clusters within. Both UMAP and t-SNE
produced a **finer** graph (more nodes/edges) than PCA here. Both outputs verified with
the same headless-browser check used for v2 (node click → distributions + gallery render
correctly, no console/page errors) — spot-checked on the UMAP output.

Outputs: `Mapper/results/v2.5/graphs/medgemma_train_mapper_umap.html`,
`Mapper/results/v2.5/graphs/medgemma_train_mapper_tsne.html`.

### Next Steps

- User to visually compare the three graphs (PCA v2, UMAP, t-SNE) — does either
  non-linear lens reveal a more spatially coherent Pneumothorax gradient than PCA's?
- `build_cover`'s `"uniform"`-only structure is now the intended integration point for
  G-Mapper (SPEC.md's deferred v3 item) — not implemented here, just structured for it.

## Run Log

Raw per-invocation diagnostics, auto-appended by
`Mapper/scripts/build_mapper_graph.py` / `build_mapper_graph_v2.py` on every run:

## Run 2026-07-30T20:34:11.887969+00:00

- Dataset: backend=medgemma, split=train, 460 rows,
  embedding_dim=1152
- Lens: PCA(n_components=2, random_state=42) on raw embeddings
- Cover: kmapper.Cover(n_cubes=10, perc_overlap=0.5)
- Clustering: sklearn.cluster.DBSCAN(eps=0.5, min_samples=5),
  fit on original 1152-d embeddings (not lens-space)
- Node coloring: mean Pneumothorax value among cluster members (0-1 scale)
- Output: (not written — see note below)
- Mapper graph: 0 nodes, 0 edges
- DBSCAN degeneracy check (whole-dataset diagnostic fit): 100.0% unclustered
  (label == -1), largest single cluster holds 0.0% of points
- FULLY DEGENERATE — 0 nodes, kmapper could not render a graph at all (every point was DBSCAN noise). No HTML written for this run.

## Run 2026-07-30T20:34:21.221621+00:00

- Dataset: backend=medgemma, split=train, 460 rows,
  embedding_dim=1152
- Lens: PCA(n_components=2, random_state=42) on raw embeddings
- Cover: kmapper.Cover(n_cubes=10, perc_overlap=0.5)
- Clustering: sklearn.cluster.DBSCAN(eps=3.5, min_samples=5),
  fit on original 1152-d embeddings (not lens-space)
- Node coloring: mean Pneumothorax value among cluster members (0-1 scale)
- Output: /home/boon/Projects/topological-data-analysis/Mapper/results/v1/graphs/medgemma_train_mapper.html
- Mapper graph: 54 nodes, 162 edges
- DBSCAN degeneracy check (whole-dataset diagnostic fit): 16.3% unclustered
  (label == -1), largest single cluster holds 83.7% of points
- Not degenerate by the >=90% noise / >=90% single-cluster threshold.

## Run 2026-07-30T20:57:10.402998+00:00 (v2 — enhanced cluster viewer)

- Dataset: backend=medgemma, split=train, 460 rows,
  embedding_dim=1152
- Lens: PCA(n_components=2, random_state=42) on raw embeddings
- Cover: kmapper.Cover(n_cubes=10, perc_overlap=0.5)
- Clustering: sklearn.cluster.DBSCAN(eps=3.5, min_samples=5),
  fit on original 1152-d embeddings (not lens-space)
- Node coloring: mean Pneumothorax value among cluster members (0-1 scale)
- Cluster-detail fields: Pneumothorax, Age
- Image kind: processed, gallery batch size: 48
- Output: /home/boon/Projects/topological-data-analysis/Mapper/results/v2/graphs/medgemma_train_mapper.html
- Mapper graph: 54 nodes, 162 edges
- DBSCAN degeneracy check (whole-dataset diagnostic fit): 16.3% unclustered
  (label == -1), largest single cluster holds 83.7% of points
- Not degenerate by the >=90% noise / >=90% single-cluster threshold.
- Images skipped (unresolvable path): 0

## Run 2026-07-30T21:17:44.864097+00:00 (v2 — enhanced cluster viewer)

- Dataset: backend=medgemma, split=train, 460 rows,
  embedding_dim=1152
- Lens: umap (n_components=2, random_state=42) on raw embeddings
- Cover: kind=uniform, kmapper.Cover(n_cubes=10, perc_overlap=0.5)
- Clustering: sklearn.cluster.DBSCAN(eps=3.5, min_samples=5),
  fit on original 1152-d embeddings (not lens-space)
- Node coloring: mean Pneumothorax value among cluster members (0-1 scale)
- Cluster-detail fields: Pneumothorax, Age
- Image kind: processed, gallery batch size: 48
- Output: Mapper/results/v2.5/graphs/medgemma_train_mapper_umap.html
- Mapper graph: 64 nodes, 181 edges
- DBSCAN degeneracy check (whole-dataset diagnostic fit): 16.3% unclustered
  (label == -1), largest single cluster holds 83.7% of points
- Not degenerate by the >=90% noise / >=90% single-cluster threshold.
- Images skipped (unresolvable path): 0

## Run 2026-07-30T21:17:58.913401+00:00 (v2 — enhanced cluster viewer)

- Dataset: backend=medgemma, split=train, 460 rows,
  embedding_dim=1152
- Lens: tsne (n_components=2, random_state=42) on raw embeddings
- Cover: kind=uniform, kmapper.Cover(n_cubes=10, perc_overlap=0.5)
- Clustering: sklearn.cluster.DBSCAN(eps=3.5, min_samples=5),
  fit on original 1152-d embeddings (not lens-space)
- Node coloring: mean Pneumothorax value among cluster members (0-1 scale)
- Cluster-detail fields: Pneumothorax, Age
- Image kind: processed, gallery batch size: 48
- Output: Mapper/results/v2.5/graphs/medgemma_train_mapper_tsne.html
- Mapper graph: 69 nodes, 189 edges
- DBSCAN degeneracy check (whole-dataset diagnostic fit): 16.3% unclustered
  (label == -1), largest single cluster holds 83.7% of points
- Not degenerate by the >=90% noise / >=90% single-cluster threshold.
- Images skipped (unresolvable path): 0
