# Experiment 2 — Mapper Graph (v1)

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

## Run Log

Raw per-invocation diagnostics, auto-appended by
`Mapper/scripts/build_mapper_graph.py` on every run:

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
