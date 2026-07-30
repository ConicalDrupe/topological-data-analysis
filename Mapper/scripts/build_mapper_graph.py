"""Build a basic Mapper graph on MedGemma embeddings and render it as static HTML.

Lens: PCA(n_components=2) on the raw 1152-d embedding vectors, fixed random_state.
Cover: kmapper.Cover(n_cubes=10, perc_overlap=0.5) — kmapper's canonical defaults.
Clustering: sklearn.cluster.DBSCAN, run on the ORIGINAL 1152-d embedding vectors
(not the 2D lens projection). This is intentional: the correct Mapper algorithm
clusters the pullback cover in the original feature space, not in lens-space —
clustering in lens-space instead is a common Mapper implementation mistake.

Example:
    uv run python Mapper/scripts/build_mapper_graph.py \\
        --backend medgemma --split train --eps 0.5 --min-samples 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import kmapper as km
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from mapper.data import REPO_ROOT, load_embeddings

RANDOM_STATE = 42
DEGENERACY_THRESHOLD = 0.90


@dataclass
class MapperConfig:
    backend: str
    split: str
    eps: float
    min_samples: int
    n_cubes: int
    perc_overlap: float
    output_html: Path
    log_path: Path


def parse_args() -> MapperConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="medgemma")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN eps, fit on raw 1152-d embeddings (sklearn default; likely "
        "needs tuning in high-dimensional space — see the degeneracy report printed "
        "after clustering)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=5, help="DBSCAN min_samples (sklearn default)"
    )
    parser.add_argument("--n-cubes", type=int, default=10)
    parser.add_argument("--perc-overlap", type=float, default=0.5)
    parser.add_argument(
        "--output-html",
        type=Path,
        default=REPO_ROOT / "Mapper" / "results" / "v1" / "graphs" / "medgemma_train_mapper.html",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=REPO_ROOT / "Mapper" / "logs" / "mapper_log.md",
    )
    args = parser.parse_args()
    return MapperConfig(
        backend=args.backend,
        split=args.split,
        eps=args.eps,
        min_samples=args.min_samples,
        n_cubes=args.n_cubes,
        perc_overlap=args.perc_overlap,
        output_html=args.output_html,
        log_path=args.log_path,
    )


def main() -> None:
    config = parse_args()

    print(f"Loading embeddings backend={config.backend!r} split={config.split!r} ...")
    df = load_embeddings(backend=config.backend, split=config.split)
    X = np.stack(df["embedding"].to_numpy())
    print(f"Loaded {X.shape[0]} rows, embedding_dim={X.shape[1]}")

    mapper = km.KeplerMapper(verbose=1)

    print("Fitting PCA(n_components=2) lens ...")
    lens = mapper.fit_transform(X, projection=PCA(n_components=2, random_state=RANDOM_STATE))

    print(
        f"Clustering with DBSCAN(eps={config.eps}, min_samples={config.min_samples}) "
        "on the original 1152-d embeddings (NOT the lens-space projection) ..."
    )
    clusterer = DBSCAN(eps=config.eps, min_samples=config.min_samples)

    # Diagnostic-only whole-dataset fit, used purely to report degeneracy below.
    # km.map() below clones/refits an equivalent clusterer per-cube internally.
    labels_full = clusterer.fit_predict(X)
    frac_noise = float(np.mean(labels_full == -1))
    unique_labels, counts = np.unique(labels_full[labels_full != -1], return_counts=True)
    frac_largest_cluster = float(counts.max() / len(labels_full)) if len(counts) else 0.0
    degenerate = frac_noise >= DEGENERACY_THRESHOLD or frac_largest_cluster >= DEGENERACY_THRESHOLD

    print(
        f"DBSCAN diagnostic (whole-dataset fit): {frac_noise:.1%} noise, "
        f"largest cluster holds {frac_largest_cluster:.1%} of points"
    )
    if degenerate:
        print("*** DEGENERATE CLUSTERING DETECTED (>=90% noise or one dominant cluster) ***")
        print("*** Proceeding anyway per spec — this will be recorded in the log. ***")

    cover = km.Cover(n_cubes=config.n_cubes, perc_overlap=config.perc_overlap)
    graph = mapper.map(lens, X, clusterer=clusterer, cover=cover)

    n_nodes = len(graph["nodes"])
    n_edges = sum(len(v) for v in graph["links"].values())
    print(f"Mapper graph: {n_nodes} nodes, {n_edges} edges")

    html_written = False
    if n_nodes == 0:
        # kmapper's own visualize() hard-fails on an empty graph rather than
        # rendering one — this is a degenerate result even more extreme than the
        # 90% threshold below (every point ended up as DBSCAN noise). Skip
        # visualize() rather than letting the script crash; the diagnostic
        # numbers are still logged below.
        print(
            "*** Mapper graph has 0 nodes — kmapper cannot render this. "
            "Skipping HTML output for this run. Re-run with a larger --eps. ***"
        )
    else:
        config.output_html.parent.mkdir(parents=True, exist_ok=True)
        mapper.visualize(
            graph,
            color_values=df["Pneumothorax"].to_numpy(),
            color_function_name="Pneumothorax (mean)",
            path_html=str(config.output_html),
            title=f"MedGemma {config.split} embeddings — Mapper graph",
            X=X,
            X_names=[f"emb_{i:04d}" for i in range(X.shape[1])],
            lens=lens,
            lens_names=["PCA-1", "PCA-2"],
            custom_tooltips=df["patient_id"].to_numpy(),
        )
        html_written = True
        print(f"Wrote HTML graph to {config.output_html}")

    _append_log_entry(
        config, df, n_nodes, n_edges, frac_noise, frac_largest_cluster, degenerate, html_written
    )
    print(f"Appended run entry to {config.log_path}")


def _append_log_entry(
    config: MapperConfig,
    df,
    n_nodes: int,
    n_edges: int,
    frac_noise: float,
    frac_largest_cluster: float,
    degenerate: bool,
    html_written: bool,
) -> None:
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    if n_nodes == 0:
        degeneracy_note = (
            "FULLY DEGENERATE — 0 nodes, kmapper could not render a graph at all "
            "(every point was DBSCAN noise). No HTML written for this run."
        )
    elif degenerate:
        degeneracy_note = (
            "DEGENERATE — clustering collapsed to >=90% noise or one dominant cluster; "
            "graph coloring/topology below is not meaningfully informative for this run."
        )
    else:
        degeneracy_note = "Not degenerate by the >=90% noise / >=90% single-cluster threshold."

    entry = f"""
## Run {timestamp}

- Dataset: backend={config.backend}, split={config.split}, {len(df)} rows,
  embedding_dim={df["embedding"].iloc[0].shape[0]}
- Lens: PCA(n_components=2, random_state={RANDOM_STATE}) on raw embeddings
- Cover: kmapper.Cover(n_cubes={config.n_cubes}, perc_overlap={config.perc_overlap})
- Clustering: sklearn.cluster.DBSCAN(eps={config.eps}, min_samples={config.min_samples}),
  fit on original {df["embedding"].iloc[0].shape[0]}-d embeddings (not lens-space)
- Node coloring: mean Pneumothorax value among cluster members (0-1 scale)
- Output: {config.output_html if html_written else "(not written — see note below)"}
- Mapper graph: {n_nodes} nodes, {n_edges} edges
- DBSCAN degeneracy check (whole-dataset diagnostic fit): {frac_noise:.1%} unclustered
  (label == -1), largest single cluster holds {frac_largest_cluster:.1%} of points
- {degeneracy_note}
"""

    if not config.log_path.exists():
        header = "# Experiment 2 — Mapper Graph (v1)\n"
        config.log_path.write_text(header + entry)
    else:
        with config.log_path.open("a") as f:
            f.write(entry)


if __name__ == "__main__":
    main()
