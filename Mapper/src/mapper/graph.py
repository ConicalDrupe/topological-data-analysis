"""Shared Mapper-graph construction: lens + cover + clustering + degeneracy check.

Used by `scripts/build_mapper_graph.py` (v1, basic graph), `scripts/build_mapper_graph_v2.py`
(v2, enhanced cluster-detail viewer), and v2.5's UMAP/t-SNE lens comparison runs, so the
cover/clustering/degeneracy diagnostic logic stays identical across all of them — only the
lens choice differs.

Clustering is intentionally run on the ORIGINAL embedding vectors, not the 2D lens
projection — the correct Mapper algorithm clusters the pullback cover in the original
feature space; clustering in lens-space is a common Mapper implementation mistake.
"""

from __future__ import annotations

from dataclasses import dataclass

import kmapper as km
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

RANDOM_STATE = 42
DEGENERACY_THRESHOLD = 0.90

LENS_CHOICES = ["pca", "tsne", "umap"]
COVER_CHOICES = ["uniform"]


@dataclass
class GraphResult:
    X: np.ndarray
    lens: np.ndarray
    graph: dict
    n_nodes: int
    n_edges: int
    frac_noise: float
    frac_largest_cluster: float
    degenerate: bool


def build_lens(name: str, random_state: int = RANDOM_STATE):
    """Returns a fit_transform-able projection object for kmapper's `projection=` arg."""
    if name == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state)
    if name == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, random_state=random_state)
    if name == "umap":
        from umap import UMAP

        return UMAP(n_components=2, random_state=random_state)
    raise ValueError(f"Unknown lens {name!r}, expected one of {LENS_CHOICES}")


def build_cover(kind: str, n_cubes: int, perc_overlap: float) -> km.Cover:
    """
    Factory for the Mapper cover, kept separate from build_graph so a future G-Mapper-
    optimized cover (SPEC.md's deferred v3 item) can be added as a new `kind` branch here
    without changing build_graph's signature or call sites.
    """
    if kind == "uniform":
        return km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap)
    raise ValueError(f"Unknown cover kind {kind!r}, expected one of {COVER_CHOICES}")


def build_graph(
    df: pd.DataFrame,
    eps: float,
    min_samples: int,
    n_cubes: int,
    perc_overlap: float,
    lens: str = "pca",
    cover_kind: str = "uniform",
    verbose: int = 1,
) -> GraphResult:
    X = np.stack(df["embedding"].to_numpy())

    mapper = km.KeplerMapper(verbose=verbose)
    lens_projection = mapper.fit_transform(X, projection=build_lens(lens))

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    # Diagnostic-only whole-dataset fit, used purely to report degeneracy below.
    # km.map() below clones/refits an equivalent clusterer per-cube internally.
    labels_full = clusterer.fit_predict(X)
    frac_noise = float(np.mean(labels_full == -1))
    unique_labels, counts = np.unique(labels_full[labels_full != -1], return_counts=True)
    frac_largest_cluster = float(counts.max() / len(labels_full)) if len(counts) else 0.0
    degenerate = frac_noise >= DEGENERACY_THRESHOLD or frac_largest_cluster >= DEGENERACY_THRESHOLD

    cover = build_cover(cover_kind, n_cubes, perc_overlap)
    graph = mapper.map(lens_projection, X, clusterer=clusterer, cover=cover)

    n_nodes = len(graph["nodes"])
    n_edges = sum(len(v) for v in graph["links"].values())

    return GraphResult(
        X=X,
        lens=lens_projection,
        graph=graph,
        n_nodes=n_nodes,
        n_edges=n_edges,
        frac_noise=frac_noise,
        frac_largest_cluster=frac_largest_cluster,
        degenerate=degenerate,
    )
