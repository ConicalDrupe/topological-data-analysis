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
COVER_CHOICES = ["uniform", "gmapper"]


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
    cover_info: dict


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


def build_cover(
    kind: str,
    n_cubes: int,
    perc_overlap: float,
    *,
    ad_threshold: float = 10.0,
    g_overlap: float = 0.1,
    gmapper_max_intervals_per_axis: int = 10,
    gmapper_iterations: int = 10,
):
    """
    Factory for the Mapper cover, kept separate from build_graph so covers can be swapped
    without changing build_graph's signature or call sites.

    "gmapper" is a reimplementation (see mapper.gmapper) of the G-Mapper adaptive cover
    (SPEC.md's deferred v3 item, https://github.com/MRC-Mapper/G-Mapper): it splits each
    lens axis independently wherever the marginal distribution fails an Anderson-Darling
    normality test, then takes the Cartesian product of the two axes' intervals as the
    cover's cubes. `n_cubes`/`perc_overlap` are ignored in this branch (gmapper has its own
    ad_threshold/g_overlap/max-intervals knobs instead).
    """
    if kind == "uniform":
        return km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap)
    if kind == "gmapper":
        from mapper.gmapper import GMapperCover

        return GMapperCover(
            ad_threshold=ad_threshold,
            g_overlap=g_overlap,
            max_intervals_per_axis=gmapper_max_intervals_per_axis,
            iterations=gmapper_iterations,
        )
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
    *,
    ad_threshold: float = 10.0,
    g_overlap: float = 0.1,
    gmapper_max_intervals_per_axis: int = 10,
    gmapper_iterations: int = 10,
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

    cover = build_cover(
        cover_kind,
        n_cubes,
        perc_overlap,
        ad_threshold=ad_threshold,
        g_overlap=g_overlap,
        gmapper_max_intervals_per_axis=gmapper_max_intervals_per_axis,
        gmapper_iterations=gmapper_iterations,
    )
    graph = mapper.map(lens_projection, X, clusterer=clusterer, cover=cover)

    n_nodes = len(graph["nodes"])
    n_edges = sum(len(v) for v in graph["links"].values())

    if cover_kind == "gmapper":
        intervals_per_axis = [len(iv) for iv in cover.intervals_per_axis_]
        cover_info = {
            "kind": "gmapper",
            "ad_threshold": ad_threshold,
            "g_overlap": g_overlap,
            "max_intervals_per_axis": gmapper_max_intervals_per_axis,
            "iterations": gmapper_iterations,
            "intervals_per_axis": intervals_per_axis,
            "n_cubes_product": int(np.prod(intervals_per_axis)),
        }
    else:
        cover_info = {"kind": "uniform", "n_cubes": n_cubes, "perc_overlap": perc_overlap}

    return GraphResult(
        X=X,
        lens=lens_projection,
        graph=graph,
        n_nodes=n_nodes,
        n_edges=n_edges,
        frac_noise=frac_noise,
        frac_largest_cluster=frac_largest_cluster,
        degenerate=degenerate,
        cover_info=cover_info,
    )
