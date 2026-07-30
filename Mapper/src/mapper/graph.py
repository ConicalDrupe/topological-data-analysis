"""Shared Mapper-graph construction: lens + cover + clustering + degeneracy check.

Used by both `scripts/build_mapper_graph.py` (v1, basic graph) and
`scripts/build_mapper_graph_v2.py` (v2, enhanced cluster-detail viewer) so the
lens/cover/clustering parameters and degeneracy diagnostic stay identical across both.

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
from sklearn.decomposition import PCA

RANDOM_STATE = 42
DEGENERACY_THRESHOLD = 0.90


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


def build_graph(
    df: pd.DataFrame,
    eps: float,
    min_samples: int,
    n_cubes: int,
    perc_overlap: float,
    verbose: int = 1,
) -> GraphResult:
    X = np.stack(df["embedding"].to_numpy())

    mapper = km.KeplerMapper(verbose=verbose)
    lens = mapper.fit_transform(X, projection=PCA(n_components=2, random_state=RANDOM_STATE))

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    # Diagnostic-only whole-dataset fit, used purely to report degeneracy below.
    # km.map() below clones/refits an equivalent clusterer per-cube internally.
    labels_full = clusterer.fit_predict(X)
    frac_noise = float(np.mean(labels_full == -1))
    unique_labels, counts = np.unique(labels_full[labels_full != -1], return_counts=True)
    frac_largest_cluster = float(counts.max() / len(labels_full)) if len(counts) else 0.0
    degenerate = frac_noise >= DEGENERACY_THRESHOLD or frac_largest_cluster >= DEGENERACY_THRESHOLD

    cover = km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap)
    graph = mapper.map(lens, X, clusterer=clusterer, cover=cover)

    n_nodes = len(graph["nodes"])
    n_edges = sum(len(v) for v in graph["links"].values())

    return GraphResult(
        X=X,
        lens=lens,
        graph=graph,
        n_nodes=n_nodes,
        n_edges=n_edges,
        frac_noise=frac_noise,
        frac_largest_cluster=frac_largest_cluster,
        degenerate=degenerate,
    )
