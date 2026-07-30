"""Build a basic Mapper graph on MedGemma embeddings and render it as static HTML.

Lens: pluggable (PCA/t-SNE/UMAP, --lens flag), fixed random_state. Defaults to PCA.
Cover: pluggable (--cover-kind flag; only "uniform" — kmapper's own Cover — is
implemented today, kept separate so a future G-Mapper-optimized cover can be added as
a new kind without changing this script). Default kmapper.Cover(n_cubes=10, perc_overlap=0.5).
Clustering: sklearn.cluster.DBSCAN, run on the ORIGINAL 1152-d embedding vectors
(not the 2D lens projection). This is intentional: the correct Mapper algorithm
clusters the pullback cover in the original feature space, not in lens-space —
clustering in lens-space instead is a common Mapper implementation mistake.

See mapper.graph.build_graph for the shared lens/cover/cluster/degeneracy-check logic
(also used by scripts/build_mapper_graph_v2.py).

Example:
    uv run python Mapper/scripts/build_mapper_graph.py \\
        --backend medgemma --split train --eps 0.5 --min-samples 5 --lens umap
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import kmapper as km

from mapper.data import REPO_ROOT, load_embeddings
from mapper.graph import COVER_CHOICES, LENS_CHOICES, RANDOM_STATE, build_graph


@dataclass
class MapperConfig:
    backend: str
    split: str
    eps: float
    min_samples: int
    n_cubes: int
    perc_overlap: float
    lens: str
    cover_kind: str
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
    parser.add_argument("--lens", default="pca", choices=LENS_CHOICES)
    parser.add_argument("--cover-kind", default="uniform", choices=COVER_CHOICES)
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
        lens=args.lens,
        cover_kind=args.cover_kind,
        output_html=args.output_html,
        log_path=args.log_path,
    )


def main() -> None:
    config = parse_args()

    print(f"Loading embeddings backend={config.backend!r} split={config.split!r} ...")
    df = load_embeddings(backend=config.backend, split=config.split)
    print(f"Loaded {len(df)} rows, embedding_dim={df['embedding'].iloc[0].shape[0]}")

    print(
        f"Building graph: {config.lens} lens, DBSCAN(eps={config.eps}, min_samples={config.min_samples}) "
        "on original embeddings (NOT lens-space), Cover("
        f"kind={config.cover_kind}, n_cubes={config.n_cubes}, perc_overlap={config.perc_overlap}) ..."
    )
    result = build_graph(
        df,
        config.eps,
        config.min_samples,
        config.n_cubes,
        config.perc_overlap,
        lens=config.lens,
        cover_kind=config.cover_kind,
    )

    print(
        f"DBSCAN diagnostic (whole-dataset fit): {result.frac_noise:.1%} noise, "
        f"largest cluster holds {result.frac_largest_cluster:.1%} of points"
    )
    if result.degenerate:
        print("*** DEGENERATE CLUSTERING DETECTED (>=90% noise or one dominant cluster) ***")
        print("*** Proceeding anyway per spec — this will be recorded in the log. ***")

    print(f"Mapper graph: {result.n_nodes} nodes, {result.n_edges} edges")

    mapper = km.KeplerMapper(verbose=1)
    html_written = False
    if result.n_nodes == 0:
        # kmapper's own visualize() hard-fails on an empty graph rather than
        # rendering one — this is a degenerate result even more extreme than the
        # 90% threshold above (every point ended up as DBSCAN noise). Skip
        # visualize() rather than letting the script crash; the diagnostic
        # numbers are still logged below.
        print(
            "*** Mapper graph has 0 nodes — kmapper cannot render this. "
            "Skipping HTML output for this run. Re-run with a larger --eps. ***"
        )
    else:
        config.output_html.parent.mkdir(parents=True, exist_ok=True)
        mapper.visualize(
            result.graph,
            color_values=df["Pneumothorax"].to_numpy(),
            color_function_name="Pneumothorax (mean)",
            path_html=str(config.output_html),
            title=f"MedGemma {config.split} embeddings — Mapper graph ({config.lens})",
            X=result.X,
            X_names=[f"emb_{i:04d}" for i in range(result.X.shape[1])],
            lens=result.lens,
            lens_names=[f"{config.lens.upper()}-1", f"{config.lens.upper()}-2"],
            custom_tooltips=df["patient_id"].to_numpy(),
        )
        html_written = True
        print(f"Wrote HTML graph to {config.output_html}")

    _append_log_entry(config, df, result, html_written)
    print(f"Appended run entry to {config.log_path}")


def _append_log_entry(config: MapperConfig, df, result, html_written: bool) -> None:
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    if result.n_nodes == 0:
        degeneracy_note = (
            "FULLY DEGENERATE — 0 nodes, kmapper could not render a graph at all "
            "(every point was DBSCAN noise). No HTML written for this run."
        )
    elif result.degenerate:
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
- Lens: {config.lens} (n_components=2, random_state={RANDOM_STATE}) on raw embeddings
- Cover: kind={config.cover_kind}, kmapper.Cover(n_cubes={config.n_cubes}, perc_overlap={config.perc_overlap})
- Clustering: sklearn.cluster.DBSCAN(eps={config.eps}, min_samples={config.min_samples}),
  fit on original {df["embedding"].iloc[0].shape[0]}-d embeddings (not lens-space)
- Node coloring: mean Pneumothorax value among cluster members (0-1 scale)
- Output: {config.output_html if html_written else "(not written — see note below)"}
- Mapper graph: {result.n_nodes} nodes, {result.n_edges} edges
- DBSCAN degeneracy check (whole-dataset diagnostic fit): {result.frac_noise:.1%} unclustered
  (label == -1), largest single cluster holds {result.frac_largest_cluster:.1%} of points
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
