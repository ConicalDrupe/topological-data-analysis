"""Build an enhanced Mapper graph: same lens/cover/clustering as v1, but the rendered
HTML's existing "Cluster Details" panel (KeplerMapper's own Jinja2/D3 template) is
extended with per-node field-distribution summaries and a browsable image gallery.

See Mapper/SPEC.md Section 6. Distributions are dtype-driven (categorical vs. continuous,
mapper.cluster_details.classify_field), configurable via --detail-fields. Images are
referenced by relative file path (never embedded as base64) and rendered in lazy, batched
thumbnail grids with a click-through lightbox, so a single node holding many thousands of
images stays fast to browse — see mapper.cluster_details / static/cluster_details.js for
the design rationale.

Example:
    uv run python Mapper/scripts/build_mapper_graph_v2.py \\
        --backend medgemma --split train --eps 3.5 --min-samples 5 \\
        --detail-fields Pneumothorax,Age,Sex
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import kmapper as km

from mapper.cluster_details import build_cluster_details_payload, inject_cluster_details
from mapper.data import REPO_ROOT, load_embeddings
from mapper.graph import COVER_CHOICES, LENS_CHOICES, RANDOM_STATE, build_graph


@dataclass
class MapperV2Config:
    backend: str
    split: str
    eps: float
    min_samples: int
    n_cubes: int
    perc_overlap: float
    lens: str
    cover_kind: str
    detail_fields: list[str]
    image_kind: str
    gallery_batch_size: int
    output_html: Path
    log_path: Path


def parse_args() -> MapperV2Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="medgemma")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps (sklearn default)")
    parser.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples (sklearn default)")
    parser.add_argument("--n-cubes", type=int, default=10)
    parser.add_argument("--perc-overlap", type=float, default=0.5)
    parser.add_argument("--lens", default="pca", choices=LENS_CHOICES)
    parser.add_argument("--cover-kind", default="uniform", choices=COVER_CHOICES)
    parser.add_argument(
        "--detail-fields",
        default="Pneumothorax,Age",
        help="Comma-separated metadata columns to summarize per cluster (SPEC.md Section 6.1 default)",
    )
    parser.add_argument("--image-kind", default="processed", choices=["processed", "raw"])
    parser.add_argument("--gallery-batch-size", type=int, default=48)
    parser.add_argument(
        "--output-html",
        type=Path,
        default=REPO_ROOT / "Mapper" / "results" / "v2" / "graphs" / "medgemma_train_mapper.html",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=REPO_ROOT / "Mapper" / "logs" / "mapper_log.md",
    )
    args = parser.parse_args()
    return MapperV2Config(
        backend=args.backend,
        split=args.split,
        eps=args.eps,
        min_samples=args.min_samples,
        n_cubes=args.n_cubes,
        perc_overlap=args.perc_overlap,
        lens=args.lens,
        cover_kind=args.cover_kind,
        detail_fields=[f.strip() for f in args.detail_fields.split(",") if f.strip()],
        image_kind=args.image_kind,
        gallery_batch_size=args.gallery_batch_size,
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

    if result.n_nodes == 0:
        print(
            "*** Mapper graph has 0 nodes — kmapper cannot render this. "
            "Skipping HTML output for this run. Re-run with a larger --eps. ***"
        )
        _append_log_entry(config, df, result, html_written=False, n_skipped_images=0)
        print(f"Appended run entry to {config.log_path}")
        return

    config.output_html.parent.mkdir(parents=True, exist_ok=True)

    mapper = km.KeplerMapper(verbose=1)
    html = mapper.visualize(
        result.graph,
        color_values=df["Pneumothorax"].to_numpy(),
        color_function_name="Pneumothorax (mean)",
        path_html=str(config.output_html),
        title=f"MedGemma {config.split} embeddings — Mapper graph (v2, {config.lens})",
        save_file=False,
        X=result.X,
        X_names=[f"emb_{i:04d}" for i in range(result.X.shape[1])],
        lens=result.lens,
        lens_names=[f"{config.lens.upper()}-1", f"{config.lens.upper()}-2"],
        custom_tooltips=df["patient_id"].to_numpy(),
    )

    print(f"Computing per-node field distributions ({', '.join(config.detail_fields)}) and image gallery data ...")
    payload, n_skipped_images = build_cluster_details_payload(
        result.graph, df, config.detail_fields, config.output_html, config.image_kind
    )
    if n_skipped_images:
        print(f"  ...skipped {n_skipped_images} member(s) with unresolvable image paths")

    html = inject_cluster_details(html, payload, config.gallery_batch_size)

    config.output_html.write_text(html, encoding="utf-8")
    print(f"Wrote enhanced HTML graph to {config.output_html}")

    _append_log_entry(config, df, result, html_written=True, n_skipped_images=n_skipped_images)
    print(f"Appended run entry to {config.log_path}")


def _append_log_entry(config: MapperV2Config, df, result, html_written: bool, n_skipped_images: int) -> None:
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
## Run {timestamp} (v2 — enhanced cluster viewer)

- Dataset: backend={config.backend}, split={config.split}, {len(df)} rows,
  embedding_dim={df["embedding"].iloc[0].shape[0]}
- Lens: {config.lens} (n_components=2, random_state={RANDOM_STATE}) on raw embeddings
- Cover: kind={config.cover_kind}, kmapper.Cover(n_cubes={config.n_cubes}, perc_overlap={config.perc_overlap})
- Clustering: sklearn.cluster.DBSCAN(eps={config.eps}, min_samples={config.min_samples}),
  fit on original {df["embedding"].iloc[0].shape[0]}-d embeddings (not lens-space)
- Node coloring: mean Pneumothorax value among cluster members (0-1 scale)
- Cluster-detail fields: {", ".join(config.detail_fields)}
- Image kind: {config.image_kind}, gallery batch size: {config.gallery_batch_size}
- Output: {config.output_html if html_written else "(not written — see note below)"}
- Mapper graph: {result.n_nodes} nodes, {result.n_edges} edges
- DBSCAN degeneracy check (whole-dataset diagnostic fit): {result.frac_noise:.1%} unclustered
  (label == -1), largest single cluster holds {result.frac_largest_cluster:.1%} of points
- {degeneracy_note}
- Images skipped (unresolvable path): {n_skipped_images}
"""

    if not config.log_path.exists():
        header = "# Experiment 2 — Mapper Graph (v1)\n"
        config.log_path.write_text(header + entry)
    else:
        with config.log_path.open("a") as f:
            f.write(entry)


if __name__ == "__main__":
    main()
