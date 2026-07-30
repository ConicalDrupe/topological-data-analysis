"""Enhanced cluster-detail payload: per-node field distributions + image gallery data.

Extends KeplerMapper's own rendered HTML (does not replace it) by post-processing the
HTML string `KeplerMapper().visualize(..., save_file=False)` returns: a JSON payload of
per-node extra data is injected as a JS global, plus JS/CSS (Mapper/src/mapper/static/
cluster_details.js, .css) that hooks into kmapper's existing `set_focus_node` panel to
render it. See Mapper/SPEC.md Section 6.

Node keys in the payload match `graph["nodes"]` keys exactly (kmapper's own
"cubeN_clusterM" strings) — these are the same strings kmapper embeds as `d.name` in its
own graph JSON, so no translation is needed to look up a clicked node's extra data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from mapper.data import resolve_image_path

STATIC_DIR = Path(__file__).resolve().parent / "static"


def classify_field(series: pd.Series, categorical_max_unique: int = 10) -> Literal["categorical", "continuous"]:
    """
    Dtype-driven classification, not per-field special-casing (SPEC.md Section 6.1):
    non-numeric dtypes (object/string/bool) are always categorical. Numeric dtypes with
    few unique values (e.g. Pneumothorax: float, 2 values) are also categorical — this
    generalizes to any low-cardinality numeric column without hardcoding field names.
    Numeric dtypes above the threshold are continuous.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "categorical"
    if series.nunique(dropna=True) <= categorical_max_unique:
        return "categorical"
    return "continuous"


def summarize_field(series: pd.Series, kind: Literal["categorical", "continuous"]) -> dict:
    missing = int(series.isna().sum())

    if kind == "categorical":
        counts = series.value_counts(dropna=True)
        total = int(counts.sum())
        return {
            "type": "categorical",
            "counts": {str(k): int(v) for k, v in counts.items()},
            "proportions": {str(k): (float(v) / total if total else 0.0) for k, v in counts.items()},
            "missing": missing,
        }

    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"type": "continuous", "mean": None, "median": None, "std": None, "min": None, "max": None, "missing": missing, "histogram": []}

    counts, bin_edges = np.histogram(values, bins=10)
    histogram = [
        {"bin_start": float(bin_edges[i]), "bin_end": float(bin_edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]
    return {
        "type": "continuous",
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "missing": missing,
        "histogram": histogram,
    }


def compute_node_distributions(
    df: pd.DataFrame,
    member_indices: list[int],
    detail_fields: list[str],
    field_types: dict[str, Literal["categorical", "continuous"]],
) -> dict[str, dict]:
    subset = df.iloc[member_indices]
    return {
        field: summarize_field(subset[field], field_types[field])
        for field in detail_fields
        if field in subset.columns
    }


def _format_tooltip(row: pd.Series, detail_fields: list[str]) -> str:
    parts = [str(row["patient_id"])]
    for field in detail_fields:
        if field in row.index:
            parts.append(f"{field}: {row[field]}")
    return " | ".join(parts)


def compute_node_images(
    df: pd.DataFrame,
    member_indices: list[int],
    output_html_path: Path,
    detail_fields: list[str],
    image_kind: str = "processed",
) -> tuple[list[dict], int]:
    """
    Returns (images, n_skipped). Unlike resolve_image_path's own contract (raise loudly —
    useful for the spot-check use case it's designed for), this walks potentially many
    rows and must not let one missing file abort the whole graph build: missing images are
    skipped and counted instead.
    """
    images = []
    n_skipped = 0
    output_dir = output_html_path.parent
    for idx in member_indices:
        row = df.iloc[idx]
        try:
            abs_path = resolve_image_path(row, kind=image_kind)
        except FileNotFoundError:
            n_skipped += 1
            continue
        rel_path = os.path.relpath(abs_path, start=output_dir)
        images.append(
            {
                "path": rel_path,
                "patient_id": row["patient_id"],
                "tooltip": _format_tooltip(row, detail_fields),
            }
        )
    return images, n_skipped


def build_cluster_details_payload(
    graph: dict,
    df: pd.DataFrame,
    detail_fields: list[str],
    output_html_path: Path,
    image_kind: str = "processed",
) -> tuple[dict, int]:
    # Classified once from the FULL column, not per-node: a field's type (e.g. Age is
    # continuous) must stay consistent across nodes regardless of how many distinct
    # values happen to appear in any single node's small member subset.
    field_types = {field: classify_field(df[field]) for field in detail_fields if field in df.columns}

    payload = {}
    total_skipped = 0
    for node_name, member_indices in graph["nodes"].items():
        images, n_skipped = compute_node_images(df, member_indices, output_html_path, detail_fields, image_kind)
        total_skipped += n_skipped
        payload[node_name] = {
            "distributions": compute_node_distributions(df, member_indices, detail_fields, field_types),
            "images": images,
        }
    return payload, total_skipped


def inject_cluster_details(html: str, payload: dict, gallery_batch_size: int) -> str:
    """Splices our own <style>/<script> blocks into kmapper's rendered HTML, just before
    </body>. Note: kmapper's HTML still loads d3/file-saver from a CDN (base.html) — an
    existing, pre-existing dependency on network access, not something this function
    changes. kmapper's own bundled static/d3.min.js is an incompatible legacy d3 version
    (no d3.scalePow, breaks the whole graph) so it is deliberately NOT used as a
    replacement here."""
    js_text = (STATIC_DIR / "cluster_details.js").read_text()
    css_text = (STATIC_DIR / "cluster_details.css").read_text()

    injected = f"""
<style>{css_text}</style>
<script>
const clusterDetailsExtra = {json.dumps(payload)};
const GALLERY_BATCH_SIZE = {gallery_batch_size};
{js_text}
</script>
"""

    if "</body>" not in html:
        raise ValueError("Expected kmapper's rendered HTML to contain a </body> tag")
    return html.replace("</body>", injected + "</body>")
