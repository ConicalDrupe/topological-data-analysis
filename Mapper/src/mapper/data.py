"""Embedding loading and image-path resolution utilities for the Mapper POC.

Mirrors the data-access conventions in `GenerateEmbeddings/src/gemma_embeddings`:
CSV-backed embeddings keyed on `output_path`, image roots under `kaggle/`.

Example:
    from mapper.data import load_embeddings, resolve_image_path

    df = load_embeddings(backend="medgemma", split="train")
    img_path = resolve_image_path(df.iloc[0], kind="processed")
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

METADATA_COLUMNS = [
    "output_path",
    "raw_path",
    "patient_id",
    "Pneumothorax",
    "cohort_split",
    "Sex",
    "Age",
    "comorbidity_count",
    "is_clean_negative",
]


def load_embeddings(backend: str = "medgemma", split: str = "train") -> pd.DataFrame:
    """
    Loads data/embeddings/processed/{backend}_{split}_embeddings.csv.
    Casts emb_* columns to float and packs them into a single `embedding` column
    of np.ndarray (shape (embedding_dim,) per row). Returns the 9 metadata columns
    unchanged plus this `embedding` column — drop the individual emb_* columns
    from the returned frame to avoid an unwieldy 1152-wide DataFrame.
    """
    csv_path = REPO_ROOT / "data" / "embeddings" / "processed" / f"{backend}_{split}_embeddings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Embeddings CSV not found: {csv_path}")

    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    df = pd.read_csv(csv_path)

    emb_cols = sorted(c for c in df.columns if c.startswith("emb_"))
    if not emb_cols:
        raise ValueError(f"No emb_* columns found in {csv_path}")

    embedding_dim = meta.get("embedding_dim", len(emb_cols))
    if len(emb_cols) != embedding_dim:
        raise ValueError(
            f"Expected {embedding_dim} embedding columns per meta.json, found {len(emb_cols)}"
        )

    embedding_matrix = df[emb_cols].astype(float).to_numpy()
    packed = pd.Series(list(embedding_matrix), index=df.index, name="embedding")

    metadata_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    result = df[metadata_cols].copy()
    result["embedding"] = packed
    return result


def resolve_image_path(row: pd.Series, kind: str = "processed") -> Path:
    """
    kind="processed" -> repo_root / "kaggle" / "processed" / row["output_path"]
    kind="raw"        -> repo_root / "kaggle" / row["raw_path"]
    Must check the resolved path exists on disk and raise if it doesn't —
    do not silently return a dangling path.
    """
    if kind == "processed":
        path = REPO_ROOT / "kaggle" / "processed" / row["output_path"]
    elif kind == "raw":
        path = REPO_ROOT / "kaggle" / row["raw_path"]
    else:
        raise ValueError(f"kind must be 'processed' or 'raw', got {kind!r}")

    if not path.exists():
        raise FileNotFoundError(f"Resolved image path does not exist: {path}")
    return path
