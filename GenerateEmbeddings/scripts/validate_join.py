"""Sanity-check that a generated embeddings CSV joins cleanly onto a
topological-data-analysis-style manifest CSV, via the shared `Path` foreign key.

Standalone: does not import tda_chexpr, just plain pandas, so it can run from either
project's environment.

Usage:
    uv run python scripts/validate_join.py \\
        --manifest-csv /path/to/preprocessing_sample.csv \\
        --embeddings-csv /path/to/embeddings.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--embeddings-csv", type=Path, required=True)
    parser.add_argument("--key-column", default="Path")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    embeddings = pd.read_csv(args.embeddings_csv)

    if args.key_column not in manifest.columns:
        raise SystemExit(f"{args.key_column!r} not in manifest columns: {list(manifest.columns)}")
    if args.key_column not in embeddings.columns:
        raise SystemExit(f"{args.key_column!r} not in embeddings columns: {list(embeddings.columns)}")

    embedding_cols = [c for c in embeddings.columns if c != args.key_column]
    if not embedding_cols:
        raise SystemExit(f"{args.embeddings_csv} has no embedding columns besides {args.key_column!r}")

    merged = manifest.merge(embeddings, on=args.key_column, how="left")
    matched = merged[embedding_cols[0]].notna().sum()
    total = len(manifest)

    print(f"Manifest rows: {total}")
    print(f"Embeddings rows: {len(embeddings)}")
    print(f"Embedding dim: {len(embedding_cols)}")
    print(f"Matched rows: {matched}" + (f" ({matched / total:.1%})" if total else ""))

    missing = merged.loc[merged[embedding_cols[0]].isna(), args.key_column]
    if len(missing):
        print(f"First {min(10, len(missing))} of {len(missing)} unmatched keys:")
        for key in missing.head(10):
            print(f"  {key}")


if __name__ == "__main__":
    main()
