"""Convenience wrapper around ``gemma_embeddings.generate_embeddings`` for this repo's
pneumothorax cohort, so callers don't have to remember --input-csv/--image-root paths.

Requires the repo-root environment to be synced (``uv sync`` from the repo root) since
gemma_embeddings is installed as part of the single shared project.

Examples:
    uv run python GenerateEmbeddings/scripts/run_pneumothorax_embeddings.py \\
        --backend siglip --split test

    uv run python GenerateEmbeddings/scripts/run_pneumothorax_embeddings.py \\
        --backend siglip --split both --batch-size 8 --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "exp1" / "v2_corrected_cohort"
IMAGE_ROOT = REPO_ROOT / "kaggle"
OUTPUT_DIR = REPO_ROOT / "results" / "exp2" / "embeddings" / "v1"


def run_split(split: str, backend: str, extra_args: list[str]) -> None:
    input_csv = DATA_DIR / f"pneumothorax_{split}_split.csv"
    output_csv = OUTPUT_DIR / f"{backend}_{split}_embeddings.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "gemma_embeddings.generate_embeddings",
        "--input-csv",
        str(input_csv),
        "--image-root",
        str(IMAGE_ROOT),
        "--output-csv",
        str(output_csv),
        "--backend",
        backend,
        *extra_args,
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--backend", default="siglip")
    args, extra_args = parser.parse_known_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        run_split(split, args.backend, extra_args)


if __name__ == "__main__":
    main()
