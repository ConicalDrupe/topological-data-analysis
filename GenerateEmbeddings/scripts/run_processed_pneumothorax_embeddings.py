"""Generate vision-encoder embeddings for the *processed* (PSPNet lung-mask crop +
conservative CLAHE, 224x224 PNG) pneumothorax dataset under ``kaggle/processed/``. See
logs/exp1_log.md Experiment 007 for how that dataset was produced, and
logs/exp2_embeddings_log.md Experiment 008 for this step.

Unlike ``run_pneumothorax_embeddings.py`` (which embeds the raw JPEGs directly, keyed on
their own ``Path`` column), this script:

- Builds a per-split "processed manifest" CSV (merging the existing
  ``pneumothorax_{split}_split.csv`` with the preprocessing manifest's ``output_path``),
  since the processed PNGs live at a different relative path than the raw JPEGs.
- Passes ``--key-column output_path`` so ``gemma_embeddings`` resolves images under
  ``kaggle/processed/`` rather than the raw ``Path`` column.
- Runs all three backends automatically per split (default), instead of one backend per
  invocation.
- Routes each backend's bare ``[output_path, emb_*]`` output to a staging directory, then
  enriches it with label columns (Pneumothorax, patient_id, ...) before writing the final
  CSV -- enriching in place would corrupt a later ``--resume`` run, since
  ``EmbeddingCsvWriter`` appends bare rows under whatever header is already there.

Requires the repo-root environment to be synced (``uv sync`` from the repo root).

Examples:
    uv run python GenerateEmbeddings/scripts/run_processed_pneumothorax_embeddings.py

    uv run python GenerateEmbeddings/scripts/run_processed_pneumothorax_embeddings.py \\
        --split test --backends siglip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO_ROOT / "data" / "exp1" / "v2_corrected_cohort"
PREPROCESSING_MANIFEST = REPO_ROOT / "results" / "exp1" / "preprocessing" / "v1" / "manifest.csv"
IMAGE_ROOT = REPO_ROOT / "kaggle" / "processed"
OUTPUT_DIR = REPO_ROOT / "data" / "embeddings" / "processed"
STAGING_DIR = OUTPUT_DIR / "_raw"

KEY_COLUMN = "output_path"
ALL_BACKENDS = ["siglip", "rad-dino", "medgemma"]
LABEL_COLUMNS = [
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


def build_processed_manifest(split: str) -> pd.DataFrame:
    """Merge pneumothorax_{split}_split.csv with the preprocessing manifest's
    output_path/cohort_split, so the result can serve as both the --input-csv fed to
    gemma_embeddings and the label source for post-hoc enrichment.
    """
    split_df = pd.read_csv(SPLIT_DIR / f"pneumothorax_{split}_split.csv")
    manifest_df = pd.read_csv(PREPROCESSING_MANIFEST)
    manifest_df = manifest_df[
        (manifest_df["status"] == "success") & (manifest_df["cohort_split"] == f"{split}_split")
    ]

    merged = split_df.merge(
        manifest_df[["path", "output_path", "cohort_split"]],
        left_on="Path",
        right_on="path",
        how="inner",
    )
    if len(merged) != len(split_df):
        raise ValueError(
            f"{split}: expected {len(split_df)} rows to match the preprocessing manifest, got {len(merged)}"
        )

    merged = merged.rename(columns={"Path": "raw_path"})
    return merged[LABEL_COLUMNS]


def write_processed_manifest(split: str) -> Path:
    out_path = OUTPUT_DIR / f"processed_manifest_{split}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_processed_manifest(split).to_csv(out_path, index=False)
    return out_path


def run_backend_split(backend: str, split: str, manifest_csv: Path, extra_args: list[str]) -> Path:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    bare_csv = STAGING_DIR / f"{backend}_{split}_embeddings.csv"

    cmd = [
        sys.executable,
        "-m",
        "gemma_embeddings.generate_embeddings",
        "--input-csv",
        str(manifest_csv),
        "--image-root",
        str(IMAGE_ROOT),
        "--output-csv",
        str(bare_csv),
        "--backend",
        backend,
        "--key-column",
        KEY_COLUMN,
        *extra_args,
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return bare_csv


def enrich_output(bare_csv: Path, manifest_df: pd.DataFrame, final_csv: Path) -> None:
    bare_df = pd.read_csv(bare_csv)
    emb_cols = [c for c in bare_df.columns if c != KEY_COLUMN]

    merged = manifest_df.merge(bare_df, on=KEY_COLUMN, how="inner")
    if len(merged) != len(bare_df):
        raise ValueError(
            f"enrich_output: {bare_csv} has {len(bare_df)} rows, only {len(merged)} matched the manifest"
        )

    final_csv.parent.mkdir(parents=True, exist_ok=True)
    merged[LABEL_COLUMNS + emb_cols].to_csv(final_csv, index=False)


def copy_meta_with_note(bare_csv: Path, final_csv: Path, manifest_csv: Path) -> None:
    bare_meta_path = bare_csv.with_suffix(bare_csv.suffix + ".meta.json")
    final_meta_path = final_csv.with_suffix(final_csv.suffix + ".meta.json")
    if not bare_meta_path.exists():
        return
    meta = json.loads(bare_meta_path.read_text())
    meta["label_columns_from"] = manifest_csv.relative_to(REPO_ROOT).as_posix()
    final_meta_path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--backends", nargs="+", choices=ALL_BACKENDS, default=ALL_BACKENDS)
    args, extra_args = parser.parse_known_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        manifest_csv = write_processed_manifest(split)
        manifest_df = pd.read_csv(manifest_csv)
        print(f"[{split}] processed manifest: {manifest_csv} ({len(manifest_df)} rows)")

        for backend in args.backends:
            bare_csv = run_backend_split(backend, split, manifest_csv, extra_args)
            final_csv = OUTPUT_DIR / f"{backend}_{split}_embeddings.csv"
            enrich_output(bare_csv, manifest_df, final_csv)
            copy_meta_with_note(bare_csv, final_csv, manifest_csv)
            print(f"[{split}/{backend}] wrote {final_csv}")


if __name__ == "__main__":
    main()
