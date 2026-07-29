"""CLI entrypoint: generate vision-encoder embeddings for a CheXpert-style CSV manifest.

Output is a CSV keyed on the same `Path` foreign key topological-data-analysis's
`tda_chexpr` uses throughout (e.g. `train/patient00001/study1/view1_frontal.jpg`), plus
a sidecar `<output_csv>.meta.json` describing the model/pooling that produced it. See
scripts/validate_join.py to check the output joins cleanly onto a manifest CSV.

Example:
    uv run python -m gemma_embeddings.generate_embeddings \\
        --input-csv path/to/preprocessing_sample.csv \\
        --image-root /path/to/kaggle \\
        --output-csv embeddings.csv \\
        --backend siglip

See config.py and encoders.py for backend/auth details (siglip is ungated and is the
fastest way to smoke-test this end-to-end before dealing with MedGemma's gated access).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gemma_embeddings.config import EmbeddingConfig
from gemma_embeddings.dataset import ImageManifestDataset, collate_skip_failed
from gemma_embeddings.encoders import BACKEND_PRESETS, get_encoder
from gemma_embeddings.io_utils import EmbeddingCsvWriter, load_done_keys, write_run_metadata


def parse_args() -> EmbeddingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--backend", default="medgemma", choices=[*BACKEND_PRESETS, "custom"])
    parser.add_argument("--key-column", default="Path")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--vision-attr", default=None)
    parser.add_argument("--pooling", default=None, choices=["pooler", "cls", "mean_patch"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1, help="print progress every N batches")
    args = parser.parse_args()

    return EmbeddingConfig(
        input_csv=args.input_csv,
        image_root=args.image_root,
        output_csv=args.output_csv,
        backend=args.backend,
        key_column=args.key_column,
        model_id=args.model_id,
        processor_id=args.processor_id,
        vision_attr=args.vision_attr,
        pooling=args.pooling,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        num_workers=args.num_workers,
        resume=args.resume,
        flush_every=args.flush_every,
    )


def main() -> None:
    config = parse_args()

    print(f"Loading backend={config.backend!r} ...")
    encoder = get_encoder(config)
    print(
        f"Loaded {encoder.model_id} (vision_attr={encoder.vision_attr!r}, "
        f"pooling={encoder.pooling!r}, embedding_dim={encoder.embedding_dim})"
    )

    done_keys = load_done_keys(config.output_csv, config.key_column) if config.resume else set()
    if done_keys:
        print(f"Resuming: {len(done_keys)} keys already present in {config.output_csv}")

    dataset = ImageManifestDataset(
        config.input_csv, config.image_root, config.key_column, done_keys=done_keys
    )
    print(f"{len(dataset)} images remaining to embed")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_skip_failed,
    )

    writer = EmbeddingCsvWriter(
        config.output_csv, config.key_column, encoder.embedding_dim, resume=config.resume
    )

    num_batches = 0
    num_embedded = 0
    try:
        for keys, images in loader:
            if not keys:
                continue
            embeddings = encoder.embed(images)
            writer.write_batch(keys, embeddings)
            num_batches += 1
            num_embedded += len(keys)
            if num_batches % max(config.flush_every, 1) == 0:
                print(f"  ...{num_embedded} images embedded")
    finally:
        writer.close()

    write_run_metadata(
        config.output_csv,
        config,
        encoder.embedding_dim,
        resolved_model_id=encoder.model_id,
        resolved_vision_attr=encoder.vision_attr,
        resolved_pooling=encoder.pooling,
    )
    print(f"Done: {num_embedded} embeddings written to {config.output_csv}")


if __name__ == "__main__":
    main()
