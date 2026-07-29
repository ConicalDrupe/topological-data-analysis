"""CSV/metadata I/O helpers for the embedding-generation pipeline."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gemma_embeddings.config import EmbeddingConfig


def embedding_column_names(dim: int) -> list[str]:
    width = len(str(dim - 1))
    return [f"emb_{i:0{width}d}" for i in range(dim)]


class EmbeddingCsvWriter:
    """Append-mode incremental CSV writer so a crash mid-run keeps completed batches."""

    def __init__(
        self,
        output_csv: Path,
        key_column: str,
        embedding_dim: int,
        resume: bool = False,
    ) -> None:
        self.output_csv = output_csv
        self.key_column = key_column
        self.columns = [key_column, *embedding_column_names(embedding_dim)]
        write_header = not (resume and output_csv.exists())
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._file = output_csv.open("a" if resume else "w", newline="")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(self.columns)
            self._file.flush()

    def write_batch(self, keys: list[str], embeddings: np.ndarray) -> None:
        for key, vector in zip(keys, embeddings):
            self._writer.writerow([key, *vector.tolist()])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def load_done_keys(output_csv: Path, key_column: str) -> set[str]:
    if not output_csv.exists():
        return set()
    return set(pd.read_csv(output_csv, usecols=[key_column])[key_column])


def write_run_metadata(
    output_csv: Path,
    config: EmbeddingConfig,
    embedding_dim: int,
    resolved_model_id: str,
    resolved_vision_attr: str,
    resolved_pooling: str,
) -> None:
    meta_path = output_csv.with_suffix(output_csv.suffix + ".meta.json")
    metadata = {
        "backend": config.backend,
        "model_id": resolved_model_id,
        "vision_attr": resolved_vision_attr,
        "pooling": resolved_pooling,
        "embedding_dim": embedding_dim,
        "key_column": config.key_column,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
