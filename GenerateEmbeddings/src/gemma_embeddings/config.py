"""Runtime configuration for the embedding-generation pipeline.

Auth, per backend:
- `siglip` (default preset): ungated, no Hugging Face account needed. Good for
  smoke-testing the pipeline first.
- `rad-dino`: confirmed ungated (MIT-licensed model card) -- no auth needed.
- `medgemma`: gated, like the rest of the Gemma family. Create an HF account, accept
  the license on the `google/medgemma-4b-it` model page, then either run
  `huggingface-cli login` once or set `HF_TOKEN` in the environment before running.
See encoders.py for the full backend preset table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Pooling = Literal["pooler", "cls", "mean_patch"]


@dataclass
class EmbeddingConfig:
    input_csv: Path
    image_root: Path
    output_csv: Path
    backend: str = "medgemma"
    key_column: str = "Path"

    # Overrides -- None means "use the chosen backend preset's value". All four must
    # be set explicitly when backend="custom".
    model_id: str | None = None
    processor_id: str | None = None
    vision_attr: str | None = None
    pooling: Pooling | None = None

    batch_size: int = 16
    device: str = "cpu"
    dtype: str = "bfloat16"
    num_workers: int = 4
    resume: bool = False
    flush_every: int = 1
