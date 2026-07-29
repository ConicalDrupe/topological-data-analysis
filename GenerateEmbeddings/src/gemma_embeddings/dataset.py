"""Reads a CheXpert-style manifest CSV and yields (key, PIL.Image) pairs.

Mirrors the relative-path convention used in topological-data-analysis's
`tda_chexpr.data.resolve_image_path` (Path column values like
`train/patient00001/study1/view1_frontal.jpg`, resolved against a root directory)
without importing that project -- `image_root` is supplied explicitly via CLI so this
project has no dependency on where that repo's `kaggle/` checkout lives.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ImageManifestDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        key_column: str = "Path",
        done_keys: set[str] | None = None,
    ) -> None:
        df = pd.read_csv(csv_path)
        if key_column not in df.columns:
            raise ValueError(f"{key_column!r} not found in {csv_path}; columns: {list(df.columns)}")
        if done_keys:
            df = df[~df[key_column].isin(done_keys)]
        self.keys = df[key_column].tolist()
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[str, Image.Image | None]:
        key = self.keys[idx]
        image_path = self.image_root / key
        try:
            with Image.open(image_path) as img:
                return key, img.convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Failed to load %s: %s", image_path, exc)
            return key, None


def collate_skip_failed(
    batch: list[tuple[str, Image.Image | None]],
) -> tuple[list[str], list[Image.Image]]:
    keys: list[str] = []
    images: list[Image.Image] = []
    for key, image in batch:
        if image is not None:
            keys.append(key)
            images.append(image)
    return keys, images
