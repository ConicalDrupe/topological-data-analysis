"""Access to the CheXpert-v1.0-small dataset under kaggle/. See CLAUDE.md."""

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_ROOT = REPO_ROOT / "kaggle"

_CSV_PATH_PREFIX = "CheXpert-v1.0-small/"

_PATH_RE = re.compile(
    r"^(?P<split>train|valid)/"
    r"(?P<patient_id>patient\d+)/"
    r"study(?P<study_number>\d+)/"
    r"(?P<view>view\d+)_(?P<orientation>frontal|lateral)\.jpg$"
)


def load_labels(split: str) -> pd.DataFrame:
    """Load train.csv or valid.csv, with Path remapped to the on-disk layout.

    The CSV's Path column is prefixed `CheXpert-v1.0-small/`, which the
    extracted kaggle/ directory does not have -- that prefix is stripped here.
    """
    if split not in ("train", "valid"):
        raise ValueError(f"split must be 'train' or 'valid', got {split!r}")
    df = pd.read_csv(KAGGLE_ROOT / f"{split}.csv")
    df["Path"] = df["Path"].str.removeprefix(_CSV_PATH_PREFIX)
    return df


def resolve_image_path(relative_path: str) -> Path:
    """Resolve an already-remapped Path value to an absolute file path."""
    return KAGGLE_ROOT / relative_path


def parse_path_components(relative_path: str) -> dict:
    """Parse patient_id/study_number/view/orientation out of a remapped Path."""
    match = _PATH_RE.match(relative_path)
    if not match:
        raise ValueError(f"Path does not match expected CheXpert layout: {relative_path!r}")
    parsed = match.groupdict()
    parsed["study_number"] = int(parsed["study_number"])
    return parsed
