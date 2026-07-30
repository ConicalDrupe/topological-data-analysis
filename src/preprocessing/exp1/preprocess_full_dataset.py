"""Full-dataset batch preprocessing for Experiment 1: Raw -> lung-mask ROI crop
(224x224) -> CLAHE -> uint8 PNG, for every row of both cohort splits. Writes processed
images to kaggle/processed/ (mirroring kaggle/'s own train/... layout) and a per-row
manifest recording success/failure. See experiments.md Experiment 1 pipeline,
logs/exp1_log.md Experiment 007.

CLAHE clip_limit=0.002 here is a deliberate, more conservative deviation from
Experiment 003's grid-searched default (clip_limit=0.01) -- see the Experiment 007 log
entry for why.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from skimage import io as skio
from skimage.util import img_as_ubyte

from tda_chexpr.data import KAGGLE_ROOT, REPO_ROOT, resolve_image_path
from tda_chexpr.eda import next_version_dir
from tda_chexpr.preprocessing import apply_normalization, load_image_grayscale
from tda_chexpr.roi import apply_roi_crop
from tda_chexpr.segmentation import get_pspnet_model

DATA_DIR = REPO_ROOT / "data" / "exp1" / "v2_corrected_cohort"
PROCESSED_ROOT = KAGGLE_ROOT / "processed"
MANIFEST_BASE_DIR = REPO_ROOT / "results" / "exp1" / "preprocessing"

SPLIT_FILES = {
    "train_split": DATA_DIR / "pneumothorax_train_split.csv",
    "test_split": DATA_DIR / "pneumothorax_test_split.csv",
}

ROI_PARAMS = {"margin_frac": 0.05, "threshold": 0.5, "size": 224}
CLAHE_PARAMS = {"clip_limit": 0.002, "kernel_size": 16, "nbins": 256}


@dataclass
class RowResult:
    path: str
    cohort_split: str
    pneumothorax: float
    status: str
    error_message: str
    output_path: str
    raw_height: int | None = None
    raw_width: int | None = None
    processed_height: int | None = None
    processed_width: int | None = None
    duration_sec: float = 0.0


def output_path_for(relative_path: str) -> Path:
    """Map a source Path ('train/patientNNNNN/studyN/viewN_frontal.jpg') to its
    kaggle/processed/ output path, preserving the folder structure and changing the
    extension to .png.
    """
    return PROCESSED_ROOT / Path(relative_path).with_suffix(".png")


def process_row(row: pd.Series, cohort_split: str) -> RowResult:
    """Run the fixed pipeline on one row, catching failures per-stage so one bad
    image never aborts the batch.
    """
    relative_path = row["Path"]
    start = time.perf_counter()
    base = dict(path=relative_path, cohort_split=cohort_split, pneumothorax=row["Pneumothorax"])

    try:
        raw = load_image_grayscale(resolve_image_path(relative_path))
    except ValueError as exc:
        return RowResult(
            **base, status="load_failed", error_message=str(exc),
            output_path="", duration_sec=time.perf_counter() - start,
        )

    try:
        cropped = apply_roi_crop(raw, "lung_mask", **ROI_PARAMS)
    except ValueError as exc:
        return RowResult(
            **base, status="roi_crop_failed", error_message=str(exc),
            output_path="", raw_height=raw.shape[0], raw_width=raw.shape[1],
            duration_sec=time.perf_counter() - start,
        )

    try:
        normalized = apply_normalization(cropped, "clahe", **CLAHE_PARAMS)
        out_path = output_path_for(relative_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        skio.imsave(out_path, img_as_ubyte(normalized), check_contrast=False)
    except Exception as exc:  # last-resort catch-all: never let one row kill the batch
        return RowResult(
            **base, status="save_failed", error_message=f"{type(exc).__name__}: {exc}",
            output_path="", raw_height=raw.shape[0], raw_width=raw.shape[1],
            duration_sec=time.perf_counter() - start,
        )

    return RowResult(
        **base, status="success", error_message="",
        output_path=out_path.relative_to(PROCESSED_ROOT).as_posix(),
        raw_height=raw.shape[0], raw_width=raw.shape[1],
        processed_height=normalized.shape[0], processed_width=normalized.shape[1],
        duration_sec=time.perf_counter() - start,
    )


def main() -> None:
    manifest_dir = next_version_dir(MANIFEST_BASE_DIR)

    load_start = time.perf_counter()
    get_pspnet_model()
    model_load_sec = time.perf_counter() - load_start
    print(f"PSPNet model loaded in {model_load_sec:.1f}s")

    results: list[RowResult] = []
    batch_start = time.perf_counter()
    for cohort_split, csv_path in SPLIT_FILES.items():
        df = pd.read_csv(csv_path)
        for i, row in df.iterrows():
            result = process_row(row, cohort_split)
            results.append(result)
            if result.status != "success":
                print(f"[{cohort_split}] FAILED {result.path}: {result.status} - {result.error_message}")
            if (i + 1) % 50 == 0:
                print(f"[{cohort_split}] {i + 1}/{len(df)} processed")
    batch_sec = time.perf_counter() - batch_start

    manifest_df = pd.DataFrame([asdict(r) for r in results])
    manifest_path = manifest_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    n_success = int((manifest_df["status"] == "success").sum())
    n_failed = len(manifest_df) - n_success
    print(f"Processed {len(manifest_df)} rows: {n_success} success, {n_failed} failed")
    print(
        f"Model load: {model_load_sec:.1f}s; batch: {batch_sec:.1f}s "
        f"({batch_sec / len(manifest_df):.2f}s/image avg)"
    )
    print(f"Manifest written to {manifest_path}")
    if n_failed:
        print(
            manifest_df.loc[
                manifest_df["status"] != "success", ["path", "cohort_split", "status", "error_message"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
