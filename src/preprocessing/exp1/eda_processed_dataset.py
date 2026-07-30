"""EDA for the Experiment 1 full-dataset preprocessing run (Raw -> lung-mask crop ->
CLAHE -> uint8 PNG). Reads results/exp1/preprocessing/<latest>/manifest.csv plus the
already-saved raw and processed images; never re-runs PSPNet. See CLAUDE.md
"Automatic Exploratory Data Analysis", logs/exp1_log.md Experiment 007.
"""

from __future__ import annotations

import json

import pandas as pd
from skimage import img_as_float
from skimage import io as skio

from tda_chexpr.data import KAGGLE_ROOT, REPO_ROOT, resolve_image_path
from tda_chexpr.eda import next_version_dir, print_summary, save_summary_json, summarize_cohort
from tda_chexpr.image_eda import image_stats, plot_intensity_histogram_comparison

DATA_DIR = REPO_ROOT / "data" / "exp1" / "v2_corrected_cohort"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"
MANIFEST_BASE_DIR = REPO_ROOT / "results" / "exp1" / "preprocessing"

N_PER_CLASS = 15
RANDOM_STATE = 42


def latest_manifest_path():
    versions = [
        p for p in MANIFEST_BASE_DIR.iterdir() if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    ]
    latest = max(versions, key=lambda p: int(p.name[1:]))
    return latest / "manifest.csv"


def dataset_summary(manifest: pd.DataFrame, version_dir) -> None:
    """CLAUDE.md Dataset Summary, augmented with preprocessing success/failure counts
    on top of the existing cohort-level summarize_cohort() stats.
    """
    for cohort_split in ("train_split", "test_split"):
        df = pd.read_csv(DATA_DIR / f"pneumothorax_{cohort_split}.csv")
        summary = summarize_cohort(df, label="Pneumothorax")
        sub = manifest[manifest["cohort_split"] == cohort_split]
        summary["preprocessing_status_counts"] = sub["status"].value_counts().to_dict()
        summary["preprocessing_mean_duration_sec"] = float(sub["duration_sec"].mean())
        print_summary(summary, title=f"Experiment 1 full preprocessing - {cohort_split}")
        save_summary_json(summary, version_dir / cohort_split / "summary.json")


def stratified_sample(manifest: pd.DataFrame) -> pd.DataFrame:
    """15 pos / 15 neg, seed=42, drawn only from successfully-processed rows (pooled
    across both cohort splits).
    """
    ok = manifest[manifest["status"] == "success"]
    pos = ok[ok["pneumothorax"] == 1.0].sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
    neg = ok[ok["pneumothorax"] == 0.0].sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
    return pd.concat([pos, neg]).reset_index(drop=True)


def load_before_after(row: pd.Series) -> tuple:
    raw = img_as_float(skio.imread(resolve_image_path(row["path"])))
    processed = img_as_float(skio.imread(KAGGLE_ROOT / "processed" / row["output_path"]))
    return raw, processed


def main() -> None:
    manifest_path = latest_manifest_path()
    manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest from {manifest_path} ({len(manifest)} rows)")

    version_dir = next_version_dir(RESULTS_DIR)

    dataset_summary(manifest, version_dir)

    sample = stratified_sample(manifest)
    sample.to_csv(version_dir / "full_preprocessing_eda_sample.csv", index=False)

    stats_records = []
    pos_images: dict = {}
    neg_images: dict = {}
    for _, row in sample.iterrows():
        raw, processed = load_before_after(row)
        label = "pos" if row["pneumothorax"] == 1.0 else "neg"
        for stage_name, image in [("raw", raw), ("processed", processed)]:
            stats_records.append(
                {"path": row["path"], "label": label, "stage": stage_name, **image_stats(image)}
            )
        bucket = pos_images if label == "pos" else neg_images
        if len(bucket) < 2:
            bucket[f"{row['path']}_raw"] = raw
            bucket[f"{row['path']}_processed"] = processed

    with open(version_dir / "image_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    plot_intensity_histogram_comparison(
        pos_images, version_dir / "intensity_histogram_positive.png",
        title="Raw vs. processed intensity (positive example)",
    )
    plot_intensity_histogram_comparison(
        neg_images, version_dir / "intensity_histogram_negative.png",
        title="Raw vs. processed intensity (negative example)",
    )

    processed_shapes = {(r["height"], r["width"]) for r in stats_records if r["stage"] == "processed"}
    print(f"Processed output shapes across sample: {processed_shapes} (expect a single (224, 224))")
    print(f"EDA artifacts written to {version_dir}")


if __name__ == "__main__":
    main()
