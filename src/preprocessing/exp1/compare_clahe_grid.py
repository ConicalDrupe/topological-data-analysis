"""Finalize the Experiment 1 preprocessing pipeline (Raw -> lung-mask crop ->
pad-to-square -> resize to 224x224 -> CLAHE) and grid search CLAHE's clip_limit at a
fixed, smaller kernel_size=8. See experiments.md, Experiment 1 pipeline, and
logs/exp1_log.md, Experiment 003.
"""

import json

import pandas as pd

from tda_chexpr.data import REPO_ROOT, resolve_image_path
from tda_chexpr.eda import next_version_dir
from tda_chexpr.image_eda import image_stats, plot_intensity_histogram_comparison
from tda_chexpr.preprocessing import apply_normalization, load_image_grayscale, plot_stage_grid
from tda_chexpr.roi import apply_roi_crop

DATA_DIR = REPO_ROOT / "data" / "exp1"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"

CLIP_LIMIT_GRID = [0.01, 0.02, 0.03, 0.04, 0.05]
FIXED_KERNEL_SIZE = 8
FIXED_SIZE = 224


def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "preprocessing_sample.csv")


def main() -> None:
    sample = load_sample()
    version_dir = next_version_dir(RESULTS_DIR)
    out_dir = version_dir / "clahe_grid_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_rows = []
    stats_records = []
    representative = {}

    for _, record in sample.iterrows():
        label = "pos" if record["Pneumothorax"] == 1.0 else "neg"
        path = record["Path"]
        raw = load_image_grayscale(resolve_image_path(path))

        roi_failed = False
        try:
            cropped = apply_roi_crop(raw, "lung_mask", margin_frac=0.05, threshold=0.5, size=FIXED_SIZE)
        except ValueError:
            roi_failed = True
            cropped = apply_roi_crop(raw, "center_crop", size=FIXED_SIZE)

        he = apply_normalization(cropped, "he")
        clahe_variants = [
            (clip, apply_normalization(cropped, "clahe", clip_limit=clip, kernel_size=FIXED_KERNEL_SIZE))
            for clip in CLIP_LIMIT_GRID
        ]

        stages = [("Raw", raw), ("Cropped+Resized", cropped), ("HE", he)]
        stages += [(f"CLAHE clip={clip}", im) for clip, im in clahe_variants]
        grid_rows.append((label, stages))

        for stage_name, image in stages:
            stats_records.append(
                {
                    "path": path,
                    "label": label,
                    "stage": stage_name,
                    "roi_crop_fallback": roi_failed,
                    **image_stats(image),
                }
            )

        if label not in representative:
            representative[label] = {"cropped": cropped, "he": he, "clahe_variants": clahe_variants}

    plot_stage_grid(
        grid_rows,
        out_dir / "clahe_grid_comparison.png",
        title="ROI crop + CLAHE clip_limit grid search (kernel_size=8)",
    )

    with open(out_dir / "image_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    for label, stages in representative.items():
        images = {"Cropped (no eq.)": stages["cropped"], "HE": stages["he"]}
        for clip, im in stages["clahe_variants"]:
            if clip in (CLIP_LIMIT_GRID[0], CLIP_LIMIT_GRID[len(CLIP_LIMIT_GRID) // 2], CLIP_LIMIT_GRID[-1]):
                images[f"CLAHE clip={clip}"] = im
        plot_intensity_histogram_comparison(
            images,
            out_dir / f"intensity_histogram_comparison_{label}.png",
            title=f"Intensity histogram comparison ({label} example)",
        )

    n_fallback = sum(1 for r in stats_records if r["roi_crop_fallback"] and r["stage"] == "Raw")
    print(f"ROI-crop fallbacks (empty lung mask -> center_crop): {n_fallback}/{len(sample)}")
    print(f"CLAHE grid search artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
