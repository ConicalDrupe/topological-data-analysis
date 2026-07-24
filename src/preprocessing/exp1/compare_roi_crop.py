"""Compare the ROI-crop pipeline stage (Raw -> center-crop -> resize) on the same
representative sample used by compare_preprocessing.py, and re-run the HE/CLAHE
method comparison on the cropped images. See experiments.md, Experiment 1 pipeline,
and logs/exp1_log.md, Experiment 003.
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from tda_chexpr.data import REPO_ROOT, resolve_image_path
from tda_chexpr.eda import next_version_dir
from tda_chexpr.image_eda import image_stats
from tda_chexpr.preprocessing import load_image_grayscale, plot_method_comparison
from tda_chexpr.roi import center_crop, resize

DATA_DIR = REPO_ROOT / "data" / "exp1"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"

RESIZE_SIZE = 224


def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "preprocessing_sample.csv")


def plot_roi_crop_comparison(stage_records: list[tuple], out_path) -> None:
    n = len(stage_records)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, (label, raw, cropped, resized) in enumerate(stage_records):
        stages = [("Raw", raw), ("Center-cropped", cropped), (f"Resized ({RESIZE_SIZE}x{RESIZE_SIZE})", resized)]
        for col, (name, im) in enumerate(stages):
            ax = axes[row, col]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(label)
    fig.suptitle("ROI crop pipeline (Raw / Center-cropped / Resized)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    sample = load_sample()
    version_dir = next_version_dir(RESULTS_DIR)
    out_dir = version_dir / "roi_crop"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_records = []
    cropped_images = []
    stats_records = []
    for _, record in sample.iterrows():
        label = "pos" if record["Pneumothorax"] == 1.0 else "neg"
        raw = load_image_grayscale(resolve_image_path(record["Path"]))
        cropped = center_crop(raw)
        resized = resize(cropped, size=RESIZE_SIZE)
        stage_records.append((label, raw, cropped, resized))
        cropped_images.append((label, resized))

        for stage_name, image in [("raw", raw), ("center_cropped", cropped), ("resized", resized)]:
            stats_records.append(
                {
                    "path": record["Path"],
                    "label": label,
                    "stage": stage_name,
                    **image_stats(image),
                }
            )

    plot_roi_crop_comparison(stage_records, out_dir / "roi_crop_comparison.png")
    plot_method_comparison(cropped_images, out_dir / "method_comparison_cropped.png")

    with open(out_dir / "roi_crop_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    resized_shapes = {(r["height"], r["width"]) for r in stats_records if r["stage"] == "resized"}
    print(f"Resized output shapes across sample: {resized_shapes} (expect a single ({RESIZE_SIZE}, {RESIZE_SIZE}))")
    print(f"ROI crop comparison artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
