"""Compare PSPNet-based lung segmentation ROI cropping against the center-crop
baseline (Experiment 003) on the same representative sample, and re-run the HE/CLAHE
comparison on the lung-mask-cropped images. See experiments.md, Experiment 1
pipeline, and logs/exp1_log.md, Experiment 004.
"""

import json

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tda_chexpr.data import REPO_ROOT, resolve_image_path
from tda_chexpr.eda import next_version_dir
from tda_chexpr.preprocessing import load_image_grayscale, plot_method_comparison
from tda_chexpr.roi import apply_roi_crop, crop_to_bbox, mask_to_bbox
from tda_chexpr.segmentation import predict_lung_mask

DATA_DIR = REPO_ROOT / "data" / "exp1"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"


def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "preprocessing_sample.csv")


def plot_lung_mask_pipeline(records: list[tuple], out_path) -> None:
    n = len(records)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, (label, raw, mask, bbox, cropped) in enumerate(records):
        ax_raw, ax_overlay, ax_cropped = axes[row]

        ax_raw.imshow(raw, cmap="gray", vmin=0, vmax=1)

        ax_overlay.imshow(raw, cmap="gray", vmin=0, vmax=1)
        ax_overlay.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.4)
        if bbox is not None:
            top, left, bottom, right = bbox
            rect = patches.Rectangle(
                (left, top), right - left, bottom - top, linewidth=1.5, edgecolor="lime", facecolor="none"
            )
            ax_overlay.add_patch(rect)

        ax_cropped.imshow(cropped, cmap="gray", vmin=0, vmax=1)

        for ax in (ax_raw, ax_overlay, ax_cropped):
            ax.set_xticks([])
            ax.set_yticks([])
        if row == 0:
            ax_raw.set_title("Raw")
            ax_overlay.set_title("Mask + bbox overlay")
            ax_cropped.set_title("Cropped")
        ax_raw.set_ylabel(label)
    fig.suptitle("PSPNet lung segmentation pipeline (Raw / Mask+bbox / Cropped)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roi_method_comparison(records: list[tuple], out_path) -> None:
    n = len(records)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, (label, raw, center_cropped, lung_cropped) in enumerate(records):
        stages = [("Raw", raw), ("Center-crop", center_cropped), ("Lung-mask crop", lung_cropped)]
        for col, (name, im) in enumerate(stages):
            ax = axes[row, col]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(label)
    fig.suptitle("ROI method comparison (Raw / Center-crop / Lung-mask crop)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    sample = load_sample()
    version_dir = next_version_dir(RESULTS_DIR)
    out_dir = version_dir / "lung_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_records = []
    method_records = []
    lung_cropped_images = []
    stats_records = []

    for _, record in sample.iterrows():
        label = "pos" if record["Pneumothorax"] == 1.0 else "neg"
        raw = load_image_grayscale(resolve_image_path(record["Path"]))
        center_cropped = apply_roi_crop(raw, "center_crop", size=224)

        mask = predict_lung_mask(raw)
        empty_mask = not mask.any()
        if empty_mask:
            bbox = None
            lung_cropped = raw
        else:
            bbox = mask_to_bbox(mask)
            lung_cropped = crop_to_bbox(raw, bbox)

        pipeline_records.append((label, raw, mask, bbox, lung_cropped))
        method_records.append((label, raw, center_cropped, lung_cropped))
        lung_cropped_images.append((label, lung_cropped))

        stats_records.append(
            {
                "path": record["Path"],
                "label": label,
                "raw_height": raw.shape[0],
                "raw_width": raw.shape[1],
                "empty_mask": empty_mask,
                "mask_fraction": float(mask.mean()),
                "bbox": list(bbox) if bbox is not None else None,
                "cropped_height": lung_cropped.shape[0],
                "cropped_width": lung_cropped.shape[1],
            }
        )

    plot_lung_mask_pipeline(pipeline_records, out_dir / "lung_mask_pipeline.png")
    plot_roi_method_comparison(method_records, out_dir / "roi_method_comparison.png")
    plot_method_comparison(lung_cropped_images, out_dir / "method_comparison_lung_cropped.png")

    with open(out_dir / "lung_mask_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    n_empty = sum(r["empty_mask"] for r in stats_records)
    print(f"Empty-mask failures: {n_empty}/{len(stats_records)}")
    print(f"Lung segmentation comparison artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
