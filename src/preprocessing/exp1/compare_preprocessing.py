"""Compare HE and CLAHE normalization against the original image, and sweep each
method's parameters, on the representative sample selected by
select_preprocessing_sample.py. See experiments.md, Experiment 1 pipeline step 2.

Also produces the Image Statistics EDA (CLAUDE.md) for the sample: dimensions,
brightness/contrast, and before/after intensity histograms.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tda_chexpr.data import REPO_ROOT, resolve_image_path
from tda_chexpr.eda import next_version_dir
from tda_chexpr.image_eda import image_stats, plot_intensity_histogram_comparison
from tda_chexpr.preprocessing import NORMALIZATION_VARIANTS, apply_normalization, load_image_grayscale

DATA_DIR = REPO_ROOT / "data" / "exp1"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"

DEFAULT_HE_PARAMS = {"nbins": 256}
DEFAULT_CLAHE_PARAMS = {"clip_limit": 0.01, "kernel_size": 32}

CLAHE_CLIP_LIMITS = [0.005, 0.01, 0.02, 0.05]
CLAHE_KERNEL_SIZES = [8, 16, 32, 64]
HE_NBINS = [32, 64, 128, 256]


def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "preprocessing_sample.csv")


def plot_method_comparison(sample: pd.DataFrame, out_path: Path) -> None:
    n = len(sample)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, (_, record) in enumerate(sample.iterrows()):
        image = load_image_grayscale(resolve_image_path(record["Path"]))
        he = apply_normalization(image, "he", **DEFAULT_HE_PARAMS)
        clahe = apply_normalization(image, "clahe", **DEFAULT_CLAHE_PARAMS)
        label = "pos" if record["Pneumothorax"] == 1.0 else "neg"
        for col, (name, im) in enumerate([("Original", image), ("HE", he), ("CLAHE", clahe)]):
            ax = axes[row, col]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(label)
    fig.suptitle("Normalization method comparison (Original / HE / CLAHE)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_clahe_parameter_grid(image: np.ndarray, out_path: Path) -> None:
    n_cols = 1 + max(len(CLAHE_CLIP_LIMITS), len(CLAHE_KERNEL_SIZES))
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

    for row in range(2):
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Original")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

    for col, clip_limit in enumerate(CLAHE_CLIP_LIMITS, start=1):
        out = apply_normalization(
            image, "clahe", clip_limit=clip_limit, kernel_size=DEFAULT_CLAHE_PARAMS["kernel_size"]
        )
        axes[0, col].imshow(out, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"clip_limit={clip_limit}")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
    for col in range(len(CLAHE_CLIP_LIMITS) + 1, n_cols):
        axes[0, col].set_axis_off()

    for col, kernel_size in enumerate(CLAHE_KERNEL_SIZES, start=1):
        out = apply_normalization(
            image, "clahe", clip_limit=DEFAULT_CLAHE_PARAMS["clip_limit"], kernel_size=kernel_size
        )
        axes[1, col].imshow(out, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"kernel_size={kernel_size}")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
    for col in range(len(CLAHE_KERNEL_SIZES) + 1, n_cols):
        axes[1, col].set_axis_off()

    fig.suptitle(
        f"CLAHE parameter sweep (row 1: clip_limit @ kernel_size={DEFAULT_CLAHE_PARAMS['kernel_size']}; "
        f"row 2: kernel_size @ clip_limit={DEFAULT_CLAHE_PARAMS['clip_limit']})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_he_parameter_grid(image: np.ndarray, out_path: Path) -> None:
    n_cols = 1 + len(HE_NBINS)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for col, nbins in enumerate(HE_NBINS, start=1):
        out = apply_normalization(image, "he", nbins=nbins)
        axes[col].imshow(out, cmap="gray", vmin=0, vmax=1)
        axes[col].set_title(f"nbins={nbins}")
        axes[col].set_xticks([])
        axes[col].set_yticks([])
    fig.suptitle("HE parameter sweep (nbins)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    sample = load_sample()
    version_dir = next_version_dir(RESULTS_DIR)
    out_dir = version_dir / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_method_comparison(sample, out_dir / "method_comparison.png")

    representative_record = sample[sample["Pneumothorax"] == 1.0].iloc[0]
    representative_image = load_image_grayscale(resolve_image_path(representative_record["Path"]))
    plot_clahe_parameter_grid(representative_image, out_dir / "clahe_parameter_grid.png")
    plot_he_parameter_grid(representative_image, out_dir / "he_parameter_grid.png")

    stats_records = []
    dimensions = []
    for _, record in sample.iterrows():
        image = load_image_grayscale(resolve_image_path(record["Path"]))
        dimensions.append({"path": record["Path"], "height": image.shape[0], "width": image.shape[1]})
        label = "positive" if record["Pneumothorax"] == 1.0 else "negative"
        for method, params in NORMALIZATION_VARIANTS:
            normalized = apply_normalization(image, method, **params)
            stats_records.append(
                {
                    "path": record["Path"],
                    "label": label,
                    "method": method,
                    "params": params,
                    **image_stats(normalized),
                }
            )
    with open(out_dir / "image_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    widths = [d["width"] for d in dimensions]
    print(f"Sample image widths: min={min(widths)} max={max(widths)} (height fixed at {dimensions[0]['height']})")

    negative_record = sample[sample["Pneumothorax"] == 0.0].iloc[0]
    for record, tag in [(representative_record, "positive"), (negative_record, "negative")]:
        image = load_image_grayscale(resolve_image_path(record["Path"]))
        images = {method: apply_normalization(image, method, **params) for method, params in NORMALIZATION_VARIANTS}
        plot_intensity_histogram_comparison(
            images,
            out_dir / f"intensity_histogram_comparison_{tag}.png",
            title=f"Intensity histogram comparison ({tag} example)",
        )

    print(f"Preprocessing comparison artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
