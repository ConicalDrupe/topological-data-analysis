"""Image Statistics EDA helpers. See CLAUDE.md, "Automatic Exploratory Data Analysis
(EDA)" -> Image Statistics, for the schema this implements.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def image_stats(image: np.ndarray, n_hist_bins: int = 16) -> dict:
    """Dimension, brightness, and contrast statistics for a single image.

    Computed directly in the array's own value range (e.g. [0, 1] float for images
    coming out of preprocessing.load_image_grayscale / apply_normalization), so
    before/after comparisons across normalization methods stay apples-to-apples.
    """
    height, width = image.shape
    counts, bin_edges = np.histogram(image, bins=n_hist_bins)
    return {
        "height": height,
        "width": width,
        "aspect_ratio": width / height,
        "min": float(image.min()),
        "max": float(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
        "histogram_counts": counts.tolist(),
        "histogram_bin_edges": bin_edges.tolist(),
    }


def plot_intensity_histogram_comparison(images: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    """Overlay intensity histograms for a {method_name: image} mapping, so
    before/after normalization effects on the pixel distribution are visible
    directly (per CLAUDE.md's before/after comparison requirement).
    """
    fig, ax = plt.subplots()
    for name, image in images.items():
        ax.hist(image.ravel(), bins=64, histtype="step", label=name, density=True)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
