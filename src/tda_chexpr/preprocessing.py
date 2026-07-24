"""Image loading and normalization (HE / CLAHE) for the shared normalization ->
filtration -> vectorization pipeline. See experiments.md, Experiment 1 pipeline step 2
and "Shared infrastructure".

Also holds the HE/CLAHE comparison plotting helpers (moved here from
preprocessing/exp1/compare_preprocessing.py so they're reusable on any set of
already-loaded images, e.g. cropped images from the ROI-crop stage).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, img_as_float, io

NORMALIZATION_VARIANTS: list[tuple[str, dict]] = [
    ("none", {}),
    ("he", {"nbins": 256}),
    ("clahe", {"clip_limit": 0.01, "kernel_size": 32}),
]

DEFAULT_HE_PARAMS = {"nbins": 256}
DEFAULT_CLAHE_PARAMS = {"clip_limit": 0.01, "kernel_size": 32}

CLAHE_CLIP_LIMITS = [0.005, 0.01, 0.02, 0.05]
CLAHE_KERNEL_SIZES = [8, 16, 32, 64]
HE_NBINS = [32, 64, 128, 256]


def load_image_grayscale(path: Path) -> np.ndarray:
    """Load a CheXpert image as a 2D float64 array in [0, 1].

    CheXpert-small images are already single-channel grayscale (PIL mode "L").
    """
    image = io.imread(path)
    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel grayscale image, got shape {image.shape}: {path}")
    return img_as_float(image)


def apply_normalization(image: np.ndarray, method: str, **params) -> np.ndarray:
    """Apply a normalization variant to a grayscale image in [0, 1].

    method:
      - "none": passthrough.
      - "he": global histogram equalization (skimage.exposure.equalize_hist).
        params: nbins (default 256).
      - "clahe": contrast-limited adaptive histogram equalization
        (skimage.exposure.equalize_adapthist). params: clip_limit (default 0.01),
        kernel_size (default None, i.e. skimage's own default of image_shape // 8),
        nbins (default 256).
    """
    if method == "none":
        return image
    if method == "he":
        return exposure.equalize_hist(image, nbins=params.get("nbins", 256))
    if method == "clahe":
        return exposure.equalize_adapthist(
            image,
            kernel_size=params.get("kernel_size"),
            clip_limit=params.get("clip_limit", 0.01),
            nbins=params.get("nbins", 256),
        )
    raise ValueError(f"Unknown normalization method: {method!r}")


def plot_method_comparison(
    images: list[tuple[str, np.ndarray]],
    out_path: Path,
    he_params: dict | None = None,
    clahe_params: dict | None = None,
) -> None:
    """Grid of [Original, HE, CLAHE], one row per (row_label, image) pair."""
    he_params = he_params if he_params is not None else DEFAULT_HE_PARAMS
    clahe_params = clahe_params if clahe_params is not None else DEFAULT_CLAHE_PARAMS
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, (label, image) in enumerate(images):
        he = apply_normalization(image, "he", **he_params)
        clahe = apply_normalization(image, "clahe", **clahe_params)
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
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
