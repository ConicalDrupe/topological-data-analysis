"""Image loading and normalization (HE / CLAHE) for the shared normalization ->
filtration -> vectorization pipeline. See experiments.md, Experiment 1 pipeline step 2
and "Shared infrastructure".
"""

from pathlib import Path

import numpy as np
from skimage import exposure, img_as_float, io

NORMALIZATION_VARIANTS: list[tuple[str, dict]] = [
    ("none", {}),
    ("he", {"nbins": 256}),
    ("clahe", {"clip_limit": 0.01, "kernel_size": 32}),
]


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
