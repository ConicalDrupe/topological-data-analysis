"""Region-of-interest cropping, upstream of normalization in the shared pipeline
(Raw -> ROI crop -> Normalize). See experiments.md, Experiment 1 pipeline.

Starts with torchxrayvision's own deterministic center-crop + resize utilities (no
model inference). A lung-mask-based variant (PSPNet) may be added later once this
baseline is reviewed -- see logs/exp1_log.md, Experiment 003.
"""

import numpy as np
from torchxrayvision.datasets import XRayCenterCrop, XRayResizer

ROI_VARIANTS: list[tuple[str, dict]] = [
    ("none", {}),
    ("center_crop", {"size": 224}),
]

_CENTER_CROP = XRayCenterCrop()


def center_crop(image: np.ndarray) -> np.ndarray:
    """Crop to a centered square using min(height, width)."""
    return _CENTER_CROP(image[None, ...])[0]


def resize(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize to a fixed size x size image."""
    return XRayResizer(size)(image[None, ...])[0]


def apply_roi_crop(image: np.ndarray, method: str, **params) -> np.ndarray:
    """Apply an ROI-crop variant to a grayscale image in [0, 1].

    method:
      - "none": passthrough.
      - "center_crop": center_crop() then resize(). params: size (default 224).
    """
    if method == "none":
        return image
    if method == "center_crop":
        return resize(center_crop(image), size=params.get("size", 224))
    raise ValueError(f"Unknown ROI crop method: {method!r}")
