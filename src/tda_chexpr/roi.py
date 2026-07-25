"""Region-of-interest cropping, upstream of normalization in the shared pipeline
(Raw -> ROI crop -> Normalize). See experiments.md, Experiment 1 pipeline.

Starts with torchxrayvision's own deterministic center-crop + resize utilities (no
model inference), and now also a content-aware lung-mask-based crop (PSPNet) -- see
logs/exp1_log.md, Experiments 003-004.
"""

import numpy as np
from torchxrayvision.datasets import XRayCenterCrop, XRayResizer

ROI_VARIANTS: list[tuple[str, dict]] = [
    ("none", {}),
    ("center_crop", {"size": 224}),
    ("lung_mask", {"margin_frac": 0.05, "threshold": 0.5}),
]

_CENTER_CROP = XRayCenterCrop()


def center_crop(image: np.ndarray) -> np.ndarray:
    """Crop to a centered square using min(height, width)."""
    return _CENTER_CROP(image[None, ...])[0]


def resize(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize to a fixed size x size image."""
    return XRayResizer(size)(image[None, ...])[0]


def mask_to_bbox(mask: np.ndarray, margin_frac: float = 0.05) -> tuple[int, int, int, int]:
    """Bounding box of a boolean mask's True region, as (top, left, bottom, right)
    for image[top:bottom, left:right]. Expanded by margin_frac of the box's own
    height/width on each side (safety margin for the lung apex/costophrenic angle),
    clipped to the mask's bounds.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        raise ValueError("mask_to_bbox: mask is empty, no True pixels")

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    bottom += 1
    right += 1

    height, width = mask.shape
    margin_y = int(round((bottom - top) * margin_frac))
    margin_x = int(round((right - left) * margin_frac))

    top = max(0, top - margin_y)
    left = max(0, left - margin_x)
    bottom = min(height, bottom + margin_y)
    right = min(width, right + margin_x)
    return int(top), int(left), int(bottom), int(right)


def crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    top, left, bottom, right = bbox
    return image[top:bottom, left:right]


def apply_roi_crop(image: np.ndarray, method: str, **params) -> np.ndarray:
    """Apply an ROI-crop variant to a grayscale image in [0, 1].

    method:
      - "none": passthrough.
      - "center_crop": center_crop() then resize(). params: size (default 224).
      - "lung_mask": PSPNet-based lung segmentation -> bounding box -> crop (keeps
        all pixel values inside the box, no pixel masking). params: margin_frac
        (default 0.05), threshold (default 0.5), size (default None -- leave at
        natural bbox size; if given, resize() the crop afterward).
    """
    if method == "none":
        return image
    if method == "center_crop":
        return resize(center_crop(image), size=params.get("size", 224))
    if method == "lung_mask":
        from tda_chexpr.segmentation import predict_lung_mask

        mask = predict_lung_mask(image, threshold=params.get("threshold", 0.5))
        bbox = mask_to_bbox(mask, margin_frac=params.get("margin_frac", 0.05))
        cropped = crop_to_bbox(image, bbox)
        size = params.get("size")
        return resize(cropped, size=size) if size is not None else cropped
    raise ValueError(f"Unknown ROI crop method: {method!r}")
