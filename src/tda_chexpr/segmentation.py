"""PSPNet-based lung segmentation (torchxrayvision). See experiments.md, Experiment 1
pipeline step 2, and logs/exp1_log.md, Experiment 004.
"""

import numpy as np
import torch
import torchxrayvision as xrv
from skimage.transform import resize as sk_resize

from tda_chexpr.roi import center_crop

_LEFT_LUNG = "Left Lung"
_RIGHT_LUNG = "Right Lung"

_MODEL = None


def get_pspnet_model():
    """Lazy module-level singleton -- PSPNet takes ~9s to load, load once per run."""
    global _MODEL
    if _MODEL is None:
        _MODEL = xrv.baseline_models.chestx_det.PSPNet()
    return _MODEL


def predict_lung_mask(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Predict a boolean lung mask for `image` (grayscale, [0, 1]), in the image's own
    coordinate frame.

    PSPNet requires a square input, so this center-crops internally (matching
    roi.center_crop's own offset math) purely to satisfy that constraint, then pastes
    the resulting mask back into the original (possibly non-square) frame -- the crop
    is not otherwise applied to the caller's image.
    """
    height, width = image.shape
    crop_size = min(height, width)
    y_offset = height // 2 - crop_size // 2
    x_offset = width // 2 - crop_size // 2

    square = center_crop(image)

    img_255 = np.clip(square * 255, 0, 255).astype(np.float32)
    img_xrv = xrv.utils.normalize(img_255, 255)  # -> roughly [-1024, 1024]
    x = torch.from_numpy(img_xrv)[None, None, ...].float()

    model = get_pspnet_model()
    with torch.no_grad():
        logits = model(x)
    probs = torch.sigmoid(logits)[0]

    left = probs[model.targets.index(_LEFT_LUNG)]
    right = probs[model.targets.index(_RIGHT_LUNG)]
    lung_prob = torch.maximum(left, right).numpy()

    lung_prob_resized = sk_resize(
        lung_prob, (crop_size, crop_size), order=1, preserve_range=True, anti_aliasing=True
    )
    square_mask = lung_prob_resized > threshold

    mask = np.zeros((height, width), dtype=bool)
    mask[y_offset : y_offset + crop_size, x_offset : x_offset + crop_size] = square_mask
    return mask
