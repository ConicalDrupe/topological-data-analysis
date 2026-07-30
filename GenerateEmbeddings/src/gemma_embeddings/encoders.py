"""Generic Hugging Face vision-encoder wrapper plus a small backend presets table.

Every preset here loads through `transformers.AutoModel` / `AutoImageProcessor`, which
covers plain SigLIP, RAD-DINO (DINOv2-based), and most MedCLIP ports (CLIP-architecture)
with the same code path. MedGemma is the one preset that sets `vision_attr`, since it's
a full multimodal checkpoint and the vision encoder is a submodule of it
(`model.vision_tower`) rather than the top-level model.

Adding a new AutoModel-compatible backend (another RAD-DINO variant, a MedCLIP port,
...) is a new `BackendPreset` entry below -- no new class needed. For a one-off
checkpoint not worth a preset, use `--backend custom` with explicit `--model-id`
`--vision-attr` / `--pooling` CLI overrides.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from gemma_embeddings.config import EmbeddingConfig, Pooling


@dataclass(frozen=True)
class BackendPreset:
    model_id: str
    vision_attr: str  # dotted attribute path to the vision submodule; "" = top-level model
    pooling: Pooling
    requires_auth: bool
    processor_id: str | None = None  # defaults to model_id
    notes: str = ""


BACKEND_PRESETS: dict[str, BackendPreset] = {
    "siglip": BackendPreset(
        model_id="google/siglip-so400m-patch14-384",
        vision_attr="vision_model",
        pooling="pooler",
        requires_auth=False,
        notes=(
            "Ungated. AutoModel resolves this checkpoint to the combined SiglipModel "
            "(image+text), not a vision-only model, so vision_attr must drill into its "
            "`vision_model` submodule (a SiglipVisionModel) to get pixel_values-only "
            "forward + a flat hidden_size on its config. Use this to smoke-test the "
            "pipeline before dealing with MedGemma auth."
        ),
    ),
    "medgemma": BackendPreset(
        model_id="google/medgemma-4b-it",
        vision_attr="vision_tower",
        pooling="pooler",
        requires_auth=True,
        notes=(
            "TODO: confirm 'vision_tower' is still the correct attribute name, and that "
            "AutoModel resolves this checkpoint at all (see the try/except fallback in "
            "HFVisionEncoder.from_pretrained), against the transformers version actually "
            "installed -- Gemma3/MedGemma multimodal support is recent and names may "
            "shift across releases."
        ),
    ),
    "rad-dino": BackendPreset(
        model_id="microsoft/rad-dino",
        vision_attr="",
        pooling="mean_patch",
        requires_auth=True,  # unconfirmed -- verify with `huggingface-cli download microsoft/rad-dino`
        notes=(
            "TODO: verify current gating status, and confirm mean-patch pooling against "
            "the model card's recommended usage (mean-patch vs. CLS) before trusting "
            "this default."
        ),
    ),
}


def _resolve_submodule(model: torch.nn.Module, vision_attr: str) -> torch.nn.Module:
    if not vision_attr:
        return model
    return functools.reduce(getattr, vision_attr.split("."), model)


class HFVisionEncoder:
    """Wraps a single HF vision encoder (or a submodule of a larger checkpoint)."""

    def __init__(
        self,
        model: torch.nn.Module,
        image_processor,
        model_id: str,
        vision_attr: str,
        pooling: Pooling,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.image_processor = image_processor
        self.model_id = model_id
        self.vision_attr = vision_attr
        self.pooling = pooling
        self.device = device
        self.dtype = dtype
        self._warned_no_pooler = False

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        processor_id: str,
        vision_attr: str,
        pooling: Pooling,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "HFVisionEncoder":
        try:
            full_model = AutoModel.from_pretrained(model_id, dtype=dtype)
        except (ValueError, KeyError):
            # TODO: some multimodal checkpoints (e.g. MedGemma) may only register under
            # a task-specific Auto class rather than the base AutoModel. Confirm which
            # class actually resolves google/medgemma-4b-it on the installed
            # transformers version and adjust this fallback accordingly.
            from transformers import AutoModelForImageTextToText

            full_model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)

        vision_model = _resolve_submodule(full_model, vision_attr)
        vision_model = vision_model.to(device=device, dtype=dtype).eval()
        image_processor = AutoImageProcessor.from_pretrained(processor_id)
        return cls(vision_model, image_processor, model_id, vision_attr, pooling, device, dtype)

    @property
    def embedding_dim(self) -> int:
        # Assumes the resolved vision submodule is itself a standard HF vision model
        # (SiglipVisionModel/Dinov2Model/CLIPVisionModel-style config); holds for every
        # current preset above.
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        outputs = self.model(pixel_values=pixel_values)

        if self.pooling == "pooler":
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                if not self._warned_no_pooler:
                    warnings.warn(
                        f"{self.model.__class__.__name__} has no pooler_output; "
                        "falling back to CLS-token pooling.",
                        stacklevel=2,
                    )
                    self._warned_no_pooler = True
                pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == "mean_patch":
            # TODO: confirm per-backend whether last_hidden_state has a leading
            # CLS/register token that should be excluded before averaging (check the
            # model card -- RAD-DINO's usage guide should specify this).
            pooled = outputs.last_hidden_state.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling!r}")

        return pooled.to(torch.float32).cpu().numpy()


def get_encoder(config: EmbeddingConfig) -> HFVisionEncoder:
    preset = BACKEND_PRESETS.get(config.backend)
    if preset is None and config.backend != "custom":
        raise ValueError(
            f"Unknown backend {config.backend!r}; choose one of "
            f"{[*BACKEND_PRESETS, 'custom']}"
        )

    model_id = config.model_id or (preset.model_id if preset else None)
    vision_attr = (
        config.vision_attr if config.vision_attr is not None
        else (preset.vision_attr if preset else None)
    )
    pooling = config.pooling or (preset.pooling if preset else None)
    processor_id = (
        config.processor_id
        or (preset.processor_id if preset else None)
        or model_id
    )

    if model_id is None or vision_attr is None or pooling is None:
        raise ValueError(
            "backend='custom' requires --model-id, --vision-attr, and --pooling to all be set."
        )

    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)

    return HFVisionEncoder.from_pretrained(
        model_id=model_id,
        processor_id=processor_id,
        vision_attr=vision_attr,
        pooling=pooling,
        device=device,
        dtype=dtype,
    )
