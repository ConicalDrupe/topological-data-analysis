"""Generic Hugging Face vision-encoder wrapper plus a small backend presets table.

Every preset here loads through `transformers.AutoModel` / `AutoImageProcessor`, which
covers plain SigLIP, RAD-DINO (DINOv2-based), and most MedCLIP ports (CLIP-architecture)
with the same code path. `vision_attr` is set whenever the checkpoint's top-level model
isn't itself the vision encoder but wraps it as a submodule -- SigLIP's checkpoint
resolves to the combined SiglipModel (`vision_model` + `text_model`), and MedGemma's to
a full multimodal model (`vision_tower` alongside the language model). RAD-DINO is the
one preset with `vision_attr=""`, since its checkpoint is vision-only end to end. A
non-empty `vision_attr` also triggers `_load_vision_submodule`, which loads just that
submodule's weights rather than the whole checkpoint -- see its docstring.

Adding a new AutoModel-compatible backend (another RAD-DINO variant, a MedCLIP port,
...) is a new `BackendPreset` entry below -- no new class needed. For a one-off
checkpoint not worth a preset, use `--backend custom` with explicit `--model-id`
`--vision-attr` / `--pooling` CLI overrides.
"""

from __future__ import annotations

import functools
import gc
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from gemma_embeddings.config import EmbeddingConfig, Pooling
from gemma_embeddings.weights import load_submodule_state_dict


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
        pooling="mean_patch",
        requires_auth=True,
        notes=(
            "Gated -- confirmed 'vision_tower' resolves correctly via AutoModel once "
            "authenticated. Its vision tower has vision_use_head=False (no attention- "
            "pooling head in this checkpoint), so pooling='pooler' would silently fall "
            "back to CLS-token indexing -- but SigLIP has no CLS token at all (patch- "
            "only sequence), so that would just grab one arbitrary patch, not a global "
            "representation. mean_patch (averaging over all patches) is the correct "
            "choice here, matching RAD-DINO's approach for the same reason."
        ),
    ),
    "rad-dino": BackendPreset(
        model_id="microsoft/rad-dino",
        vision_attr="",
        pooling="mean_patch",
        requires_auth=False,  # confirmed ungated (MIT-licensed model card), no HF auth needed
        notes=(
            "DINOv2-base finetuned on chest X-rays, 86.6M params, vision-only "
            "checkpoint (no text tower, so vision_attr stays ''). mean-patch pooling "
            "per the model card's recommended usage."
        ),
    ),
}


def _resolve_submodule(model: torch.nn.Module, vision_attr: str) -> torch.nn.Module:
    if not vision_attr:
        return model
    return functools.reduce(getattr, vision_attr.split("."), model)


def _build_skeleton(config, dtype: torch.dtype) -> torch.nn.Module:
    try:
        return AutoModel.from_config(config, dtype=dtype)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_config(config, dtype=dtype)


def _load_vision_submodule(
    model_id: str, vision_attr: str, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
    """Build only the `vision_attr` submodule of `model_id` and load just its weights.

    Downloading/holding the rest of a large multimodal checkpoint (e.g. MedGemma's ~4B
    total params to use its ~400M-param vision tower) wastes both bandwidth and VRAM.
    `load_submodule_state_dict` avoids downloading shards that don't contain the vision
    tower's tensors at all (see weights.py).

    We build the *architecture* (config -> module graph) for the full model on CPU
    rather than using a `torch.device("meta")` skeleton + `to_empty()`: some vision
    submodules register non-persistent buffers (e.g. position-id index tensors) that
    aren't part of any checkpoint's state dict by design -- `to_empty()` would leave
    those as uninitialized garbage instead of the values their real `__init__` computes
    (confirmed empirically: SiglipVisionModel has 448 persistent state_dict entries but
    449 buffers total). A real (non-meta) init pays a one-time, purely-in-memory random-
    init cost for the discarded siblings (e.g. the language model) -- no network/disk
    I/O, freed as soon as this function returns -- which is the trade we want here.

    The checkpoint's raw tensor names aren't guaranteed to line up with the currently-
    installed transformers version's live module attribute structure -- confirmed with
    MedGemma: its checkpoint stores `vision_tower.vision_model.*` (an extra nesting
    level, from whatever transformers version it was saved with), but transformers
    5.14.1's `Gemma3Model.vision_tower` is a flat `SiglipVisionModel` with no such
    `.vision_model` wrapper, so a naive prefix-stripped load would silently leave the
    submodule at its random init -- garbage embeddings with no error. We verify the
    checkpoint's keys (after stripping `vision_attr`) exactly match the built module's
    `state_dict()` keys before trusting the fast path; any mismatch falls back to the
    slower but guaranteed-correct route of loading the full checkpoint through the
    official `from_pretrained`, which carries transformers' own legacy-key-remapping
    logic for exactly this kind of drift.
    """
    config = AutoConfig.from_pretrained(model_id)
    skeleton = _build_skeleton(config, dtype)
    vision_model = _resolve_submodule(skeleton, vision_attr)
    del skeleton  # drop references to sibling submodules (e.g. the language model)

    state_dict = load_submodule_state_dict(model_id, vision_attr + ".")
    target_keys = set(vision_model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    if target_keys != checkpoint_keys:
        missing = target_keys - checkpoint_keys
        unexpected = checkpoint_keys - target_keys
        warnings.warn(
            f"{model_id}'s {vision_attr!r} submodule, as built by the installed "
            f"transformers version, doesn't line up with the checkpoint's raw tensor "
            f"names under that prefix ({len(missing)} missing, {len(unexpected)} "
            "unexpected) -- likely a naming change since this checkpoint was saved. "
            "Falling back to the full official from_pretrained load (downloads the "
            "whole checkpoint, but its own key-remapping logic keeps this correct).",
            stacklevel=2,
        )
        return _load_vision_submodule_via_full_load(model_id, vision_attr, device, dtype)

    vision_model.load_state_dict({key: value.to(dtype=dtype) for key, value in state_dict.items()})
    return vision_model.to(device=device, dtype=dtype).eval()


def _load_vision_submodule_via_full_load(
    model_id: str, vision_attr: str, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
    """Fallback for when the fast prefix-based partial load doesn't cleanly apply:
    load the full checkpoint the standard way (correct by construction), keep only the
    resolved submodule, and drop the rest."""
    try:
        full_model = AutoModel.from_pretrained(model_id, dtype=dtype)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        full_model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)

    vision_model = _resolve_submodule(full_model, vision_attr).to(device=device, dtype=dtype).eval()
    del full_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vision_model


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
        if vision_attr:
            # A non-empty vision_attr means the vision encoder is a submodule of a
            # larger checkpoint (e.g. MedGemma's 4B-param multimodal model) -- avoid
            # downloading/materializing the rest of it. See _load_vision_submodule.
            vision_model = _load_vision_submodule(model_id, vision_attr, device, dtype)
        else:
            try:
                full_model = AutoModel.from_pretrained(model_id, dtype=dtype)
            except (ValueError, KeyError):
                # TODO: some multimodal checkpoints (e.g. MedGemma) may only register
                # under a task-specific Auto class rather than the base AutoModel.
                # Confirm which class actually resolves google/medgemma-4b-it on the
                # installed transformers version and adjust this fallback accordingly.
                from transformers import AutoModelForImageTextToText

                full_model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)

            vision_model = full_model.to(device=device, dtype=dtype).eval()

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
