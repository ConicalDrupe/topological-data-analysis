"""Selective checkpoint loading: pull only one submodule's tensors out of a (possibly
much larger) Hugging Face checkpoint.

Used so that loading, say, MedGemma's vision tower doesn't require downloading the full
4B-parameter multimodal checkpoint -- only the shard(s) containing tensors under the
resolved `vision_attr` prefix (e.g. `"vision_tower."`) are ever fetched. For a
*sharded* checkpoint this is a real download saving (only the relevant shard files are
pulled). For a single-file checkpoint there's no way around downloading that one file in
full -- safetensors' header/offsets would allow a partial byte-range fetch in principle,
but neither `huggingface_hub` nor this module implements that; not worth the complexity
for the checkpoints in play here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open


def _download(model_id: str, filename: str) -> Path:
    return Path(hf_hub_download(model_id, filename))


def _shards_for_prefix(model_id: str, prefix: str) -> dict[Path, list[str]]:
    """Map {local shard path: [full tensor keys under `prefix`]}, downloading only the
    shard(s) that actually contain a matching tensor."""
    try:
        index_path = _download(model_id, "model.safetensors.index.json")
    except EntryNotFoundError:
        single_path = _download(model_id, "model.safetensors")
        with safe_open(single_path, framework="pt") as f:
            matches = [key for key in f.keys() if key.startswith(prefix)]
        return {single_path: matches} if matches else {}

    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    matches_by_shard_name: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        if key.startswith(prefix):
            matches_by_shard_name.setdefault(shard_name, []).append(key)

    return {
        _download(model_id, shard_name): keys
        for shard_name, keys in matches_by_shard_name.items()
    }


def load_submodule_state_dict(model_id: str, prefix: str) -> dict[str, torch.Tensor]:
    """Return a state dict (keys stripped of `prefix`) for only the tensors under
    `prefix` in `model_id`'s checkpoint -- e.g. prefix="vision_tower." to load just a
    multimodal model's vision encoder. Tensors are returned on CPU in their stored
    dtype; the caller is responsible for `.to(device=..., dtype=...)`.
    """
    by_shard = _shards_for_prefix(model_id, prefix)
    if not by_shard:
        raise ValueError(f"No tensors with prefix {prefix!r} found in {model_id}'s checkpoint")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_path, keys in by_shard.items():
        with safe_open(shard_path, framework="pt") as f:
            for key in keys:
                state_dict[key[len(prefix):]] = f.get_tensor(key)
    return state_dict
