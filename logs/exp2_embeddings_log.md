# Experiment 2 Embeddings Log

- **Current experiment:** Experiment 007 (embedding generation) — done for all three
  backends (SigLIP, RAD-DINO, MedGemma), both splits.
- **Preprocessing pipeline:** none — raw images loaded via PIL, resized/normalized by
  each backend's own `AutoImageProcessor` (no CLAHE/ROI-crop from the Experiment 1
  pipeline applied here).
- **Dataset version:** `data/exp1/v2_corrected_cohort/pneumothorax_{train,test}_split.csv`
  (460/115 rows, Experiment 006 corrected cohort) joined to images under `kaggle/` on
  `Path`.
- **Model versions:**
  - `google/siglip-so400m-patch14-384` (`--backend siglip`, ungated). Vision tower
    (`vision_model` submodule, 428M params, embedding_dim=1152, `pooling="pooler"`) via
    the fast lazy-load path (verified bit-exact against a full-checkpoint load).
  - `microsoft/rad-dino` (`--backend rad-dino`, DINOv2-base finetuned on chest X-rays,
    confirmed ungated). 86.6M params, embedding_dim=768, `pooling="mean_patch"`,
    vision-only checkpoint (no lazy submodule loading needed, `vision_attr=""`).
  - `google/medgemma-4b-it` (`--backend medgemma`, gated — ran after the user
    authenticated via `hf auth login`). Vision tower (`vision_tower` submodule,
    embedding_dim=1152, `pooling="mean_patch"`). See "Bugs found and fixed" below —
    this backend needed two real fixes before its output could be trusted.
- **Random seed:** n/a — feature extraction is deterministic given a fixed checkpoint
  (no sampling/augmentation).
- **Feature extraction method:** pooled vision-encoder output via
  `GenerateEmbeddings/src/gemma_embeddings`, run through
  `GenerateEmbeddings/scripts/run_pneumothorax_embeddings.py`.
- **Parameters:** `--batch-size 16` (default), `--dtype bfloat16` (default), `--device
  cuda` (GPU: RTX 3090, driver 560.94 / CUDA 12.6; torch pinned to the `pytorch-cu126`
  wheel index in the root `pyproject.toml`).
- **Evaluation metric:** none yet — this stage only produces the embedding CSVs feeding
  Experiment 2's planned topology-of-embedding-space analysis (H0/H1 persistent
  homology / Mapper on the pooled vectors, per `experiments.md`).
- **Efficiency fix:** `encoders.py`'s `HFVisionEncoder.from_pretrained` previously loaded
  the *entire* checkpoint via `AutoModel`/`AutoModelForImageTextToText` even when only a
  submodule (`vision_attr`) was used. Added `weights.py` + `_load_vision_submodule`:
  builds just the target submodule's architecture and loads only the checkpoint tensors
  under its prefix, skipping any safetensors shard that doesn't contain one.
- **Bugs found and fixed while enabling MedGemma:**
  1. **Silent random-init bug (serious).** MedGemma's checkpoint stores vision-tower
     tensors as `vision_tower.vision_model.*` — an extra `.vision_model` nesting level
     left over from whatever transformers version the checkpoint was saved with. The
     currently-installed transformers (5.14.1) builds `vision_tower` as a *flat*
     `SiglipVisionModel` with no such wrapper. The lazy-load path's naive prefix-strip
     therefore matched zero real keys (437 missing, 437 unexpected out of ~437),
     `load_state_dict(strict=False)` silently accepted the empty overlap, and the first
     run completed "successfully" while actually embedding from a **randomly-initialized**
     vision tower — no exception, no useful signal beyond the (easy-to-miss) warning.
     Fix: `_load_vision_submodule` now verifies the checkpoint's keys (after stripping
     the prefix) exactly match the built module's `state_dict()` keys before trusting the
     fast path; on any mismatch it falls back to `_load_vision_submodule_via_full_load`,
     which uses the standard `from_pretrained` (transformers' own legacy-key-remapping
     logic handles the drift correctly, at the cost of downloading the full checkpoint
     for this backend only). The first (corrupted) `medgemma_test_embeddings.csv` was
     deleted and regenerated.
  2. **Wrong pooling strategy.** This checkpoint's vision tower has
     `vision_use_head=False` (no attention-pooling head), so `pooling="pooler"` (the
     preset's original value, inherited from the SigLIP preset) fell back to CLS-token
     indexing (`last_hidden_state[:, 0, :]`) — but SigLIP has no CLS token at all
     (patch-only sequence), so that fallback just returns one arbitrary patch, not a
     meaningful global representation. Fixed the preset to `pooling="mean_patch"`
     (same reasoning as RAD-DINO). The second `medgemma_test_embeddings.csv` (still
     CLS-indexed) was also discarded before the final, correctly-pooled regeneration.
  - Both fixes are generic (apply to any future backend hitting the same failure modes),
    not MedGemma-specific hacks. Final MedGemma output was sanity-checked (no NaNs, no
    all-zero or duplicate rows, healthy per-row variance) in addition to the 100% join
    check below.
- **Current status:** `results/exp2/embeddings/v1/` has
  `{siglip,rad-dino,medgemma}_{train,test}_embeddings.csv` + `.meta.json` sidecars, all
  validated at 100% `Path` join match via `scripts/validate_join.py`. All three backends
  planned for Experiment 2 are now available. Next: the actual topology-of-embedding-
  space analysis (persistent homology / Mapper on these pooled vectors).
