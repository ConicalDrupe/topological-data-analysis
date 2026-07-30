# Experiment 2 Embeddings Log

- **Current experiment:** Experiment 008 (embeddings from the processed/CLAHE dataset,
  below) — done for all three backends, both splits. Experiment 007 (raw-image
  embeddings) is complete and unchanged; both sets of embeddings now exist side by side.

---

# Experiment 007

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

---

# Experiment 008

- **Goal:** generate embeddings for the *processed* pneumothorax dataset (PSPNet
  lung-mask crop + conservative CLAHE, 224x224 PNG, `kaggle/processed/` — see
  `logs/exp1_log.md` Experiment 007), so the planned topology-of-embedding-space
  analysis can compare raw-image embeddings (Experiment 007, above) against
  processed-image embeddings.
- **Preprocessing pipeline:** PSPNet lung-mask crop (`margin_frac=0.05`,
  `threshold=0.5`) → direct resize to 224x224 → CLAHE (`clip_limit=0.002`,
  `kernel_size=16`) → 8-bit PNG. Full detail/rationale in `logs/exp1_log.md` Experiment
  007, not reproduced here. Each backend's own `AutoImageProcessor` still applies its
  own resize/normalize on top of the already-224x224 PNG, exactly as it did for the raw
  images in Experiment 007 above — no special-casing was needed for the smaller input
  size.
- **Dataset version:** new `data/embeddings/processed/processed_manifest_{train,test}.csv`
  (460/115 rows), built by merging `data/exp1/v2_corrected_cohort/
  pneumothorax_{train,test}_split.csv` with `results/exp1/preprocessing/v1/manifest.csv`'s
  `output_path`/`cohort_split` columns on `Path`==`path` — verified 100% match (460/460,
  115/115 rows), preserving the exact same patient-level 80/20 split used everywhere else
  in this project (`tda_chexpr.split.stratified_split`, `random_state=42`); no new split
  was computed. Images are resolved from `kaggle/processed/` via the manifest's
  `output_path` column (the raw `Path` column's `.jpg` values don't exist under
  `kaggle/processed/`, only the `.png` outputs do).
- **Model versions:** unchanged from Experiment 007 above (same three presets/
  checkpoints/pooling strategies) — see that entry for full detail.
- **Random seed:** n/a, same reasoning as Experiment 007.
- **Feature extraction method:** new
  `GenerateEmbeddings/scripts/run_processed_pneumothorax_embeddings.py`. Mirrors
  `run_pneumothorax_embeddings.py`'s subprocess-per-(backend,split) pattern, but points
  `--image-root` at `kaggle/processed/`, passes `--key-column output_path` explicitly,
  and runs all three backends automatically per split in one invocation (rather than one
  backend per invocation). Each backend's bare `[output_path, emb_*]` output is first
  written to a staging directory (`data/embeddings/processed/_raw/`), then enriched with
  label columns (`raw_path`, `patient_id`, `Pneumothorax`, `cohort_split`, `Sex`, `Age`,
  `comorbidity_count`, `is_clean_negative`) via a post-hoc merge on `output_path` before
  writing the final CSV — staging avoids a real hazard where enriching in place would
  corrupt a later `--resume` run (`EmbeddingCsvWriter` appends bare rows under whatever
  header is already there). Unlike Experiment 007's output, these CSVs are
  denormalized/labeled directly — no separate join against the split CSVs is needed
  downstream.
- **Parameters:** same defaults as Experiment 007 (`--batch-size 16`, `--dtype
  bfloat16`, `--device cuda`, GPU: RTX 3090) plus `--key-column output_path`.
- **Evaluation metric:** none yet, same as Experiment 007.
- **Current status:** `data/embeddings/processed/` has
  `{siglip,rad-dino,medgemma}_{train,test}_embeddings.csv` + `.meta.json` sidecars (each
  `.meta.json` additionally records `label_columns_from`, pointing at the
  `processed_manifest_{split}.csv` used), plus the staged bare outputs under `_raw/` and
  the two `processed_manifest_{train,test}.csv` input manifests. All 6
  (backend × split) combinations verified: row counts match 460/115 exactly, 100%
  `output_path` join match via `scripts/validate_join.py`, and spot-checked embedding
  sanity (0 NaNs, 0 zero-norm rows, 0 duplicate rows, healthy per-row variance:
  SigLIP row-norms 20.3–23.5, RAD-DINO 5.2–21.8, MedGemma 46.3–52.3) across all three
  backends. Next: decide whether the planned topology-of-embedding-space analysis should
  run against the raw-image embeddings (Experiment 007), the processed-image embeddings
  (this experiment), or both compared side by side.
