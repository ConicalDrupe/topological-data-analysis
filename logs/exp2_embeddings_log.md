# Experiment 2 Embeddings Log

- **Current experiment:** Experiment 007 (embedding generation, SigLIP smoke test) —
  in progress.
- **Preprocessing pipeline:** none — raw images loaded via PIL, resized/normalized by
  each backend's own `AutoImageProcessor` (no CLAHE/ROI-crop from the Experiment 1
  pipeline applied here).
- **Dataset version:** `data/exp1/v2_corrected_cohort/pneumothorax_{train,test}_split.csv`
  (460/115 rows, Experiment 006 corrected cohort) joined to images under `kaggle/` on
  `Path`.
- **Model version:** `google/siglip-so400m-patch14-384` (`--backend siglip`, ungated,
  ran first as an end-to-end smoke test). `medgemma` (`google/medgemma-4b-it`, gated)
  and `rad-dino` (`microsoft/rad-dino`, gating unconfirmed) presets exist in
  `GenerateEmbeddings/src/gemma_embeddings/encoders.py` and are planned next, not yet
  run.
- **Random seed:** n/a — feature extraction is deterministic given a fixed checkpoint
  (no sampling/augmentation).
- **Feature extraction method:** pooled vision-encoder output (`pooling="pooler"` for
  SigLIP) via `GenerateEmbeddings/src/gemma_embeddings`, run through
  `GenerateEmbeddings/scripts/run_pneumothorax_embeddings.py`.
- **Parameters:** `--batch-size 16` (default), `--dtype bfloat16` (default),
  `--device cuda` (GPU: driver 560.94 / CUDA 12.6; torch pinned to the
  `pytorch-cu126` wheel index in the root `pyproject.toml`).
- **Evaluation metric:** none yet — this stage only produces the embedding CSVs feeding
  Experiment 2's planned topology-of-embedding-space analysis (H0/H1 persistent
  homology / Mapper on the pooled vectors, per `experiments.md`).
- **Current status:** `GenerateEmbeddings/` consolidated into the top-level uv
  environment (previously a separate `requires-python>=3.13` project, now merged into
  the root `pyproject.toml` at `==3.12.*`). Output convention:
  `results/exp2/embeddings/v1/{backend}_{split}_embeddings.csv` +
  `<output_csv>.meta.json` sidecar. Next: run the SigLIP smoke test end-to-end, validate
  the join back onto the split CSVs, then run MedGemma/RAD-DINO once gated access is
  sorted out.
