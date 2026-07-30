# GemmaGenerateEmbeddings

Generates chest-xray embeddings from a MedGemma/SigLIP-family vision encoder and writes
them to a CSV keyed on `Path`, for use as a companion feature table alongside a
CheXpert-style manifest (e.g. `topological-data-analysis/data/exp1/preprocessing_sample.csv`).

## Setup

This package is part of the single top-level `topological-data-analysis` uv project —
there is no separate environment here anymore. From the repo root:

```bash
uv sync
```

installs `gemma_embeddings` (this package) alongside the rest of the repo's
dependencies into one shared `.venv`. `torch`/`torchvision` are pinned to the
CUDA 12.6 wheel index in the root `pyproject.toml` (see `[tool.uv.index]` /
`[tool.uv.sources]`), matching a driver reporting `CUDA Version: 12.6`.

Fastest way to run against this repo's pneumothorax cohort — use the wrapper script
instead of calling the module directly:

```bash
uv run python GenerateEmbeddings/scripts/run_pneumothorax_embeddings.py \
    --backend siglip --split test
```

It fills in `--input-csv`/`--image-root`/`--output-csv` for
`data/exp1/v2_corrected_cohort/pneumothorax_{train,test}_split.csv` and writes to
`results/exp2/embeddings/v1/`. Run `--split both` (the default) to do train+test in one
go, or pass extra flags (`--device`, `--batch-size`, `--resume`, ...) through — see below
for what's available.

Auth, per `--backend`:

| backend    | gated?      | what you need                                                                 |
|------------|-------------|--------------------------------------------------------------------------------|
| `siglip`   | no          | nothing — works anonymously                                                    |
| `medgemma` | yes         | HF account, accept license on `google/medgemma-4b-it`'s model page, then `huggingface-cli login` or `export HF_TOKEN=...` |
| `rad-dino` | unconfirmed | try it; if it 401s, same flow as medgemma                                      |
| `custom`   | depends     | whatever the checkpoint you point `--model-id` at requires                     |

Start with `--backend siglip` to confirm the pipeline runs end-to-end before dealing with
gated access.

## Generate embeddings

For an arbitrary manifest CSV (not this repo's pneumothorax split — use
`scripts/run_pneumothorax_embeddings.py` for that, above), call the module directly from
the repo root:

```bash
uv run python -m gemma_embeddings.generate_embeddings \
    --input-csv /path/to/preprocessing_sample.csv \
    --image-root /path/to/kaggle \
    --output-csv results/embeddings.csv \
    --backend siglip
```

- `--input-csv` — any CSV with a `Path` column in the CheXpert relative-path form
  (`train/patient00001/study1/view1_frontal.jpg`). Only `Path` is read; other columns
  (labels, `patient_id`, etc.) are ignored and untouched.
- `--image-root` — directory those `Path` values are relative to (your local `kaggle/`
  checkout). Not hardcoded, since this project doesn't depend on
  `topological-data-analysis`'s layout.
- `--output-csv` — where the embeddings CSV is written. A sidecar
  `<output-csv>.meta.json` is written alongside it (backend, model id, pooling,
  embedding dimension, timestamp).

### Key parameters

| flag              | default             | notes                                                              |
|--------------------|--------------------|----------------------------------------------------------------------|
| `--backend`         | `medgemma`          | `siglip` \| `medgemma` \| `rad-dino` \| `custom`                     |
| `--model-id`        | preset default      | override to point at a specific checkpoint                          |
| `--vision-attr`     | preset default      | dotted path to the vision submodule (`"vision_tower"` for MedGemma, `""` for a vision-only checkpoint) — required if `--backend custom` |
| `--pooling`         | preset default      | `pooler` \| `cls` \| `mean_patch` — required if `--backend custom`  |
| `--batch-size`      | `16`                | lower if you hit OOM                                                 |
| `--device`          | `cuda` if available | `cuda` \| `cpu`                                                      |
| `--dtype`           | `bfloat16`          | use `float32` on CPU                                                 |
| `--resume`          | off                 | skip `Path` values already present in `--output-csv` (safe to re-run after a crash) |

Run `--help` for the full list.

## Output format

```
Path,emb_0000,emb_0001,...,emb_{D-1}
train/patient00001/study1/view1_frontal.jpg,0.0123,-0.0456,...
```

`D` (embedding dimension) depends on the backend and is recorded in
`<output-csv>.meta.json`, never hardcoded.

## Integrating with topological-data-analysis

Join on `Path` — no changes to that repo needed:

```python
import pandas as pd

cohort = pd.read_csv("data/exp1/preprocessing_sample.csv")
embeddings = pd.read_csv("results/embeddings.csv")
merged = cohort.merge(embeddings, on="Path", how="left")
```

Before wiring that into an experiment, sanity-check the join (run from the repo root):

```bash
uv run python GenerateEmbeddings/scripts/validate_join.py \
    --manifest-csv data/exp1/v2_corrected_cohort/pneumothorax_test_split.csv \
    --embeddings-csv results/exp2/embeddings/v1/siglip_test_embeddings.csv
```

Reports matched-row count and prints any unmatched `Path` values.

## Extending to another backbone (RAD-DINO, a MedCLIP port, etc.)

If the checkpoint loads via `transformers.AutoModel` / `AutoImageProcessor`, add a
`BackendPreset` entry in `src/gemma_embeddings/encoders.py` (model id, vision submodule
attribute path, pooling strategy) — no new code path required. For a one-off checkpoint,
skip the preset and pass `--backend custom --model-id ... --vision-attr ... --pooling
...` directly.

## Known TODOs

See the `TODO` comments in `src/gemma_embeddings/encoders.py`:
- MedGemma's exact `vision_tower` attribute path and whether `AutoModel` resolves the
  checkpoint at all — confirm against the installed `transformers` version.
- RAD-DINO's gating status and its model card's recommended pooling (mean-patch vs. CLS).
