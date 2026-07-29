# GemmaGenerateEmbeddings

Generates chest-xray embeddings from a MedGemma/SigLIP-family vision encoder and writes
them to a CSV keyed on `Path`, for use as a companion feature table alongside a
CheXpert-style manifest (e.g. `topological-data-analysis/data/exp1/preprocessing_sample.csv`).

## Setup

```bash
uv sync
```

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

Before wiring that into an experiment, sanity-check the join:

```bash
uv run python scripts/validate_join.py \
    --manifest-csv /path/to/preprocessing_sample.csv \
    --embeddings-csv results/embeddings.csv
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
