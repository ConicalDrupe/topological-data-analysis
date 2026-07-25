# AGENTS.md

Guidance for working in this repository. See [`experiments.md`](./experiments.md) for the
actual research plan — this file covers data access and tooling only.

## Data access

Dataset (CheXpert-v1.0-small) lives at **`kaggle/`, relative to the repo root** — always
reference it relatively; there is no dataset at an absolute `/kaggle/` path.

### Layout

```
kaggle/
├── train/            64,540 patient folders (~223,414 images)
├── train.csv         per-image metadata + labels for train/
├── valid/             200 patient folders
└── valid.csv         per-image metadata + labels for valid/
```

Nesting: `kaggle/train/patient00001/study1/view1_frontal.jpg`, `view2_lateral.jpg`, etc.

- Many patients have multiple `studyN` (multi-visit data, some with dozens of studies) —
  this is what Experiment 3 in `experiments.md` uses. No date field exists; order studies
  by study number and/or the per-row `Age` column, not a timestamp.
- **Path prefix gotcha:** `Path` column values are prefixed
  `CheXpert-v1.0-small/train/...`, but the on-disk layout under `kaggle/` is not — strip
  the prefix before resolving a CSV row to a file.
- `archive.zip` (~11.5GB, repo root) is the original download `kaggle/` was extracted
  from — not needed once `kaggle/` exists.

### CSV schema (`train.csv` / `valid.csv`, identical columns)

```
Path, Sex, Age, Frontal/Lateral, AP/PA,
No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion,
Pleural Other, Fracture, Support Devices
```

The 14 pathology columns use the standard CheXpert label convention:

| Value  | Meaning                |
|--------|------------------------|
| `1.0`  | Positive               |
| `0.0`  | Negative               |
| `-1.0` | Uncertain              |
| blank  | Not mentioned in report |

## Environment & libraries

This project is managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` /
`uv.lock`). Use `uv sync` to install and `uv run <cmd>` to execute within the project
environment.

**[giotto-tda](https://giotto-ai.github.io/gtda-docs/) (`gtda`) is the primary TDA
library** for all new experiment work described in `experiments.md`.

TTK (Topology ToolKit) and RIVET are mentioned in `README.md` as possible tools for
pointcloud visualization and two-parameter persistence, respectively. They're not set up
or documented further here — treat them as future/optional, per `experiments.md`.

## Automatic Exploratory Data Analysis (EDA)

Whenever a new dataset, sample, split, preprocessing stage, or transformed dataset is
created, run EDA (below) before training or evaluation, and before implementing the
experiment (see Research Workflow step 3).

### Dataset Summary

- Number of samples
- Number of unique patients
- Number of studies
- Number of frontal vs lateral images
- Number of AP vs PA views
- Age statistics
- Sex distribution
- Missing values
- Duplicate paths
- Class frequencies for every pathology

### Image Statistics

Only relevant when preprocessing images or gathering representative class samples for
classification. Compute:

- image dimensions
- aspect ratio distribution
- pixel intensity statistics
- histogram summaries
- grayscale range
- brightness statistics
- contrast statistics

If preprocessing has been applied (HE, CLAHE, AGC, etc.), compare before/after statistics.

### Visualizations

Generate figures whenever practical (sample images, preprocessing comparison grids).
Store under `results/<experiment>/eda/`. Never overwrite previous plots.

## Research Workflow

For every new experiment, follow this workflow unless explicitly instructed otherwise.

1. Understand the experiment objective from `experiments.md`.
2. Inspect the current codebase before implementing changes.
3. If creating a new dataset or transformed sample, run EDA (above) and update the
   experiment log before implementing.
4. Implement the experiment.
5. Evaluate the results.
6. Record observations, limitations, and next steps.
7. Never overwrite previous experiment outputs — create new versioned directories instead.

### Experiment Log

Maintain a running markdown file. Keep this file minimal and factual.

logs/<experiment>_log.md

This file should always contain:

- current experiment
- preprocessing pipeline
- dataset version
- model version
- random seed
- feature extraction method
- parameters for applicable preprocessing, model, etc.
- evaluation metric
- current status

Each experiment entry should include:

# Experiment 004

## Goal

...

## Dataset

...

## Parameters

...

## Results

...

## Observations

...

## Next Steps

...
