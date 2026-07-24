# AGENTS.md

Guidance for working in this repository. See [`experiments.md`](./experiments.md) for the
actual research plan — this file covers data access and tooling only.

## Data access

The dataset (CheXpert-v1.0-small) lives at **`kaggle/`, relative to the repo root**
(`/home/boon/Projects/topological-data-analysis/kaggle/`) — there is no dataset at an
absolute `/kaggle/` path on the filesystem. Always reference it relative to the repo root.

### Layout

```
kaggle/
├── train/            64,540 patient folders (~223,414 images)
├── train.csv         per-image metadata + labels for train/
├── valid/             200 patient folders
└── valid.csv         per-image metadata + labels for valid/
```

Each patient folder nests further by study (visit) and view:

```
kaggle/train/patient00001/study1/view1_frontal.jpg
                                 /view2_lateral.jpg
                         /study2/view1_frontal.jpg
                         ...
```

Many patients have more than one `studyN` — this is real multi-visit data (some patients
have dozens of studies) and is what Experiment 3 in `experiments.md` uses. There is no
absolute date field; only an `Age` column per study row, so ordering studies means sorting
by study number and/or `Age`, not a calendar timestamp.

### CSV schema (`train.csv` / `valid.csv`, identical columns)

```
Path, Sex, Age, Frontal/Lateral, AP/PA,
No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion,
Pleural Other, Fracture, Support Devices
```

The 14 pathology columns use the standard CheXpert label convention:

| Value  | Meaning              |
|--------|-----------------------|
| `1.0`  | Positive               |
| `0.0`  | Negative               |
| `-1.0` | Uncertain              |
| blank  | Not mentioned in report |

### Known gotcha: path prefix mismatch

The `Path` column values look like `CheXpert-v1.0-small/train/patientXXXXX/studyN/...`,
but the actual on-disk layout under `kaggle/` **does not have that
`CheXpert-v1.0-small/` prefix** — it's just `train/patientXXXXX/studyN/...`. Any code that
joins CSV rows to image files needs to strip that prefix before resolving the path.

### Other files at repo root

- `archive.zip` (~11.5GB) — the original dataset download that `kaggle/` was extracted
  from. Not needed once `kaggle/` exists.

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

Whenever a new dataset, sample, split, preprocessing stage, or transformed dataset is created,
automatically perform exploratory data analysis before training or evaluation.

The EDA should include, where applicable:

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
Only relevant when we are pre-processing images or gathering representative class samples in classification.
Compute:

- image dimensions
- aspect ratio distribution
- pixel intensity statistics
- histogram summaries
- grayscale range
- brightness statistics
- contrast statistics

If preprocessing has been applied (HE, CLAHE, AGC, etc.), compare before/after statistics.

### Visualizations

Generate figures whenever practical:

- sample images
- preprocessing comparison grids

Store plots under

results/<experiment>/eda/

Never overwrite previous plots.

## Research Workflow

For every new experiment, follow this workflow unless explicitly instructed otherwise.

1. Understand the experiment objective from `experiments.md`.
2. Inspect the current codebase before implementing changes.
3. If creating a new dataset or transformed sample:
   - Run exploratory data analysis (EDA).
   - Save EDA artifacts.
   - Update the experiment log. (ex. /logs/exp1.md)
4. Implement the experiment.
5. Evaluate the results.
6. Record observations, limitations, and next steps.
7. Never overwrite previous experiment outputs. Create new versioned directories instead.
