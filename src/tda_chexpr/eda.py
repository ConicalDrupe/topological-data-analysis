"""Dataset Summary EDA helpers. See CLAUDE.md, "Automatic Exploratory Data Analysis
(EDA)" for the schema this implements, and experiments.md, "Shared infrastructure".
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tda_chexpr.data import PATHOLOGY_COLUMNS


def summarize_cohort(df: pd.DataFrame, label: str) -> dict:
    """CLAUDE.md's "Dataset Summary" schema: counts, breakdowns, and data-quality
    checks (missing values, duplicate paths) for a cohort/split CSV.
    """
    return {
        "n_records": len(df),
        "n_patients": df["patient_id"].nunique(),
        "n_studies": df[["patient_id", "study_number"]].drop_duplicates().shape[0],
        f"{label}_counts": df[label].value_counts(dropna=False).to_dict(),
        "pathology_counts": {
            col: df[col].value_counts(dropna=False).to_dict()
            for col in PATHOLOGY_COLUMNS
            if col in df.columns
        },
        "ap_pa_counts": df["AP/PA"].value_counts(dropna=False).to_dict(),
        "frontal_lateral_counts": df["Frontal/Lateral"].value_counts(dropna=False).to_dict(),
        "sex_counts": df["Sex"].value_counts(dropna=False).to_dict(),
        "age_stats": df["Age"].describe().to_dict(),
        "missing_value_counts": df.isna().sum().to_dict(),
        "duplicate_path_count": int(df["Path"].duplicated().sum()),
    }


def print_summary(summary: dict, title: str) -> None:
    print(f"--- {title} ---")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print()


def save_summary_json(summary: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def plot_age_histogram(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=20)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def next_version_dir(base_dir: Path) -> Path:
    """Return the next unused `base_dir/vN` directory (creating it), so repeated
    runs never overwrite a previous run's artifacts (per CLAUDE.md).
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name[1:])
        for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    ]
    next_n = max(existing, default=0) + 1
    version_dir = base_dir / f"v{next_n}"
    version_dir.mkdir(parents=True)
    return version_dir
