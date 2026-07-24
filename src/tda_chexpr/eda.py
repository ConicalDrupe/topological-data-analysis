"""Basic count/EDA helpers for a constructed cohort. See experiments.md,
"Shared infrastructure" -> Basic EDA / count analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize_cohort(df: pd.DataFrame, label: str) -> dict:
    return {
        "n_records": len(df),
        "n_patients": df["patient_id"].nunique(),
        f"{label}_counts": df[label].value_counts(dropna=False).to_dict(),
        "ap_pa_counts": df["AP/PA"].value_counts(dropna=False).to_dict(),
        "frontal_lateral_counts": df["Frontal/Lateral"].value_counts(dropna=False).to_dict(),
        "sex_counts": df["Sex"].value_counts(dropna=False).to_dict(),
        "age_stats": df["Age"].describe().to_dict(),
    }


def print_summary(summary: dict, title: str) -> None:
    print(f"--- {title} ---")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print()


def plot_age_histogram(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=20)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
