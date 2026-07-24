"""Dataset Summary EDA for the Experiment 1 cohort and train/test split. See
CLAUDE.md, "Automatic Exploratory Data Analysis (EDA)", and experiments.md,
"Shared infrastructure" -> Basic EDA / count analysis. Run after build_cohort.py
and split_cohort.py.
"""

import pandas as pd

from tda_chexpr.data import REPO_ROOT
from tda_chexpr.eda import next_version_dir, plot_age_histogram, print_summary, save_summary_json, summarize_cohort

DATA_DIR = REPO_ROOT / "data" / "exp1"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"

DATASETS = [
    "pneumothorax_cohort_train",
    "pneumothorax_cohort_valid",
    "pneumothorax_train_split",
    "pneumothorax_test_split",
]


def main() -> None:
    version_dir = next_version_dir(RESULTS_DIR)
    for name in DATASETS:
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        summary = summarize_cohort(df, label="Pneumothorax")
        print_summary(summary, title=f"Experiment 1 - {name}")

        out_dir = version_dir / name
        save_summary_json(summary, out_dir / "summary.json")
        plot_age_histogram(df, out_dir / "age_histogram.png", title=f"Age distribution ({name})")

    print(f"EDA artifacts written to {version_dir}")


if __name__ == "__main__":
    main()
