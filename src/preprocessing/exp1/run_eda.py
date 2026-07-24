"""Basic EDA / count analysis over the Experiment 1 cohort. See experiments.md,
"Shared infrastructure" -> Basic EDA / count analysis. Run after build_cohort.py.
"""

import pandas as pd

from tda_chexpr.data import REPO_ROOT
from tda_chexpr.eda import plot_age_histogram, print_summary, summarize_cohort

DATA_DIR = REPO_ROOT / "data" / "exp1"
PLOTS_DIR = REPO_ROOT / "plots" / "exp1"


def main() -> None:
    for split in ("train", "valid"):
        csv_path = DATA_DIR / f"pneumothorax_cohort_{split}.csv"
        df = pd.read_csv(csv_path)
        summary = summarize_cohort(df, label="Pneumothorax")
        print_summary(summary, title=f"Experiment 1 cohort - {split}")
        plot_age_histogram(
            df,
            PLOTS_DIR / f"age_histogram_{split}.png",
            title=f"Age distribution ({split})",
        )


if __name__ == "__main__":
    main()
