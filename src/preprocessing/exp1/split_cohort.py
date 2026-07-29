"""Split the Experiment 1 train cohort into our own train/test sets.

CheXpert's valid.csv is too small after the AP/no-devices filters (77 rows, 1
positive case) to serve as a reliable held-out set for this cohort, so this
experiment carves its own stratified split out of the filtered train cohort.

Sourced from the corrected "clean" cohort (positives require
comorbidity_count == 0, negatives restricted to No Finding == 1.0) rather than
the reference cohort, since Experiment 1 uses the clean cohort as its
primary/active dataset -- see experiments.md, Experiment 1, and
cohort_validation.md.
"""

import pandas as pd

from tda_chexpr.data import REPO_ROOT
from tda_chexpr.eda import print_summary, summarize_cohort
from tda_chexpr.split import stratified_split

DATA_DIR = REPO_ROOT / "data" / "exp1" / "v2_corrected_cohort"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "pneumothorax_cohort_train_clean.csv")
    train_df, test_df = stratified_split(df, label="Pneumothorax", test_frac=0.2, random_state=42)

    train_df.to_csv(DATA_DIR / "pneumothorax_train_split.csv", index=False)
    test_df.to_csv(DATA_DIR / "pneumothorax_test_split.csv", index=False)

    for name, split_df in (("train_split", train_df), ("test_split", test_df)):
        print_summary(summarize_cohort(split_df, label="Pneumothorax"), title=f"Experiment 1 {name}")


if __name__ == "__main__":
    main()
