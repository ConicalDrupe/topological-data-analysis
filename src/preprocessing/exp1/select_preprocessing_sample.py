"""Select a small representative image sample for the normalization (HE/CLAHE)
comparison. See experiments.md, Experiment 1 pipeline step 2.

Stratified 4 positive / 4 negative sample, seed=42 (matching the split's seed
convention), drawn from the train split so the sample is disjoint from the test split.
"""

import pandas as pd

from tda_chexpr.data import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "exp1"
N_PER_CLASS = 4
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(DATA_DIR / "pneumothorax_train_split.csv")

    positive = df[df["Pneumothorax"] == 1.0].sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
    negative = df[df["Pneumothorax"] == 0.0].sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
    sample = pd.concat([positive, negative]).reset_index(drop=True)

    out_path = DATA_DIR / "preprocessing_sample.csv"
    sample.to_csv(out_path, index=False)

    print(f"Selected {len(sample)} images ({N_PER_CLASS} positive / {N_PER_CLASS} negative) -> {out_path}")
    print(sample[["Path", "Pneumothorax"]].to_string(index=False))


if __name__ == "__main__":
    main()
