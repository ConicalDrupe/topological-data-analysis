"""Build the Experiment 1 Pneumothorax classification cohort.

Filters CheXpert train/valid to Pneumothorax in {0.0, 1.0}, AP view only, and
confirmed absence of support devices, then keeps each patient's earliest study
that qualifies on all three criteria at once. See experiments.md, Experiment 1.
"""

from tda_chexpr.cohort import build_cohort
from tda_chexpr.data import REPO_ROOT, load_labels

OUTPUT_DIR = REPO_ROOT / "data" / "exp1"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        df = load_labels(split)
        cohort = build_cohort(
            df,
            label="Pneumothorax",
            mode="first_qualifying",
            ap_only=True,
            require_no_support_devices=True,
        )
        out_path = OUTPUT_DIR / f"pneumothorax_cohort_{split}.csv"
        cohort.to_csv(out_path, index=False)
        print(
            f"{split}: {len(df)} raw rows -> {len(cohort)} cohort rows "
            f"({cohort['patient_id'].nunique()} patients) -> {out_path}"
        )


if __name__ == "__main__":
    main()
