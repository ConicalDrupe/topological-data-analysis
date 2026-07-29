"""Build the Experiment 1 Pneumothorax classification cohort.

Filters CheXpert train/valid to Pneumothorax in {0.0, 1.0}, AP view only, and
confirmed absence of support devices, then keeps each patient's earliest study
that qualifies on all criteria at once -- including no confounding comorbidity:
positives require comorbidity_count == 0, negatives require No Finding == 1.0.
See experiments.md (Experiment 1) and cohort_validation.md for the criteria and
the correction history.

Also writes an unfiltered-by-comorbidity reference cohort (same label/view/
device filters, no comorbidity requirement) alongside the corrected "clean"
cohort -- "the larger dataset" to revisit once the classification step exists.
"""

from tda_chexpr.cohort import add_clean_negative_flag, add_comorbidity_count, build_cohort
from tda_chexpr.data import REPO_ROOT, load_labels

OUTPUT_DIR = REPO_ROOT / "data" / "exp1" / "v2_corrected_cohort"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        df = load_labels(split)

        reference = build_cohort(
            df,
            label="Pneumothorax",
            mode="first_qualifying",
            ap_only=True,
            require_no_support_devices=True,
        )
        reference = add_comorbidity_count(reference, target_label="Pneumothorax")
        reference = add_clean_negative_flag(reference)
        ref_path = OUTPUT_DIR / f"pneumothorax_cohort_{split}.csv"
        reference.to_csv(ref_path, index=False)
        print(
            f"{split}: {len(df)} raw rows -> {len(reference)} reference cohort rows "
            f"({reference['patient_id'].nunique()} patients) -> {ref_path}"
        )

        clean = build_cohort(
            df,
            label="Pneumothorax",
            mode="first_qualifying",
            ap_only=True,
            require_no_support_devices=True,
            require_no_comorbidity=True,
        )
        clean_path = OUTPUT_DIR / f"pneumothorax_cohort_{split}_clean.csv"
        clean.to_csv(clean_path, index=False)
        pos = int((clean["Pneumothorax"] == 1.0).sum())
        neg = int((clean["Pneumothorax"] == 0.0).sum())
        print(
            f"{split}: {len(df)} raw rows -> {len(clean)} clean cohort rows "
            f"({clean['patient_id'].nunique()} patients, {pos} positive / {neg} negative) -> {clean_path}"
        )


if __name__ == "__main__":
    main()
