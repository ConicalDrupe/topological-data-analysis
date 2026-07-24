"""Stratified train/test split for a cohort too small to rely on CheXpert's own
valid.csv -- see experiments.md, Experiment 1.
"""

import pandas as pd


def stratified_split(
    df: pd.DataFrame, label: str, test_frac: float = 0.2, random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train, test) by patient, preserving each label value's proportion.

    first_qualifying dedup selects one *study* per patient, but that study can
    still contain more than one frontal AP image (multiple views), so a row-level
    split could put two images of the same patient/study on opposite sides of the
    split. Splitting is done on unique patient_id instead, then every row for a
    chosen patient goes to that patient's side of the split.
    """
    patients = df.drop_duplicates(subset="patient_id")[["patient_id", label]]

    train_patient_parts, test_patient_parts = [], []
    for _, group in patients.groupby(label):
        test_group = group.sample(frac=test_frac, random_state=random_state)
        train_group = group.drop(test_group.index)
        train_patient_parts.append(train_group)
        test_patient_parts.append(test_group)

    train_patient_ids = pd.concat(train_patient_parts)["patient_id"]
    test_patient_ids = pd.concat(test_patient_parts)["patient_id"]

    train_df = (
        df[df["patient_id"].isin(train_patient_ids)]
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    test_df = (
        df[df["patient_id"].isin(test_patient_ids)]
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    return train_df, test_df
