"""Cohort construction: label filtering and per-patient study selection.

Shared between Experiment 1 (one study per patient) and Experiment 3 (all
qualifying studies per patient, ordered) -- see experiments.md, "Shared
infrastructure".
"""

import pandas as pd

from tda_chexpr.data import parse_path_components


def add_path_components(df: pd.DataFrame) -> pd.DataFrame:
    """Attach patient_id/study_number/view/orientation columns parsed from Path."""
    parsed = df["Path"].apply(parse_path_components).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)


def filter_binary_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep only rows where `label` is exactly 0.0 or 1.0 (drop -1.0 and blank)."""
    return df[df[label].isin([0.0, 1.0])].copy()


def select_studies(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Select which studies to keep per patient.

    mode="first_qualifying": one study per patient -- the earliest
      (lowest study_number) among the rows remaining after filtering.
      This is Experiment 1's rule.
    mode="all_ordered": keep every remaining study per patient, sorted by
      study_number. This is Experiment 3's rule.
    """
    if mode == "first_qualifying":
        min_study = df.groupby("patient_id")["study_number"].transform("min")
        return df[df["study_number"] == min_study].copy()
    if mode == "all_ordered":
        return df.sort_values(["patient_id", "study_number"]).reset_index(drop=True)
    raise ValueError(f"Unknown mode: {mode!r}")


def filter_frontal_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Frontal/Lateral"] == "Frontal"].copy()


def build_cohort(
    df: pd.DataFrame,
    label: str,
    mode: str,
    frontal_only: bool = False,
) -> pd.DataFrame:
    """Full cohort construction pipeline: filter -> select studies -> narrow view."""
    df = add_path_components(df)
    df = filter_binary_label(df, label)
    df = select_studies(df, mode=mode)
    if frontal_only:
        df = filter_frontal_only(df)
    return df.reset_index(drop=True)
