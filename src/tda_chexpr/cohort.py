"""Cohort construction: label filtering and per-patient study selection.

Shared between Experiment 1 (one study per patient) and Experiment 3 (all
qualifying studies per patient, ordered) -- see experiments.md, "Shared
infrastructure".
"""

import pandas as pd

from tda_chexpr.data import PATHOLOGY_COLUMNS, parse_path_components

_NON_DISEASE_COLUMNS = ("No Finding", "Support Devices")


def add_path_components(df: pd.DataFrame) -> pd.DataFrame:
    """Attach patient_id/study_number/view/orientation columns parsed from Path."""
    parsed = df["Path"].apply(parse_path_components).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)


def filter_binary_label(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep only rows where `label` is exactly 0.0 or 1.0 (drop -1.0 and blank)."""
    return df[df[label].isin([0.0, 1.0])].copy()


def select_studies(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Select which studies to keep per patient.

    mode="first_qualifying": exactly one row per patient -- the earliest
      (lowest study_number) among the rows remaining after filtering, and
      within that study the lowest view number (a study can have more than
      one qualifying view, e.g. view1 + view2 both AP frontal -- without this
      tiebreak that would leave two rows for one patient). This is
      Experiment 1's rule.
    mode="all_ordered": keep every remaining study per patient, sorted by
      study_number. This is Experiment 3's rule.
    """
    if mode == "first_qualifying":
        min_study = df.groupby("patient_id")["study_number"].transform("min")
        df = df[df["study_number"] == min_study]
        min_view = df.groupby("patient_id")["view"].transform("min")
        return df[df["view"] == min_view].copy()
    if mode == "all_ordered":
        return df.sort_values(["patient_id", "study_number"]).reset_index(drop=True)
    raise ValueError(f"Unknown mode: {mode!r}")


def filter_frontal_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Frontal/Lateral"] == "Frontal"].copy()


def filter_ap_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep AP view only (drop PA and lateral) -- AP is preferred for pneumothorax dx.

    AP/PA is only ever set on frontal rows (lateral rows have it blank), so this
    also implies frontal-only.
    """
    return df[df["AP/PA"] == "AP"].copy()


def filter_no_support_devices(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with confirmed absence of support devices (strict: == 0.0).

    -1.0 (uncertain) and blank (unmentioned) are excluded too, not treated as
    "no device" -- the cleanest label guarantee at the cost of a smaller cohort.
    """
    return df[df["Support Devices"] == 0.0].copy()


def add_comorbidity_count(df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    """Count of OTHER confirmed-positive (==1.0) pathology columns per row --
    excludes target_label itself and non-disease columns (No Finding, Support
    Devices). Uncertain (-1.0) values are not counted -- strict/confirmed-only
    convention, matching filter_no_support_devices.
    """
    other_cols = [
        c for c in PATHOLOGY_COLUMNS
        if c != target_label and c not in _NON_DISEASE_COLUMNS
    ]
    df = df.copy()
    df["comorbidity_count"] = (df[other_cols] == 1.0).sum(axis=1)
    return df


def add_clean_negative_flag(df: pd.DataFrame) -> pd.DataFrame:
    """True where the labeler found nothing at all (No Finding == 1.0) -- the
    strictest available "truly healthy" signal.
    """
    df = df.copy()
    df["is_clean_negative"] = df["No Finding"] == 1.0
    return df


def filter_no_comorbidity(df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    """Keep only rows with no confounding comorbidity, asymmetric by label:

    - target_label == 1.0 (positive): require comorbidity_count == 0 -- no
      other pathology column confirmed positive. There's no "No Finding"-style
      explicit signal available for positives (it's mutually exclusive with
      any positive pathology label, including the target itself).
    - target_label == 0.0 (negative): require is_clean_negative (No Finding
      == 1.0) -- the labeler's own explicit "nothing found" signal, stricter
      than comorbidity_count == 0 alone (it also excludes rows where another
      finding is merely uncertain/-1.0, not just confirmed positive).

    Requires add_comorbidity_count and add_clean_negative_flag to have been
    run already.
    """
    is_positive = df[target_label] == 1.0
    qualifies = (is_positive & (df["comorbidity_count"] == 0)) | (
        ~is_positive & df["is_clean_negative"]
    )
    return df[qualifies].copy()


def build_cohort(
    df: pd.DataFrame,
    label: str,
    mode: str,
    frontal_only: bool = False,
    ap_only: bool = False,
    require_no_support_devices: bool = False,
    require_no_comorbidity: bool = False,
) -> pd.DataFrame:
    """Full cohort construction pipeline: filter -> narrow view/devices/comorbidity
    -> select studies.

    All row-level filters run *before* select_studies so that "qualifying" is a
    joint condition -- a patient's earliest study must satisfy every active
    filter at once (label, view, devices, and comorbidity), not just the label
    filter. This matters for require_no_comorbidity in particular: without it,
    a patient could be anchored to an early study that fails the comorbidity
    check even though a later study of theirs would have qualified cleanly.
    """
    df = add_path_components(df)
    df = filter_binary_label(df, label)
    if frontal_only:
        df = filter_frontal_only(df)
    if ap_only:
        df = filter_ap_only(df)
    if require_no_support_devices:
        df = filter_no_support_devices(df)
    if require_no_comorbidity:
        df = add_comorbidity_count(df, target_label=label)
        df = add_clean_negative_flag(df)
        df = filter_no_comorbidity(df, target_label=label)
    df = select_studies(df, mode=mode)
    return df.reset_index(drop=True)
