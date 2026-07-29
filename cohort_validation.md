# Experiment 1 Cohort Validation

Validation of the Pneumothorax cohort-selection pipeline against the four criteria
below, and correction of the issues found. See `experiments.md` (Experiment 1) for the
full pipeline spec and `logs/exp1_log.md` for the run log.

## Target criteria

1. Target pathology is Pneumothorax.
2. Positive cases (`Pneumothorax == 1.0`) have **no comorbidity** — no other pathology
   present at the same time.
3. Negative cases (`Pneumothorax == 0.0`) have **no other diseases** — a "pure false."
4. One row per patient — the patient's **earliest study** among those that qualify on
   criteria 1–3. If a patient qualifies at more than one study, keep the earliest.

## Pipeline walkthrough (source: `kaggle/train.csv`, `kaggle/valid.csv`)

| Step | What it does | Criterion |
|---|---|---|
| `filter_binary_label` | Keep `Pneumothorax` in `{0.0, 1.0}` only (drop `-1.0`/blank) | 1 |
| `filter_ap_only` | Keep `AP/PA == "AP"` (also implies frontal-only) | — (view narrowing, not one of the 4 criteria but part of the established Exp 1 baseline) |
| `filter_no_support_devices` | Keep `Support Devices == 0.0` (confirmed absence) | — (same) |
| `filter_no_comorbidity` (**new**) | Positive rows: require `comorbidity_count == 0`. Negative rows: require `No Finding == 1.0` | 2, 3 |
| `select_studies(mode="first_qualifying")` | Per patient, keep the row with the lowest `study_number`, then (**new**) the lowest `view` number if more than one view in that study still qualifies | 4 |

`comorbidity_count` = number of the 11 other pathology columns (all 14 minus
`Pneumothorax`, `No Finding`, `Support Devices`) that are confirmed-positive (`== 1.0`);
uncertain (`-1.0`) values are not counted. `No Finding == 1.0` is the labeler's own
explicit "nothing found" signal — used for negatives because it is stricter than
`comorbidity_count == 0` alone (it also excludes rows where another finding is merely
uncertain, not just confirmed). There's no equivalent explicit signal for positives
(`No Finding` is mutually exclusive with any positive pathology label, including
Pneumothorax itself), so positives use `comorbidity_count == 0` directly. This asymmetry
is intentional, not an oversight.

## Issues found and fixed

### Bug 1 — positives were never filtered for comorbidity (criterion 2 violated)

The old pipeline (`filter_clean_negatives` in `tda_chexpr/cohort.py`) kept **every**
`Pneumothorax == 1.0` row unconditionally and only restricted negatives to
`No Finding == 1.0`. This was a deliberate, documented decision at the time
(`experiments.md`: "positives are kept regardless of their own comorbidities") — but it
does not satisfy criterion 2 as stated. It also meant most of the persistence-diagram
signal Experiment 1 measures for "Pneumothorax-positive" was potentially confounded by
other visible pathology.

**Fix:** added `filter_no_comorbidity()`, applied to both positive and negative rows
(asymmetric rule above), replacing `filter_clean_negatives()`.

### Bug 2 — comorbidity filtering ran after per-patient dedup (criterion 4 undermined)

`comorbidity_count`/`is_clean_negative` were computed and the negative-only filter
applied *after* `select_studies()` picked each patient's earliest study. So "earliest
qualifying study" only accounted for label/view/device — not comorbidity. A patient could
be anchored to an early study that fails the comorbidity check even though a later study
of theirs would have qualified cleanly.

**Fix:** comorbidity computation and filtering now run *before* `select_studies()`, so
"qualifies" means "satisfies criteria 1–3 jointly," matching the literal reading of
criterion 4.

### Bug 3 — a "study" can still be more than one row (criterion 4 violated)

Even with the study-level dedup, a single qualifying study can have more than one
qualifying view (e.g. `view1_frontal.jpg` and `view2_frontal.jpg`, both AP). This left
duplicate `patient_id` rows in the cohort — confirmed present in **both** the old and new
pipeline before this fix (33 duplicate rows in the old full cohort, 23 in the old
clean-negatives cohort). This was previously masked, not fixed: the 80/20 split already
splits on unique `patient_id` rather than rows (documented in `logs/exp1_log.md` as the
fix for an 8-patient train/test leak), so it never *leaked* across train/test, but the
cohort itself still had 1 patient → 2+ rows, which directly violates criterion 4's "we
don't want to corrupt our data with several observations of the same person."

**Fix:** `select_studies()` now also breaks ties by lowest `view` number within the
selected study, guaranteeing exactly one row per patient. Verified: `n_records == n_patients`
in every output file below.

## Before vs. after

All counts below are rows / unique patients. "Reference" = label + AP + no-support-devices
filters only (no comorbidity requirement, kept for future comparison per
`experiments.md`). "Clean" = reference + `filter_no_comorbidity` (the corrected
primary/active dataset). Old outputs are under `data/exp1/`; new outputs are under
`data/exp1/v2_corrected_cohort/` (old files left in place, not overwritten).

| Dataset | Old (buggy) | New (corrected) |
|---|---|---|
| Reference — train | 2,299 rows / 2,266 patients (869 pos / 1,430 neg) — **33 patients had 2 rows** | 2,266 rows / 2,266 patients (848 pos / 1,418 neg) |
| Reference — valid | 77 / 77 (1 pos / 76 neg) | 77 / 77 (1 pos / 76 neg) |
| Clean — train | 1,189 / 1,166 (869 pos / 320 neg) — **23 patients had 2 rows** | 575 / 575 (**253 pos** / 322 neg) |
| Clean — valid | 13 / 13 (1 pos / 12 neg) | 12 / 12 (**0 pos** / 12 neg) |
| train_split (80%) | 953 / 932 (698 pos / 255 neg) | 460 / 460 (202 pos / 258 neg) |
| test_split (20%) | 236 / 234 (171 pos / 65 neg) | 115 / 115 (51 pos / 64 neg) |

**Headline finding:** enforcing "no comorbidity" on positives (criterion 2, previously
unenforced) removes **71% of the previous positive class** (869 → 253 in the train
cohort) — most Pneumothorax-positive studies in this dataset also have at least one other
confirmed finding. The corrected primary cohort is roughly half the size of the old one
(1,189 → 575 rows) but is now balanced much closer to 50/50 (was 73%/27% pos/neg, now
44%/56%).

**Valid-set note:** the single Pneumothorax-positive row in the old `valid` clean cohort
had a comorbidity, so it's dropped in the corrected version — the clean valid cohort now
has 0 positive cases. This doesn't block anything (per `experiments.md`, `valid.csv` was
already too small to serve as the held-out set — Experiment 1 carves its own 80/20 split
from the train cohort), but the valid clean cohort is now negative-only and not useful for
even a spot-check on positives.

## Verification performed

- `comorbidity_count == 0` for every positive row in the clean cohort (checked directly).
- `is_clean_negative == True` for every negative row in the clean cohort.
- `n_records == n_patients` in every output file (zero duplicate patients).
- Zero patient overlap between `train_split` and `test_split`.
- `len(train_split) + len(test_split) == len(clean train cohort)` (460 + 115 = 575).
- Full EDA (dataset summary: counts, patients, studies, view/orientation breakdown, age,
  sex, missing values, duplicates, class frequencies) run on all 6 datasets — see
  `results/exp1/eda/v9/`.

## Downstream impact (not addressed here)

Experiments 002–005 (ROI crop, CLAHE normalization, cubical filtration, denoising) were
all developed and tuned against the old, buggy cohort/split sample
(`data/exp1/pneumothorax_train_split.csv`). Since the corrected cohort is a different
(smaller, more balanced) set of patients, that sample is stale. Regenerating the
preprocessing sample and re-validating those steps against the corrected split is the
next step before Experiment 1 resumes — out of scope for this cohort-correction task.
