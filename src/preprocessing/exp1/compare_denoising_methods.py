"""Denoising for cubical persistence diagrams: Anscombe+denoise+inverse (image-space,
pre-CLAHE), persistence thresholding, and confidence-set/bottleneck bootstrap
(diagram-space), compared against the raw (undenoised) baseline. See experiments.md,
Experiment 1 pipeline step 4, and logs/exp1_log.md, Experiment 005.

Sublevel filtration only -- see logs/exp1_log.md, Experiment 005 for the scope
rationale.
"""

import json

import numpy as np
import pandas as pd

from tda_chexpr.data import REPO_ROOT, resolve_image_path
from tda_chexpr.denoising import bottleneck_confidence_cutoff, denoise_anscombe
from tda_chexpr.eda import next_version_dir
from tda_chexpr.filtration import (
    compute_persistence_diagram,
    diagram_summary_stats,
    plot_persistence_diagram_grid,
    threshold_diagram,
)
from tda_chexpr.preprocessing import DEFAULT_CLAHE_PARAMS, apply_normalization, load_image_grayscale, plot_stage_grid
from tda_chexpr.roi import apply_roi_crop

DATA_DIR = REPO_ROOT / "data" / "exp1"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"

FIXED_SIZE = 224
THRESHOLD_EPSILONS = [0.02, 0.05, 0.1]
REPRESENTATIVE_THRESHOLD_EPSILON = 0.05
N_BOOTSTRAP = 20
BLOCK_SIZE = 16
RNG_SEED = 42


def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "preprocessing_sample.csv")


def main() -> None:
    sample = load_sample()
    rng = np.random.default_rng(RNG_SEED)
    version_dir = next_version_dir(RESULTS_DIR)
    out_dir = version_dir / "denoising_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_images = []
    for _, record in sample.iterrows():
        label = "pos" if record["Pneumothorax"] == 1.0 else "neg"
        raw = load_image_grayscale(resolve_image_path(record["Path"]))
        try:
            cropped = apply_roi_crop(raw, "lung_mask", margin_frac=0.05, threshold=0.5, size=FIXED_SIZE)
        except ValueError:
            cropped = apply_roi_crop(raw, "center_crop", size=FIXED_SIZE)
        baseline = apply_normalization(cropped, "clahe", **DEFAULT_CLAHE_PARAMS)
        anscombe_denoised = denoise_anscombe(cropped)
        anscombe_image = apply_normalization(anscombe_denoised, "clahe", **DEFAULT_CLAHE_PARAMS)
        pipeline_images.append(
            {
                "path": record["Path"],
                "label": label,
                "cropped": cropped,
                "baseline": baseline,
                "anscombe_image": anscombe_image,
            }
        )

    representative_pos = next(e for e in pipeline_images if e["label"] == "pos")
    representative_neg = next(e for e in pipeline_images if e["label"] == "neg")

    # 1. Before/after grid: all 10 samples, Raw crop vs Anscombe+denoise (pre-CLAHE).
    before_after_rows = [
        (entry["label"], [("Raw crop", entry["cropped"]), ("Anscombe+denoise (pre-CLAHE)", entry["anscombe_image"])])
        for entry in pipeline_images
    ]
    plot_stage_grid(
        before_after_rows,
        out_dir / "anscombe_denoise_before_after.png",
        title="Anscombe transform + wavelet denoise + inverse, applied before CLAHE",
    )

    # 2. Full quantitative sweep: all 10 images, 4 methods (raw / anscombe / threshold x3 / confidence-set).
    stats_records = []
    representative_diagrams = {}

    for entry in pipeline_images:
        diagram_raw = compute_persistence_diagram(entry["baseline"])
        stats_records.append(
            {"path": entry["path"], "label": entry["label"], "method": "raw", **diagram_summary_stats(diagram_raw)}
        )

        diagram_anscombe = compute_persistence_diagram(entry["anscombe_image"])
        stats_records.append(
            {
                "path": entry["path"],
                "label": entry["label"],
                "method": "anscombe_denoise",
                **diagram_summary_stats(diagram_anscombe),
            }
        )

        for eps in THRESHOLD_EPSILONS:
            diagram_thresh = threshold_diagram(diagram_raw, eps)
            stats_records.append(
                {
                    "path": entry["path"],
                    "label": entry["label"],
                    "method": "threshold",
                    "epsilon": eps,
                    **diagram_summary_stats(diagram_thresh),
                }
            )

        c_n, _ = bottleneck_confidence_cutoff(
            entry["baseline"], n_bootstrap=N_BOOTSTRAP, block_size=BLOCK_SIZE, rng=rng
        )
        diagram_confidence = threshold_diagram(diagram_raw, 2 * c_n)
        stats_records.append(
            {
                "path": entry["path"],
                "label": entry["label"],
                "method": "confidence_set",
                "c_n": c_n,
                "epsilon": 2 * c_n,
                "n_bootstrap": N_BOOTSTRAP,
                "block_size": BLOCK_SIZE,
                **diagram_summary_stats(diagram_confidence),
            }
        )

        if entry is representative_pos or entry is representative_neg:
            representative_diagrams[entry["label"]] = {
                "Raw baseline": diagram_raw,
                "Anscombe+denoise": diagram_anscombe,
                f"Thresholded (eps={REPRESENTATIVE_THRESHOLD_EPSILON})": threshold_diagram(
                    diagram_raw, REPRESENTATIVE_THRESHOLD_EPSILON
                ),
                "Confidence-set": diagram_confidence,
            }

    with open(out_dir / "denoising_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    # 3. Persistence diagram grid: 2 representative images x 4 methods.
    diagram_rows = [
        (label, list(representative_diagrams[label].items())) for label in (representative_pos["label"], representative_neg["label"])
    ]
    plot_persistence_diagram_grid(
        diagram_rows,
        out_dir / "persistence_diagram_method_comparison.png",
        title="Denoising method comparison (sublevel filtration)",
    )

    pos_path = representative_pos["path"]
    raw_pos = next(r for r in stats_records if r["path"] == pos_path and r["method"] == "raw")
    anscombe_pos = next(r for r in stats_records if r["path"] == pos_path and r["method"] == "anscombe_denoise")
    conf_pos = next(r for r in stats_records if r["path"] == pos_path and r["method"] == "confidence_set")
    print(f"Representative positive sample: raw n_points={raw_pos['n_points']}")
    print(f"  anscombe_denoise n_points={anscombe_pos['n_points']}")
    print(f"  confidence_set n_points={conf_pos['n_points']} (c_n={conf_pos['c_n']:.4f}, epsilon={conf_pos['epsilon']:.4f})")
    for eps in THRESHOLD_EPSILONS:
        t = next(r for r in stats_records if r["path"] == pos_path and r["method"] == "threshold" and r["epsilon"] == eps)
        print(f"  threshold(eps={eps}) n_points={t['n_points']}")
    print(f"Denoising comparison artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
