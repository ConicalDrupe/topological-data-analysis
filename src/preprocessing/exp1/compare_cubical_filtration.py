"""Cubical persistence filtration: sweep smoothing sigma x filtration direction on the
finalized preprocessing pipeline output, and produce persistence diagrams. See
experiments.md, Experiment 1 pipeline step 4, and logs/exp1_log.md, Experiment 004.
"""

import json

import pandas as pd

from tda_chexpr.data import REPO_ROOT, resolve_image_path
from tda_chexpr.eda import next_version_dir
from tda_chexpr.filtration import (
    FILTRATION_DIRECTIONS,
    SMOOTHING_SIGMAS,
    apply_direction,
    apply_smoothing,
    compute_persistence_diagram,
    diagram_summary_stats,
    plot_persistence_diagram_grid,
)
from tda_chexpr.preprocessing import DEFAULT_CLAHE_PARAMS, apply_normalization, load_image_grayscale, plot_stage_grid
from tda_chexpr.roi import apply_roi_crop

DATA_DIR = REPO_ROOT / "data" / "exp1"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"

FIXED_SIZE = 224
REPRESENTATIVE_SIGMA = 1.0


def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "preprocessing_sample.csv")


def main() -> None:
    sample = load_sample()
    version_dir = next_version_dir(RESULTS_DIR)
    out_dir = version_dir / "cubical_filtration"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_images = []
    for _, record in sample.iterrows():
        label = "pos" if record["Pneumothorax"] == 1.0 else "neg"
        raw = load_image_grayscale(resolve_image_path(record["Path"]))
        try:
            cropped = apply_roi_crop(raw, "lung_mask", margin_frac=0.05, threshold=0.5, size=FIXED_SIZE)
        except ValueError:
            cropped = apply_roi_crop(raw, "center_crop", size=FIXED_SIZE)
        clahe = apply_normalization(cropped, "clahe", **DEFAULT_CLAHE_PARAMS)
        pipeline_images.append({"path": record["Path"], "label": label, "image": clahe})

    representative_pos = next(e for e in pipeline_images if e["label"] == "pos")
    representative_neg = next(e for e in pipeline_images if e["label"] == "neg")

    # 1. Smoothing sigma grid -- single representative (positive) image.
    sigma_stage = [
        (f"sigma={sigma}", apply_smoothing(representative_pos["image"], sigma)) for sigma in SMOOTHING_SIGMAS
    ]
    plot_stage_grid(
        [(representative_pos["label"], sigma_stage)],
        out_dir / "smoothing_sigma_grid.png",
        title="Gaussian smoothing sigma sweep (pre-filtration)",
    )

    # 2. Full quantitative sweep: all 10 images x 5 sigmas x 2 directions.
    stats_records = []
    for entry in pipeline_images:
        for sigma in SMOOTHING_SIGMAS:
            smoothed = apply_smoothing(entry["image"], sigma)
            for direction in FILTRATION_DIRECTIONS:
                directed = apply_direction(smoothed, direction)
                diagram = compute_persistence_diagram(directed)
                stats_records.append(
                    {
                        "path": entry["path"],
                        "label": entry["label"],
                        "sigma": sigma,
                        "direction": direction,
                        **diagram_summary_stats(diagram),
                    }
                )
    with open(out_dir / "filtration_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    # 3. Persistence diagram grid: 2 representative images x 5 sigmas, sublevel fixed.
    diagram_rows = []
    for entry in (representative_pos, representative_neg):
        columns = []
        for sigma in SMOOTHING_SIGMAS:
            smoothed = apply_smoothing(entry["image"], sigma)
            diagram = compute_persistence_diagram(apply_direction(smoothed, "sublevel"))
            columns.append((f"sigma={sigma}", diagram))
        diagram_rows.append((entry["label"], columns))
    plot_persistence_diagram_grid(
        diagram_rows,
        out_dir / "persistence_diagram_sigma_grid.png",
        title="Persistence diagrams across smoothing sigma (sublevel filtration)",
    )

    # 4. Direction comparison: same 2 representative images, sigma fixed.
    direction_rows = []
    for entry in (representative_pos, representative_neg):
        smoothed = apply_smoothing(entry["image"], REPRESENTATIVE_SIGMA)
        columns = [
            (direction, compute_persistence_diagram(apply_direction(smoothed, direction)))
            for direction in FILTRATION_DIRECTIONS
        ]
        direction_rows.append((entry["label"], columns))
    plot_persistence_diagram_grid(
        direction_rows,
        out_dir / "persistence_diagram_direction_comparison.png",
        title=f"Sublevel vs superlevel persistence diagrams (sigma={REPRESENTATIVE_SIGMA})",
    )

    sigma0 = next(r for r in stats_records if r["path"] == representative_pos["path"] and r["sigma"] == 0 and r["direction"] == "sublevel")
    sigma4 = next(r for r in stats_records if r["path"] == representative_pos["path"] and r["sigma"] == 4 and r["direction"] == "sublevel")
    sub1 = next(r for r in stats_records if r["path"] == representative_pos["path"] and r["sigma"] == REPRESENTATIVE_SIGMA and r["direction"] == "sublevel")
    sup1 = next(r for r in stats_records if r["path"] == representative_pos["path"] and r["sigma"] == REPRESENTATIVE_SIGMA and r["direction"] == "superlevel")
    print(f"Representative positive sample: n_points sigma=0 -> {sigma0['n_points']}, sigma=4 -> {sigma4['n_points']}")
    print(f"At sigma={REPRESENTATIVE_SIGMA}: sublevel n_points={sub1['n_points']}, superlevel n_points={sup1['n_points']}")
    print(f"Cubical filtration artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
