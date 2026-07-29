"""Cubical persistence filtration: sublevel vs superlevel direction on the finalized
preprocessing pipeline output (no smoothing/denoising step). See experiments.md,
Experiment 1 pipeline step 4, and logs/exp1_log.md, Experiment 004.
"""

import json

import pandas as pd

from tda_chexpr.data import REPO_ROOT, resolve_image_path
from tda_chexpr.eda import next_version_dir
from tda_chexpr.filtration import (
    FILTRATION_DIRECTIONS,
    apply_direction,
    compute_persistence_diagram,
    diagram_summary_stats,
    plot_persistence_diagram_grid,
)
from tda_chexpr.preprocessing import DEFAULT_CLAHE_PARAMS, apply_normalization, load_image_grayscale, plot_stage_grid
from tda_chexpr.roi import apply_roi_crop

DATA_DIR = REPO_ROOT / "data" / "exp1"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"

FIXED_SIZE = 224


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
        pipeline_images.append({"path": record["Path"], "label": label, "raw": raw, "image": clahe})

    representative_pos = next(e for e in pipeline_images if e["label"] == "pos")
    representative_neg = next(e for e in pipeline_images if e["label"] == "neg")

    # 1. Before/after grid: all 10 samples, Raw vs Postprocessed (cropped+resized+CLAHE).
    before_after_rows = [
        (entry["label"], [("Raw", entry["raw"]), ("Postprocessed (CLAHE)", entry["image"])])
        for entry in pipeline_images
    ]
    plot_stage_grid(
        before_after_rows,
        out_dir / "postprocessing_before_after.png",
        title="Postprocessing pipeline: Raw vs. lung-crop + resize + CLAHE (kernel_size=16, clip_limit=0.01)",
    )

    # 2. Full quantitative sweep: all 10 images x 2 directions, no smoothing.
    stats_records = []
    for entry in pipeline_images:
        for direction in FILTRATION_DIRECTIONS:
            directed = apply_direction(entry["image"], direction)
            diagram = compute_persistence_diagram(directed)
            stats_records.append(
                {
                    "path": entry["path"],
                    "label": entry["label"],
                    "direction": direction,
                    **diagram_summary_stats(diagram),
                }
            )
    with open(out_dir / "filtration_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    # 3. Persistence diagram grid: 2 representative images x 2 directions.
    direction_rows = []
    for entry in (representative_pos, representative_neg):
        columns = [
            (direction, compute_persistence_diagram(apply_direction(entry["image"], direction)))
            for direction in FILTRATION_DIRECTIONS
        ]
        direction_rows.append((entry["label"], columns))
    plot_persistence_diagram_grid(
        direction_rows,
        out_dir / "persistence_diagram_direction_comparison.png",
        title="Sublevel vs superlevel persistence diagrams (no smoothing)",
    )

    sub_pos = next(r for r in stats_records if r["path"] == representative_pos["path"] and r["direction"] == "sublevel")
    sup_pos = next(r for r in stats_records if r["path"] == representative_pos["path"] and r["direction"] == "superlevel")
    print(f"Representative positive sample: sublevel n_points={sub_pos['n_points']}, superlevel n_points={sup_pos['n_points']}")
    print(f"Cubical filtration artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
