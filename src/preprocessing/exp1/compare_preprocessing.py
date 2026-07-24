"""Compare HE and CLAHE normalization against the original image, and sweep each
method's parameters, on the representative sample selected by
select_preprocessing_sample.py. See experiments.md, Experiment 1 pipeline step 2.

Also produces the Image Statistics EDA (CLAUDE.md) for the sample: dimensions,
brightness/contrast, and before/after intensity histograms.
"""

import json

import pandas as pd

from tda_chexpr.data import REPO_ROOT, resolve_image_path
from tda_chexpr.eda import next_version_dir
from tda_chexpr.image_eda import image_stats, plot_intensity_histogram_comparison
from tda_chexpr.preprocessing import (
    NORMALIZATION_VARIANTS,
    apply_normalization,
    load_image_grayscale,
    plot_clahe_parameter_grid,
    plot_he_parameter_grid,
    plot_method_comparison,
)

DATA_DIR = REPO_ROOT / "data" / "exp1"
RESULTS_DIR = REPO_ROOT / "results" / "exp1" / "eda"


def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "preprocessing_sample.csv")


def main() -> None:
    sample = load_sample()
    version_dir = next_version_dir(RESULTS_DIR)
    out_dir = version_dir / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [
        ("pos" if record["Pneumothorax"] == 1.0 else "neg", load_image_grayscale(resolve_image_path(record["Path"])))
        for _, record in sample.iterrows()
    ]
    plot_method_comparison(images, out_dir / "method_comparison.png")

    representative_record = sample[sample["Pneumothorax"] == 1.0].iloc[0]
    representative_image = load_image_grayscale(resolve_image_path(representative_record["Path"]))
    plot_clahe_parameter_grid(representative_image, out_dir / "clahe_parameter_grid.png")
    plot_he_parameter_grid(representative_image, out_dir / "he_parameter_grid.png")

    stats_records = []
    dimensions = []
    for _, record in sample.iterrows():
        image = load_image_grayscale(resolve_image_path(record["Path"]))
        dimensions.append({"path": record["Path"], "height": image.shape[0], "width": image.shape[1]})
        label = "positive" if record["Pneumothorax"] == 1.0 else "negative"
        for method, params in NORMALIZATION_VARIANTS:
            normalized = apply_normalization(image, method, **params)
            stats_records.append(
                {
                    "path": record["Path"],
                    "label": label,
                    "method": method,
                    "params": params,
                    **image_stats(normalized),
                }
            )
    with open(out_dir / "image_stats.json", "w") as f:
        json.dump(stats_records, f, indent=2)

    widths = [d["width"] for d in dimensions]
    print(f"Sample image widths: min={min(widths)} max={max(widths)} (height fixed at {dimensions[0]['height']})")

    negative_record = sample[sample["Pneumothorax"] == 0.0].iloc[0]
    for record, tag in [(representative_record, "positive"), (negative_record, "negative")]:
        image = load_image_grayscale(resolve_image_path(record["Path"]))
        images = {method: apply_normalization(image, method, **params) for method, params in NORMALIZATION_VARIANTS}
        plot_intensity_histogram_comparison(
            images,
            out_dir / f"intensity_histogram_comparison_{tag}.png",
            title=f"Intensity histogram comparison ({tag} example)",
        )

    print(f"Preprocessing comparison artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
