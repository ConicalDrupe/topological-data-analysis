"""Cubical persistence filtration (gtda.homology.CubicalPersistence) and its
preprocessing-level parameters. See experiments.md, Experiment 1 pipeline step 4, and
logs/exp1_log.md, Experiment 004.

CubicalPersistence itself has almost no tunable filtration-shaping parameters -- the
real lever is which structures (dark vs bright) are born first in the filtration. No
smoothing/denoising is applied before filtration -- persistence is computed directly on
the postprocessing pipeline's CLAHE output.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gtda.homology import CubicalPersistence

FILTRATION_DIRECTIONS: list[str] = ["sublevel", "superlevel"]

_HOMOLOGY_DIM_COLORS = {0: "tab:blue", 1: "tab:orange"}


def apply_direction(image: np.ndarray, direction: str) -> np.ndarray:
    """Orient `image` (float, [0, 1]) for a sublevel- or superlevel-set filtration.

    "sublevel": passthrough -- dark structures (e.g. pneumothorax air lucency) are
    born first. "superlevel": `1.0 - image` -- bright structures (e.g. bone, pleural
    line) are born first.
    """
    if direction == "sublevel":
        return image
    if direction == "superlevel":
        return 1.0 - image
    raise ValueError(f"Unknown filtration direction: {direction!r}")


def compute_persistence_diagram(image: np.ndarray, homology_dimensions: tuple[int, ...] = (0, 1)) -> np.ndarray:
    """Cubical persistence diagram for a single grayscale image.

    Returns an (n_points, 3) array of (birth, death, homology_dimension) triples.
    """
    cp = CubicalPersistence(homology_dimensions=homology_dimensions, n_jobs=1)
    return cp.fit_transform(image[None, :, :])[0]


def diagram_summary_stats(diagram: np.ndarray, persistence_threshold: float = 0.1) -> dict:
    """Summary statistics for a persistence diagram -- the EDA-equivalent for
    diagrams (mirrors image_eda.image_stats's role for images).
    """
    births, deaths, dims = diagram[:, 0], diagram[:, 1], diagram[:, 2]
    finite = np.isfinite(deaths)
    persistence = deaths[finite] - births[finite]
    return {
        "n_points": int(diagram.shape[0]),
        "n_h0": int((dims == 0).sum()),
        "n_h1": int((dims == 1).sum()),
        "max_persistence": float(persistence.max()) if persistence.size else 0.0,
        "n_points_persistence_above_0.1": int((persistence > persistence_threshold).sum()),
    }


def plot_persistence_diagram(ax: plt.Axes, diagram: np.ndarray, title: str) -> None:
    """Scatter a persistence diagram (birth vs death, colored by homology dimension,
    with a diagonal reference line) onto a given Axes.
    """
    births, deaths, dims = diagram[:, 0], diagram[:, 1], diagram[:, 2]
    finite = np.isfinite(deaths)
    max_val = max(births.max(), deaths[finite].max()) if finite.any() else 1.0

    for dim, color in _HOMOLOGY_DIM_COLORS.items():
        mask = (dims == dim) & finite
        ax.scatter(births[mask], deaths[mask], s=5, color=color, alpha=0.5, label=f"H{int(dim)}")

    ax.plot([0, max_val], [0, max_val], color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_title(title)


def plot_persistence_diagram_grid(
    rows: list[tuple[str, list[tuple[str, np.ndarray]]]],
    out_path: Path,
    title: str,
) -> None:
    """Grid of persistence diagrams, one row per (row_label, diagrams) pair, where
    `diagrams` is an ordered list of (column_name, diagram) pairs. Mirrors
    preprocessing.plot_stage_grid's layout, for diagrams instead of images.
    """
    n_rows = len(rows)
    n_cols = len(rows[0][1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    for row, (label, diagrams) in enumerate(rows):
        for col, (name, diagram) in enumerate(diagrams):
            ax = axes[row][col]
            plot_persistence_diagram(ax, diagram, title=name if row == 0 else "")
            if col == 0:
                ax.set_ylabel(label)
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
