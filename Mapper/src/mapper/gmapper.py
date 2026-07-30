"""G-Mapper adaptive cover: reimplementation of the DFS gmeans-splitting algorithm from
https://github.com/MRC-Mapper/G-Mapper (mapper_gmean_cover.py), ported and adapted to
satisfy kmapper's duck-typed Cover contract.

kmapper's `KeplerMapper.map()` calls `bins = cover.fit(lens)` (return value only consumed
for a verbose `len()` print) then iterates `cover.transform(lens)` (called with the lens
array only, no second arg), expecting an iterable of 2D arrays — each a subset of rows of
`lens` (kmapper prepends an index column as column 0) belonging to one cube. It also reads
`cover.n_cubes` / `cover.perc_overlap` attributes for `graph["meta_data"]`. A custom cover
does not need to inherit `kmapper.Cover` — its `transform_single` assumes one fixed radius
per axis shared by every cube along that axis, which cannot represent G-Mapper's irregular
per-interval widths.

G-Mapper's own repo is not pip-installable and its own `Cover`/graph-builder bypass kmapper
entirely (different graph data structure) — only `ad_test`'s and `gm_split`'s math and the
DFS iteration loop are ported here; kmapper's own `mapper.map()` graph builder, visualize(),
and cluster_details pipeline are reused unchanged.

2-D extension: G-Mapper's published algorithm only covers a 1-D lens. We run it
independently per lens axis (`gmeans_intervals_1d`), then take the Cartesian product of the
axes' interval lists as the cover's cubes — the same independent-per-axis-then-product
generalization kmapper's own CubicalCover uses for uniform covers. Known limitation:
axis-independent splitting cannot detect joint bimodality invisible in both marginals.
"""

from __future__ import annotations

import warnings
from itertools import product

import numpy as np
from scipy.stats import anderson
from sklearn.mixture import GaussianMixture

from mapper.graph import RANDOM_STATE

# Below this many points, a 2-component GMM fit and an Anderson-Darling normality test
# aren't statistically meaningful (not enough points to estimate two means/stds robustly),
# and scipy.stats.anderson can NaN/raise on very small or constant-valued samples. Treat
# such intervals as non-splittable rather than crashing or producing a degenerate fit.
MIN_INTERVAL_SIZE = 10


def ad_test(values: np.ndarray, min_interval_size: int = MIN_INTERVAL_SIZE) -> float:
    """Corrected Anderson-Darling statistic vs. the normal distribution, ported from
    G-Mapper's `ad_test`. Returns 0.0 (never triggers a split) for intervals too small to
    fit meaningfully, or with zero variance (all-identical values) — the reference only
    guards `n == 0`; this guard is intentionally wider since our intervals can end up much
    smaller than the reference's typical inputs once nested inside a 2D product cover.
    """
    n = len(values)
    if n < min_interval_size or np.ptp(values) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # scipy>=1.17's method= migration notice
        statistic = anderson(values, dist="norm").statistic
    return float(statistic * (1 + 4 / n - 25 / (n**2)))


def gm_split(interval: list[float], members: np.ndarray, g_overlap: float) -> np.ndarray | None:
    """Near-identical port of G-Mapper's `gm_split`: fits a 2-component GMM (means
    initialized at c +/- sqrt(2/pi)*std, the closed-form split point for a Gaussian into two
    equal-mass halves) and derives two overlapping sub-intervals whose overlap width is
    controlled by `g_overlap`.

    Returns None (signals "could not split, treat interval as final") on any GaussianMixture
    fit failure — e.g. a near-singular covariance on a sample with many duplicate values.
    The reference has no such guard; `min_interval_size` alone doesn't rule this out.

    Deviation from the reference: `random_state=RANDOM_STATE` is passed to GaussianMixture
    for reproducibility, matching every other stochastic step in this repo (PCA/TSNE/UMAP
    lenses are all seeded); the reference omits this.
    """
    lo, hi = interval
    center, std = float(np.mean(members)), float(np.std(members, ddof=1))
    shift = np.sqrt(2 / np.pi) * std
    reshaped = members.reshape(-1, 1)
    try:
        gmm = GaussianMixture(
            n_components=2,
            means_init=[[center + shift], [center - shift]],
            covariance_type="full",
            random_state=RANDOM_STATE,
        ).fit(reshaped)
    except Exception:
        return None

    left_idx, right_idx = int(np.argmin(gmm.means_)), int(np.argmax(gmm.means_))
    left_mean, left_std = float(np.min(gmm.means_)), float(np.sqrt(gmm.covariances_[left_idx])[0][0])
    right_mean, right_std = float(np.max(gmm.means_)), float(np.sqrt(gmm.covariances_[right_idx])[0][0])

    left_interval = [
        lo,
        min(left_mean + (1 + g_overlap) * left_std / (left_std + right_std) * (right_mean - left_mean), hi),
    ]
    right_interval = [
        max(right_mean - (1 + g_overlap) * right_std / (left_std + right_std) * (right_mean - left_mean), lo),
        hi,
    ]
    return np.array([left_interval, right_interval])


def gmeans_intervals_1d(
    values: np.ndarray,
    ad_threshold: float,
    g_overlap: float,
    max_intervals: int,
    iterations: int,
    min_interval_size: int = MIN_INTERVAL_SIZE,
) -> np.ndarray:
    """DFS variant of G-Mapper's gmeans_cover — the only method implemented (BFS/randomized
    only change which failing interval is split first; no clear benefit at this dataset
    size, and DFS is the reference's own default).

    Starts from one interval spanning [min(values), max(values)]. Each iteration scans
    intervals left-to-right; the first one whose ad_test(members) > ad_threshold is split
    via gm_split, the two children are reordered so the higher-AD-scoring one comes first
    (matches the reference's left-to-right AD-descending bias), then the scan restarts from
    the top. Converges (returns early) once a full pass finds nothing left to split, or once
    the interval count exceeds max_intervals. Overlap membership uses inclusive bounds, so a
    point in the overlap of two intervals belongs to both.
    """
    lo0, hi0 = float(np.min(values)), float(np.max(values))
    intervals: list[list[float]] = [[lo0, hi0]]

    def members_in(interval: list[float]) -> np.ndarray:
        lo, hi = interval
        return values[(values >= lo) & (values <= hi)]

    for _ in range(iterations):
        split_happened = False
        for idx, interval in enumerate(intervals):
            members = members_in(interval)
            if ad_test(members, min_interval_size) <= ad_threshold:
                continue
            new_pair = gm_split(interval, members, g_overlap)
            if new_pair is None:
                continue  # GMM fit failed; treat as non-splittable, keep scanning
            left_score = ad_test(members_in(new_pair[0]), min_interval_size)
            right_score = ad_test(members_in(new_pair[1]), min_interval_size)
            if right_score > left_score:
                new_pair = new_pair[::-1]
            intervals = intervals[:idx] + list(new_pair) + intervals[idx + 1 :]
            split_happened = True
            break
        if not split_happened or len(intervals) > max_intervals:
            break
    return np.array(intervals)


class GMapperCover:
    """Satisfies kmapper's duck-typed Cover contract (`.fit`, `.transform`, `.n_cubes`,
    `.perc_overlap`) without inheriting `kmapper.Cover`. Cubes are the Cartesian product of
    per-axis G-Mapper intervals — works for both a 1D lens (n_dims == 1, general-purpose,
    not used by this repo's scripts) and a 2D lens (n_dims == 2, the PCA/t-SNE/UMAP lenses
    actually in use).
    """

    def __init__(
        self,
        ad_threshold: float = 10.0,
        g_overlap: float = 0.1,
        max_intervals_per_axis: int = 10,
        iterations: int = 10,
        min_interval_size: int = MIN_INTERVAL_SIZE,
        verbose: int = 0,
    ):
        self.ad_threshold = ad_threshold
        self.g_overlap = g_overlap
        self.max_intervals_per_axis = max_intervals_per_axis
        self.iterations = iterations
        self.min_interval_size = min_interval_size
        self.verbose = verbose

        self.intervals_per_axis_: list[np.ndarray] | None = None
        # Populated in fit(); read by kmapper for graph["meta_data"] — a list is fine, kmapper
        # only stores it verbatim, never arithmetic on it.
        self.n_cubes: list[int] | None = None
        self.perc_overlap = g_overlap

    def fit(self, data: np.ndarray):
        values = data[:, 1:]  # strip kmapper's prepended index column
        n_dims = values.shape[1]
        self.intervals_per_axis_ = [
            gmeans_intervals_1d(
                values[:, d],
                self.ad_threshold,
                self.g_overlap,
                self.max_intervals_per_axis,
                self.iterations,
                self.min_interval_size,
            )
            for d in range(n_dims)
        ]
        self.n_cubes = [len(iv) for iv in self.intervals_per_axis_]
        centers = list(product(*self.intervals_per_axis_))
        if self.verbose > 0:
            print(f" - GMapperCover - intervals per axis: {self.n_cubes}, {len(centers)} product cubes")
        return centers

    def transform(self, data: np.ndarray):
        if self.intervals_per_axis_ is None:
            raise RuntimeError("GMapperCover.transform called before fit")
        values = data[:, 1:]
        n_dims = values.shape[1]
        axis_masks = [
            [(values[:, d] >= lo) & (values[:, d] <= hi) for lo, hi in self.intervals_per_axis_[d]]
            for d in range(n_dims)
        ]
        return [data[np.logical_and.reduce(combo)] for combo in product(*axis_masks)]
