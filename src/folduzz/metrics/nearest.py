"""Nearest-neighbour and diversity metrics: memorisation and mode collapse.

Two failure modes that every other metric in this package would happily call a
success:

* **Memorisation.** A generator that reproduces training windows scores
  perfectly on validity and distribution. The nearest-neighbour RMSE from each
  sample to the closest *training* matrix catches it: near zero means copying.
* **Mode collapse.** A generator that emits one plausible matrix over and over
  also scores well on per-matrix metrics. Mean pairwise RMSE among the samples
  (self-diversity) and coverage -- the share of samples that pick a *distinct*
  nearest training neighbour -- catch that.

Distances are RMSE over matrix entries, in angstroms, computed through the
`||a||^2 + ||b||^2 - 2<a, b>` identity so a few hundred samples against a few
thousand references is one BLAS call.
"""

from __future__ import annotations

import numpy as np

from folduzz.errors import InvalidInputError

#: Cap on reference matrices per comparison, to keep the Gram matrix small.
DEFAULT_MAX_REFERENCE = 3000
#: Cap on samples used for the O(n^2) self-diversity computation.
DEFAULT_MAX_DIVERSITY = 256


def _flatten(matrices: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(matrices, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise InvalidInputError(
            f"{label}: expected a (count, n, n) stack of square matrices, got {array.shape}"
        )
    if array.shape[0] == 0:
        raise InvalidInputError(f"{label}: no matrices")
    return array.reshape(array.shape[0], -1)


def pairwise_rmse(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """`(len(query), len(reference))` matrix of entrywise RMSE values."""
    left = _flatten(query, "query")
    right = _flatten(reference, "reference")
    if left.shape[1] != right.shape[1]:
        raise InvalidInputError(
            f"matrix sizes differ: query has {left.shape[1]} entries, "
            f"reference has {right.shape[1]}"
        )
    squared = (
        np.square(left).sum(axis=1)[:, None]
        + np.square(right).sum(axis=1)[None, :]
        - 2.0 * (left @ right.T)
    )
    return np.sqrt(np.clip(squared, 0.0, None) / left.shape[1])


def nearest_neighbour_rmse(
    query: np.ndarray,
    reference: np.ndarray,
    max_reference: int = DEFAULT_MAX_REFERENCE,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query distance to, and index of, the closest reference matrix."""
    if max_reference < 1:
        raise InvalidInputError(f"max_reference must be >= 1, got {max_reference}")
    pool = np.asarray(reference)
    if pool.shape[0] > max_reference:
        chosen = np.random.default_rng(seed).permutation(pool.shape[0])[:max_reference]
        pool = pool[np.sort(chosen)]
    distances = pairwise_rmse(query, pool)
    return distances.min(axis=1), distances.argmin(axis=1)


def mean_pairwise_rmse(
    matrices: np.ndarray, max_count: int = DEFAULT_MAX_DIVERSITY
) -> float:
    """Mean RMSE between distinct pairs of `matrices`. NaN for a single matrix."""
    if max_count < 1:
        raise InvalidInputError(f"max_count must be >= 1, got {max_count}")
    subset = np.asarray(matrices)[:max_count]
    if subset.shape[0] < 2:
        return float("nan")
    distances = pairwise_rmse(subset, subset)
    upper = np.triu_indices(distances.shape[0], k=1)
    return float(distances[upper].mean())


def nearest_neighbour_report(
    generated: np.ndarray,
    reference: np.ndarray,
    max_reference: int = DEFAULT_MAX_REFERENCE,
    max_diversity: int = DEFAULT_MAX_DIVERSITY,
    seed: int = 0,
) -> dict[str, float]:
    """Memorisation and mode-collapse summary for one source of matrices."""
    distances, indices = nearest_neighbour_rmse(
        generated, reference, max_reference=max_reference, seed=seed
    )
    unique = float(np.unique(indices).size)
    return {
        "nn_rmse_mean_angstrom": float(distances.mean()),
        "nn_rmse_median_angstrom": float(np.median(distances)),
        "nn_rmse_min_angstrom": float(distances.min()),
        "self_diversity_rmse_angstrom": mean_pairwise_rmse(
            generated, max_count=max_diversity
        ),
        "coverage": unique / float(len(indices)),
        "distinct_neighbours": unique,
    }
