"""Trivial baselines, so "the GAN scored X" means something.

Three controls, each designed to be good at exactly one thing and hopeless at
the rest:

* **gaussian** -- i.i.d. normal entries matched to the real mean and standard
  deviation, clipped to [0, max_distance]. The floor: right magnitudes, no
  structure, not even symmetry.
* **shuffled_distances** -- a real matrix's upper triangle randomly permuted and
  mirrored. Keeps the real distance histogram *exactly*, keeps symmetry and the
  zero diagonal, destroys the geometry. It shows how little a matching distance
  histogram proves.
* **residue_permuted** -- a real matrix with rows and columns jointly permuted.
  Still an exact Euclidean distance matrix of the same points, just relabelled,
  so it is perfect on every validity metric and on MDS stress while being
  nonsense as a chain. It shows how little perfect validity proves.

All functions take and return distance matrices in angstroms and never mutate
their input.
"""

from __future__ import annotations

import numpy as np

from folduzz import config
from folduzz.errors import InvalidInputError

BASELINE_NAMES = (
    "baseline_gaussian",
    "baseline_shuffled_distances",
    "baseline_residue_permuted",
)


def _validate(matrices: np.ndarray) -> np.ndarray:
    array = np.asarray(matrices, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise InvalidInputError(
            f"expected a (count, n, n) stack of square matrices, got {array.shape}"
        )
    if array.shape[0] == 0:
        raise InvalidInputError("no real matrices to build baselines from")
    return array


def _selection(count: int, available: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of `count` real matrices, sampling with replacement if needed."""
    if count < 1:
        raise InvalidInputError(f"count must be >= 1, got {count}")
    if count <= available:
        return rng.permutation(available)[:count]
    return rng.integers(0, available, size=count)


def gaussian_baseline(
    real: np.ndarray,
    count: int,
    rng: np.random.Generator,
    max_distance: float = config.MAX_DISTANCE_ANGSTROM,
) -> np.ndarray:
    """i.i.d. normal matrices matched to the real mean/std, clipped to range."""
    array = _validate(real)
    if count < 1:
        raise InvalidInputError(f"count must be >= 1, got {count}")
    size = array.shape[1]
    samples = rng.normal(array.mean(), array.std(), size=(count, size, size))
    return np.clip(samples, 0.0, max_distance)


def shuffled_distance_baseline(real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute each matrix's upper triangle and mirror it."""
    array = _validate(real)
    size = array.shape[1]
    rows, cols = np.triu_indices(size, k=1)

    out = np.zeros_like(array)
    for index, matrix in enumerate(array):
        values = rng.permutation(matrix[rows, cols])
        out[index, rows, cols] = values
        out[index, cols, rows] = values
    return out


def residue_permuted_baseline(real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Jointly permute rows and columns: the same geometry, relabelled."""
    array = _validate(real)
    size = array.shape[1]
    out = np.empty_like(array)
    for index, matrix in enumerate(array):
        order = rng.permutation(size)
        out[index] = matrix[np.ix_(order, order)]
    return out


def build_baselines(
    real: np.ndarray,
    count: int,
    seed: int = 0,
    max_distance: float = config.MAX_DISTANCE_ANGSTROM,
) -> dict[str, np.ndarray]:
    """All three baselines, each with `count` matrices, from one seed."""
    array = _validate(real)
    rng = np.random.default_rng(seed)
    chosen = array[_selection(count, array.shape[0], rng)]
    return {
        "baseline_gaussian": gaussian_baseline(
            array, count=count, rng=rng, max_distance=max_distance
        ),
        "baseline_shuffled_distances": shuffled_distance_baseline(chosen, rng=rng),
        "baseline_residue_permuted": residue_permuted_baseline(chosen, rng=rng),
    }
