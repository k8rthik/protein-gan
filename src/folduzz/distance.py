"""Residue-residue distance matrices: build, crop into fixed windows, scale.

This is the heart of the data contract. The DCGAN only ever sees a
`MATRIX_SIZE x MATRIX_SIZE` array of values in [-1, 1], where

    value = 2 * clip(d_angstrom, 0, MAX_DISTANCE_ANGSTROM) / MAX_DISTANCE_ANGSTROM - 1

so -1 means "0 A apart" and +1 means ">= MAX_DISTANCE_ANGSTROM apart". The scale
is a fixed global constant, not a per-matrix min/max, which is what makes a
generated matrix interpretable in angstroms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from folduzz import config
from folduzz.errors import InsufficientResiduesError, InvalidInputError


@dataclass(frozen=True)
class Window:
    """One fixed-size crop of a chain, with its distance matrix in angstroms."""

    start: int
    matrix: np.ndarray


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Full pairwise Euclidean distance matrix of `coords` (n, 3), in angstroms."""
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != config.EMBED_DIMENSIONS:
        raise InvalidInputError(
            f"coordinates must have shape (n, 3), got {array.shape}"
        )
    if array.shape[0] == 0:
        raise InvalidInputError("coordinates are empty")
    if not np.isfinite(array).all():
        raise InvalidInputError("coordinates contain NaN or infinity")

    deltas = array[:, None, :] - array[None, :, :]
    matrix = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
    # Kill float asymmetry/diagonal noise so "is it symmetric" stays a question
    # about the *model*, not about our own arithmetic.
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def window_starts(
    length: int,
    size: int = config.MATRIX_SIZE,
    stride: int = config.WINDOW_STRIDE,
) -> tuple[int, ...]:
    """Start indices of `size`-long windows over `length` residues.

    Windows advance by `stride`; if the last one would leave a tail uncovered, a
    final flush-right window is added. No zero padding is ever used.
    """
    if size <= 0 or stride <= 0:
        raise InvalidInputError(f"size and stride must be positive, got {size}/{stride}")
    if length < size:
        raise InsufficientResiduesError(
            f"need at least {size} residues to crop a window, got {length}"
        )

    starts = list(range(0, length - size + 1, stride))
    last_covered = starts[-1] + size
    if last_covered < length:
        starts.append(length - size)
    return tuple(starts)


def crop_windows(
    coords: np.ndarray,
    size: int = config.MATRIX_SIZE,
    stride: int = config.WINDOW_STRIDE,
) -> tuple[Window, ...]:
    """Crop `coords` into immutable `Window`s of `size` residues each."""
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2:
        raise InvalidInputError(f"coordinates must be 2-D, got shape {array.shape}")

    windows = []
    for start in window_starts(array.shape[0], size=size, stride=stride):
        matrix = distance_matrix(array[start : start + size])
        matrix.flags.writeable = False
        windows.append(Window(start=start, matrix=matrix))
    return tuple(windows)


def normalize(
    matrix: np.ndarray, max_distance: float = config.MAX_DISTANCE_ANGSTROM
) -> np.ndarray:
    """Angstroms -> [-1, 1] using a fixed global scale. Returns a new array."""
    if max_distance <= 0:
        raise InvalidInputError(f"max_distance must be positive, got {max_distance}")
    clipped = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, max_distance)
    return (clipped / np.float32(max_distance)) * np.float32(2.0) - np.float32(1.0)


def denormalize(
    matrix: np.ndarray, max_distance: float = config.MAX_DISTANCE_ANGSTROM
) -> np.ndarray:
    """[-1, 1] -> angstroms. Negative distances are clipped to 0. New array."""
    if max_distance <= 0:
        raise InvalidInputError(f"max_distance must be positive, got {max_distance}")
    scaled = (np.asarray(matrix, dtype=np.float32) + np.float32(1.0)) * np.float32(
        max_distance / 2.0
    )
    return np.clip(scaled, 0.0, None)


def clipped_fraction(
    matrix: np.ndarray, max_distance: float = config.MAX_DISTANCE_ANGSTROM
) -> float:
    """Fraction of entries that `normalize` would saturate. Used for reporting."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.count_nonzero(array > max_distance) / array.size)
