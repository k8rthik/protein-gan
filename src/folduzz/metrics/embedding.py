"""Try to turn a generated distance matrix into 3D coordinates, and score it.

This is where the project's honest limit shows up. A 64x64 matrix of numbers is
only a protein if some set of 64 points in space reproduces it, so the test is:
run classical multidimensional scaling (Torgerson scaling) on the matrix, keep
three dimensions, and measure what was lost.

* **Kruskal stress-1** -- `sqrt(sum (d - d_hat)^2 / sum d^2)` between the input
  distances and the distances of the 3D embedding. 0 is perfect; by the usual
  rule of thumb >0.2 is a poor fit.
* **Negative eigenvalue mass** -- the double-centred matrix of a true Euclidean
  distance matrix is positive semi-definite. The share of absolute eigenvalue
  mass that is negative measures how non-Euclidean the matrix is.
* **Top-3 variance ratio** -- how much of the positive eigenvalue mass fits in
  the three dimensions a protein actually has.
* **Bond plausibility** -- after embedding, are consecutive residues ~3.8 A
  apart? A matrix can embed beautifully and still describe something that is not
  a polypeptide chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from folduzz import config
from folduzz.errors import InvalidInputError
from folduzz.metrics.validity import DEFAULT_BOND_TOLERANCE

#: Default cap on how many matrices to embed (MDS is O(n^3) per matrix).
DEFAULT_MAX_EMBED = 128


@dataclass(frozen=True)
class EmbeddingResult:
    stress1: float
    negative_eigenvalue_mass: float
    top_dim_variance_ratio: float
    reconstruction_rmse_angstrom: float
    bond_mean_angstrom: float
    bond_std_angstrom: float
    bond_fraction_plausible: float


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise InvalidInputError(f"expected one square matrix, got shape {array.shape}")
    symmetric = 0.5 * (array + array.T)
    np.fill_diagonal(symmetric, 0.0)
    return symmetric


def classical_mds(
    matrix: np.ndarray, dim: int = config.EMBED_DIMENSIONS
) -> tuple[np.ndarray, np.ndarray]:
    """Torgerson classical MDS. Returns `(coords (n, dim), eigenvalues desc)`.

    The input is symmetrised first, because MDS is undefined otherwise;
    asymmetry is measured separately in `validity.py` rather than hidden here.
    Dimensions whose eigenvalue is negative contribute zero coordinates -- the
    honest representation of "this distance cannot be realised in 3D".
    """
    symmetric = _symmetrized(matrix)
    size = symmetric.shape[0]
    if dim < 1:
        raise InvalidInputError(f"dim must be >= 1, got {dim}")
    if dim > size:
        raise InvalidInputError(f"dim {dim} exceeds the {size} points available")

    squared = np.square(symmetric)
    centering = np.eye(size) - np.full((size, size), 1.0 / size)
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    kept = np.clip(eigenvalues[:dim], 0.0, None)
    coords = eigenvectors[:, :dim] * np.sqrt(kept)[None, :]
    return coords, eigenvalues


def _pairwise(coords: np.ndarray) -> np.ndarray:
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))


def embed_one(
    matrix: np.ndarray,
    dim: int = config.EMBED_DIMENSIONS,
    ideal_bond: float = config.CA_CA_BOND_ANGSTROM,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
) -> EmbeddingResult:
    """Embed one matrix into `dim` dimensions and score the result."""
    symmetric = _symmetrized(matrix)
    coords, eigenvalues = classical_mds(symmetric, dim=dim)
    embedded = _pairwise(coords)

    residual = symmetric - embedded
    denominator = float(np.square(symmetric).sum())
    stress = float(np.sqrt(np.square(residual).sum() / denominator)) if denominator > 0 else 0.0

    absolute = np.abs(eigenvalues)
    total = float(absolute.sum())
    negative_mass = (
        float(absolute[eigenvalues < 0].sum() / total) if total > 0 else 0.0
    )
    positive = eigenvalues[eigenvalues > 0]
    positive_total = float(positive.sum())
    top_ratio = (
        float(positive[:dim].sum() / positive_total) if positive_total > 0 else 0.0
    )

    bonds = np.diagonal(embedded, offset=1)
    offdiag_count = symmetric.size - symmetric.shape[0]
    rmse = (
        float(np.sqrt(np.square(residual)[~np.eye(symmetric.shape[0], dtype=bool)].sum() / offdiag_count))
        if offdiag_count
        else 0.0
    )

    return EmbeddingResult(
        stress1=stress,
        negative_eigenvalue_mass=negative_mass,
        top_dim_variance_ratio=top_ratio,
        reconstruction_rmse_angstrom=rmse,
        bond_mean_angstrom=float(bonds.mean()),
        bond_std_angstrom=float(bonds.std()),
        bond_fraction_plausible=float(np.mean(np.abs(bonds - ideal_bond) <= bond_tolerance)),
    )


def embedding_report(
    matrices: np.ndarray,
    max_count: int = DEFAULT_MAX_EMBED,
    dim: int = config.EMBED_DIMENSIONS,
) -> dict[str, float]:
    """Embed up to `max_count` matrices and summarise mean/median per metric."""
    array = np.asarray(matrices, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise InvalidInputError(
            f"expected a (count, n, n) stack of square matrices, got {array.shape}"
        )
    if max_count < 1:
        raise InvalidInputError(f"max_count must be >= 1, got {max_count}")

    results = [embed_one(matrix, dim=dim) for matrix in array[:max_count]]
    fields = (
        "stress1",
        "negative_eigenvalue_mass",
        "top_dim_variance_ratio",
        "reconstruction_rmse_angstrom",
        "bond_mean_angstrom",
        "bond_std_angstrom",
        "bond_fraction_plausible",
    )
    report: dict[str, float] = {"matrices_embedded": float(len(results))}
    for field in fields:
        values = np.array([getattr(result, field) for result in results], dtype=np.float64)
        report[f"{field}_mean"] = float(values.mean())
        report[f"{field}_median"] = float(np.median(values))
    return report
