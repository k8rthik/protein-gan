"""Is a generated array actually a distance matrix?

A DCGAN's output is an image. Nothing in the architecture forces it to obey the
three properties every Euclidean distance matrix has -- symmetry, a zero
diagonal, and the triangle inequality -- and nothing forces consecutive residues
to sit one virtual CA-CA bond apart. These functions measure each of those, in
angstroms, so the answer is a number instead of an impression.

Non-negativity is the one property that comes for free: the generator's Tanh
output maps into [-1, 1], which denormalises to [0, max_distance]. It is
guaranteed by construction, not learned, so it is not reported as a success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from folduzz import config
from folduzz.errors import InvalidInputError

#: Cap on how many matrices the O(n^3) triangle check looks at.
DEFAULT_MAX_TRIANGLE_MATRICES = 64
#: A consecutive CA-CA distance counts as plausible within this many angstroms.
DEFAULT_BOND_TOLERANCE = 0.5


def _validate_stack(matrices: np.ndarray) -> np.ndarray:
    array = np.asarray(matrices, dtype=np.float64)
    if array.ndim != 3:
        raise InvalidInputError(
            f"expected a (count, n, n) stack of matrices, got shape {array.shape}"
        )
    if array.shape[1] != array.shape[2]:
        raise InvalidInputError(f"matrices must be square, got {array.shape[1:]}")
    if array.shape[1] < 3:
        raise InvalidInputError(f"matrices must be at least 3x3, got {array.shape[1]}")
    return array


@dataclass(frozen=True)
class SymmetryStats:
    mean_abs_asymmetry_angstrom: float
    max_abs_asymmetry_angstrom: float
    relative_asymmetry: float

    # Short aliases keep the test and report code readable.
    @property
    def mean_abs_asymmetry(self) -> float:
        return self.mean_abs_asymmetry_angstrom

    @property
    def max_abs_asymmetry(self) -> float:
        return self.max_abs_asymmetry_angstrom

    @property
    def relative(self) -> float:
        return self.relative_asymmetry


@dataclass(frozen=True)
class DiagonalStats:
    mean_abs_diagonal_angstrom: float
    max_abs_diagonal_angstrom: float

    @property
    def mean_abs_diagonal(self) -> float:
        return self.mean_abs_diagonal_angstrom

    @property
    def max_abs_diagonal(self) -> float:
        return self.max_abs_diagonal_angstrom


@dataclass(frozen=True)
class TriangleStats:
    violation_rate: float
    mean_excess_angstrom: float
    max_excess_angstrom: float
    matrices_checked: float


@dataclass(frozen=True)
class BondStats:
    mean_angstrom: float
    std_angstrom: float
    fraction_plausible: float


def symmetry_stats(matrices: np.ndarray) -> SymmetryStats:
    """How far from `M == M.T` the matrices are, absolutely and relatively."""
    array = _validate_stack(matrices)
    asymmetry = np.abs(array - np.transpose(array, (0, 2, 1)))
    mean_distance = float(np.mean(np.abs(array)))
    denominator = mean_distance if mean_distance > 0 else 1.0
    return SymmetryStats(
        mean_abs_asymmetry_angstrom=float(asymmetry.mean()),
        max_abs_asymmetry_angstrom=float(asymmetry.max()),
        relative_asymmetry=float(asymmetry.mean() / denominator),
    )


def diagonal_stats(matrices: np.ndarray) -> DiagonalStats:
    """How far the self-distances are from zero."""
    array = _validate_stack(matrices)
    diagonals = np.abs(np.diagonal(array, axis1=1, axis2=2))
    return DiagonalStats(
        mean_abs_diagonal_angstrom=float(diagonals.mean()),
        max_abs_diagonal_angstrom=float(diagonals.max()),
    )


def triangle_stats(
    matrices: np.ndarray,
    tolerance: float = config.TRIANGLE_TOLERANCE_ANGSTROM,
    max_matrices: int = DEFAULT_MAX_TRIANGLE_MATRICES,
) -> TriangleStats:
    """Fraction of (i, j, k) triples where `d_ik > d_ij + d_jk + tolerance`.

    Checked exhaustively per matrix (n^3 triples) but over at most
    `max_matrices` matrices, which is enough to pin the rate to well under a
    percentage point while staying fast.
    """
    if tolerance < 0:
        raise InvalidInputError(f"tolerance must be >= 0, got {tolerance}")
    if max_matrices < 1:
        raise InvalidInputError(f"max_matrices must be >= 1, got {max_matrices}")

    array = _validate_stack(matrices)
    subset = array[:max_matrices]
    violations = 0
    triples = 0
    excess_sum = 0.0
    excess_max = 0.0

    for matrix in subset:
        # excess[i, j, k] = d_ik - (d_ij + d_jk); positive means a violation.
        excess = matrix[:, None, :] - (matrix[:, :, None] + matrix[None, :, :])
        offending = excess > tolerance
        violations += int(offending.sum())
        triples += int(excess.size)
        if offending.any():
            excess_sum += float(excess[offending].sum())
            excess_max = max(excess_max, float(excess.max()))

    return TriangleStats(
        violation_rate=violations / triples if triples else 0.0,
        mean_excess_angstrom=excess_sum / violations if violations else 0.0,
        max_excess_angstrom=excess_max,
        matrices_checked=float(subset.shape[0]),
    )


def backbone_bond_stats(
    matrices: np.ndarray,
    ideal: float = config.CA_CA_BOND_ANGSTROM,
    tolerance: float = DEFAULT_BOND_TOLERANCE,
) -> BondStats:
    """Statistics of the first off-diagonal, i.e. consecutive-residue distances."""
    if tolerance <= 0:
        raise InvalidInputError(f"tolerance must be > 0, got {tolerance}")
    array = _validate_stack(matrices)
    bonds = np.diagonal(array, offset=1, axis1=1, axis2=2)
    return BondStats(
        mean_angstrom=float(bonds.mean()),
        std_angstrom=float(bonds.std()),
        fraction_plausible=float(np.mean(np.abs(bonds - ideal) <= tolerance)),
    )


def validity_report(
    matrices: np.ndarray,
    tolerance: float = config.TRIANGLE_TOLERANCE_ANGSTROM,
    max_triangle_matrices: int = DEFAULT_MAX_TRIANGLE_MATRICES,
) -> dict[str, dict[str, float]]:
    """All validity metrics, as nested plain-float dicts ready for JSON."""
    sections: dict[str, Any] = {
        "symmetry": symmetry_stats(matrices),
        "diagonal": diagonal_stats(matrices),
        "triangle": triangle_stats(
            matrices, tolerance=tolerance, max_matrices=max_triangle_matrices
        ),
        "backbone_bond": backbone_bond_stats(matrices),
    }
    return {
        name: {key: float(value) for key, value in asdict(stats).items()}
        for name, stats in sections.items()
    }
