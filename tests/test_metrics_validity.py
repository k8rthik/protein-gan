"""Tests for the "is this even a distance matrix" metrics."""

from __future__ import annotations

import numpy as np
import pytest

from folduzz.errors import InvalidInputError
from folduzz.metrics.validity import (
    backbone_bond_stats,
    diagonal_stats,
    symmetry_stats,
    triangle_stats,
    validity_report,
)


def real_matrices(count: int = 4, size: int = 16, seed: int = 0) -> np.ndarray:
    """Genuine Euclidean distance matrices from random 3D points."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        coords = rng.normal(scale=8.0, size=(size, 3))
        deltas = coords[:, None, :] - coords[None, :, :]
        out.append(np.sqrt((deltas**2).sum(-1)))
    return np.stack(out).astype(np.float32)


def chain_matrices(count: int = 3, size: int = 16, bond: float = 3.8) -> np.ndarray:
    """Distance matrices of straight chains with exact `bond` spacing."""
    out = []
    for _ in range(count):
        coords = np.zeros((size, 3))
        coords[:, 0] = np.arange(size) * bond
        deltas = coords[:, None, :] - coords[None, :, :]
        out.append(np.sqrt((deltas**2).sum(-1)))
    return np.stack(out).astype(np.float32)


class TestSymmetryStats:
    def test_perfect_on_real_matrices(self):
        stats = symmetry_stats(real_matrices())
        assert stats.mean_abs_asymmetry == pytest.approx(0.0, abs=1e-4)
        assert stats.relative_asymmetry == pytest.approx(0.0, abs=1e-5)

    def test_detects_asymmetry(self):
        matrices = real_matrices(count=1).copy()
        matrices[0, 0, 1] += 10.0
        stats = symmetry_stats(matrices)
        assert stats.mean_abs_asymmetry > 0.0
        assert stats.max_abs_asymmetry == pytest.approx(10.0, abs=1e-3)

    def test_relative_is_scaled_by_mean_distance(self):
        matrices = real_matrices(count=2)
        doubled = matrices * 2
        assert symmetry_stats(matrices).relative_asymmetry == pytest.approx(
            symmetry_stats(doubled).relative_asymmetry, abs=1e-6
        )

    def test_rejects_non_square(self):
        with pytest.raises(InvalidInputError):
            symmetry_stats(np.zeros((2, 4, 5)))

    def test_rejects_wrong_rank(self):
        with pytest.raises(InvalidInputError):
            symmetry_stats(np.zeros((4, 4)))


class TestDiagonalStats:
    def test_zero_for_real_matrices(self):
        stats = diagonal_stats(real_matrices())
        assert stats.mean_abs_diagonal == pytest.approx(0.0, abs=1e-5)

    def test_detects_nonzero_diagonal(self):
        matrices = real_matrices(count=1).copy()
        matrices[0, 3, 3] = 7.0
        stats = diagonal_stats(matrices)
        assert stats.max_abs_diagonal == pytest.approx(7.0, abs=1e-3)
        assert stats.mean_abs_diagonal > 0.0


class TestTriangleStats:
    def test_real_matrices_do_not_violate(self):
        stats = triangle_stats(real_matrices(), tolerance=1e-3)
        assert stats.violation_rate == pytest.approx(0.0, abs=1e-6)
        assert stats.mean_excess_angstrom == pytest.approx(0.0, abs=1e-6)

    def test_violating_matrix_is_caught(self):
        # A "long way round is shorter" matrix: d(0,2) >> d(0,1) + d(1,2).
        matrix = np.array(
            [[0.0, 1.0, 50.0], [1.0, 0.0, 1.0], [50.0, 1.0, 0.0]], dtype=np.float32
        )[None]
        stats = triangle_stats(matrix, tolerance=0.05)
        assert stats.violation_rate > 0.0
        assert stats.max_excess_angstrom == pytest.approx(48.0, abs=1e-3)

    def test_tolerance_suppresses_float_noise(self):
        matrices = real_matrices(count=2)
        noisy = matrices + np.float32(1e-4)
        np.fill_diagonal(noisy[0], 0.0)
        np.fill_diagonal(noisy[1], 0.0)
        assert triangle_stats(noisy, tolerance=1.0).violation_rate == pytest.approx(0.0)

    def test_rate_is_between_zero_and_one(self):
        rng = np.random.default_rng(1)
        noise = np.abs(rng.normal(20, 10, size=(3, 12, 12))).astype(np.float32)
        rate = triangle_stats(noise).violation_rate
        assert 0.0 < rate < 1.0

    def test_subsampling_matches_full_computation_roughly(self):
        matrices = real_matrices(count=2, size=20)
        full = triangle_stats(matrices).violation_rate
        subset = triangle_stats(matrices, max_matrices=1).violation_rate
        assert full == pytest.approx(subset, abs=0.01)

    def test_rejects_bad_tolerance(self):
        with pytest.raises(InvalidInputError):
            triangle_stats(real_matrices(), tolerance=-1.0)


class TestBackboneBondStats:
    def test_ideal_chain_has_ideal_bonds(self):
        stats = backbone_bond_stats(chain_matrices())
        assert stats.mean_angstrom == pytest.approx(3.8, abs=1e-3)
        assert stats.std_angstrom == pytest.approx(0.0, abs=1e-3)
        assert stats.fraction_plausible == pytest.approx(1.0)

    def test_random_matrices_have_implausible_bonds(self):
        stats = backbone_bond_stats(real_matrices())
        assert stats.fraction_plausible < 0.5

    def test_tolerance_is_respected(self):
        matrices = chain_matrices(bond=5.0)
        assert backbone_bond_stats(matrices, tolerance=0.5).fraction_plausible == 0.0
        assert backbone_bond_stats(matrices, tolerance=2.0).fraction_plausible == 1.0


class TestValidityReport:
    def test_assembles_all_sections(self):
        report = validity_report(real_matrices())
        assert set(report) == {"symmetry", "diagonal", "triangle", "backbone_bond"}
        assert report["symmetry"]["mean_abs_asymmetry_angstrom"] == pytest.approx(0.0, abs=1e-4)

    def test_values_are_plain_floats_for_json(self):
        report = validity_report(real_matrices())
        for section in report.values():
            assert all(isinstance(value, float) for value in section.values())
