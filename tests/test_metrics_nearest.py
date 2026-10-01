"""Tests for nearest-neighbour and diversity (mode-collapse) metrics."""

from __future__ import annotations

import numpy as np
import pytest

from folduzz.errors import InvalidInputError
from folduzz.metrics.nearest import (
    mean_pairwise_rmse,
    nearest_neighbour_report,
    nearest_neighbour_rmse,
    pairwise_rmse,
)
from tests.test_metrics_validity import real_matrices


class TestPairwiseRmse:
    def test_matches_the_direct_formula(self):
        a = real_matrices(count=3, size=8, seed=1)
        b = real_matrices(count=4, size=8, seed=2)
        distances = pairwise_rmse(a, b)
        assert distances.shape == (3, 4)
        expected = np.sqrt(np.mean((a[1] - b[2]) ** 2))
        assert distances[1, 2] == pytest.approx(expected, rel=1e-6)

    def test_self_distance_is_zero(self):
        a = real_matrices(count=3, size=8)
        np.testing.assert_allclose(np.diag(pairwise_rmse(a, a)), 0.0, atol=1e-5)

    def test_never_negative(self):
        a = real_matrices(count=4, size=8, seed=3)
        assert pairwise_rmse(a, a).min() >= 0.0

    def test_rejects_size_mismatch(self):
        with pytest.raises(InvalidInputError):
            pairwise_rmse(real_matrices(size=8), real_matrices(size=12))


class TestNearestNeighbourRmse:
    def test_identical_matrices_have_zero_distance(self):
        reference = real_matrices(count=5, size=12, seed=4)
        distances, indices = nearest_neighbour_rmse(reference[:2], reference)
        np.testing.assert_allclose(distances, 0.0, atol=1e-5)
        assert indices.tolist() == [0, 1]

    def test_finds_the_closest_reference(self):
        reference = real_matrices(count=4, size=12, seed=5)
        query = reference[2:3] + 0.001
        distances, indices = nearest_neighbour_rmse(query, reference)
        assert indices[0] == 2
        assert distances[0] < 0.01

    def test_subsamples_large_references_reproducibly(self):
        reference = real_matrices(count=20, size=12, seed=6)
        query = real_matrices(count=3, size=12, seed=7)
        first, _ = nearest_neighbour_rmse(query, reference, max_reference=5, seed=1)
        second, _ = nearest_neighbour_rmse(query, reference, max_reference=5, seed=1)
        np.testing.assert_array_equal(first, second)

    def test_rejects_bad_max_reference(self):
        with pytest.raises(InvalidInputError):
            nearest_neighbour_rmse(real_matrices(), real_matrices(), max_reference=0)


class TestMeanPairwiseRmse:
    def test_zero_for_identical_copies(self):
        matrix = real_matrices(count=1, size=12)
        collapsed = np.repeat(matrix, 6, axis=0)
        assert mean_pairwise_rmse(collapsed) == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_varied_matrices(self):
        assert mean_pairwise_rmse(real_matrices(count=6, size=12, seed=8)) > 1.0

    def test_single_matrix_is_nan(self):
        assert np.isnan(mean_pairwise_rmse(real_matrices(count=1, size=12)))


class TestNearestNeighbourReport:
    def test_detects_memorisation(self):
        """Samples copied from the reference set should score ~0 NN RMSE."""
        reference = real_matrices(count=8, size=12, seed=9)
        report = nearest_neighbour_report(reference[:4], reference)
        assert report["nn_rmse_mean_angstrom"] == pytest.approx(0.0, abs=1e-4)
        assert report["coverage"] == pytest.approx(1.0)

    def test_detects_mode_collapse(self):
        reference = real_matrices(count=8, size=12, seed=10)
        collapsed = np.repeat(real_matrices(count=1, size=12, seed=11), 6, axis=0)
        report = nearest_neighbour_report(collapsed, reference)
        assert report["self_diversity_rmse_angstrom"] == pytest.approx(0.0, abs=1e-5)
        # Six identical samples all pick the same neighbour: coverage 1/6.
        assert report["coverage"] == pytest.approx(1 / 6)

    def test_healthy_samples_have_diversity_and_coverage(self):
        reference = real_matrices(count=12, size=12, seed=12)
        query = real_matrices(count=6, size=12, seed=13)
        report = nearest_neighbour_report(query, reference)
        assert report["self_diversity_rmse_angstrom"] > 1.0
        assert report["nn_rmse_mean_angstrom"] > 0.0
        assert 0.0 < report["coverage"] <= 1.0

    def test_all_values_are_floats(self):
        report = nearest_neighbour_report(
            real_matrices(count=3, size=12), real_matrices(count=5, size=12)
        )
        assert all(isinstance(value, float) for value in report.values())
