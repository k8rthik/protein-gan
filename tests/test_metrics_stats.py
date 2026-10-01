"""Tests for the distribution-matching statistics."""

from __future__ import annotations

import numpy as np
import pytest

from folduzz.errors import InvalidInputError
from folduzz.metrics.stats import (
    contact_density,
    distance_histogram,
    distribution_report,
    js_divergence,
    radius_of_gyration,
    relative_contact_order,
    wasserstein_1d,
)
from tests.test_metrics_validity import chain_matrices, real_matrices


def coords_to_matrix(coords: np.ndarray) -> np.ndarray:
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((deltas**2).sum(-1))[None]


class TestContactDensity:
    def test_straight_chain_has_only_local_contacts(self):
        # Residues 6 apart on a 3.8 A straight chain are 22.8 A apart: no
        # non-local contacts at all.
        assert contact_density(chain_matrices(count=2), min_separation=6)[0] == 0.0

    def test_collapsed_structure_is_all_contacts(self):
        matrices = np.zeros((1, 16, 16))
        assert contact_density(matrices, min_separation=6)[0] == pytest.approx(1.0)

    def test_returns_one_value_per_matrix(self):
        assert contact_density(real_matrices(count=5)).shape == (5,)

    def test_threshold_is_monotonic(self):
        matrices = real_matrices(count=3)
        assert contact_density(matrices, threshold=8.0).mean() <= contact_density(
            matrices, threshold=16.0
        ).mean()

    def test_rejects_bad_separation(self):
        with pytest.raises(InvalidInputError):
            contact_density(real_matrices(), min_separation=0)

    def test_rejects_separation_larger_than_matrix(self):
        with pytest.raises(InvalidInputError):
            contact_density(real_matrices(size=8), min_separation=20)


class TestRelativeContactOrder:
    def test_nan_when_there_are_no_contacts(self):
        value = relative_contact_order(chain_matrices(count=1), min_separation=6)[0]
        assert np.isnan(value)

    def test_known_value_for_a_single_contact(self):
        size = 16
        matrix = np.full((1, size, size), 100.0)
        matrix[0, 0, 10] = matrix[0, 10, 0] = 1.0
        # One contact at |i - j| = 10, normalised by chain length 16.
        assert relative_contact_order(matrix, min_separation=6)[0] == pytest.approx(10 / size)

    def test_local_contacts_give_lower_order_than_distant_ones(self):
        size = 32
        local = np.full((1, size, size), 100.0)
        distant = np.full((1, size, size), 100.0)
        local[0, 0, 7] = local[0, 7, 0] = 1.0
        distant[0, 0, 31] = distant[0, 31, 0] = 1.0
        assert (
            relative_contact_order(local, min_separation=6)[0]
            < relative_contact_order(distant, min_separation=6)[0]
        )


class TestRadiusOfGyration:
    def test_matches_the_coordinate_definition(self):
        rng = np.random.default_rng(3)
        coords = rng.normal(scale=10.0, size=(40, 3))
        centred = coords - coords.mean(axis=0)
        expected = np.sqrt((centred**2).sum(axis=1).mean())
        assert radius_of_gyration(coords_to_matrix(coords))[0] == pytest.approx(
            expected, rel=1e-6
        )

    def test_zero_for_a_collapsed_structure(self):
        assert radius_of_gyration(np.zeros((1, 8, 8)))[0] == pytest.approx(0.0)

    def test_scales_linearly(self):
        matrices = real_matrices(count=2)
        np.testing.assert_allclose(
            radius_of_gyration(matrices * 3), radius_of_gyration(matrices) * 3, rtol=1e-6
        )


class TestDistanceHistogram:
    def test_sums_to_one(self):
        histogram, edges = distance_histogram(real_matrices(), bins=20, max_distance=50.0)
        assert histogram.sum() == pytest.approx(1.0)
        assert edges.shape == (21,)

    def test_ignores_the_diagonal(self):
        """Including the zero diagonal would put a spurious spike in bin 0."""
        matrices = chain_matrices(count=1, size=8)
        histogram, _ = distance_histogram(matrices, bins=10, max_distance=50.0)
        chain_pairs = 8 * 7
        assert histogram[0] * chain_pairs == pytest.approx(
            np.count_nonzero(
                (matrices[0] > 0) & (matrices[0] < 5.0) & ~np.eye(8, dtype=bool)
            ),
            abs=1.0,
        )

    def test_rejects_bad_bins(self):
        with pytest.raises(InvalidInputError):
            distance_histogram(real_matrices(), bins=0)


class TestDivergences:
    def test_js_is_zero_for_identical_distributions(self):
        histogram, _ = distance_histogram(real_matrices(), bins=20)
        assert js_divergence(histogram, histogram) == pytest.approx(0.0, abs=1e-12)

    def test_js_is_bounded_by_one(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1.0])
        assert js_divergence(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_js_is_symmetric(self):
        a = np.array([0.5, 0.3, 0.2])
        b = np.array([0.2, 0.3, 0.5])
        assert js_divergence(a, b) == pytest.approx(js_divergence(b, a))

    def test_js_rejects_mismatched_lengths(self):
        with pytest.raises(InvalidInputError):
            js_divergence(np.ones(3), np.ones(4))

    def test_wasserstein_zero_for_identical_samples(self):
        values = np.linspace(0, 10, 50)
        assert wasserstein_1d(values, values) == pytest.approx(0.0)

    def test_wasserstein_equals_shift(self):
        values = np.linspace(0, 10, 100)
        assert wasserstein_1d(values, values + 4.0) == pytest.approx(4.0, abs=1e-6)

    def test_wasserstein_ignores_nans(self):
        a = np.array([1.0, 2.0, np.nan])
        b = np.array([1.0, 2.0])
        assert wasserstein_1d(a, b) == pytest.approx(0.0)

    def test_wasserstein_all_nan_returns_nan(self):
        assert np.isnan(wasserstein_1d(np.array([np.nan]), np.array([1.0])))


class TestDistributionReport:
    def test_identical_inputs_score_zero_distance(self):
        matrices = real_matrices(count=6, size=20)
        report = distribution_report(matrices, matrices)
        assert report["distance_histogram"]["js_divergence"] == pytest.approx(0.0, abs=1e-9)
        assert report["distance_histogram"]["wasserstein_angstrom"] == pytest.approx(0.0)
        assert report["contact_density"]["wasserstein"] == pytest.approx(0.0)

    def test_different_inputs_score_nonzero(self):
        real = real_matrices(count=6, size=20, seed=1)
        fake = real_matrices(count=6, size=20, seed=2) * 2.0
        report = distribution_report(fake, real)
        assert report["distance_histogram"]["wasserstein_angstrom"] > 1.0

    def test_report_includes_both_sides(self):
        real = real_matrices(count=4, size=16)
        report = distribution_report(real, real)
        assert report["contact_density"]["generated_mean"] == pytest.approx(
            report["contact_density"]["real_mean"]
        )
        assert set(report) == {
            "distance_histogram",
            "contact_density",
            "relative_contact_order",
            "radius_of_gyration",
        }

    def test_all_values_are_floats(self):
        report = distribution_report(real_matrices(), real_matrices())
        for section in report.values():
            assert all(isinstance(value, float) for value in section.values())

    def test_rejects_size_mismatch(self):
        with pytest.raises(InvalidInputError):
            distribution_report(real_matrices(size=16), real_matrices(size=20))
