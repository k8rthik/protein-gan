"""Tests for the trivial baselines the GAN has to beat."""

from __future__ import annotations

import numpy as np
import pytest

from folduzz.errors import InvalidInputError
from folduzz.metrics.baselines import (
    build_baselines,
    gaussian_baseline,
    residue_permuted_baseline,
    shuffled_distance_baseline,
)
from folduzz.metrics.stats import distance_histogram
from folduzz.metrics.validity import symmetry_stats, triangle_stats
from tests.test_metrics_validity import real_matrices


class TestGaussianBaseline:
    def test_shape_and_range(self):
        real = real_matrices(count=4, size=16)
        fake = gaussian_baseline(real, count=7, rng=np.random.default_rng(0), max_distance=50.0)
        assert fake.shape == (7, 16, 16)
        assert fake.min() >= 0.0 and fake.max() <= 50.0

    def test_matches_mean_roughly(self):
        real = real_matrices(count=8, size=20)
        fake = gaussian_baseline(real, count=64, rng=np.random.default_rng(1), max_distance=200.0)
        assert fake.mean() == pytest.approx(real.mean(), rel=0.25)

    def test_is_reproducible(self):
        real = real_matrices(count=2, size=12)
        a = gaussian_baseline(real, count=5, rng=np.random.default_rng(3))
        b = gaussian_baseline(real, count=5, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_is_not_symmetric(self):
        """An unconstrained i.i.d. baseline should fail symmetry, which is what
        makes it a meaningful floor for that metric."""
        real = real_matrices(count=4, size=16)
        fake = gaussian_baseline(real, count=16, rng=np.random.default_rng(0))
        assert symmetry_stats(fake).mean_abs_asymmetry > 1.0

    def test_rejects_bad_count(self):
        with pytest.raises(InvalidInputError):
            gaussian_baseline(real_matrices(), count=0, rng=np.random.default_rng(0))


class TestShuffledDistanceBaseline:
    def test_preserves_the_distance_histogram_exactly(self):
        real = real_matrices(count=5, size=16)
        shuffled = shuffled_distance_baseline(real, rng=np.random.default_rng(0))
        real_hist, _ = distance_histogram(real, bins=30, max_distance=80.0)
        fake_hist, _ = distance_histogram(shuffled, bins=30, max_distance=80.0)
        np.testing.assert_allclose(real_hist, fake_hist, atol=1e-12)

    def test_is_symmetric_with_zero_diagonal(self):
        shuffled = shuffled_distance_baseline(
            real_matrices(count=3, size=16), rng=np.random.default_rng(0)
        )
        np.testing.assert_allclose(shuffled, np.transpose(shuffled, (0, 2, 1)))
        np.testing.assert_allclose(np.diagonal(shuffled, axis1=1, axis2=2), 0.0)

    def test_destroys_the_triangle_inequality(self):
        shuffled = shuffled_distance_baseline(
            real_matrices(count=4, size=16), rng=np.random.default_rng(0)
        )
        assert triangle_stats(shuffled).violation_rate > 0.01

    def test_does_not_mutate_input(self):
        real = real_matrices(count=2, size=12)
        before = real.copy()
        shuffled_distance_baseline(real, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(real, before)


class TestResiduePermutedBaseline:
    def test_stays_a_valid_distance_matrix(self):
        """Relabelling residues keeps the geometry, so this baseline is perfect
        on every validity metric. It isolates how much of a "good" validity
        score is actually evidence of protein-like structure: none."""
        permuted = residue_permuted_baseline(
            real_matrices(count=4, size=16), rng=np.random.default_rng(0)
        )
        assert symmetry_stats(permuted).mean_abs_asymmetry == pytest.approx(0.0, abs=1e-9)
        assert triangle_stats(permuted).violation_rate == pytest.approx(0.0, abs=1e-9)

    def test_actually_permutes(self):
        real = real_matrices(count=3, size=16)
        permuted = residue_permuted_baseline(real, rng=np.random.default_rng(0))
        assert not np.allclose(permuted, real)

    def test_preserves_the_multiset_of_distances(self):
        real = real_matrices(count=2, size=12)
        permuted = residue_permuted_baseline(real, rng=np.random.default_rng(0))
        np.testing.assert_allclose(np.sort(real.ravel()), np.sort(permuted.ravel()), atol=1e-9)


class TestBuildBaselines:
    def test_returns_all_baselines_with_matching_shapes(self):
        real = real_matrices(count=6, size=16)
        baselines = build_baselines(real, count=6, seed=0, max_distance=80.0)
        assert set(baselines) == {
            "baseline_gaussian",
            "baseline_shuffled_distances",
            "baseline_residue_permuted",
        }
        for matrices in baselines.values():
            assert matrices.shape == (6, 16, 16)

    def test_is_reproducible_for_a_seed(self):
        real = real_matrices(count=4, size=16)
        first = build_baselines(real, count=4, seed=11)
        second = build_baselines(real, count=4, seed=11)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_count_larger_than_real_set_is_tiled(self):
        real = real_matrices(count=2, size=16)
        baselines = build_baselines(real, count=5, seed=0)
        assert baselines["baseline_residue_permuted"].shape[0] == 5
