"""Tests for classical-MDS embedding of distance matrices into 3D."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from folduzz.errors import InvalidInputError
from folduzz.metrics.embedding import (
    classical_mds,
    embed_one,
    embedding_report,
)
from tests.test_metrics_validity import chain_matrices, real_matrices


def helix(n: int = 24) -> np.ndarray:
    turn = 100.0 * np.pi / 180.0
    radius, rise = 2.3, 1.5
    return np.stack(
        [radius * np.cos(np.arange(n) * turn), radius * np.sin(np.arange(n) * turn),
         np.arange(n) * rise],
        axis=1,
    )


def matrix_of(coords: np.ndarray) -> np.ndarray:
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((deltas**2).sum(-1))


class TestClassicalMds:
    def test_recovers_3d_geometry_exactly(self):
        coords = helix(30)
        embedded, eigenvalues = classical_mds(matrix_of(coords), dim=3)
        assert embedded.shape == (30, 3)
        # Distances are invariant to the rotation/reflection MDS is free to pick.
        np.testing.assert_allclose(matrix_of(embedded), matrix_of(coords), atol=1e-6)
        assert eigenvalues[:3].min() > 0

    def test_planar_points_need_only_two_dimensions(self):
        rng = np.random.default_rng(0)
        coords = np.concatenate([rng.normal(size=(20, 2)), np.zeros((20, 1))], axis=1)
        _, eigenvalues = classical_mds(matrix_of(coords), dim=3)
        assert abs(eigenvalues[2]) < 1e-8

    def test_non_euclidean_matrix_produces_negative_eigenvalues(self):
        matrix = np.array(
            [[0.0, 1.0, 50.0], [1.0, 0.0, 1.0], [50.0, 1.0, 0.0]], dtype=np.float64
        )
        _, eigenvalues = classical_mds(matrix, dim=2)
        assert eigenvalues.min() < -1e-6

    def test_asymmetric_input_is_symmetrized_not_rejected(self):
        matrix = matrix_of(helix(12))
        matrix[0, 1] += 5.0
        coords, _ = classical_mds(matrix, dim=3)
        assert np.isfinite(coords).all()

    def test_rejects_bad_dim(self):
        with pytest.raises(InvalidInputError):
            classical_mds(matrix_of(helix(8)), dim=0)

    def test_rejects_dim_above_point_count(self):
        with pytest.raises(InvalidInputError):
            classical_mds(matrix_of(helix(4)), dim=10)

    def test_rejects_non_square(self):
        with pytest.raises(InvalidInputError):
            classical_mds(np.zeros((4, 5)))


class TestEmbedOne:
    def test_perfect_embedding_scores_zero_stress(self):
        result = embed_one(matrix_of(helix(30)))
        assert result.stress1 == pytest.approx(0.0, abs=1e-6)
        assert result.negative_eigenvalue_mass == pytest.approx(0.0, abs=1e-6)
        assert result.top_dim_variance_ratio == pytest.approx(1.0, abs=1e-6)
        assert result.reconstruction_rmse_angstrom == pytest.approx(0.0, abs=1e-5)

    def test_helix_bonds_are_plausible(self):
        result = embed_one(matrix_of(helix(30)))
        assert result.bond_mean_angstrom == pytest.approx(3.8, abs=0.3)
        assert result.bond_fraction_plausible > 0.9

    def test_noise_matrix_scores_high_stress(self):
        rng = np.random.default_rng(5)
        noise = np.abs(rng.normal(20, 8, size=(24, 24)))
        noise = 0.5 * (noise + noise.T)
        np.fill_diagonal(noise, 0.0)
        result = embed_one(noise)
        assert result.stress1 > 0.1
        assert result.negative_eigenvalue_mass > 0.05

    def test_results_are_immutable(self):
        result = embed_one(matrix_of(helix(10)))
        with pytest.raises(FrozenInstanceError):
            result.stress1 = 0.0  # type: ignore[misc]


class TestEmbeddingReport:
    def test_aggregates_over_a_stack(self):
        matrices = np.stack([matrix_of(helix(24)) for _ in range(3)])
        report = embedding_report(matrices)
        assert report["matrices_embedded"] == 3.0
        assert report["stress1_mean"] == pytest.approx(0.0, abs=1e-6)
        assert report["stress1_median"] == pytest.approx(0.0, abs=1e-6)

    def test_limits_how_many_matrices_are_embedded(self):
        matrices = real_matrices(count=10, size=16)
        assert embedding_report(matrices, max_count=4)["matrices_embedded"] == 4.0

    def test_real_random_points_embed_well_chains_do_not(self):
        """Random 3D points are exactly embeddable; their *bonds* are not
        protein-like. The two must be reported separately."""
        points = embedding_report(real_matrices(count=4, size=16))
        assert points["stress1_mean"] == pytest.approx(0.0, abs=1e-6)
        assert points["bond_fraction_plausible_mean"] < 0.3

        chains = embedding_report(chain_matrices(count=4, size=16))
        assert chains["bond_fraction_plausible_mean"] == pytest.approx(1.0)

    def test_all_values_are_floats(self):
        report = embedding_report(real_matrices(count=2, size=12))
        assert all(isinstance(value, float) for value in report.values())

    def test_rejects_bad_max_count(self):
        with pytest.raises(InvalidInputError):
            embedding_report(real_matrices(), max_count=0)
