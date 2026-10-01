"""Tests for the distance-matrix contract: construction, cropping, scaling."""

from __future__ import annotations

import numpy as np
import pytest

from folduzz import config
from folduzz.distance import (
    crop_windows,
    denormalize,
    distance_matrix,
    normalize,
    window_starts,
)
from folduzz.errors import InsufficientResiduesError, InvalidInputError


def _line_coords(n: int, spacing: float = 3.8) -> np.ndarray:
    """n points on a straight line, `spacing` apart."""
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n) * spacing
    return coords


class TestDistanceMatrix:
    def test_known_distances_on_a_line(self):
        matrix = distance_matrix(_line_coords(4, spacing=1.0))
        expected = np.array(
            [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=np.float64
        )
        np.testing.assert_allclose(matrix, expected)

    def test_is_symmetric_with_zero_diagonal(self):
        rng = np.random.default_rng(0)
        matrix = distance_matrix(rng.normal(size=(12, 3)))
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)

    def test_does_not_mutate_input(self):
        coords = _line_coords(5)
        before = coords.copy()
        distance_matrix(coords)
        np.testing.assert_array_equal(coords, before)

    def test_rejects_wrong_shape(self):
        with pytest.raises(InvalidInputError):
            distance_matrix(np.zeros((5, 2)))

    def test_rejects_non_finite(self):
        coords = _line_coords(5)
        broken = coords.copy()
        broken[2, 0] = np.nan
        with pytest.raises(InvalidInputError):
            distance_matrix(broken)


class TestWindowStarts:
    def test_exact_fit_yields_single_window(self):
        assert window_starts(64, size=64, stride=32) == (0,)

    def test_overlapping_windows(self):
        assert window_starts(128, size=64, stride=32) == (0, 32, 64)

    def test_tail_is_covered_by_a_final_flush_window(self):
        # 100 residues, stride 32: 0, 32 then the tail window must end at 100.
        assert window_starts(100, size=64, stride=32) == (0, 32, 36)

    def test_too_short_raises(self):
        with pytest.raises(InsufficientResiduesError):
            window_starts(10, size=64, stride=32)

    def test_invalid_stride_raises(self):
        with pytest.raises(InvalidInputError):
            window_starts(128, size=64, stride=0)


class TestCropWindows:
    def test_window_count_and_shape(self):
        coords = _line_coords(130)
        windows = crop_windows(coords, size=64, stride=32)
        assert [w.start for w in windows] == [0, 32, 64, 66]
        for w in windows:
            assert w.matrix.shape == (64, 64)

    def test_window_matrix_matches_direct_computation(self):
        coords = _line_coords(100)
        windows = crop_windows(coords, size=64, stride=32)
        first = windows[0]
        np.testing.assert_allclose(first.matrix, distance_matrix(coords[0:64]))

    def test_windows_are_read_only(self):
        windows = crop_windows(_line_coords(64), size=64, stride=32)
        with pytest.raises(ValueError):
            windows[0].matrix[0, 0] = 1.0


class TestNormalize:
    def test_round_trip_within_range(self):
        matrix = np.array([[0.0, 10.0], [10.0, 0.0]])
        restored = denormalize(normalize(matrix))
        np.testing.assert_allclose(restored, matrix, atol=1e-6)

    def test_maps_zero_to_minus_one_and_max_to_plus_one(self):
        matrix = np.array([[0.0, config.MAX_DISTANCE_ANGSTROM], [config.MAX_DISTANCE_ANGSTROM, 0.0]])
        scaled = normalize(matrix)
        assert scaled[0, 0] == pytest.approx(-1.0)
        assert scaled[0, 1] == pytest.approx(1.0)

    def test_clips_above_max(self):
        matrix = np.full((2, 2), config.MAX_DISTANCE_ANGSTROM * 3)
        assert normalize(matrix).max() == pytest.approx(1.0)

    def test_scaling_is_global_not_per_matrix(self):
        """Two matrices with different maxima must not both saturate at 1.

        This is the bug in the original preprocessing: per-matrix min-max
        normalisation destroyed the absolute A scale, so a compact 20 A window
        and a sprawling 60 A window produced identical value ranges.
        """
        small = normalize(np.array([[0.0, 10.0], [10.0, 0.0]]))
        large = normalize(np.array([[0.0, 30.0], [30.0, 0.0]]))
        assert small[0, 1] < large[0, 1]

    def test_denormalize_is_clipped_to_non_negative(self):
        assert denormalize(np.array([[-5.0]]))[0, 0] >= 0.0

    def test_normalize_does_not_mutate(self):
        matrix = np.array([[0.0, 5.0], [5.0, 0.0]])
        before = matrix.copy()
        normalize(matrix)
        np.testing.assert_array_equal(matrix, before)
