"""Tests for the dependency-free PNG preview writer."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from folduzz.errors import InvalidInputError
from folduzz.preview import (
    DEFAULT_PAD,
    PNG_SIGNATURE,
    tile_grid,
    to_grayscale,
    write_comparison,
    write_png,
)


class TestToGrayscale:
    def test_maps_range_to_full_byte_range(self):
        values = np.array([[[-1.0, 1.0], [1.0, -1.0]]])
        grey = to_grayscale(values)
        assert grey.dtype == np.uint8
        assert grey[0, 0, 0] == 255  # 0 A is white
        assert grey[0, 0, 1] == 0  # max distance is black

    def test_clips_out_of_range_values(self):
        grey = to_grayscale(np.array([[[-5.0, 5.0], [5.0, -5.0]]]))
        assert grey.min() == 0 and grey.max() == 255

    def test_rejects_empty(self):
        with pytest.raises(InvalidInputError):
            to_grayscale(np.zeros((0, 4, 4)))


class TestTileGrid:
    def test_grid_dimensions(self):
        images = np.zeros((4, 8, 8), dtype=np.uint8)
        grid = tile_grid(images, columns=2, pad=1, scale=1)
        assert grid.shape == (2 * 8 + 3, 2 * 8 + 3)

    def test_scaling_enlarges_tiles(self):
        images = np.zeros((1, 4, 4), dtype=np.uint8)
        grid = tile_grid(images, columns=1, pad=0, scale=3)
        assert grid.shape == (12, 12)

    def test_partial_last_row_is_allowed(self):
        grid = tile_grid(np.zeros((3, 4, 4), dtype=np.uint8), columns=2, pad=0, scale=1)
        assert grid.shape == (8, 8)

    def test_tile_content_is_preserved(self):
        images = np.full((1, 4, 4), 200, dtype=np.uint8)
        grid = tile_grid(images, columns=1, pad=1, scale=1)
        assert grid[1, 1] == 200

    @pytest.mark.parametrize(
        "kwargs", [{"columns": 0}, {"pad": -1}, {"scale": 0}]
    )
    def test_rejects_bad_arguments(self, kwargs):
        with pytest.raises(InvalidInputError):
            tile_grid(np.zeros((2, 4, 4), dtype=np.uint8), **kwargs)

    def test_rejects_wrong_rank(self):
        with pytest.raises(InvalidInputError):
            tile_grid(np.zeros((4, 4), dtype=np.uint8))


class TestWritePng:
    def test_writes_a_decodable_png(self, tmp_path):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        path = write_png(tmp_path / "out.png", image)
        data = path.read_bytes()
        assert data.startswith(PNG_SIGNATURE)

        # Parse IHDR and IDAT back out, to prove the file is well formed.
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (16, 16)
        start = data.index(b"IDAT") + 4
        length = struct.unpack(">I", data[start - 8 : start - 4])[0]
        raw = zlib.decompress(data[start : start + length])
        rows = [raw[i * 17 : (i + 1) * 17] for i in range(16)]
        assert all(row[0] == 0 for row in rows)  # filter type None
        np.testing.assert_array_equal(
            np.frombuffer(b"".join(row[1:] for row in rows), dtype=np.uint8).reshape(16, 16),
            image,
        )

    def test_creates_parent_directory(self, tmp_path):
        path = write_png(tmp_path / "a" / "b" / "out.png", np.zeros((4, 4), dtype=np.uint8))
        assert path.is_file()

    def test_rejects_non_uint8(self, tmp_path):
        with pytest.raises(InvalidInputError):
            write_png(tmp_path / "out.png", np.zeros((4, 4), dtype=np.float32))

    def test_rejects_wrong_rank(self, tmp_path):
        with pytest.raises(InvalidInputError):
            write_png(tmp_path / "out.png", np.zeros((2, 4, 4), dtype=np.uint8))


class TestWriteComparison:
    def test_writes_two_rows(self, tmp_path):
        rng = np.random.default_rng(0)
        real = rng.uniform(-1, 1, size=(8, 16, 16))
        fake = rng.uniform(-1, 1, size=(8, 16, 16))
        path = write_comparison(tmp_path / "cmp.png", real, fake, per_row=4, scale=1)
        data = path.read_bytes()
        width, height = struct.unpack(">II", data[16:24])
        # 4 columns, 2 rows, DEFAULT_PAD = 2 between and around tiles.
        assert width == 4 * 16 + 5 * DEFAULT_PAD
        assert height == 2 * 16 + 3 * DEFAULT_PAD

    def test_rejects_empty_inputs(self, tmp_path):
        with pytest.raises(InvalidInputError):
            write_comparison(
                tmp_path / "cmp.png", np.zeros((0, 8, 8)), np.zeros((2, 8, 8))
            )
