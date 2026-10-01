"""Tests for loading the processed arrays into a torch Dataset."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from folduzz import config
from folduzz.dataset import DistanceMatrixDataset, load_manifest, load_split
from folduzz.errors import DatasetError, InvalidInputError


@pytest.fixture
def processed(tmp_path):
    rng = np.random.default_rng(0)
    directory = tmp_path / "processed"
    directory.mkdir()
    train = rng.uniform(-1, 1, size=(7, 64, 64)).astype(np.float32)
    val = rng.uniform(-1, 1, size=(3, 64, 64)).astype(np.float32)
    np.save(directory / "train.npy", train)
    np.save(directory / "val.npy", val)
    (directory / config.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "matrix_size": 64,
                "max_distance_angstrom": 50.0,
                "window_stride": 32,
                "windows": [],
            }
        )
    )
    return directory, train, val


class TestLoadSplit:
    def test_loads_requested_split(self, processed):
        directory, train, _ = processed
        np.testing.assert_array_equal(load_split(directory, "train"), train)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(DatasetError):
            load_split(tmp_path / "nope", "train")

    def test_missing_file_raises(self, tmp_path):
        directory = tmp_path / "processed"
        directory.mkdir()
        with pytest.raises(DatasetError):
            load_split(directory, "train")

    def test_bad_split_name_raises(self, processed):
        directory, _, _ = processed
        with pytest.raises(InvalidInputError):
            load_split(directory, "test")

    def test_empty_split_raises(self, tmp_path):
        directory = tmp_path / "processed"
        directory.mkdir()
        np.save(directory / "train.npy", np.zeros((0, 64, 64), dtype=np.float32))
        with pytest.raises(DatasetError):
            load_split(directory, "train")

    def test_wrong_rank_raises(self, tmp_path):
        directory = tmp_path / "processed"
        directory.mkdir()
        np.save(directory / "train.npy", np.zeros((4, 64), dtype=np.float32))
        with pytest.raises(DatasetError):
            load_split(directory, "train")

    def test_non_square_raises(self, tmp_path):
        directory = tmp_path / "processed"
        directory.mkdir()
        np.save(directory / "train.npy", np.zeros((4, 64, 32), dtype=np.float32))
        with pytest.raises(DatasetError):
            load_split(directory, "train")

    def test_out_of_range_values_raise(self, tmp_path):
        directory = tmp_path / "processed"
        directory.mkdir()
        np.save(directory / "train.npy", np.full((2, 64, 64), 5.0, dtype=np.float32))
        with pytest.raises(DatasetError):
            load_split(directory, "train")


class TestLoadManifest:
    def test_reads_manifest(self, processed):
        directory, _, _ = processed
        assert load_manifest(directory)["matrix_size"] == 64

    def test_missing_manifest_raises(self, tmp_path):
        directory = tmp_path / "processed"
        directory.mkdir()
        with pytest.raises(DatasetError):
            load_manifest(directory)

    def test_corrupt_manifest_raises(self, tmp_path):
        directory = tmp_path / "processed"
        directory.mkdir()
        (directory / config.MANIFEST_NAME).write_text("{not json")
        with pytest.raises(DatasetError):
            load_manifest(directory)


class TestDistanceMatrixDataset:
    def test_length_and_item_shape(self, processed):
        directory, train, _ = processed
        dataset = DistanceMatrixDataset.from_directory(directory, "train")
        assert len(dataset) == len(train)
        assert dataset[0].shape == (1, 64, 64)
        assert dataset[0].dtype == torch.float32

    def test_values_are_passed_through_unchanged(self, processed):
        """The original dataloader re-ran min-max normalisation per sample inside
        __getitem__, on top of the per-matrix normalisation already applied in
        preprocessing. Values must now arrive exactly as stored."""
        directory, train, _ = processed
        dataset = DistanceMatrixDataset.from_directory(directory, "train")
        np.testing.assert_allclose(dataset[3].squeeze(0).numpy(), train[3], atol=0)

    def test_indexing_out_of_range_raises(self, processed):
        directory, _, _ = processed
        dataset = DistanceMatrixDataset.from_directory(directory, "train")
        with pytest.raises(IndexError):
            dataset[99]

    def test_negative_index_works(self, processed):
        directory, train, _ = processed
        dataset = DistanceMatrixDataset.from_directory(directory, "train")
        np.testing.assert_allclose(dataset[-1].squeeze(0).numpy(), train[-1])

    def test_works_with_a_dataloader(self, processed):
        directory, _, _ = processed
        dataset = DistanceMatrixDataset.from_directory(directory, "train")
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch.shape == (4, 1, 64, 64)

    def test_rejects_wrong_array_rank(self):
        with pytest.raises(DatasetError):
            DistanceMatrixDataset(np.zeros((4, 8), dtype=np.float32))

    def test_matrix_size_property(self, processed):
        directory, _, _ = processed
        assert DistanceMatrixDataset.from_directory(directory, "val").matrix_size == 64
