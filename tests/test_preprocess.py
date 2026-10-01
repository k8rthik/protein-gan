"""Tests for the raw-PDB -> normalised matrix dataset pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from folduzz import config
from folduzz.errors import InvalidInputError
from folduzz.preprocess import (
    PreprocessOptions,
    preprocess_directory,
    preprocess_structure,
)
from tests.test_pdb_parse import write_pdb


def _helix(n: int, offset: float = 0.0) -> list[tuple[float, float, float]]:
    """A crude alpha-helix-ish curve with ~3.8 A CA-CA spacing."""
    radius, rise, turn = 2.3, 1.5, 100.0 * np.pi / 180.0
    points = []
    for i in range(n):
        points.append(
            (
                offset + radius * float(np.cos(i * turn)),
                radius * float(np.sin(i * turn)),
                i * rise,
            )
        )
    return points


@pytest.fixture
def raw_dir(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    write_pdb(directory / "1abc.pdb", {"A": _helix(100)})
    write_pdb(directory / "2def.pdb", {"A": _helix(70), "B": _helix(70, offset=80.0)})
    write_pdb(directory / "3ghi.pdb", {"A": _helix(20)})  # too short, skipped
    return directory


class TestPreprocessStructure:
    def test_produces_normalised_windows(self, raw_dir):
        records = preprocess_structure(raw_dir / "1abc.pdb")
        # 100 residues, stride 32 -> window starts 0, 32 and a flush-right 36
        assert [r.start for r in records] == [0, 32, 36]
        for record in records:
            assert record.matrix.shape == (config.MATRIX_SIZE, config.MATRIX_SIZE)
            assert record.matrix.dtype == np.float32
            assert record.matrix.min() >= -1.0
            assert record.matrix.max() <= 1.0

    def test_records_provenance(self, raw_dir):
        record = preprocess_structure(raw_dir / "1abc.pdb")[0]
        assert record.pdb_id == "1ABC"
        assert record.chain_id == "A"
        assert record.start == 0

    def test_diagonal_maps_to_minus_one(self, raw_dir):
        record = preprocess_structure(raw_dir / "1abc.pdb")[0]
        np.testing.assert_allclose(np.diag(record.matrix), -1.0, atol=1e-6)

    def test_each_chain_contributes_windows(self, raw_dir):
        records = preprocess_structure(raw_dir / "2def.pdb")
        assert {r.chain_id for r in records} == {"A", "B"}

    def test_short_structure_yields_nothing(self, raw_dir):
        assert preprocess_structure(raw_dir / "3ghi.pdb") == ()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(InvalidInputError):
            preprocess_structure(tmp_path / "nope.pdb")


class TestPreprocessDirectory:
    def test_writes_arrays_and_manifest(self, raw_dir, tmp_path):
        out = tmp_path / "processed"
        report = preprocess_directory(raw_dir, out, PreprocessOptions(val_fraction=0.0))
        assert report.window_count > 0
        assert report.structures_used == 2
        assert report.structures_skipped == 1

        train = np.load(out / "train.npy")
        assert train.shape == (report.window_count, config.MATRIX_SIZE, config.MATRIX_SIZE)
        manifest = json.loads((out / config.MANIFEST_NAME).read_text())
        assert manifest["matrix_size"] == config.MATRIX_SIZE
        assert manifest["max_distance_angstrom"] == config.MAX_DISTANCE_ANGSTROM
        assert len(manifest["windows"]) == report.window_count

    def test_split_is_by_structure(self, raw_dir, tmp_path):
        out = tmp_path / "processed"
        preprocess_directory(raw_dir, out, PreprocessOptions(val_fraction=1.0))
        manifest = json.loads((out / config.MANIFEST_NAME).read_text())
        assert {w["split"] for w in manifest["windows"]} == {"val"}
        assert np.load(out / "train.npy").shape[0] == 0

    def test_reports_clipping_rate(self, raw_dir, tmp_path):
        report = preprocess_directory(raw_dir, tmp_path / "p", PreprocessOptions(val_fraction=0.0))
        assert 0.0 <= report.clipped_fraction <= 1.0

    def test_is_deterministic(self, raw_dir, tmp_path):
        a = preprocess_directory(raw_dir, tmp_path / "a", PreprocessOptions(val_fraction=0.2))
        b = preprocess_directory(raw_dir, tmp_path / "b", PreprocessOptions(val_fraction=0.2))
        assert a.window_count == b.window_count
        np.testing.assert_array_equal(
            np.load(tmp_path / "a" / "train.npy"), np.load(tmp_path / "b" / "train.npy")
        )

    def test_empty_raw_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(InvalidInputError):
            preprocess_directory(empty, tmp_path / "out", PreprocessOptions())

    def test_missing_raw_dir_raises(self, tmp_path):
        with pytest.raises(InvalidInputError):
            preprocess_directory(tmp_path / "nope", tmp_path / "out", PreprocessOptions())

    def test_unreadable_file_is_skipped_not_fatal(self, raw_dir, tmp_path):
        (raw_dir / "4jkl.pdb").write_text("this is not a pdb file\n")
        report = preprocess_directory(raw_dir, tmp_path / "out", PreprocessOptions(val_fraction=0.0))
        assert report.structures_skipped == 2
        assert report.window_count > 0


class TestPreprocessOptions:
    def test_rejects_bad_size(self):
        with pytest.raises(InvalidInputError):
            PreprocessOptions(matrix_size=0)

    def test_rejects_bad_stride(self):
        with pytest.raises(InvalidInputError):
            PreprocessOptions(stride=-1)

    def test_rejects_bad_max_distance(self):
        with pytest.raises(InvalidInputError):
            PreprocessOptions(max_distance=0.0)
