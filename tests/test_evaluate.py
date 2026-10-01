"""Tests for the evaluation driver and its report rendering."""

from __future__ import annotations

import json

import numpy as np
import pytest

from folduzz import config
from folduzz.distance import normalize
from folduzz.errors import DatasetError, InvalidInputError
from folduzz.evaluate import (
    JSON_NAME,
    MARKDOWN_NAME,
    REAL_SOURCE,
    evaluate,
    evaluate_to_files,
    load_samples,
    to_markdown,
)
from tests.test_metrics_validity import real_matrices

MAX_DISTANCE = 50.0


@pytest.fixture
def processed(tmp_path):
    directory = tmp_path / "processed"
    directory.mkdir()
    # Fixed seeds: hash() is salted per process, which would make the fixture
    # (and therefore the float32 round-off in these assertions) non-reproducible.
    for split, count, seed in (("train", 8, 41), ("val", 6, 42)):
        matrices = real_matrices(count=count, size=16, seed=seed)
        np.save(directory / f"{split}.npy", normalize(matrices, max_distance=MAX_DISTANCE))
    (directory / config.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "matrix_size": 16,
                "max_distance_angstrom": MAX_DISTANCE,
                "window_stride": 8,
                "windows": [],
            }
        )
    )
    return directory


@pytest.fixture
def samples(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "samples.npy"
    np.save(path, rng.uniform(-1, 1, size=(6, 16, 16)).astype(np.float32))
    return path


class TestLoadSamples:
    def test_loads_valid_file(self, samples):
        assert load_samples(samples).shape == (6, 16, 16)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(InvalidInputError):
            load_samples(tmp_path / "nope.npy")

    def test_out_of_range_raises(self, tmp_path):
        path = tmp_path / "bad.npy"
        np.save(path, np.full((2, 8, 8), 4.0, dtype=np.float32))
        with pytest.raises(DatasetError):
            load_samples(path)


class TestEvaluate:
    def test_includes_every_source(self, processed, samples):
        results = evaluate(samples, processed, split="val", embed_count=4)
        assert set(results["sources"]) == {
            "dcgan_raw",
            "dcgan_symmetrized",
            REAL_SOURCE,
            "baseline_gaussian",
            "baseline_shuffled_distances",
            "baseline_residue_permuted",
        }

    def test_real_source_is_a_perfect_reference(self, processed, samples):
        real = evaluate(samples, processed, split="val", embed_count=6)["sources"][REAL_SOURCE]
        assert real["validity"]["symmetry"]["mean_abs_asymmetry_angstrom"] < 1e-2
        assert real["validity"]["triangle"]["violation_rate"] < 1e-6
        assert real["distribution"]["distance_histogram"]["js_divergence"] < 1e-9
        # Not exactly zero: the matrices round-trip through float32 [-1, 1]
        # normalisation, which quantises distances to ~50/2^24 A steps.
        assert real["embedding"]["stress1_mean"] < 1e-2

    def test_symmetrized_source_fixes_symmetry(self, processed, samples):
        sources = evaluate(samples, processed, split="val", embed_count=4)["sources"]
        raw = sources["dcgan_raw"]["validity"]["symmetry"]["mean_abs_asymmetry_angstrom"]
        fixed = sources["dcgan_symmetrized"]["validity"]["symmetry"][
            "mean_abs_asymmetry_angstrom"
        ]
        assert fixed < raw

    def test_nearest_neighbour_reference_is_the_training_split(self, processed, samples):
        results = evaluate(samples, processed, split="val", embed_count=2)
        assert results["nn_reference_split"] == "train"
        nearest = results["sources"][REAL_SOURCE]["nearest"]
        assert nearest["nn_rmse_mean_angstrom"] > 0.0
        assert 0.0 < nearest["coverage"] <= 1.0

    def test_records_the_contract_it_used(self, processed, samples):
        results = evaluate(samples, processed, split="val", embed_count=2)
        assert results["max_distance_angstrom"] == MAX_DISTANCE
        assert results["matrix_size"] == 16

    def test_prefers_the_sidecar_clip(self, processed, samples):
        sidecar = samples.with_suffix(".npy.meta.json")
        sidecar.write_text(json.dumps({"max_distance_angstrom": 30.0}))
        results = evaluate(samples, processed, split="val", embed_count=2)
        assert results["max_distance_angstrom"] == 30.0

    def test_corrupt_sidecar_raises(self, processed, samples):
        samples.with_suffix(".npy.meta.json").write_text("{nope")
        with pytest.raises(InvalidInputError):
            evaluate(samples, processed, split="val", embed_count=2)

    def test_size_mismatch_raises(self, processed, tmp_path):
        wrong = tmp_path / "wrong.npy"
        np.save(wrong, np.zeros((3, 32, 32), dtype=np.float32))
        with pytest.raises(InvalidInputError):
            evaluate(wrong, processed, split="val", embed_count=2)

    def test_is_reproducible(self, processed, samples):
        first = evaluate(samples, processed, split="val", seed=5, embed_count=4)
        second = evaluate(samples, processed, split="val", seed=5, embed_count=4)
        assert first["sources"] == second["sources"]

    def test_results_are_json_serialisable(self, processed, samples):
        results = evaluate(samples, processed, split="val", embed_count=2)
        assert json.loads(json.dumps(results))["split"] == "val"


class TestToMarkdown:
    def test_has_a_row_per_source(self, processed, samples):
        results = evaluate(samples, processed, split="val", embed_count=2)
        table = to_markdown(results)
        for name in results["sources"]:
            assert f"`{name}`" in table
        assert "MDS stress-1" in table

    def test_handles_nan_cells(self, processed, samples):
        results = evaluate(samples, processed, split="val", embed_count=2)
        results["sources"]["dcgan_raw"]["distribution"]["relative_contact_order"][
            "generated_mean"
        ] = float("nan")
        assert "nan" in to_markdown(results)

    def test_handles_missing_cells(self, processed, samples):
        results = evaluate(samples, processed, split="val", embed_count=2)
        del results["sources"]["dcgan_raw"]["embedding"]["stress1_mean"]
        assert "n/a" in to_markdown(results)


class TestEvaluateToFiles:
    def test_writes_both_reports(self, processed, samples, tmp_path):
        paths = evaluate_to_files(
            samples, processed, split="val", out_dir=tmp_path / "reports", embed_count=2
        )
        assert paths.json_path.name == JSON_NAME
        assert paths.markdown_path.name == MARKDOWN_NAME
        assert json.loads(paths.json_path.read_text())["split"] == "val"
        assert paths.markdown_path.read_text().startswith("Matrices: 16x16")

    def test_creates_the_output_directory(self, processed, samples, tmp_path):
        out = tmp_path / "nested" / "reports"
        evaluate_to_files(samples, processed, split="val", out_dir=out, embed_count=2)
        assert out.is_dir()
