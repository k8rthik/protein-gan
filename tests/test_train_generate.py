"""Smoke tests for the training loop and the generate path.

The loop itself is excluded from the coverage target (it is slow and stochastic),
but it must still be proven to run, checkpoint, resume and produce samples --
which the original `train_gan.py` could not do at all.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from folduzz import config
from folduzz.checkpoints import load_checkpoint
from folduzz.errors import CheckpointError, InvalidInputError
from folduzz.generate import generate_matrices, generate_to_file
from folduzz.train import LAST_CHECKPOINT_NAME, LOG_NAME, train

TINY = config.DEFAULT_TRAIN_CONFIG.evolve(
    epochs=2, batch_size=4, latent_dim=8, matrix_size=32, checkpoint_every=1, log_every=1
)


@pytest.fixture
def tiny_data(tmp_path):
    rng = np.random.default_rng(0)
    directory = tmp_path / "processed"
    directory.mkdir()
    np.save(directory / "train.npy", rng.uniform(-1, 1, (12, 32, 32)).astype(np.float32))
    np.save(directory / "val.npy", rng.uniform(-1, 1, (4, 32, 32)).astype(np.float32))
    (directory / config.MANIFEST_NAME).write_text(
        json.dumps(
            {"matrix_size": 32, "max_distance_angstrom": 50.0, "window_stride": 32, "windows": []}
        )
    )
    return directory


class TestTrain:
    def test_runs_and_writes_checkpoint_and_log(self, tiny_data, tmp_path):
        run_dir = tmp_path / "run"
        result = train(tiny_data, run_dir, TINY, device_name="cpu")
        assert result.epochs_completed == 2
        assert result.checkpoint_path == run_dir / LAST_CHECKPOINT_NAME
        assert result.checkpoint_path.is_file()

        lines = (run_dir / LOG_NAME).read_text().strip().splitlines()
        assert len(lines) == 2
        assert set(json.loads(lines[0])) >= {"epoch", "d_loss", "g_loss", "seconds"}

    def test_records_config_and_contract(self, tiny_data, tmp_path):
        run_dir = tmp_path / "run"
        train(tiny_data, run_dir, TINY, device_name="cpu")
        saved = json.loads((run_dir / "train_config.json").read_text())
        assert saved["contract"]["matrix_size"] == 32
        assert saved["dataset_size"] == 12

    def test_checkpoint_carries_the_data_contract(self, tiny_data, tmp_path):
        run_dir = tmp_path / "run"
        result = train(tiny_data, run_dir, TINY, device_name="cpu")
        checkpoint = load_checkpoint(result.checkpoint_path)
        assert checkpoint.contract.max_distance_angstrom == 50.0
        assert checkpoint.epoch == 2

    def test_resume_continues_epoch_numbering(self, tiny_data, tmp_path):
        run_dir = tmp_path / "run"
        train(tiny_data, run_dir, TINY, device_name="cpu")
        train(tiny_data, run_dir, TINY, device_name="cpu", resume=True)
        assert load_checkpoint(run_dir / LAST_CHECKPOINT_NAME).epoch == 4
        assert len((run_dir / LOG_NAME).read_text().strip().splitlines()) == 4

    def test_resume_without_checkpoint_raises(self, tiny_data, tmp_path):
        with pytest.raises(CheckpointError):
            train(tiny_data, tmp_path / "fresh", TINY, device_name="cpu", resume=True)

    def test_resume_rejects_mismatched_contract(self, tiny_data, tmp_path):
        run_dir = tmp_path / "run"
        train(tiny_data, run_dir, TINY, device_name="cpu")
        manifest_path = tiny_data / config.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["max_distance_angstrom"] = 40.0
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(CheckpointError):
            train(tiny_data, run_dir, TINY, device_name="cpu", resume=True)

    def test_losses_are_finite(self, tiny_data, tmp_path):
        result = train(tiny_data, tmp_path / "run", TINY, device_name="cpu")
        assert np.isfinite(result.final_d_loss) and np.isfinite(result.final_g_loss)

    def test_generator_actually_learns_something(self, tiny_data, tmp_path):
        """A loop that cannot change its weights is not a training loop."""
        result = train(tiny_data, tmp_path / "run", TINY, device_name="cpu")
        checkpoint = load_checkpoint(result.checkpoint_path)
        fresh = type(checkpoint.build_generator())(
            latent_dim=TINY.latent_dim, output_size=32
        ).state_dict()
        trained = checkpoint.generator_state
        assert any(
            not torch.allclose(trained[key], fresh[key])
            for key in fresh
            if trained[key].dtype.is_floating_point
        )


class TestGenerate:
    @pytest.fixture
    def checkpoint(self, tiny_data, tmp_path):
        return train(tiny_data, tmp_path / "run", TINY, device_name="cpu").checkpoint_path

    def test_shape_dtype_and_range(self, checkpoint):
        matrices, summary = generate_matrices(checkpoint, count=5, device_name="cpu")
        assert matrices.shape == (5, 32, 32)
        assert matrices.dtype == np.float32
        assert matrices.min() >= -1.0 and matrices.max() <= 1.0
        assert summary.matrix_size == 32

    def test_seed_is_reproducible(self, checkpoint):
        a, _ = generate_matrices(checkpoint, count=4, seed=7, device_name="cpu")
        b, _ = generate_matrices(checkpoint, count=4, seed=7, device_name="cpu")
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, checkpoint):
        a, _ = generate_matrices(checkpoint, count=4, seed=1, device_name="cpu")
        b, _ = generate_matrices(checkpoint, count=4, seed=2, device_name="cpu")
        assert not np.array_equal(a, b)

    def test_chunking_does_not_change_output(self, checkpoint):
        a, _ = generate_matrices(checkpoint, count=10, seed=3, device_name="cpu", chunk=10)
        b, _ = generate_matrices(checkpoint, count=10, seed=3, device_name="cpu", chunk=3)
        np.testing.assert_allclose(a[:3], b[:3])

    def test_symmetrize_option(self, checkpoint):
        matrices, summary = generate_matrices(
            checkpoint, count=3, device_name="cpu", symmetrize_output=True
        )
        np.testing.assert_allclose(matrices, np.transpose(matrices, (0, 2, 1)), atol=1e-6)
        np.testing.assert_allclose(np.diagonal(matrices, axis1=1, axis2=2), -1.0)
        assert summary.symmetrized is True

    def test_write_to_file_with_sidecar(self, checkpoint, tmp_path):
        out = tmp_path / "gen" / "samples.npy"
        summary = generate_to_file(checkpoint, out, count=6, device_name="cpu")
        assert np.load(out).shape == (6, 32, 32)
        meta = json.loads(out.with_suffix(".npy.meta.json").read_text())
        assert meta["count"] == 6
        assert meta["max_distance_angstrom"] == 50.0
        assert summary.path == out

    def test_rejects_non_npy_output(self, checkpoint, tmp_path):
        with pytest.raises(InvalidInputError):
            generate_to_file(checkpoint, tmp_path / "samples.csv", count=2, device_name="cpu")

    def test_rejects_zero_count(self, checkpoint):
        with pytest.raises(InvalidInputError):
            generate_matrices(checkpoint, count=0, device_name="cpu")

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(CheckpointError):
            generate_matrices(tmp_path / "nope.pt", count=2, device_name="cpu")
