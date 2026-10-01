"""Tests for checkpoint writing/reading and device resolution."""

from __future__ import annotations

import pytest
import torch

from folduzz import config
from folduzz.checkpoints import (
    CHECKPOINT_VERSION,
    DataContract,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)
from folduzz.errors import CheckpointError, InvalidInputError
from folduzz.models import Discriminator, Generator


@pytest.fixture
def pieces():
    generator = Generator(latent_dim=8, output_size=32)
    discriminator = Discriminator(input_size=32)
    cfg = config.DEFAULT_TRAIN_CONFIG.evolve(latent_dim=8, matrix_size=32, epochs=2)
    contract = DataContract(matrix_size=32, max_distance_angstrom=50.0, window_stride=32)
    return generator, discriminator, cfg, contract


class TestSaveLoadRoundTrip:
    def test_round_trip_restores_weights(self, tmp_path, pieces):
        generator, discriminator, cfg, contract = pieces
        path = tmp_path / "last.pt"
        save_checkpoint(path, generator, discriminator, cfg, contract, epoch=3)

        checkpoint = load_checkpoint(path)
        assert checkpoint.epoch == 3
        assert checkpoint.version == CHECKPOINT_VERSION
        assert checkpoint.config.latent_dim == 8
        assert checkpoint.contract.matrix_size == 32

        restored = Generator(latent_dim=8, output_size=32)
        restored.load_state_dict(checkpoint.generator_state)
        for a, b in zip(
            generator.state_dict().values(), restored.state_dict().values(), strict=True
        ):
            torch.testing.assert_close(a, b)

    def test_build_generator_helper_returns_usable_model(self, tmp_path, pieces):
        generator, discriminator, cfg, contract = pieces
        path = tmp_path / "last.pt"
        save_checkpoint(path, generator, discriminator, cfg, contract, epoch=1)
        rebuilt = load_checkpoint(path).build_generator()
        with torch.no_grad():
            assert rebuilt(torch.randn(2, 8)).shape == (2, 1, 32, 32)

    def test_optimizer_states_are_optional(self, tmp_path, pieces):
        generator, discriminator, cfg, contract = pieces
        path = tmp_path / "last.pt"
        save_checkpoint(path, generator, discriminator, cfg, contract, epoch=1)
        assert load_checkpoint(path).optimizer_g_state is None

    def test_optimizer_states_round_trip(self, tmp_path, pieces):
        generator, discriminator, cfg, contract = pieces
        opt_g = torch.optim.Adam(generator.parameters())
        opt_d = torch.optim.Adam(discriminator.parameters())
        path = tmp_path / "last.pt"
        save_checkpoint(
            path,
            generator,
            discriminator,
            cfg,
            contract,
            epoch=1,
            optimizer_g=opt_g,
            optimizer_d=opt_d,
        )
        assert load_checkpoint(path).optimizer_g_state is not None

    def test_parent_directory_is_created(self, tmp_path, pieces):
        generator, discriminator, cfg, contract = pieces
        path = tmp_path / "nested" / "deep" / "last.pt"
        save_checkpoint(path, generator, discriminator, cfg, contract, epoch=0)
        assert path.is_file()

    def test_write_is_atomic(self, tmp_path, pieces):
        """A `.part` file must not survive a successful save."""
        generator, discriminator, cfg, contract = pieces
        path = tmp_path / "last.pt"
        save_checkpoint(path, generator, discriminator, cfg, contract, epoch=0)
        assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]


class TestLoadErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(CheckpointError):
            load_checkpoint(tmp_path / "nope.pt")

    def test_garbage_file_raises(self, tmp_path):
        path = tmp_path / "bad.pt"
        path.write_bytes(b"not a torch file")
        with pytest.raises(CheckpointError):
            load_checkpoint(path)

    def test_missing_keys_raise(self, tmp_path):
        path = tmp_path / "partial.pt"
        torch.save({"epoch": 1}, path)
        with pytest.raises(CheckpointError):
            load_checkpoint(path)

    def test_future_version_raises(self, tmp_path, pieces):
        generator, discriminator, cfg, contract = pieces
        path = tmp_path / "last.pt"
        save_checkpoint(path, generator, discriminator, cfg, contract, epoch=0)
        payload = torch.load(path, weights_only=False)
        payload["version"] = CHECKPOINT_VERSION + 99
        torch.save(payload, path)
        with pytest.raises(CheckpointError):
            load_checkpoint(path)

    def test_negative_epoch_rejected(self, tmp_path, pieces):
        generator, discriminator, cfg, contract = pieces
        with pytest.raises(InvalidInputError):
            save_checkpoint(tmp_path / "x.pt", generator, discriminator, cfg, contract, epoch=-1)


class TestDataContract:
    def test_from_manifest(self):
        contract = DataContract.from_manifest(
            {"matrix_size": 64, "max_distance_angstrom": 50.0, "window_stride": 32}
        )
        assert contract.matrix_size == 64

    def test_missing_field_raises(self):
        with pytest.raises(CheckpointError):
            DataContract.from_manifest({"matrix_size": 64})

    def test_mismatch_detection(self):
        a = DataContract(matrix_size=64, max_distance_angstrom=50.0, window_stride=32)
        b = DataContract(matrix_size=64, max_distance_angstrom=40.0, window_stride=32)
        assert a.describe_mismatch(b) is not None
        assert a.describe_mismatch(a) is None


class TestResolveDevice:
    def test_cpu_is_always_available(self):
        assert resolve_device("cpu").type == "cpu"

    def test_auto_returns_something_usable(self):
        assert resolve_device(None).type in {"cpu", "mps", "cuda"}

    def test_unknown_device_raises(self):
        with pytest.raises(InvalidInputError):
            resolve_device("tpu")
