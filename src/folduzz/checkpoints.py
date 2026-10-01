"""Checkpoint format and device selection.

A checkpoint carries the *data contract* it was trained under (matrix size,
distance clip, window stride) alongside the weights. Without that, a saved
generator is ambiguous: the same tensor means different angstrom values under a
different clip, and nothing in the original code recorded which was used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from folduzz import config
from folduzz.errors import CheckpointError, InvalidInputError
from folduzz.models import Discriminator, Generator

CHECKPOINT_VERSION = 1
REQUIRED_KEYS = ("version", "epoch", "generator", "discriminator", "config", "contract")
SUPPORTED_DEVICES = ("cpu", "mps", "cuda")


@dataclass(frozen=True)
class DataContract:
    """What the numbers in a matrix mean. Recorded at train time."""

    matrix_size: int
    max_distance_angstrom: float
    window_stride: int

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> DataContract:
        try:
            return cls(
                matrix_size=int(manifest["matrix_size"]),
                max_distance_angstrom=float(manifest["max_distance_angstrom"]),
                window_stride=int(manifest["window_stride"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"manifest is missing preprocessing contract: {exc}") from exc

    def describe_mismatch(self, other: DataContract) -> str | None:
        """Human-readable difference, or None when the contracts agree."""
        differences = [
            f"{field}: checkpoint {getattr(self, field)} vs data {getattr(other, field)}"
            for field in ("matrix_size", "max_distance_angstrom", "window_stride")
            if getattr(self, field) != getattr(other, field)
        ]
        return "; ".join(differences) or None


@dataclass(frozen=True)
class Checkpoint:
    version: int
    epoch: int
    config: config.TrainConfig
    contract: DataContract
    generator_state: dict[str, Any]
    discriminator_state: dict[str, Any]
    optimizer_g_state: dict[str, Any] | None
    optimizer_d_state: dict[str, Any] | None

    def build_generator(self) -> Generator:
        """A generator with these weights, in eval mode."""
        generator = Generator(
            latent_dim=self.config.latent_dim, output_size=self.contract.matrix_size
        )
        generator.load_state_dict(self.generator_state)
        return generator.eval()

    def build_discriminator(self) -> Discriminator:
        discriminator = Discriminator(input_size=self.contract.matrix_size)
        discriminator.load_state_dict(self.discriminator_state)
        return discriminator.eval()


def save_checkpoint(
    path: Path | str,
    generator: Generator,
    discriminator: Discriminator,
    train_config: config.TrainConfig,
    contract: DataContract,
    epoch: int,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
) -> Path:
    """Write a checkpoint atomically (via `.part` + rename)."""
    if epoch < 0:
        raise InvalidInputError(f"epoch must be >= 0, got {epoch}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "config": asdict(train_config),
        "contract": asdict(contract),
        "generator": {k: v.cpu() for k, v in generator.state_dict().items()},
        "discriminator": {k: v.cpu() for k, v in discriminator.state_dict().items()},
        "optimizer_g": optimizer_g.state_dict() if optimizer_g is not None else None,
        "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
        "torch_version": torch.__version__,
    }
    partial = target.with_suffix(target.suffix + ".part")
    torch.save(payload, partial)
    partial.replace(target)
    return target


def load_checkpoint(path: Path | str) -> Checkpoint:
    """Read a checkpoint, validating version and required keys."""
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"could not read checkpoint {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"{source}: expected a dict payload")

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise CheckpointError(f"{source}: checkpoint is missing keys {missing}")
    if payload["version"] > CHECKPOINT_VERSION:
        raise CheckpointError(
            f"{source}: checkpoint version {payload['version']} is newer than this "
            f"code supports ({CHECKPOINT_VERSION})"
        )

    try:
        train_config = config.TrainConfig(**payload["config"])
        contract = DataContract(**payload["contract"])
    except TypeError as exc:
        raise CheckpointError(f"{source}: unreadable config/contract ({exc})") from exc

    return Checkpoint(
        version=int(payload["version"]),
        epoch=int(payload["epoch"]),
        config=train_config,
        contract=contract,
        generator_state=payload["generator"],
        discriminator_state=payload["discriminator"],
        optimizer_g_state=payload.get("optimizer_g"),
        optimizer_d_state=payload.get("optimizer_d"),
    )


def resolve_device(name: str | None = None) -> torch.device:
    """Pick a torch device. `None` prefers MPS on Apple silicon, then CUDA, then CPU.

    The original training script hard-coded `torch.device("mps")`, so the file
    simply crashed on any machine without Metal.
    """
    if name is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    lowered = name.strip().lower()
    if lowered not in SUPPORTED_DEVICES:
        raise InvalidInputError(f"device must be one of {SUPPORTED_DEVICES}, got {name!r}")
    if lowered == "mps" and not torch.backends.mps.is_available():
        raise InvalidInputError("MPS requested but not available on this machine")
    if lowered == "cuda" and not torch.cuda.is_available():
        raise InvalidInputError("CUDA requested but not available on this machine")
    return torch.device(lowered)
