"""Sample distance matrices from a trained generator.

Samples are written as a single `(count, size, size)` float32 `.npy` in the same
[-1, 1] convention as the training data, plus a small JSON sidecar recording the
checkpoint, seed and data contract they came from, so `folduzz evaluate` can
never silently score samples against the wrong normalisation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from folduzz import config
from folduzz.checkpoints import load_checkpoint, resolve_device
from folduzz.errors import InvalidInputError
from folduzz.models.ops import symmetrize

#: Samples are generated in chunks so that `--count 100000` cannot exhaust memory.
GENERATION_CHUNK = 256
SIDECAR_SUFFIX = ".meta.json"


@dataclass(frozen=True)
class GenerationSummary:
    path: Path
    count: int
    matrix_size: int
    checkpoint: str
    epoch: int
    seed: int
    symmetrized: bool
    max_distance_angstrom: float


def generate_matrices(
    checkpoint_path: Path | str,
    count: int = config.DEFAULT_NUM_SAMPLES,
    seed: int = 0,
    device_name: str | None = None,
    symmetrize_output: bool = False,
    chunk: int = GENERATION_CHUNK,
) -> tuple[np.ndarray, GenerationSummary]:
    """Draw `count` samples. Returns `(array, summary)` without touching disk."""
    if count < 1:
        raise InvalidInputError(f"count must be >= 1, got {count}")
    if chunk < 1:
        raise InvalidInputError(f"chunk must be >= 1, got {chunk}")

    checkpoint = load_checkpoint(checkpoint_path)
    device = resolve_device(device_name)
    generator = checkpoint.build_generator().to(device)
    latent_dim = checkpoint.config.latent_dim
    size = checkpoint.contract.matrix_size

    # All noise is drawn up front, in one call, so the output depends only on
    # (seed, count) and not on how the forward pass happens to be chunked.
    # count * latent_dim floats is kilobytes; the samples are the large part.
    generator_rng = torch.Generator(device="cpu").manual_seed(seed)
    all_noise = torch.randn(count, latent_dim, generator=generator_rng)

    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, count, chunk):
            noise = all_noise[start : start + chunk].to(device)
            samples = generator(noise)
            if symmetrize_output:
                samples = symmetrize(samples, diagonal_value=-1.0)
            pieces.append(samples.squeeze(1).cpu().numpy().astype(np.float32))

    matrices = np.concatenate(pieces, axis=0)
    summary = GenerationSummary(
        path=Path(),
        count=int(matrices.shape[0]),
        matrix_size=size,
        checkpoint=str(checkpoint_path),
        epoch=checkpoint.epoch,
        seed=seed,
        symmetrized=symmetrize_output,
        max_distance_angstrom=checkpoint.contract.max_distance_angstrom,
    )
    return matrices, summary


def generate_to_file(
    checkpoint_path: Path | str,
    out_path: Path | str = config.GENERATED_DIR / "samples.npy",
    count: int = config.DEFAULT_NUM_SAMPLES,
    seed: int = 0,
    device_name: str | None = None,
    symmetrize: bool = False,
) -> GenerationSummary:
    """Generate samples and write them (plus a provenance sidecar) to disk."""
    matrices, summary = generate_matrices(
        checkpoint_path=checkpoint_path,
        count=count,
        seed=seed,
        device_name=device_name,
        symmetrize_output=symmetrize,
    )
    target = Path(out_path)
    if target.suffix != ".npy":
        raise InvalidInputError(f"output path must end in .npy, got {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)

    # np.save() appends ".npy" unless it is already there, so write through an
    # open handle to keep the temporary name exactly as intended.
    partial = target.with_suffix(".npy.part")
    with partial.open("wb") as handle:
        np.save(handle, matrices)
    partial.replace(target)

    final = GenerationSummary(**{**asdict(summary), "path": target})
    sidecar = target.with_suffix(target.suffix + SIDECAR_SUFFIX)
    sidecar.write_text(
        json.dumps({**asdict(final), "path": str(target)}, indent=2) + "\n"
    )
    return final
