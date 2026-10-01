"""Turn `data/raw/*.pdb` into the arrays the DCGAN trains on.

Output layout (all regenerable, none of it committed):

    data/processed/train.npy      (N_train, 64, 64) float32 in [-1, 1]
    data/processed/val.npy        (N_val,   64, 64) float32 in [-1, 1]
    data/processed/manifest.json  provenance for every window + the contract

The manifest is the point: every row of `train.npy` can be traced back to a PDB
ID, a chain and a residue offset, and the normalisation constants used to build
it are recorded alongside, so a checkpoint is never ambiguous about what its
numbers mean.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from folduzz import config
from folduzz.distance import clipped_fraction, crop_windows, normalize
from folduzz.errors import InvalidInputError, StructureParseError
from folduzz.pdb_parse import load_ca_segments, usable_segments
from folduzz.splits import Split, assign_split


@dataclass(frozen=True)
class PreprocessOptions:
    matrix_size: int = config.MATRIX_SIZE
    stride: int = config.WINDOW_STRIDE
    max_distance: float = config.MAX_DISTANCE_ANGSTROM
    min_chain_length: int = config.MIN_CHAIN_LENGTH
    val_fraction: float = config.VAL_FRACTION
    split_salt: str = config.SPLIT_SALT

    def __post_init__(self) -> None:
        if self.matrix_size < 2:
            raise InvalidInputError(f"matrix_size must be >= 2, got {self.matrix_size}")
        if self.stride < 1:
            raise InvalidInputError(f"stride must be >= 1, got {self.stride}")
        if self.max_distance <= 0:
            raise InvalidInputError(f"max_distance must be > 0, got {self.max_distance}")
        if self.min_chain_length < self.matrix_size:
            raise InvalidInputError("min_chain_length must be >= matrix_size")
        if not 0.0 <= self.val_fraction <= 1.0:
            raise InvalidInputError(f"val_fraction must be in [0, 1], got {self.val_fraction}")


@dataclass(frozen=True)
class WindowRecord:
    """One training example plus where it came from."""

    pdb_id: str
    chain_id: str
    start: int
    matrix: np.ndarray
    raw_clipped_fraction: float

    def provenance(self, split: Split, index: int) -> dict[str, object]:
        return {
            "pdb_id": self.pdb_id,
            "chain_id": self.chain_id,
            "residue_start": self.start,
            "split": split,
            "index": index,
        }


@dataclass(frozen=True)
class PreprocessReport:
    structures_used: int
    structures_skipped: int
    window_count: int
    train_count: int
    val_count: int
    clipped_fraction: float
    skipped: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def preprocess_structure(
    pdb_path: Path | str, options: PreprocessOptions | None = None
) -> tuple[WindowRecord, ...]:
    """Normalised windows for one PDB file. Empty tuple if nothing is usable."""
    opts = options or PreprocessOptions()
    path = Path(pdb_path)
    segments = usable_segments(
        load_ca_segments(path), min_length=max(opts.min_chain_length, opts.matrix_size)
    )

    pdb_id = path.stem[:4].upper()
    records: list[WindowRecord] = []
    for segment in segments:
        for window in crop_windows(segment.coords, size=opts.matrix_size, stride=opts.stride):
            records.append(
                WindowRecord(
                    pdb_id=pdb_id,
                    chain_id=segment.chain_id,
                    start=segment.start_index + window.start,
                    matrix=normalize(window.matrix, max_distance=opts.max_distance),
                    raw_clipped_fraction=clipped_fraction(
                        window.matrix, max_distance=opts.max_distance
                    ),
                )
            )
    return tuple(records)


def _stack(matrices: Sequence[np.ndarray], size: int) -> np.ndarray:
    if not matrices:
        return np.zeros((0, size, size), dtype=np.float32)
    return np.stack(matrices).astype(np.float32, copy=False)


def preprocess_directory(
    raw_dir: Path | str = config.RAW_DIR,
    out_dir: Path | str = config.PROCESSED_DIR,
    options: PreprocessOptions | None = None,
) -> PreprocessReport:
    """Preprocess every `*.pdb` in `raw_dir`, writing arrays + manifest."""
    opts = options or PreprocessOptions()
    source = Path(raw_dir)
    destination = Path(out_dir)
    if not source.is_dir():
        raise InvalidInputError(f"raw data directory does not exist: {source}")

    pdb_files = sorted(source.glob("*.pdb"))
    if not pdb_files:
        raise InvalidInputError(f"no .pdb files in {source} (run `folduzz fetch` first)")

    buckets: dict[Split, list[np.ndarray]] = {"train": [], "val": []}
    provenance: list[dict[str, object]] = []
    skipped: list[tuple[str, str]] = []
    clipped_total = 0.0
    used = 0

    for pdb_file in pdb_files:
        try:
            records = preprocess_structure(pdb_file, opts)
        except (StructureParseError, InvalidInputError) as exc:
            skipped.append((pdb_file.name, str(exc)))
            continue
        if not records:
            skipped.append((pdb_file.name, "no chain long enough for one window"))
            continue

        used += 1
        split = assign_split(
            records[0].pdb_id, val_fraction=opts.val_fraction, salt=opts.split_salt
        )
        for record in records:
            provenance.append(record.provenance(split, len(buckets[split])))
            buckets[split].append(record.matrix)
            clipped_total += record.raw_clipped_fraction

    window_count = len(provenance)
    if window_count == 0:
        raise InvalidInputError(
            f"no usable windows produced from {source}: "
            f"{len(skipped)} structures skipped (first: {skipped[0] if skipped else 'n/a'})"
        )

    destination.mkdir(parents=True, exist_ok=True)
    for split, matrices in buckets.items():
        np.save(destination / f"{split}.npy", _stack(matrices, opts.matrix_size))

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "matrix_size": opts.matrix_size,
        "window_stride": opts.stride,
        "max_distance_angstrom": opts.max_distance,
        "min_chain_length": opts.min_chain_length,
        "representative_atom": config.REPRESENTATIVE_ATOM,
        "normalization": "value = 2 * clip(d, 0, max_distance) / max_distance - 1",
        "val_fraction": opts.val_fraction,
        "split_salt": opts.split_salt,
        "structures_used": used,
        "structures_skipped": len(skipped),
        "clipped_fraction": clipped_total / window_count,
        "windows": provenance,
    }
    (destination / config.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    return PreprocessReport(
        structures_used=used,
        structures_skipped=len(skipped),
        window_count=window_count,
        train_count=len(buckets["train"]),
        val_count=len(buckets["val"]),
        clipped_fraction=clipped_total / window_count,
        skipped=tuple(skipped),
    )
