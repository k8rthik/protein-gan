"""Deterministic train/validation split, decided per *structure*.

Splitting per window would leak: overlapping windows of the same chain are
near-duplicates, so a window in validation would have an almost identical twin
in training. Hashing the PDB ID keeps the split stable across machines and runs
without storing a split file, and guarantees every window of a structure lands
on the same side.
"""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import blake2b
from typing import Literal

from folduzz import config
from folduzz.errors import InvalidInputError

Split = Literal["train", "val"]
_HASH_BYTES = 8
_HASH_RANGE = float(2 ** (8 * _HASH_BYTES))


def _unit_hash(pdb_id: str, salt: str) -> float:
    digest = blake2b(f"{salt}:{pdb_id.upper()}".encode(), digest_size=_HASH_BYTES).digest()
    return int.from_bytes(digest, "big") / _HASH_RANGE


def assign_split(
    pdb_id: str,
    val_fraction: float = config.VAL_FRACTION,
    salt: str = config.SPLIT_SALT,
) -> Split:
    """Return the split a PDB ID belongs to."""
    if not isinstance(pdb_id, str) or not pdb_id.strip():
        raise InvalidInputError("pdb_id must be a non-empty string")
    if not 0.0 <= val_fraction <= 1.0:
        raise InvalidInputError(f"val_fraction must be in [0, 1], got {val_fraction}")
    return "val" if _unit_hash(pdb_id.strip(), salt) < val_fraction else "train"


def split_counts(
    pdb_ids: Iterable[str],
    val_fraction: float = config.VAL_FRACTION,
    salt: str = config.SPLIT_SALT,
) -> dict[str, int]:
    counts = {"train": 0, "val": 0}
    for pdb_id in pdb_ids:
        counts[assign_split(pdb_id, val_fraction=val_fraction, salt=salt)] += 1
    return counts
