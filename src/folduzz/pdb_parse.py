"""Extract CA coordinate chains from PDB files.

Two decisions here differ deliberately from the original `preprocess_data.py`:

1. Chains are kept separate. The original looped over every model and every
   chain and appended all CA atoms into one list, so for a 3-chain porin it
   produced a matrix full of inter-chain distances that no single fold explains.
2. Chain breaks are honoured. A missing loop in a crystal structure leaves two
   residues adjacent in sequence but far apart in space; we split there instead
   of pretending the backbone is continuous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from folduzz import config
from folduzz.errors import InvalidInputError, StructureParseError

#: A consecutive CA-CA distance above this (A) means a chain break, not a bond.
#: Ideal is 3.8 A; 4.5 allows for strained geometry and refinement error.
MAX_CA_GAP_ANGSTROM = 4.5


@dataclass(frozen=True)
class ChainSegment:
    """A contiguous run of residues from one chain, as CA coordinates."""

    chain_id: str
    start_index: int
    coords: np.ndarray

    @property
    def length(self) -> int:
        return int(self.coords.shape[0])


def _read_only(array: np.ndarray) -> np.ndarray:
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.flags.writeable = False
    return frozen


def load_ca_segments(path: Path | str) -> tuple[ChainSegment, ...]:
    """Read every chain of the first model as one `ChainSegment` per chain.

    Only standard amino-acid residues that actually contain a CA atom count.
    Chain breaks are *not* applied here; see `usable_segments`.
    """
    pdb_path = Path(path)
    if not pdb_path.is_file():
        raise InvalidInputError(f"PDB file does not exist: {pdb_path}")

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as exc:  # Biopython raises a wide variety of errors
        raise StructureParseError(f"could not parse {pdb_path}: {exc}") from exc

    models = list(structure.get_models())
    if not models:
        raise StructureParseError(f"no models in {pdb_path}")

    segments: list[ChainSegment] = []
    for chain in models[0].get_chains():
        coords = [
            residue[config.REPRESENTATIVE_ATOM].coord
            for residue in chain.get_residues()
            if is_aa(residue, standard=True) and config.REPRESENTATIVE_ATOM in residue
        ]
        if coords:
            segments.append(
                ChainSegment(
                    chain_id=str(chain.id).strip() or "_",
                    start_index=0,
                    coords=_read_only(np.array(coords, dtype=np.float64)),
                )
            )

    if not segments:
        raise StructureParseError(
            f"no standard amino-acid CA atoms found in {pdb_path}"
        )
    return tuple(segments)


def split_contiguous(
    coords: np.ndarray, max_gap: float = MAX_CA_GAP_ANGSTROM
) -> tuple[np.ndarray, ...]:
    """Split `coords` wherever consecutive CA atoms are further than `max_gap`."""
    if max_gap <= 0:
        raise InvalidInputError(f"max_gap must be positive, got {max_gap}")
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != config.EMBED_DIMENSIONS:
        raise InvalidInputError(f"expected (n, 3) coordinates, got {array.shape}")
    if array.shape[0] < 2:
        return (array,)

    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    break_after = np.flatnonzero(steps > max_gap) + 1
    bounds = [0, *break_after.tolist(), array.shape[0]]
    pairs = zip(bounds[:-1], bounds[1:], strict=True)
    return tuple(array[lo:hi] for lo, hi in pairs if hi > lo)


def usable_segments(
    segments: tuple[ChainSegment, ...],
    min_length: int = config.MIN_CHAIN_LENGTH,
    max_gap: float = MAX_CA_GAP_ANGSTROM,
) -> tuple[ChainSegment, ...]:
    """Split chains at breaks and keep only the runs long enough to crop."""
    if min_length <= 0:
        raise InvalidInputError(f"min_length must be positive, got {min_length}")

    usable: list[ChainSegment] = []
    for segment in segments:
        offset = 0
        for part in split_contiguous(segment.coords, max_gap=max_gap):
            if part.shape[0] >= min_length:
                usable.append(
                    ChainSegment(
                        chain_id=segment.chain_id,
                        start_index=offset,
                        coords=_read_only(part),
                    )
                )
            offset += part.shape[0]
    return tuple(usable)
