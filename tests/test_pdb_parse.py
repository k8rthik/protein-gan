"""Tests for CA extraction from PDB files."""

from __future__ import annotations

import numpy as np
import pytest

from folduzz.errors import InvalidInputError, StructureParseError
from folduzz.pdb_parse import (
    ChainSegment,
    load_ca_segments,
    split_contiguous,
    usable_segments,
)


def _atom_line(serial: int, resseq: int, chain: str, xyz: tuple[float, float, float]) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5} "
        f" CA  ALA {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C  "
    )


def write_pdb(path, chains: dict[str, list[tuple[float, float, float]]]) -> None:
    """Minimal but genuinely parseable PDB file with CA atoms only."""
    lines = []
    serial = 1
    for chain_id, coords in chains.items():
        for index, xyz in enumerate(coords, start=1):
            lines.append(_atom_line(serial, index, chain_id, xyz))
            serial += 1
        lines.append(f"TER   {serial:>5}      ALA {chain_id}{len(coords):>4}")
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _chain(n: int, spacing: float = 3.8, offset: float = 0.0) -> list[tuple[float, float, float]]:
    return [(offset + i * spacing, 0.0, 0.0) for i in range(n)]


class TestLoadCaSegments:
    def test_reads_single_chain(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        write_pdb(pdb, {"A": _chain(10)})
        segments = load_ca_segments(pdb)
        assert len(segments) == 1
        assert segments[0].chain_id == "A"
        assert segments[0].coords.shape == (10, 3)

    def test_reads_multiple_chains_separately(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        write_pdb(pdb, {"A": _chain(8), "B": _chain(5, offset=100.0)})
        segments = load_ca_segments(pdb)
        assert sorted(s.chain_id for s in segments) == ["A", "B"]

    def test_chains_are_not_concatenated(self, tmp_path):
        """The original preprocessing appended every chain of every model into a
        single coordinate list, so its "distance matrix" mixed inter-chain
        distances into what was supposed to be one fold."""
        pdb = tmp_path / "test.pdb"
        write_pdb(pdb, {"A": _chain(8), "B": _chain(8, offset=500.0)})
        segments = load_ca_segments(pdb)
        assert all(s.coords.shape[0] == 8 for s in segments)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(InvalidInputError):
            load_ca_segments(tmp_path / "nope.pdb")

    def test_file_without_ca_atoms_raises(self, tmp_path):
        pdb = tmp_path / "empty.pdb"
        pdb.write_text("HEADER    NOTHING\nEND\n")
        with pytest.raises(StructureParseError):
            load_ca_segments(pdb)

    def test_segments_are_immutable(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        write_pdb(pdb, {"A": _chain(5)})
        segment = load_ca_segments(pdb)[0]
        with pytest.raises(ValueError):
            segment.coords[0, 0] = 9.0


class TestSplitContiguous:
    def test_no_break_returns_one_segment(self):
        coords = np.array(_chain(6), dtype=np.float64)
        parts = split_contiguous(coords, max_gap=4.5)
        assert len(parts) == 1
        assert parts[0].shape == (6, 3)

    def test_break_splits(self):
        coords = np.array(_chain(4) + _chain(4, offset=200.0), dtype=np.float64)
        parts = split_contiguous(coords, max_gap=4.5)
        assert [p.shape[0] for p in parts] == [4, 4]

    def test_rejects_bad_gap(self):
        with pytest.raises(InvalidInputError):
            split_contiguous(np.zeros((3, 3)), max_gap=0.0)


class TestUsableSegments:
    def test_filters_short_segments(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        write_pdb(pdb, {"A": _chain(70), "B": _chain(10, offset=900.0)})
        usable = usable_segments(load_ca_segments(pdb), min_length=64)
        assert [s.chain_id for s in usable] == ["A"]

    def test_splits_chain_with_a_gap_then_filters(self, tmp_path):
        # 70 good residues, a 200 A jump, then 30 more: only the first survives.
        coords = _chain(70) + _chain(30, offset=900.0)
        pdb = tmp_path / "test.pdb"
        write_pdb(pdb, {"A": coords})
        usable = usable_segments(load_ca_segments(pdb), min_length=64)
        assert len(usable) == 1
        assert usable[0].coords.shape[0] == 70

    def test_returns_empty_when_nothing_long_enough(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        write_pdb(pdb, {"A": _chain(12)})
        assert usable_segments(load_ca_segments(pdb), min_length=64) == ()


class TestChainSegment:
    def test_length_property(self):
        segment = ChainSegment(chain_id="A", start_index=0, coords=np.zeros((7, 3)))
        assert segment.length == 7
