"""End-to-end checks against the committed real PDB fixtures.

These are the tests that would have caught the original pipeline's bugs: they
run real crystal structures through parsing, cropping and normalisation and then
assert that the result is actually a plausible protein distance matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from folduzz import config
from folduzz.dataset import DistanceMatrixDataset, load_manifest
from folduzz.distance import denormalize
from folduzz.preprocess import PreprocessOptions, preprocess_directory

FIXTURE_RAW = config.FIXTURE_DIR / "raw"


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixture-processed")
    report = preprocess_directory(FIXTURE_RAW, out, PreprocessOptions())
    return out, report


def test_fixtures_are_present():
    assert len(list(FIXTURE_RAW.glob("*.pdb"))) == 6


def test_report_counts(processed):
    _, report = processed
    # 1aie.pdb has no contiguous 64-residue run and must be skipped.
    assert report.structures_used == 5
    assert report.structures_skipped == 1
    assert report.window_count == 20
    assert report.train_count > 0 and report.val_count > 0


def test_manifest_traces_every_window_to_a_structure(processed):
    out, report = processed
    manifest = load_manifest(out)
    assert len(manifest["windows"]) == report.window_count
    assert {w["pdb_id"] for w in manifest["windows"]} == {
        "1AE4",
        "1UHA",
        "2ACY",
        "1ACF",
        "1J3A",
    }


def test_split_does_not_mix_structures(processed):
    out, _ = processed
    manifest = load_manifest(out)
    by_id: dict[str, set[str]] = {}
    for window in manifest["windows"]:
        by_id.setdefault(window["pdb_id"], set()).add(window["split"])
    assert all(len(splits) == 1 for splits in by_id.values())


def test_real_matrices_look_like_distance_matrices(processed):
    out, _ = processed
    matrices = DistanceMatrixDataset.from_directory(out, "train").matrices
    angstroms = denormalize(matrices)
    # Symmetric, zero diagonal.
    np.testing.assert_allclose(angstroms, np.transpose(angstroms, (0, 2, 1)), atol=1e-3)
    np.testing.assert_allclose(np.diagonal(angstroms, axis1=1, axis2=2), 0.0, atol=1e-3)


def test_consecutive_residues_are_a_virtual_bond_apart(processed):
    out, _ = processed
    matrices = DistanceMatrixDataset.from_directory(out, "train").matrices
    angstroms = denormalize(matrices)
    offdiag = np.diagonal(angstroms, offset=1, axis1=1, axis2=2)
    assert offdiag.mean() == pytest.approx(config.CA_CA_BOND_ANGSTROM, abs=0.15)
    assert offdiag.std() < 0.5


def test_contact_density_is_in_the_expected_range(processed):
    out, _ = processed
    matrices = DistanceMatrixDataset.from_directory(out, "train").matrices
    angstroms = denormalize(matrices)
    upper = np.triu_indices(config.MATRIX_SIZE, k=1)
    contact_fraction = np.mean(angstroms[:, upper[0], upper[1]] < config.CONTACT_THRESHOLD_ANGSTROM)
    # Real 64-residue windows sit near 12% of pairs in contact.
    assert 0.05 < contact_fraction < 0.25


def test_clipping_rate_is_small(processed):
    _, report = processed
    assert report.clipped_fraction < 0.05
