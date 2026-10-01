"""Every tunable constant in folduzz lives here.

Nothing else in the package is allowed to hard-code a magic number: the whole
point is that the preprocessing contract (matrix size, distance scale) and the
training contract (latent size, learning rate) are readable in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root unless overridden on the CLI)
# --------------------------------------------------------------------------- #
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIXTURE_DIR = DATA_DIR / "fixtures"
GENERATED_DIR = DATA_DIR / "generated"
RUNS_DIR = Path("runs")
REPORTS_DIR = Path("reports")
PDB_ID_FILE = DATA_DIR / "pdb_ids.txt"
MANIFEST_NAME = "manifest.json"

# --------------------------------------------------------------------------- #
# Structure parsing / distance-matrix contract
# --------------------------------------------------------------------------- #
#: Side length of every distance matrix the model ever sees. 64 keeps the DCGAN
#: a 4-stage stack (4 -> 8 -> 16 -> 32 -> 64) and keeps one window inside a
#: single compact domain.
MATRIX_SIZE = 64

#: Residue-window stride when cropping a chain into MATRIX_SIZE windows.
#: 32 => 50% overlap, which roughly doubles the sample count per chain.
WINDOW_STRIDE = 32

#: Chains shorter than this are skipped outright. We never zero-pad: padding a
#: distance matrix with zeros invents a block of "all residues coincide", which
#: the discriminator learns to detect instead of learning protein geometry.
MIN_CHAIN_LENGTH = MATRIX_SIZE

#: Atom used as the residue representative.
REPRESENTATIVE_ATOM = "CA"

#: Distances are clipped here before scaling to [-1, 1]. Chosen from the data:
#: over the 64-residue windows of the training set the CA-CA distance
#: distribution has p50 = 17.3 A, p99 = 47.9 A, max = 74.3 A, so clipping at
#: 50 A saturates 0.66% of pairs (40 A would have saturated 3.7%).
MAX_DISTANCE_ANGSTROM = 50.0

#: Two residues are "in contact" below this CA-CA distance. 8 A is the usual
#: CASP/contact-prediction convention for CA-CA contact maps.
CONTACT_THRESHOLD_ANGSTROM = 8.0

#: |i - j| >= this is required for a contact to count as non-local. Excludes the
#: trivial backbone band from contact-order statistics.
MIN_CONTACT_SEPARATION = 6

#: Ideal consecutive CA-CA virtual bond length in a protein backbone.
CA_CA_BOND_ANGSTROM = 3.8

#: Tolerance (A) on the triangle inequality d_ik <= d_ij + d_jk, to avoid
#: counting float noise as a geometric violation.
TRIANGLE_TOLERANCE_ANGSTROM = 0.05

# --------------------------------------------------------------------------- #
# Dataset splitting
# --------------------------------------------------------------------------- #
#: Fraction of PDB entries held out for evaluation. The split is by structure
#: (never by window), hashed from the PDB ID, so it is stable across runs and
#: no window of a held-out structure can leak into training.
VAL_FRACTION = 0.15
SPLIT_SALT = "folduzz-v1"

# --------------------------------------------------------------------------- #
# RCSB fetching
# --------------------------------------------------------------------------- #
RCSB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"
FETCH_TIMEOUT_SECONDS = 30.0
#: Politeness delay between requests. RCSB asks for considerate use of the
#: file service; this caps us at ~3 requests/second.
FETCH_DELAY_SECONDS = 0.34
FETCH_MAX_RETRIES = 3
FETCH_BACKOFF_SECONDS = 2.0
FETCH_USER_AGENT = "folduzz/0.2 (https://github.com/k8rthik/protein-gan)"
#: A PDB file smaller than this is treated as a truncated download, not data.
MIN_PDB_BYTES = 1_000

# --------------------------------------------------------------------------- #
# Model / training
# --------------------------------------------------------------------------- #
LATENT_DIM = 100
GENERATOR_BASE_CHANNELS = 256
DISCRIMINATOR_BASE_CHANNELS = 32
LEAKY_RELU_SLOPE = 0.2


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training hyperparameters. Use `replace(cfg, epochs=...)`."""

    epochs: int = 120
    batch_size: int = 64
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    latent_dim: int = LATENT_DIM
    matrix_size: int = MATRIX_SIZE
    #: One-sided label smoothing on the real label; standard GAN stabiliser.
    real_label: float = 0.9
    fake_label: float = 0.0
    checkpoint_every: int = 20
    log_every: int = 10
    seed: int = 0
    num_workers: int = 0

    def evolve(self, **changes: object) -> TrainConfig:
        return replace(self, **changes)  # type: ignore[arg-type]


DEFAULT_TRAIN_CONFIG = TrainConfig()

# --------------------------------------------------------------------------- #
# Generation / evaluation
# --------------------------------------------------------------------------- #
DEFAULT_NUM_SAMPLES = 512
#: Number of 3D coordinates per embedded structure == residues per window.
EMBED_DIMENSIONS = 3
#: Number of bins for distance-histogram comparisons.
HISTOGRAM_BINS = 40
