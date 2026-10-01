"""Score generated matrices against real ones and against trivial baselines.

Produces a JSON blob with every metric and a Markdown summary table. Each row is
one *source* of matrices, always including the real held-out matrices themselves
(as an upper reference) and the three baselines from `metrics/baselines.py`, so a
number for the GAN is never reported without something to compare it to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from folduzz import config
from folduzz.dataset import load_manifest, load_split, validate_matrices
from folduzz.distance import denormalize
from folduzz.errors import InvalidInputError
from folduzz.generate import SIDECAR_SUFFIX
from folduzz.metrics.baselines import build_baselines
from folduzz.metrics.embedding import embedding_report
from folduzz.metrics.stats import distribution_report
from folduzz.metrics.validity import validity_report

JSON_NAME = "evaluation.json"
MARKDOWN_NAME = "evaluation.md"
RAW_SOURCE = "dcgan_raw"
SYMMETRIZED_SOURCE = "dcgan_symmetrized"
REAL_SOURCE = "real_held_out"


@dataclass(frozen=True)
class ReportPaths:
    json_path: Path
    markdown_path: Path


def _symmetrized_stack(matrices: np.ndarray) -> np.ndarray:
    averaged = 0.5 * (matrices + np.transpose(matrices, (0, 2, 1)))
    out = averaged.copy()
    identity = np.eye(out.shape[1], dtype=bool)
    out[:, identity] = 0.0
    return out


def _resolve_max_distance(samples_path: Path, manifest: dict[str, Any]) -> float:
    """Prefer the clip recorded with the samples; fall back to the dataset's."""
    sidecar = samples_path.with_suffix(samples_path.suffix + SIDECAR_SUFFIX)
    if sidecar.is_file():
        try:
            recorded = json.loads(sidecar.read_text()).get("max_distance_angstrom")
        except json.JSONDecodeError as exc:
            raise InvalidInputError(f"{sidecar} is not valid JSON: {exc}") from exc
        if recorded is not None:
            return float(recorded)
    return float(manifest["max_distance_angstrom"])


def load_samples(path: Path | str) -> np.ndarray:
    """Load and validate a generated-sample `.npy` in [-1, 1]."""
    samples_path = Path(path)
    if not samples_path.is_file():
        raise InvalidInputError(f"samples file does not exist: {samples_path}")
    try:
        array = np.load(samples_path)
    except (ValueError, OSError) as exc:
        raise InvalidInputError(f"could not read {samples_path}: {exc}") from exc
    return validate_matrices(array, source=str(samples_path))


def evaluate_sources(
    sources: dict[str, np.ndarray],
    real: np.ndarray,
    embed_count: int = 128,
) -> dict[str, dict[str, Any]]:
    """Run every metric family over every source. All inputs in angstroms."""
    if not sources:
        raise InvalidInputError("no sources to evaluate")
    return {
        name: {
            "count": int(matrices.shape[0]),
            "validity": validity_report(matrices),
            "distribution": distribution_report(matrices, real),
            "embedding": embedding_report(matrices, max_count=embed_count),
        }
        for name, matrices in sources.items()
    }


def evaluate(
    samples_path: Path | str,
    data_dir: Path | str = config.PROCESSED_DIR,
    split: str = "val",
    seed: int = 0,
    embed_count: int = 128,
) -> dict[str, Any]:
    """Full evaluation: GAN (raw + symmetrized), real held-out set, baselines."""
    samples_file = Path(samples_path)
    manifest = load_manifest(data_dir)
    max_distance = _resolve_max_distance(samples_file, manifest)

    generated_normalized = load_samples(samples_file)
    real_normalized = load_split(data_dir, split)
    if generated_normalized.shape[1] != real_normalized.shape[1]:
        raise InvalidInputError(
            f"generated matrices are {generated_normalized.shape[1]}x"
            f"{generated_normalized.shape[1]} but the {split} set is "
            f"{real_normalized.shape[1]}x{real_normalized.shape[1]}"
        )

    generated = denormalize(generated_normalized, max_distance=max_distance).astype(np.float64)
    real = denormalize(real_normalized, max_distance=max_distance).astype(np.float64)

    sources: dict[str, np.ndarray] = {
        RAW_SOURCE: generated,
        SYMMETRIZED_SOURCE: _symmetrized_stack(generated),
        REAL_SOURCE: real,
        **build_baselines(
            real, count=generated.shape[0], seed=seed, max_distance=max_distance
        ),
    }

    return {
        "samples": str(samples_file),
        "data_dir": str(data_dir),
        "split": split,
        "max_distance_angstrom": max_distance,
        "matrix_size": int(real.shape[1]),
        "real_count": int(real.shape[0]),
        "generated_count": int(generated.shape[0]),
        "seed": seed,
        "embed_count": embed_count,
        "contact_threshold_angstrom": config.CONTACT_THRESHOLD_ANGSTROM,
        "triangle_tolerance_angstrom": config.TRIANGLE_TOLERANCE_ANGSTROM,
        "sources": evaluate_sources(sources, real, embed_count=embed_count),
    }


_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("symmetry err (A)", "validity.symmetry.mean_abs_asymmetry_angstrom", "{:.2f}"),
    ("diag err (A)", "validity.diagonal.mean_abs_diagonal_angstrom", "{:.2f}"),
    ("triangle viol %", "validity.triangle.violation_rate", "{:.2%}"),
    ("bond mean (A)", "validity.backbone_bond.mean_angstrom", "{:.2f}"),
    ("bond ok %", "validity.backbone_bond.fraction_plausible", "{:.1%}"),
    ("contact dens", "distribution.contact_density.generated_mean", "{:.3f}"),
    ("rel contact order", "distribution.relative_contact_order.generated_mean", "{:.3f}"),
    ("Rg (A)", "distribution.radius_of_gyration.generated_mean", "{:.1f}"),
    ("dist JS", "distribution.distance_histogram.js_divergence", "{:.3f}"),
    ("dist W1 (A)", "distribution.distance_histogram.wasserstein_angstrom", "{:.2f}"),
    ("MDS stress-1", "embedding.stress1_mean", "{:.3f}"),
    ("neg eig mass", "embedding.negative_eigenvalue_mass_mean", "{:.3f}"),
    ("3D bond ok %", "embedding.bond_fraction_plausible_mean", "{:.1%}"),
)


def _dig(blob: dict[str, Any], dotted: str) -> float:
    current: Any = blob
    for key in dotted.split("."):
        current = current[key]
    return float(current)


def to_markdown(results: dict[str, Any]) -> str:
    """Render the evaluation as a Markdown table plus a short header."""
    sources = results["sources"]
    header = "| source | " + " | ".join(name for name, _, _ in _COLUMNS) + " |"
    divider = "|---" * (len(_COLUMNS) + 1) + "|"
    lines = [
        f"Matrices: {results['matrix_size']}x{results['matrix_size']}, "
        f"clip {results['max_distance_angstrom']:.0f} A, split `{results['split']}` "
        f"({results['real_count']} real windows), {results['generated_count']} generated.",
        "",
        header,
        divider,
    ]
    for name, blob in sources.items():
        cells = []
        for _, dotted, fmt in _COLUMNS:
            try:
                value = _dig(blob, dotted)
            except (KeyError, TypeError):
                cells.append("n/a")
                continue
            cells.append("nan" if not np.isfinite(value) else fmt.format(value))
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def evaluate_to_files(
    samples_path: Path | str,
    data_dir: Path | str = config.PROCESSED_DIR,
    split: str = "val",
    out_dir: Path | str = config.REPORTS_DIR,
    seed: int = 0,
    embed_count: int = 128,
) -> ReportPaths:
    """Run the evaluation and write `evaluation.json` + `evaluation.md`."""
    results = evaluate(
        samples_path=samples_path,
        data_dir=data_dir,
        split=split,
        seed=seed,
        embed_count=embed_count,
    )
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / JSON_NAME
    markdown_path = directory / MARKDOWN_NAME
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    markdown_path.write_text(to_markdown(results))
    return ReportPaths(json_path=json_path, markdown_path=markdown_path)
