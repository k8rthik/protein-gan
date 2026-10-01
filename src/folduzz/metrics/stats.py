"""Do generated matrices have the *statistics* of real protein windows?

Validity (see `validity.py`) asks whether an array could be a distance matrix at
all. These metrics ask the harder question: given that it is one, does it look
like a protein?

* **Contact density** -- fraction of residue pairs closer than 8 A, excluding
  near-diagonal neighbours. Real compact windows sit around 7-9%.
* **Relative contact order** -- mean sequence separation of those contacts,
  divided by chain length. It separates a local helix bundle from a long-range
  beta sheet.
* **Radius of gyration** -- computable from the distance matrix alone via
  Rg^2 = (1 / 2N^2) * sum_ij d_ij^2, so it needs no 3D embedding.
* **Distance histogram** -- the whole pairwise distance distribution, compared
  with Jensen-Shannon divergence and the 1-Wasserstein distance in angstroms.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from folduzz import config
from folduzz.errors import InvalidInputError

EPSILON = 1e-12


def _validate_stack(matrices: np.ndarray) -> np.ndarray:
    array = np.asarray(matrices, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise InvalidInputError(
            f"expected a (count, n, n) stack of square matrices, got {array.shape}"
        )
    return array


def _separation_mask(size: int, min_separation: int) -> np.ndarray:
    """Upper-triangular mask of pairs at least `min_separation` apart in sequence."""
    if min_separation < 1:
        raise InvalidInputError(f"min_separation must be >= 1, got {min_separation}")
    indices = np.arange(size)
    separation = np.abs(indices[:, None] - indices[None, :])
    mask = (separation >= min_separation) & (indices[:, None] < indices[None, :])
    if not mask.any():
        raise InvalidInputError(
            f"min_separation {min_separation} leaves no pairs in a {size}x{size} matrix"
        )
    return mask


def contact_density(
    matrices: np.ndarray,
    threshold: float = config.CONTACT_THRESHOLD_ANGSTROM,
    min_separation: int = config.MIN_CONTACT_SEPARATION,
) -> np.ndarray:
    """Per-matrix fraction of non-local residue pairs in contact."""
    array = _validate_stack(matrices)
    mask = _separation_mask(array.shape[1], min_separation)
    pairs = array[:, mask]
    return np.mean(pairs < threshold, axis=1)


def relative_contact_order(
    matrices: np.ndarray,
    threshold: float = config.CONTACT_THRESHOLD_ANGSTROM,
    min_separation: int = config.MIN_CONTACT_SEPARATION,
) -> np.ndarray:
    """Per-matrix mean |i - j| over contacts, divided by chain length.

    NaN when a matrix has no qualifying contact, which is itself informative: a
    fully extended generated matrix has no contacts to order.
    """
    array = _validate_stack(matrices)
    size = array.shape[1]
    mask = _separation_mask(size, min_separation)
    indices = np.arange(size)
    separations = np.abs(indices[:, None] - indices[None, :])[mask]

    pairs = array[:, mask]
    in_contact = pairs < threshold
    counts = in_contact.sum(axis=1)
    totals = (in_contact * separations[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / np.maximum(counts, 1) / size, np.nan)


def radius_of_gyration(matrices: np.ndarray) -> np.ndarray:
    """Per-matrix radius of gyration in angstroms, from distances alone."""
    array = _validate_stack(matrices)
    size = array.shape[1]
    squared = np.square(array).sum(axis=(1, 2))
    return np.sqrt(squared / (2.0 * size * size))


def distance_histogram(
    matrices: np.ndarray,
    bins: int = config.HISTOGRAM_BINS,
    max_distance: float = config.MAX_DISTANCE_ANGSTROM,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised histogram of all off-diagonal pairwise distances."""
    if bins < 1:
        raise InvalidInputError(f"bins must be >= 1, got {bins}")
    if max_distance <= 0:
        raise InvalidInputError(f"max_distance must be > 0, got {max_distance}")
    array = _validate_stack(matrices)
    size = array.shape[1]
    offdiag = ~np.eye(size, dtype=bool)
    values = array[:, offdiag].ravel()
    counts, edges = np.histogram(values, bins=bins, range=(0.0, max_distance))
    total = counts.sum()
    density = counts / total if total else counts.astype(np.float64)
    return density, edges


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, in [0, 1]. Inputs need not be normalised."""
    first = np.asarray(p, dtype=np.float64)
    second = np.asarray(q, dtype=np.float64)
    if first.shape != second.shape:
        raise InvalidInputError(
            f"distributions must have the same shape, got {first.shape} and {second.shape}"
        )
    if (first < 0).any() or (second < 0).any():
        raise InvalidInputError("distributions must be non-negative")

    first = first / max(first.sum(), EPSILON)
    second = second / max(second.sum(), EPSILON)
    mixture = 0.5 * (first + second)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        support = a > 0
        return float(np.sum(a[support] * np.log2(a[support] / np.maximum(b[support], EPSILON))))

    return float(0.5 * kl(first, mixture) + 0.5 * kl(second, mixture))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein distance between two 1-D samples, ignoring NaNs."""
    first = np.asarray(a, dtype=np.float64).ravel()
    second = np.asarray(b, dtype=np.float64).ravel()
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if first.size == 0 or second.size == 0:
        return float("nan")
    return float(wasserstein_distance(first, second))


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    """Mean and std over the finite entries. All-NaN is a real outcome -- a
    fully extended generated matrix has no contacts to take a contact order of --
    so it is reported as NaN rather than warned about."""
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def _scalar_section(generated: np.ndarray, real: np.ndarray) -> dict[str, float]:
    generated_mean, generated_std = _mean_std(generated)
    real_mean, real_std = _mean_std(real)
    return {
        "generated_mean": generated_mean,
        "generated_std": generated_std,
        "real_mean": real_mean,
        "real_std": real_std,
        "wasserstein": wasserstein_1d(generated, real),
        "generated_undefined_fraction": float(np.mean(~np.isfinite(generated)))
        if generated.size
        else float("nan"),
    }


def distribution_report(
    generated: np.ndarray,
    real: np.ndarray,
    bins: int = config.HISTOGRAM_BINS,
    max_distance: float = config.MAX_DISTANCE_ANGSTROM,
    threshold: float = config.CONTACT_THRESHOLD_ANGSTROM,
    min_separation: int = config.MIN_CONTACT_SEPARATION,
) -> dict[str, dict[str, float]]:
    """Compare generated and real matrices on every statistic above."""
    fake = _validate_stack(generated)
    true = _validate_stack(real)
    if fake.shape[1] != true.shape[1]:
        raise InvalidInputError(
            f"generated matrices are {fake.shape[1]}x{fake.shape[1]} but real ones are "
            f"{true.shape[1]}x{true.shape[1]}"
        )

    fake_histogram, _ = distance_histogram(fake, bins=bins, max_distance=max_distance)
    true_histogram, _ = distance_histogram(true, bins=bins, max_distance=max_distance)
    offdiag = ~np.eye(fake.shape[1], dtype=bool)

    return {
        "distance_histogram": {
            "js_divergence": js_divergence(fake_histogram, true_histogram),
            "wasserstein_angstrom": wasserstein_1d(
                fake[:, offdiag].ravel(), true[:, offdiag].ravel()
            ),
            "generated_mean_angstrom": float(fake[:, offdiag].mean()),
            "real_mean_angstrom": float(true[:, offdiag].mean()),
        },
        "contact_density": _scalar_section(
            contact_density(fake, threshold=threshold, min_separation=min_separation),
            contact_density(true, threshold=threshold, min_separation=min_separation),
        ),
        "relative_contact_order": _scalar_section(
            relative_contact_order(fake, threshold=threshold, min_separation=min_separation),
            relative_contact_order(true, threshold=threshold, min_separation=min_separation),
        ),
        "radius_of_gyration": _scalar_section(
            radius_of_gyration(fake), radius_of_gyration(true)
        ),
    }
