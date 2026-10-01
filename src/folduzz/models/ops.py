"""Small shared tensor helpers for the DCGAN."""

from __future__ import annotations

import math

import torch

from folduzz.errors import InvalidInputError

#: Resolution the latent vector is first projected to, before any upsampling.
BASE_RESOLUTION = 4


def stage_count(size: int) -> int:
    """Number of 2x stages between `BASE_RESOLUTION` and `size`.

    Raises if `size` is not `BASE_RESOLUTION * 2**k` for some k >= 1, because a
    DCGAN stack cannot hit any other size exactly, and silently resizing with
    interpolation (as the original generator did) blurs every sample.
    """
    if size < BASE_RESOLUTION * 2:
        raise InvalidInputError(
            f"matrix size must be at least {BASE_RESOLUTION * 2}, got {size}"
        )
    ratio = size / BASE_RESOLUTION
    stages = int(round(math.log2(ratio)))
    if BASE_RESOLUTION * 2**stages != size:
        raise InvalidInputError(
            f"matrix size must be {BASE_RESOLUTION} * a power of two "
            f"(8, 16, 32, 64, 128, ...), got {size}"
        )
    return stages


def symmetrize(matrix: torch.Tensor, diagonal_value: float = -1.0) -> torch.Tensor:
    """Return `(M + M^T) / 2` with the diagonal forced to `diagonal_value`.

    This is a *post-processing* option, deliberately not part of the generator:
    whether the raw model learns symmetry on its own is one of the things the
    evaluation measures. `diagonal_value` defaults to -1.0 because that is what
    0 angstroms maps to under the normalisation in `folduzz.distance`.
    """
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise InvalidInputError(
            f"expected a square trailing shape, got {tuple(matrix.shape)}"
        )
    averaged = 0.5 * (matrix + matrix.transpose(-1, -2))
    size = averaged.shape[-1]
    eye = torch.eye(size, dtype=averaged.dtype, device=averaged.device)
    return averaged * (1.0 - eye) + diagonal_value * eye
