"""Render distance matrices as a grayscale PNG, with no plotting dependency.

A table of numbers does not show what a generated matrix looks like; the
characteristic cross-diagonal bands of a real protein window, or their absence,
are obvious at a glance. This writes a bare 8-bit grayscale PNG by hand (zlib +
four chunks) so the report can include a picture without pulling in matplotlib.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from folduzz.errors import InvalidInputError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: Grayscale, 8 bits per sample.
_COLOR_TYPE_GRAY = 0
_BIT_DEPTH = 8
DEFAULT_PAD = 2
DEFAULT_SCALE = 2
#: Value used for the padding between tiles (mid grey).
PAD_VALUE = 128


def to_grayscale(matrices: np.ndarray) -> np.ndarray:
    """Map matrices in [-1, 1] to uint8, with 0 A (= -1) white and far = black.

    Near pairs are bright, so contacts read as light bands, which is the
    convention used for contact maps.
    """
    array = np.asarray(matrices, dtype=np.float64)
    if array.size == 0:
        raise InvalidInputError("no matrices to render")
    clipped = np.clip(array, -1.0, 1.0)
    return np.round((1.0 - (clipped + 1.0) / 2.0) * 255.0).astype(np.uint8)


def tile_grid(
    images: np.ndarray,
    columns: int = 4,
    pad: int = DEFAULT_PAD,
    scale: int = DEFAULT_SCALE,
) -> np.ndarray:
    """Lay 2-D uint8 images out in a padded grid, nearest-neighbour upscaled."""
    stack = np.asarray(images)
    if stack.ndim != 3:
        raise InvalidInputError(f"expected a (count, h, w) stack, got {stack.shape}")
    if columns < 1 or pad < 0 or scale < 1:
        raise InvalidInputError(
            f"columns>=1, pad>=0, scale>=1 required; got {columns}/{pad}/{scale}"
        )

    count, height, width = stack.shape
    rows = (count + columns - 1) // columns
    tile_h, tile_w = height * scale, width * scale
    canvas = np.full(
        (rows * tile_h + (rows + 1) * pad, columns * tile_w + (columns + 1) * pad),
        PAD_VALUE,
        dtype=np.uint8,
    )
    for index in range(count):
        row, column = divmod(index, columns)
        tile = np.kron(stack[index], np.ones((scale, scale), dtype=np.uint8))
        top = pad + row * (tile_h + pad)
        left = pad + column * (tile_w + pad)
        canvas[top : top + tile_h, left : left + tile_w] = tile
    return canvas


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path | str, image: np.ndarray) -> Path:
    """Write a 2-D uint8 array as an 8-bit grayscale PNG."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise InvalidInputError(f"expected a 2-D image, got shape {array.shape}")
    if array.dtype != np.uint8:
        raise InvalidInputError(f"expected uint8 pixels, got {array.dtype}")

    height, width = array.shape
    header = struct.pack(
        ">IIBBBBB", width, height, _BIT_DEPTH, _COLOR_TYPE_GRAY, 0, 0, 0
    )
    # Each scanline is prefixed with filter type 0 (None).
    raw = b"".join(b"\x00" + row.tobytes() for row in array)
    payload = (
        PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def write_comparison(
    path: Path | str,
    real_normalized: np.ndarray,
    generated_normalized: np.ndarray,
    per_row: int = 6,
    scale: int = DEFAULT_SCALE,
) -> Path:
    """One PNG: a row of real windows above a row of generated ones."""
    real = np.asarray(real_normalized)[:per_row]
    generated = np.asarray(generated_normalized)[:per_row]
    if real.shape[0] == 0 or generated.shape[0] == 0:
        raise InvalidInputError("need at least one real and one generated matrix")
    count = min(real.shape[0], generated.shape[0])
    combined = np.concatenate([real[:count], generated[:count]], axis=0)
    grid = tile_grid(to_grayscale(combined), columns=count, scale=scale)
    return write_png(path, grid)
