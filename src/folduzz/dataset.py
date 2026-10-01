"""Load the preprocessed arrays as a torch `Dataset`.

The original `matrix_dataloader.py` read one CSV per sample and then re-ran
min-max normalisation inside `__getitem__`, on top of the normalisation already
done in preprocessing. That silently rescaled every sample to fill [-1, 1]
regardless of how compact or extended the fold was, which is exactly the
information a distance matrix is supposed to carry. Here the data is validated
once on load and then handed to the model untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from folduzz import config
from folduzz.errors import DatasetError, InvalidInputError

VALID_SPLITS = ("train", "val")
#: Slack on the [-1, 1] bound, to tolerate float32 round-off from preprocessing.
RANGE_TOLERANCE = 1e-4


def load_manifest(data_dir: Path | str) -> dict[str, Any]:
    """Read `manifest.json`, which records the preprocessing contract."""
    path = Path(data_dir) / config.MANIFEST_NAME
    if not path.is_file():
        raise DatasetError(
            f"no {config.MANIFEST_NAME} in {data_dir} (run `folduzz preprocess` first)"
        )
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DatasetError(f"{path} must contain a JSON object")
    return loaded


def load_split(data_dir: Path | str, split: str) -> np.ndarray:
    """Load and validate `<data_dir>/<split>.npy` as (N, size, size) float32."""
    if split not in VALID_SPLITS:
        raise InvalidInputError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    directory = Path(data_dir)
    if not directory.is_dir():
        raise DatasetError(
            f"processed data directory does not exist: {directory} "
            "(run `folduzz preprocess` first)"
        )
    path = directory / f"{split}.npy"
    if not path.is_file():
        raise DatasetError(f"missing {path} (run `folduzz preprocess` first)")

    try:
        array = np.load(path)
    except (ValueError, OSError) as exc:
        raise DatasetError(f"could not read {path}: {exc}") from exc
    return validate_matrices(array, source=str(path))


def validate_matrices(array: np.ndarray, source: str = "<array>") -> np.ndarray:
    """Check the shape, dtype and value range of a matrix stack."""
    if array.ndim != 3:
        raise DatasetError(f"{source}: expected a 3-D (N, size, size) array, got {array.shape}")
    if array.shape[0] == 0:
        raise DatasetError(f"{source}: contains no matrices")
    if array.shape[1] != array.shape[2]:
        raise DatasetError(f"{source}: matrices must be square, got {array.shape[1:]}")
    finite = np.isfinite(array)
    if not finite.all():
        raise DatasetError(f"{source}: contains {np.count_nonzero(~finite)} non-finite values")
    low, high = float(array.min()), float(array.max())
    if low < -1.0 - RANGE_TOLERANCE or high > 1.0 + RANGE_TOLERANCE:
        raise DatasetError(
            f"{source}: values must lie in [-1, 1], got [{low:.4f}, {high:.4f}]"
        )
    return array.astype(np.float32, copy=False)


class DistanceMatrixDataset(Dataset):
    """Serves `(1, size, size)` float32 tensors in [-1, 1]."""

    def __init__(self, matrices: np.ndarray) -> None:
        self._matrices = validate_matrices(np.asarray(matrices))

    @classmethod
    def from_directory(cls, data_dir: Path | str, split: str = "train") -> DistanceMatrixDataset:
        return cls(load_split(data_dir, split))

    @property
    def matrix_size(self) -> int:
        return int(self._matrices.shape[1])

    @property
    def matrices(self) -> np.ndarray:
        """The underlying stack. Read-only view; callers must not mutate it."""
        view = self._matrices.view()
        view.flags.writeable = False
        return view

    def __len__(self) -> int:
        return int(self._matrices.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        count = len(self)
        if not -count <= index < count:
            raise IndexError(f"index {index} out of range for {count} matrices")
        return torch.from_numpy(np.array(self._matrices[index], copy=True)).unsqueeze(0)
