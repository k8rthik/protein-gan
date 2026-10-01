"""Exception types. Every failure the user can cause gets a specific class."""

from __future__ import annotations


class FolduzzError(Exception):
    """Base class for all errors raised deliberately by folduzz."""


class InvalidInputError(FolduzzError):
    """A CLI argument or config value failed validation."""


class StructureParseError(FolduzzError):
    """A PDB file could not be parsed into usable CA coordinates."""


class InsufficientResiduesError(StructureParseError):
    """A structure has no chain long enough to crop a single window."""


class FetchError(FolduzzError):
    """A structure could not be downloaded from RCSB."""


class DatasetError(FolduzzError):
    """The processed dataset is missing, empty or malformed."""


class CheckpointError(FolduzzError):
    """A checkpoint could not be read, or does not match the current config."""
