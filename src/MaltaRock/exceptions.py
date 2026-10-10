"""Exception hierarchy used by the public MaltaRock API."""

from __future__ import annotations


class MaltaRockError(Exception):
    """Base class for every error raised intentionally by MaltaRock."""


class InvalidPathError(MaltaRockError, ValueError):
    """A user supplied path cannot be used (e.g. it points to a regular file)."""


class DatasetNotFoundError(MaltaRockError, FileNotFoundError):
    """The local dataset is absent or incomplete and downloading was disabled."""


class DatasetDownloadError(MaltaRockError, ConnectionError):
    """The dataset could not be resolved or downloaded from the Hugging Face Hub."""


class DatasetIntegrityError(MaltaRockError):
    """A downloaded or cached file does not match the metadata published on the Hub."""


class SchemaError(MaltaRockError, ValueError):
    """A table does not contain the expected columns or column types."""


class SplitError(MaltaRockError, ValueError):
    """A train/test well assignment is invalid (overlapping, unknown or empty)."""
