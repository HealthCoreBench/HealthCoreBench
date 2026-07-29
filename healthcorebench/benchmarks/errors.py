"""Benchmark data / adapter errors.

These are surfaced clearly to the user rather than silently downloading substitute data or
returning an empty dataset.
"""

from __future__ import annotations


class BenchmarkError(Exception):
    """Base class for benchmark loading/validation errors."""


class BenchmarkNotRegisteredError(BenchmarkError):
    """The requested benchmark name is not in the registry."""


class BenchmarkDataNotFoundError(BenchmarkError):
    """The fixed benchmark directory or required files are missing."""


class BenchmarkFormatNotImplementedError(BenchmarkError):
    """Source files were found but this benchmark's concrete parser is not implemented yet."""


class BenchmarkSplitNotFoundError(BenchmarkError):
    """The requested split does not exist in the fixed files. No auto-fallback."""
