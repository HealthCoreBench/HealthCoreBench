"""Benchmark adapters and the fixed-directory registry.

Adapters only load fixed local data, normalize samples, build messages, parse responses and
score. They never deploy or call models, implement concurrency, or write summaries. Data is
read only from the fixed ``benchmarks/medical_llm_benchmarks/`` tree resolved via the registry;
no network downloads.
"""

from healthcorebench.benchmarks.errors import (
    BenchmarkError,
    BenchmarkDataNotFoundError,
    BenchmarkFormatNotImplementedError,
    BenchmarkSplitNotFoundError,
    BenchmarkNotRegisteredError,
)
from healthcorebench.benchmarks.registry import (
    BenchmarkRegistryEntry,
    get_registry,
    get_entry,
    resolve_benchmark_keys,
    list_benchmarks,
    get_adapter,
)
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter

__all__ = [
    "BenchmarkError",
    "BenchmarkDataNotFoundError",
    "BenchmarkFormatNotImplementedError",
    "BenchmarkSplitNotFoundError",
    "BenchmarkNotRegisteredError",
    "BenchmarkRegistryEntry",
    "get_registry",
    "get_entry",
    "resolve_benchmark_keys",
    "list_benchmarks",
    "get_adapter",
    "BaseBenchmarkAdapter",
]
