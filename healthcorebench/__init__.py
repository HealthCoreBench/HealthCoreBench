"""HealthCoreBench: a unified, resumable, auditable OpenAI-compatible medical model evaluation framework.

This package replaces the previous model-specific evaluation framework. All model
inference now goes through a single OpenAI-compatible client (see ``healthcorebench.clients``);
benchmark adapters (see ``healthcorebench.benchmarks``) only handle data loading, sample
normalization, prompt construction, answer parsing and scoring.
"""

from healthcorebench.version import (
    FRAMEWORK_VERSION,
    SCHEMA_VERSION,
    SUMMARY_CODE_VERSION,
)

__all__ = [
    "FRAMEWORK_VERSION",
    "SCHEMA_VERSION",
    "SUMMARY_CODE_VERSION",
]
