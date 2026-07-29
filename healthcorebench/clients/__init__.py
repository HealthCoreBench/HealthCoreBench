"""Unified OpenAI-compatible client layer.

The client is the *only* place model inference happens. Benchmark adapters never call it
directly; the runtime orchestrates it. It returns a structured ``ModelResponse`` (never a
bare content string) so every downstream record can capture identity, usage and timing.
"""

from healthcorebench.clients.errors import (
    ErrorType,
    ClientError,
    classify_exception,
    redact_secrets,
    RETRYABLE_ERROR_TYPES,
)
from healthcorebench.clients.openai_client import OpenAICompatibleClient, ModelResponse

__all__ = [
    "ErrorType",
    "ClientError",
    "classify_exception",
    "redact_secrets",
    "RETRYABLE_ERROR_TYPES",
    "OpenAICompatibleClient",
    "ModelResponse",
]
