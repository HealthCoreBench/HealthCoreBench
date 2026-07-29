"""Retry policy: exponential backoff with jitter, honouring Retry-After.

Only retryable error types (see ``healthcorebench.clients.errors.RETRYABLE_ERROR_TYPES``) are
retried. Backoff is ``min(max_delay, initial * 2^(n-1)) + jitter``; a server-provided
``Retry-After`` takes precedence when present. On exhaustion the caller records a failed
result — the sample is never scored as a wrong answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from healthcorebench.clients.errors import ClientError


@dataclass
class RetryPolicy:
    max_retries: int = 5
    initial_seconds: float = 2.0
    max_seconds: float = 60.0
    jitter_fraction: float = 0.25

    def should_retry(self, error: ClientError, attempt_number: int) -> bool:
        """Retry only retryable errors and only while attempts remain."""
        if attempt_number > self.max_retries:
            return False
        return bool(error.retryable)

    def backoff_seconds(self, attempt_number: int, error: ClientError | None = None) -> float:
        """Compute the delay before the next attempt."""
        if error is not None and error.retry_after_seconds is not None:
            # Respect server guidance, but still add a little jitter to avoid thundering herd.
            base = error.retry_after_seconds
        else:
            base = min(self.max_seconds, self.initial_seconds * (2 ** (attempt_number - 1)))
        jitter = base * self.jitter_fraction * random.random()
        return min(self.max_seconds, base + jitter)


def compute_backoff(attempt_number: int, initial: float, maximum: float, jitter_fraction: float = 0.25) -> float:
    """Standalone backoff helper (used in tests and by RetryPolicy)."""
    base = min(maximum, initial * (2 ** (attempt_number - 1)))
    return min(maximum, base + base * jitter_fraction * random.random())
