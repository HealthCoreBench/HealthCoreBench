"""API attempt record schema (one line per real request attempt in attempts.jsonl).

Every attempt — including retries and failures — is recorded separately and never
overwrites a previous attempt. Retries for one logical response share a
``request_group_id``; each attempt has its own ``attempt_id``.

When a timeout leaves it unknown whether the server produced output, ``usage_status`` is
set to ``unknown_due_to_timeout`` rather than recording zero token usage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator

from healthcorebench.version import SCHEMA_VERSION


def _non_negative(v):
    if v is not None and v < 0:
        raise ValueError("must be non-negative")
    return v


class UsageInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    image_tokens: int | None = None
    audio_tokens: int | None = None

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens",
                     "cached_input_tokens", "reasoning_tokens", "image_tokens", "audio_tokens")
    @classmethod
    def _nn(cls, v):
        return _non_negative(v)


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    attempt_id: str
    request_group_id: str
    parent_request_id: str | None = None
    run_id: str
    sample_id: str
    sample_repeat_index: int = 0
    attempt_number: int = 1
    request_purpose: Literal["model_inference", "evaluation_judge"] = "model_inference"

    request_start_time: str | None = None
    request_end_time: str | None = None
    latency_seconds: float | None = None

    provider: str | None = None
    requested_model_name: str | None = None
    returned_model_name: str | None = None
    system_fingerprint: str | None = None
    provider_request_id: str | None = None

    status: Literal["success", "error"] = "success"
    http_status: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    exception_class: str | None = None
    retryable: bool | None = None
    content_filter_status: str | None = None
    finish_reason: str | None = None
    requested_max_tokens: int | None = None
    effective_max_tokens: int | None = None
    max_tokens_fallback_index: int = 0
    max_tokens_adjustment_reason: str | None = None
    adaptive_retry_count: int = 0

    usage: UsageInfo = Field(default_factory=UsageInfo)
    usage_status: Literal["known", "unknown_due_to_timeout"] = "known"

    raw_response: dict | None = None
    provider_metadata: dict = Field(default_factory=dict)
    timestamp: str | None = None

    @field_validator("latency_seconds")
    @classmethod
    def _nn_latency(cls, v):
        return _non_negative(v)
