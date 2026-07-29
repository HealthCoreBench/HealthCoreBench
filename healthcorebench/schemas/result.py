"""Per-sample final result schema (one line per logical response in results.jsonl).

A result is one final logical model response for a sample repeat
(``sample_id`` + ``sample_repeat_index``). It always keeps the raw output alongside the
parsed and normalized answers and the reference — a parser error must never overwrite the
raw output, and re-parsing only updates the parsed/normalized fields and parser version.

A failed inference is recorded with ``status="error"`` and null answers; it is NOT scored
as a wrong answer.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator

from healthcorebench.version import SCHEMA_VERSION


class ResultRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    result_id: str
    run_id: str
    sample_id: str
    sample_repeat_index: int = 0
    request_group_id: str | None = None
    successful_attempt_id: str | None = None

    benchmark_name: str
    benchmark_version: str | None = None
    benchmark_split: str | None = None

    model_name: str | None = None
    actual_model_version: str | None = None
    model_role: str = "evaluation"

    # grouping metadata (copied from the sample for standalone aggregation)
    difficulty: str | None = None
    component: str | None = None
    capability: str | None = None
    specialty: str | None = None
    language: str | None = None
    modality: str | None = None
    task_type: str | None = None

    formatted_prompt: Any | None = None
    prompt_hash: str | None = None
    request_payload_hash: str | None = None
    media_hashes: list[str] = Field(default_factory=list)

    raw_response: str | None = None
    raw_response_object: dict | None = None
    parsed_answer: Any | None = None
    normalized_answer: Any | None = None
    reference_answer: Any | None = None

    parser_name: str | None = None
    parser_version: str | None = None
    parser_prompt_version: str | None = None
    parse_timestamp: str | None = None

    finish_reason: str | None = None
    requested_max_tokens: int | None = None
    effective_max_tokens: int | None = None
    max_tokens_fallback_index: int = 0
    max_tokens_adjustment_reason: str | None = None
    adaptive_retry_count: int = 0
    response_format: str = "text"
    logprobs: Any | None = None
    option_probabilities: dict | None = None
    model_confidence: float | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    image_tokens: int | None = None
    audio_tokens: int | None = None

    request_start_time: str | None = None
    request_end_time: str | None = None
    latency_seconds: float | None = None
    time_to_first_token: float | None = None
    generation_time_seconds: float | None = None

    status: Literal["success", "error"] = "success"
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    parsing_status: Literal["pending", "success", "error", "not_applicable"] = "pending"
    evaluation_status: Literal[
        "pending", "success", "error", "skipped", "not_applicable"
    ] = "pending"
    evaluation_skip_reason: str | None = None

    provider_metadata: dict = Field(default_factory=dict)
    timestamp: str | None = None

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens",
                     "cached_input_tokens", "reasoning_tokens", "image_tokens",
                     "audio_tokens", "latency_seconds")
    @classmethod
    def _non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError("must be non-negative")
        return v
