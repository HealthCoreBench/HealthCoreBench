"""Scoring / judgment record schema (one line per judgment in judgments.jsonl).

Scoring is separated from results: one result may have several judgments (exact match,
normalized match, benchmark metric, LLM judge, physician review). This lets scoring be
re-run without touching the raw responses.

Distinguishes ``raw_score`` (benchmark-native metric), ``normalized_score`` (mapped to a
fixed range, typically [0,1]), and ``is_correct`` (binary, or ``null`` when a task has no
meaningful notion of correctness).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict

from healthcorebench.version import SCHEMA_VERSION


class JudgmentRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    judgment_id: str
    run_id: str
    result_id: str
    sample_id: str

    evaluator_type: Literal["rule_based", "llm_judge", "human"] = "rule_based"
    evaluator_name: str
    evaluator_version: str = "1.0"
    evaluator_prompt_version: str | None = None
    evaluator_prompt: Any | None = None
    evaluator_prompt_hash: str | None = None
    evaluator_request_hash: str | None = None
    evaluator_response_format: Any | None = None

    raw_judgment: dict = Field(default_factory=dict)
    parsed_judgment: dict = Field(default_factory=dict)
    raw_score: float | None = None
    normalized_score: float | None = None
    is_correct: bool | None = None

    evaluation_status: Literal["success", "error", "skipped"] = "success"
    evaluation_error: str | None = None

    # LLM judge fields (null for rule-based)
    judge_model: str | None = None
    judge_returned_model: str | None = None
    judge_system_fingerprint: str | None = None
    judge_raw_response: str | None = None
    judge_rationale: str | None = None
    judge_prompt_tokens: int | None = None
    judge_completion_tokens: int | None = None
    judge_reasoning_tokens: int | None = None
    judge_total_tokens: int | None = None
    judge_latency_seconds: float | None = None
    judge_attempt_id: str | None = None

    provider_metadata: dict = Field(default_factory=dict)
    timestamp: str | None = None
