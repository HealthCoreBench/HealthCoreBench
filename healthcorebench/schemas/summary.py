"""Run summary schema (summary.json).

Every field here is recomputed from results.jsonl + judgments.jsonl by the aggregation
code — never taken from in-memory run state. Records the denominator policy and the hashes
of the source files it was computed from, so a summary can be verified against a
recomputation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from healthcorebench.version import SCHEMA_VERSION, SUMMARY_CODE_VERSION


class Counts(BaseModel):
    """Sample dispositions.

    Every successful response lands in exactly one scoring disposition, so the invariant
    ``num_successful == num_scored + num_missing_scoring + num_evaluation_errors +
    num_evaluation_skipped + num_unscorable`` always holds and no sample can disappear
    from the accounting.
    """

    model_config = ConfigDict(extra="allow")
    num_total: int = 0
    num_unique_samples: int = 0
    num_logical_responses: int = 0
    num_attempted: int = 0
    num_successful: int = 0
    num_failed: int = 0
    num_scored: int = 0
    num_parsing_errors: int = 0
    num_evaluation_errors: int = 0
    num_evaluation_skipped: int = 0
    # Responses the evaluator ran on and deliberately declined to score, because the sample
    # gives it nothing to score against (empty gold, a reference that tokenizes to nothing,
    # no extractable field pair). Kept apart from ``num_evaluation_errors``: nothing went
    # wrong at evaluation time, so folding these together made a benchmark whose references
    # are two-thirds empty look like a benchmark whose evaluator was two-thirds broken.
    num_unscorable: int = 0
    unscorable_reasons: dict[str, int] = Field(default_factory=dict)
    num_refusals: int = 0
    num_content_filtered: int = 0
    num_max_length: int = 0
    num_missing_scoring: int = 0
    # Inputs whose middles were deleted to fit the context window. A truncated input is still
    # inferred and scored, so without this count a heavily truncated task is indistinguishable
    # from a clean one.
    num_context_truncated: int = 0
    # Source records the adapter refused to normalize (malformed / out-of-scope rows). ``None``
    # means the adapter does not report a drop count, which is not the same as zero drops.
    num_source_records_dropped: int | None = None
    source_record_drop_reasons: dict[str, int] = Field(default_factory=dict)


class Metrics(BaseModel):
    model_config = ConfigDict(extra="allow")
    score: float | None = None
    macro_score: float | None = None
    micro_score: float | None = None
    confidence_interval: list[float] | None = None
    confidence_interval_method: str | None = None
    score_denominator_policy: str = "successful_and_scored_only"


class Summary(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    benchmark_name: str
    counts: Counts = Field(default_factory=Counts)
    metrics: Metrics = Field(default_factory=Metrics)
    tokens: dict = Field(default_factory=dict)
    timing: dict = Field(default_factory=dict)
    groups: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None
    summary_code_version: str = SUMMARY_CODE_VERSION
    source_files: dict = Field(default_factory=dict)
