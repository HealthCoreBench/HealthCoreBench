"""LongHealth must apply the shared context-window policy before requests are built."""

from __future__ import annotations

import pytest

from healthcorebench.benchmarks import get_adapter
from healthcorebench.benchmarks.context_window import ContextOverflowError
from healthcorebench.schemas.config import RunConfig


def _config(policy: str) -> RunConfig:
    return RunConfig.model_validate({
        "experiment": {"experiment_id": "longhealth", "run_name": "longhealth"},
        "benchmark": {"name": "LongHealth/mcqa"},
        "model": {"base_url": "http://localhost:8000/v1", "requested_model_name": "model"},
        "generation": {
            "max_tokens": 64,
            "context_token_reserve": 0,
            "context_overflow_policy": policy,
        },
        "hardware": {"max_model_len": 240},
    })


def _raw_sample() -> dict:
    choices = ["first", "second", "third", "fourth", "fifth"]
    return {
        "patient_id": "patient_01",
        "context": "x" * 4_000,
        "question": {"No": 1, "question": "Which option is correct?"},
        "choices": choices,
        "correct": choices[1],
        "source_file_rel": "22_LongHealth/data/benchmark_v5.json",
        "source_record_index": 1,
    }


def test_longhealth_head_tail_policy_is_applied_before_message_construction():
    adapter = get_adapter("LongHealth/mcqa", _config("head_tail"))
    sample = adapter.normalize_sample(_raw_sample(), 0)

    assert sample.metadata["context_truncated"] is True
    assert sample.metadata["context_truncation_strategy"] == "head_tail"
    assert "middle of source context omitted" in sample.source_content["context"]
    assert sample.input_hash


def test_longhealth_strict_policy_fails_before_provider_request():
    adapter = get_adapter("LongHealth/mcqa", _config("error"))

    with pytest.raises(ContextOverflowError):
        adapter.normalize_sample(_raw_sample(), 0)


def test_longhealth_protects_benchmark_answer_location_during_truncation():
    adapter = get_adapter("LongHealth/mcqa", _config("head_tail"))
    raw = _raw_sample()
    raw["context"] = "H" * 1_500 + "EVIDENCE" + "T" * 2_500
    raw["evidence_spans"] = [(1_500, 1_508)]

    sample = adapter.normalize_sample(raw, 0)

    assert "EVIDENCE" in sample.source_content["context"]
    assert sample.metadata["context_truncation_strategy"] == "protected_head_tail"
    assert sample.metadata["protected_span_source"] == "benchmark_answer_location"
    assert sample.metadata["protected_spans_retained"] is True
    assert sample.metadata["request_max_tokens"] == 64
