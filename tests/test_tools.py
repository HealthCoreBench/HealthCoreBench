"""Unit tests for offline tools: reparse, rescore, migrate-legacy, confidence interval."""

import json
from types import SimpleNamespace

from healthcorebench.aggregation.confidence_interval import wilson_interval
from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.tools import migrate_legacy, reparse_run, rescore_run
from healthcorebench.utils.jsonl import read_jsonl, append_jsonl, atomic_write_json


def test_wilson_interval_basic():
    lo, hi = wilson_interval(7, 8)
    assert 0 <= lo <= 7 / 8 <= hi <= 1
    assert wilson_interval(0, 0) is None
    # perfect score interval stays within [0,1]
    lo2, hi2 = wilson_interval(10, 10)
    assert hi2 <= 1.0 and lo2 < 1.0


def test_migrate_legacy_list_format(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    atomic_write_json(legacy / "results.json", [
        {"id": "1", "question": "q1", "response": "A", "answer": "A", "correct": True},
        {"id": "2", "question": "q2", "response": "B", "answer": "C", "correct": False},
    ])
    out = tmp_path / "imported"
    report = migrate_legacy(legacy, out)
    assert report["results"] == 2 and report["judgments"] == 2
    results = read_jsonl(out / "results.jsonl")
    assert all(r["provenance"] == "legacy_import" and r["legacy_schema"] for r in results)
    # no fabricated tokens
    assert all(r["prompt_tokens"] is None for r in results)
    assert all(not any("cost" in key for key in r) for r in results)
    judgments = read_jsonl(out / "judgments.jsonl")
    assert {j["is_correct"] for j in judgments} == {True, False}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["provenance"] == "legacy_import"
    assert "prompt_tokens" in manifest["missing_fields"]


def test_migrate_legacy_missing_fields_recorded(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    atomic_write_json(legacy / "results.json", [{"id": "1", "response": "x"}])
    out = tmp_path / "out"
    migrate_legacy(legacy, out)
    r = read_jsonl(out / "results.jsonl")[0]
    assert "completion_tokens" in r["missing_fields"]
    assert r["status"] == "success"


def test_rescore_explicitly_replaces_or_preserves_primary(tmp_path):
    append_jsonl(tmp_path / "samples.jsonl", {
        "sample_id": "s", "benchmark_name": "B", "reference_answer": "A",
    })
    append_jsonl(tmp_path / "results.jsonl", {
        "run_id": "run", "result_id": "r", "sample_id": "s", "benchmark_name": "B",
        "status": "success", "parsed_answer": "B", "reference_answer": "A",
    })
    append_jsonl(tmp_path / "judgments.jsonl", {
        "run_id": "run", "result_id": "r", "sample_id": "s",
        "evaluator_name": "llm_judge", "evaluator_type": "llm_judge",
        "evaluation_status": "success", "normalized_score": 1.0, "is_correct": True,
        "provider_metadata": {"primary_metric": True},
    })
    atomic_write_json(tmp_path / "manifest.json", {
        "full_config": {"evaluation": {"evaluator": "multiple_choice"}},
    })

    rescore_run(tmp_path, evaluator_name="multiple_choice", replace_primary=False)
    assert summarize_run(tmp_path).metrics.score == 1.0
    rescore_run(tmp_path, evaluator_name="multiple_choice", replace_primary=True)
    assert summarize_run(tmp_path).metrics.score == 0.0


def test_secondary_rescore_with_same_evaluator_name_preserves_primary(tmp_path):
    append_jsonl(tmp_path / "samples.jsonl", {
        "sample_id": "s", "benchmark_name": "B", "reference_answer": "A",
    })
    append_jsonl(tmp_path / "results.jsonl", {
        "run_id": "run", "result_id": "r", "sample_id": "s", "benchmark_name": "B",
        "status": "success", "parsed_answer": "B", "reference_answer": "A",
    })
    append_jsonl(tmp_path / "judgments.jsonl", {
        "run_id": "run", "result_id": "r", "sample_id": "s",
        "evaluator_name": "multiple_choice_accuracy", "evaluator_type": "rule_based",
        "evaluation_status": "success", "normalized_score": 1.0, "is_correct": True,
        "provider_metadata": {"primary_metric": True},
    })
    atomic_write_json(tmp_path / "manifest.json", {
        "full_config": {"evaluation": {"evaluator": "multiple_choice"}},
    })

    rescore_run(tmp_path, evaluator_name="multiple_choice", replace_primary=False)
    summary = summarize_run(tmp_path)
    assert summary.metrics.score == 1.0
    assert "multiple_choice_accuracy__secondary" in summary.metrics_by_evaluator


def test_offline_reparse_does_not_score_marked_incomplete_results(tmp_path, monkeypatch):
    for sample_id in ("complete", "truncated"):
        append_jsonl(tmp_path / "samples.jsonl", {
            "sample_id": sample_id,
            "benchmark_name": "Fake/mcqa",
            "reference_answer": "A",
        })
        append_jsonl(tmp_path / "results.jsonl", {
            "run_id": "run",
            "result_id": f"result-{sample_id}",
            "sample_id": sample_id,
            "benchmark_name": "Fake/mcqa",
            "status": "success",
            "finish_reason": "length" if sample_id == "truncated" else "stop",
            "raw_response": "A",
        })
    atomic_write_json(tmp_path / "manifest.json", {
        "run_id": "run",
        "benchmark": {"name": "Fake/mcqa"},
        "generation": {"length_finish_policy": "mark_incomplete"},
        "full_config": {"evaluation": {"evaluator": "multiple_choice"}},
    })

    adapter = SimpleNamespace(
        adapter_version="test",
        parse_response=lambda sample, response: response,
    )
    monkeypatch.setattr("healthcorebench.tools.reparse.get_adapter", lambda *args, **kwargs: adapter)

    report = reparse_run(tmp_path, regenerate_summary=True)
    assert report["reparsed"] == 2
    assert report["skipped_incomplete"] == 1
    assert report["parser_version"] == "1.3"
    judgments = read_jsonl(tmp_path / "judgments.jsonl")
    assert {judgment["result_id"] for judgment in judgments} == {"result-complete"}
    summary = summarize_run(tmp_path)
    assert summary.counts.num_scored == 1
    assert summary.counts.num_max_length == 1
    assert summary.counts.num_missing_scoring == 1


def test_offline_rescore_does_not_score_marked_incomplete_results(tmp_path):
    append_jsonl(tmp_path / "samples.jsonl", {
        "sample_id": "truncated", "benchmark_name": "Fake/mcqa", "reference_answer": "A",
    })
    append_jsonl(tmp_path / "results.jsonl", {
        "run_id": "run", "result_id": "result-truncated", "sample_id": "truncated",
        "benchmark_name": "Fake/mcqa", "status": "success", "finish_reason": "length",
        "parsed_answer": "A",
    })
    atomic_write_json(tmp_path / "manifest.json", {
        "run_id": "run", "benchmark": {"name": "Fake/mcqa"},
        "generation": {"length_finish_policy": "mark_incomplete"},
        "full_config": {"evaluation": {"evaluator": "multiple_choice"}},
    })

    report = rescore_run(tmp_path)
    assert report["rescored"] == 0
    assert report["skipped_incomplete"] == 1
    assert read_jsonl(tmp_path / "judgments.jsonl") == []


def test_summary_excludes_stale_judgment_for_marked_incomplete_result(tmp_path):
    append_jsonl(tmp_path / "samples.jsonl", {
        "sample_id": "truncated", "benchmark_name": "Fake/mcqa", "reference_answer": "A",
    })
    append_jsonl(tmp_path / "results.jsonl", {
        "run_id": "run", "result_id": "result-truncated", "sample_id": "truncated",
        "benchmark_name": "Fake/mcqa", "status": "success", "finish_reason": "length",
        "parsed_answer": "A",
    })
    append_jsonl(tmp_path / "judgments.jsonl", {
        "run_id": "run", "judgment_id": "stale", "result_id": "result-truncated",
        "sample_id": "truncated", "evaluator_name": "multiple_choice_accuracy",
        "evaluator_type": "rule_based", "evaluation_status": "success",
        "normalized_score": 1.0, "is_correct": True,
        "provider_metadata": {"primary_metric": True},
    })
    atomic_write_json(tmp_path / "manifest.json", {
        "run_id": "run", "benchmark": {"name": "Fake/mcqa"},
        "generation": {"length_finish_policy": "mark_incomplete"},
    })

    summary = summarize_run(tmp_path)
    assert summary.metrics.score is None
    assert summary.counts.num_scored == 0
    assert summary.counts.num_max_length == 1
    assert summary.counts.num_missing_scoring == 1
    assert summary.metrics_by_evaluator == {}


def test_metrics_by_evaluator_excludes_judgments_for_superseded_results(tmp_path):
    append_jsonl(tmp_path / "samples.jsonl", {
        "sample_id": "s", "benchmark_name": "B", "reference_answer": "A",
    })
    append_jsonl(tmp_path / "results.jsonl", {
        "run_id": "run", "result_id": "old", "sample_id": "s", "benchmark_name": "B",
        "status": "success", "parsed_answer": "A", "reference_answer": "A",
    })
    append_jsonl(tmp_path / "judgments.jsonl", {
        "run_id": "run", "result_id": "old", "sample_id": "s",
        "evaluator_name": "multiple_choice_accuracy", "evaluator_type": "rule_based",
        "evaluation_status": "success", "normalized_score": 1.0, "is_correct": True,
        "provider_metadata": {"primary_metric": True},
    })
    append_jsonl(tmp_path / "results.jsonl", {
        "run_id": "run", "result_id": "new", "sample_id": "s", "benchmark_name": "B",
        "status": "error", "error_type": "api_timeout",
    })
    atomic_write_json(tmp_path / "manifest.json", {
        "run_id": "run", "benchmark": {"name": "B"},
        "full_config": {"evaluation": {"evaluator": "multiple_choice"}},
    })

    summary = summarize_run(tmp_path)
    assert summary.metrics.score is None
    assert summary.metrics_by_evaluator == {}
