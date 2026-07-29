"""Cross-task result table: heterogeneous metric mapping and exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from healthcorebench.aggregation.batch_results import (
    build_batch_result_rows, write_batch_result_files,
)


def _run(tmp_path, key, *, evaluator, use_judge=False, extra_evaluators=None,
         score=0.0, metrics=None,
         status="completed", failed=0, parse_errors=0, evaluation_errors=0,
         max_length=0, missing_scoring=0,
         vlm_profile_metrics=None):
    run_dir = tmp_path / key
    run_dir.mkdir(parents=True)
    benchmark = key.split("/", 1)[0]
    manifest = {
        "run_status": status,
        "benchmark": {"registry_key": key},
        "full_config": {"evaluation": {
            "evaluator": evaluator, "use_llm_judge": use_judge,
            "extra_evaluators": extra_evaluators or [],
        }},
    }
    summary = {
        "benchmark_name": benchmark,
        "counts": {
            "num_total": 5, "num_successful": 5 - failed, "num_failed": failed,
            "num_scored": 5 - failed - evaluation_errors,
            "num_parsing_errors": parse_errors,
            "num_evaluation_errors": evaluation_errors,
            "num_max_length": max_length,
            "num_missing_scoring": missing_scoring,
        },
        "metrics": {
            "score": score,
            "confidence_interval": [0.1, 0.9] if score is not None else None,
            "confidence_interval_method": "test_95",
            "score_denominator_policy": "successful_and_scored_only",
            "sample_weight_sum": 5 - failed - evaluation_errors,
        },
        "metrics_by_evaluator": metrics or {},
        "vlm_profile_metrics": vlm_profile_metrics,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_batch_result_metric_mapping_and_exports(tmp_path):
    mcqa = _run(tmp_path, "MMLU/mcqa", evaluator="multiple_choice", score=0.8, metrics={
        "multiple_choice_accuracy": {"mean_score": 0.8, "accuracy": 0.8, "n": 5},
    })
    judged = _run(
        tmp_path, "CareQA/open", evaluator=None, use_judge=True,
        extra_evaluators=["rouge"], score=0.6, metrics={
        "llm_judge": {"mean_score": 0.6, "accuracy": 0.6, "n": 5},
        "rouge": {"mean_score": 0.2, "subscores": {
            "rouge1.precision": 0.4, "rouge1.recall": 0.3, "rouge1.fmeasure": 0.35,
            "rouge2.precision": 0.2, "rouge2.recall": 0.1, "rouge2.fmeasure": 0.13,
            "rougeL.precision": 0.3, "rougeL.recall": 0.2, "rougeL.fmeasure": 0.24,
        }, "n": 5},
        },
    )
    text = _run(tmp_path, "BioASQ/factoid", evaluator="text_f1_em", score=0.0, metrics={
        "text_f1_em": {"mean_score": 0.1, "accuracy": 0.0,
                       "subscores": {"em": 0.0, "f1": 0.1}, "n": 5},
    })
    summary = _run(
        tmp_path, "MeQSum/summarization", evaluator="rouge",
        extra_evaluators=["bleu"], score=0.3, metrics={
        "rouge": {"mean_score": 0.3, "subscores": {
            "rouge1.precision": 0.5, "rouge1.recall": 0.4, "rouge1.fmeasure": 0.44,
            "rouge2.precision": 0.3, "rouge2.recall": 0.2, "rouge2.fmeasure": 0.24,
            "rougeL.precision": 0.4, "rougeL.recall": 0.3, "rougeL.fmeasure": 0.34,
        }, "n": 5},
        "bleu": {"mean_score": 0.1, "subscores": {
            "bleu1": 40.0, "bleu2": 30.0, "bleu3": 20.0, "bleu4": 10.0,
        }, "n": 5},
        },
    )
    error = _run(tmp_path, "Broken/open", evaluator=None, use_judge=True, score=None,
                 status="completed_with_errors", evaluation_errors=5, metrics={})

    rows = build_batch_result_rows([mcqa, judged, text, summary, error])
    by_key = {row["task_key"]: row for row in rows}

    assert by_key["MMLU/mcqa"]["accuracy"] == 0.8
    assert by_key["CareQA/open"]["judge_score"] == 0.6
    assert by_key["CareQA/open"]["rouge_l_f1"] == 0.24
    assert by_key["BioASQ/factoid"]["primary_score"] == 0.0
    assert by_key["BioASQ/factoid"]["exact_match"] == 0.0
    assert by_key["BioASQ/factoid"]["token_f1"] == 0.1
    assert by_key["MeQSum/summarization"]["rouge_1_precision"] == 0.5
    assert by_key["MeQSum/summarization"]["rouge_1_recall"] == 0.4
    assert by_key["MeQSum/summarization"]["rouge_1_f1"] == 0.44
    assert by_key["MeQSum/summarization"]["rouge_2_f1"] == 0.24
    assert by_key["MeQSum/summarization"]["rouge_l_f1"] == 0.34
    assert by_key["MeQSum/summarization"]["bleu_1"] == 40.0
    assert by_key["MeQSum/summarization"]["bleu_2"] == 30.0
    assert by_key["MeQSum/summarization"]["bleu_3"] == 20.0
    assert by_key["MeQSum/summarization"]["bleu_4"] == 10.0
    assert by_key["Broken/open"]["judge_score"] is None
    assert by_key["Broken/open"]["errors"] == "0/0/5/0/0"

    paths = write_batch_result_files(rows, tmp_path / "exports")
    assert set(paths) == {"json", "csv", "markdown"}
    exported = json.loads((tmp_path / "exports" / "all_tasks_results.json").read_text())
    assert exported["num_tasks"] == 5
    with open(tmp_path / "exports" / "all_tasks_results.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames or []) >= {
            "rouge_1_precision", "rouge_1_recall", "rouge_1_f1",
            "rouge_2_precision", "rouge_2_recall", "rouge_2_f1",
            "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
            "bleu_1", "bleu_2", "bleu_3", "bleu_4",
            "multilabel_subset_accuracy", "multilabel_precision", "multilabel_recall",
            "multilabel_f1", "multilabel_hamming_loss", "document_field_pair_em",
            "document_field_precision", "document_field_recall", "document_field_f1",
        }
        assert len(list(reader)) == 5
    markdown = (tmp_path / "exports" / "all_tasks_results.md").read_text()
    assert "## Accuracy\n\n| Benchmark | Accuracy |" in markdown
    assert "## Judge + ROUGE\n\n| Benchmark | Judge Score | ROUGE-1 Precision |" in markdown
    assert "## Token-F1 / EM\n\n| Benchmark | Exact Match | Token-F1 |" in markdown
    assert "## ROUGE + BLEU" in markdown
    assert "| Benchmark | ROUGE-1 Precision |" in markdown
    assert "| MMLU/mcqa | 0.800 |" in markdown
    assert "| Broken/open | N/A | incomplete_scoring | 0/5 |" in markdown
    assert "Status" in markdown
    assert "Scored" in markdown
    assert "Core Score" not in markdown
    assert markdown.count("| Benchmark |") == 5


def test_same_metric_tasks_share_one_markdown_table(tmp_path):
    first = _run(tmp_path, "MMLU/mcqa", evaluator="multiple_choice", score=0.8, metrics={
        "multiple_choice_accuracy": {"mean_score": 0.8, "accuracy": 0.8, "n": 5},
    })
    second = _run(tmp_path, "PubMedQA/classification", evaluator="classification", score=0.6,
                  metrics={
                      "classification_accuracy": {
                          "mean_score": 0.6, "accuracy": 0.6, "n": 5,
                      },
                  })
    rows = build_batch_result_rows([first, second])

    paths = write_batch_result_files(rows, tmp_path / "grouped")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert markdown.count("## Accuracy") == 1
    assert markdown.count("| Benchmark | Accuracy |") == 1
    assert "| MMLU/mcqa | 0.800 |" in markdown
    assert "| PubMedQA/classification | 0.600 |" in markdown


def test_metric_grouping_is_independent_of_evaluator_order(tmp_path):
    first = _run(
        tmp_path,
        "First/open",
        evaluator=None,
        use_judge=True,
        extra_evaluators=["rouge", "bleu"],
    )
    second = _run(
        tmp_path,
        "Second/open",
        evaluator=None,
        use_judge=True,
        extra_evaluators=["bleu", "rouge"],
    )

    rows = build_batch_result_rows([first, second])
    paths = write_batch_result_files(rows, tmp_path / "order-independent")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert markdown.count("## Judge + ROUGE + BLEU") == 1
    assert markdown.count("| Benchmark | Judge Score | ROUGE-1 Precision |") == 1
    assert "| First/open |" in markdown
    assert "| Second/open |" in markdown


def test_vlm_text_overlap_tasks_share_complete_markdown_table(tmp_path):
    metrics = {
        "vlm_text_overlap": {
            "mean_score": 0.5,
            "subscores": {
                "exact_match_raw": 0.2, "exact_match_normalized": 0.3,
                "precision_token": 0.6, "recall_token": 0.4, "f1_token": 0.48,
                "bleu1": 40.0, "bleu2": 30.0, "bleu3": 20.0, "bleu4": 10.0,
                "rouge1.fmeasure": 0.44, "rouge2.fmeasure": 0.24,
                "rougeL.fmeasure": 0.34,
            },
            "n": 5,
        },
    }
    first = _run(tmp_path, "VQA-RAD/open", evaluator="vlm_text_overlap", metrics=metrics)
    second = _run(tmp_path, "SLAKE/open", evaluator="vlm_text_overlap", metrics=metrics)

    rows = build_batch_result_rows([first, second])
    paths = write_batch_result_files(rows, tmp_path / "vlm")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert markdown.count("## VLM Text Overlap") == 1
    assert markdown.count("| Benchmark | EM Raw | EM Normalized |") == 1
    assert "Token Precision | Token Recall | Token F1" in markdown
    assert "BLEU-1 (%) | BLEU-2 (%) | BLEU-3 (%) | BLEU-4 (%)" in markdown
    assert "ROUGE-1 | ROUGE-2 | ROUGE-L" in markdown
    assert "| VQA-RAD/open | 0.200 | 0.300 |" in markdown
    assert "| SLAKE/open | 0.200 | 0.300 |" in markdown


def test_incomplete_scoring_preserves_partial_score_and_displays_coverage(tmp_path):
    run = _run(
        tmp_path,
        "GeneTuring/open",
        evaluator=None,
        use_judge=True,
        score=0.0,
        max_length=4,
        missing_scoring=4,
        metrics={"llm_judge": {"mean_score": 0.0, "accuracy": 0.0, "n": 1}},
    )
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["counts"]["num_scored"] = 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    rows = build_batch_result_rows([run])
    row = rows[0]
    assert row["primary_score"] == 0.0
    assert row["score_display"] == 0.0
    assert row["status"] == "incomplete_scoring"
    assert row["scored"] == "1/5"
    assert row["errors"] == "0/0/0/4/4"

    markdown = Path(write_batch_result_files(rows, tmp_path / "incomplete")["markdown"]).read_text()
    assert "| GeneTuring/open | 0.000 | incomplete_scoring | 1/5 | 4 | 4 |" in markdown
    assert "[0.100, 0.900]" in markdown


def test_vlm_profile_aggregates_are_exported_to_markdown(tmp_path):
    profile = {
        "task_profile": "closed",
        "classification": {
            "accuracy": 0.8,
            "precision_macro_label": 0.7, "recall_macro_label": 0.6,
            "f1_macro_label": 0.65,
            "precision_micro_label": 0.8, "recall_micro_label": 0.8,
            "f1_micro_label": 0.8,
            "precision_weighted_label": 0.78, "recall_weighted_label": 0.8,
            "f1_weighted_label": 0.79, "invalid_output_rate": 0.1,
        },
    }
    run = _run(
        tmp_path, "VQA-RAD/closed", evaluator="classification",
        metrics={"classification_accuracy": {"mean_score": 0.8, "accuracy": 0.8}},
        vlm_profile_metrics=profile,
    )

    paths = write_batch_result_files(build_batch_result_rows([run]), tmp_path / "profile")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert "## VLM Classification" in markdown
    assert "Macro Precision | Macro Recall | Macro F1" in markdown
    assert "Micro Precision | Micro Recall | Micro F1" in markdown
    assert "Weighted Precision | Weighted Recall | Weighted F1" in markdown
    assert "Invalid Output Rate" in markdown
    assert "| VQA-RAD/closed | 0.800 | 0.700 |" in markdown


def test_document_full_parsing_keeps_field_metrics_primary_when_judged(tmp_path):
    profile = {
        "task_profile": "document_parse",
        "document_fields": {
            "field_name_exact_match": 0.8,
            "field_value_exact_match": 0.7,
            "field_pair_exact_match": 0.6,
            "precision_micro_field": 0.75,
            "recall_micro_field": 0.65,
            "f1_micro_field": 0.696,
            "precision_macro_field": 0.7,
            "recall_macro_field": 0.6,
            "f1_macro_field": 0.646,
            "missing_critical_field_rate": 0.2,
            "invented_field_rate": 0.1,
        },
        "llm_judge": {"overall": 0.9},
    }
    run = _run(
        tmp_path,
        "MedDocBench/ltr_full_parsing",
        evaluator="document_fields",
        use_judge=True,
        score=0.696,
        metrics={
            "vlm_document_fields": {"mean_score": 0.696},
            "llm_judge": {"mean_score": 0.9},
        },
        vlm_profile_metrics=profile,
    )

    row = build_batch_result_rows([run])[0]

    assert row["primary_evaluator"] == "document_fields"
    assert row["primary_score"] == 0.696


def test_report_header_states_evaluator_versions_and_flags_a_mix():
    """A score is only comparable against a score from the same evaluator version.

    Evaluator 1.1 stopped scoring an unparsed answer as a hard zero, which moved IgakuQA from
    0.4897 to 0.5234 with the model untouched. Nothing in the score columns shows that, so the
    version belongs in the header and a mix of versions has to be called out.
    """
    from healthcorebench.aggregation.batch_results import _render_report_header

    header = "\n".join(_render_report_header([
        {"task_key": "A/x", "has_summary": True, "status": "completed",
         "metrics_by_evaluator": {"multiple_choice": {"evaluator_versions": ["1.1"]}}},
        {"task_key": "B/y", "has_summary": True, "status": "completed",
         "metrics_by_evaluator": {"multiple_choice": {"evaluator_versions": ["1.0"]},
                                  "rouge": {"evaluator_versions": ["1.1"]}}},
    ]))
    assert "Evaluator version(s): multiple_choice 1.0/1.1, rouge 1.1" in header
    assert "WARNING: multiple_choice appears at more than one version" in header

    consistent = "\n".join(_render_report_header([
        {"task_key": "A/x", "has_summary": True, "status": "completed",
         "metrics_by_evaluator": {"multiple_choice": {"evaluator_versions": ["1.1"]}}},
    ]))
    assert "Evaluator version(s): multiple_choice 1.1" in consistent
    assert "WARNING" not in consistent


def test_summarize_records_the_evaluator_version_it_scored_with(tmp_path):
    """The version has to reach summary.json, or the batch report has nothing to read."""
    from healthcorebench.aggregation.summarize import _metrics_by_evaluator

    judgments = {
        ("res1", "multiple_choice"): {
            "result_id": "res1", "evaluator_name": "multiple_choice",
            "evaluator_version": "1.1", "evaluation_status": "success",
            "normalized_score": 1.0, "is_correct": True,
        },
        ("res2", "multiple_choice"): {
            "result_id": "res2", "evaluator_name": "multiple_choice",
            "evaluator_version": "1.0", "evaluation_status": "success",
            "normalized_score": 0.0, "is_correct": False,
        },
        # A judgment written before evaluator_version existed must not be silently dropped.
        ("res3", "rouge"): {
            "result_id": "res3", "evaluator_name": "rouge",
            "evaluation_status": "success", "normalized_score": 0.5,
        },
    }
    out = _metrics_by_evaluator(judgments, {"res1", "res2", "res3"}, {})
    assert out["multiple_choice"]["evaluator_versions"] == ["1.0", "1.1"]
    assert out["rouge"]["evaluator_versions"] == ["unknown"]


def test_registry_overlap_notes_reach_the_report():
    """A task that is largely another task's questions has to say so where scores are compared.

    MMedBench's English/Chinese/French splits *are* MedQA_USMLE, MedQA_MCMLE and FrenchMedMCQA:
    5,020 of 8,178 items. It stays enabled because its Russian/Spanish/Japanese splits are real
    added coverage, so the safeguard has to be disclosure rather than removal — and ``overlap_note``
    used to live in the registry without any report consuming it.
    """
    from healthcorebench.aggregation.batch_results import _overlap_note, _render_report_header

    note = _overlap_note("MMedBench/mcqa")
    assert note and "MedQA_USMLE/mcqa" in note
    assert _overlap_note("MMLU/mcqa") is None

    header = "\n".join(_render_report_header([
        {"task_key": "MMedBench/mcqa", "has_summary": True, "status": "completed",
         "overlap_note": note},
        {"task_key": "MMLU/mcqa", "has_summary": True, "status": "completed",
         "overlap_note": None},
    ]))
    assert "Tasks whose content overlaps another task in this suite:" in header
    assert "MMedBench/mcqa overlaps" in header
    assert "MMLU/mcqa overlaps" not in header

    # Silent when nothing overlaps, so the line means something when it appears.
    clean = "\n".join(_render_report_header([
        {"task_key": "MMLU/mcqa", "has_summary": True, "status": "completed"},
    ]))
    assert "overlaps" not in clean


def test_every_overlap_note_names_a_task_that_exists():
    """A note pointing at a renamed key is worse than no note: it reads as a checked relation."""
    import re

    from healthcorebench.benchmarks.registry import get_registry

    registry = get_registry()
    for key, entry in registry.items():
        if not entry.overlap_note:
            continue
        referenced = set(re.findall(r"\b[\w.-]+/[\w.-]+\b", entry.overlap_note))
        assert referenced, f"{key}: overlap_note names no task key"
        missing = {name for name in referenced if name not in registry}
        assert not missing, f"{key}: overlap_note points at unknown {sorted(missing)}"


def test_summarize_counts_attempts_missing_usage(tmp_path):
    from healthcorebench.aggregation.summarize import summarize_run
    from healthcorebench.utils.jsonl import append_jsonl

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for attempt in [
        {"request_purpose": "model_inference", "usage": {"total_tokens": 100}},
        {"request_purpose": "model_inference", "usage": {"total_tokens": 50}},
        # Rejected by the endpoint: billed by the provider, no usage reported.
        {"request_purpose": "model_inference", "usage": None},
        {"request_purpose": "model_inference"},
        {"request_purpose": "evaluation_judge", "usage": {"total_tokens": 7}},
    ]:
        append_jsonl(run_dir / "attempts.jsonl", attempt)

    tokens = summarize_run(run_dir).tokens
    assert tokens["cumulative_inference_attempt_tokens"] == 150
    assert tokens["cumulative_evaluation_attempt_tokens"] == 7
    assert tokens["attempts_with_usage"] == 3
    assert tokens["attempts_without_usage"] == 2


def test_zero_score_rows_say_whether_a_re_run_would_fill_them():
    """Every metric cell reads N/A when a task scored nothing, which hides *why*.

    A round where the judge endpoint returned 401 and a task that is structurally unscorable
    render identically. The first is worth re-running and the second is not, so the header
    separates them and names the command.
    """
    from healthcorebench.aggregation.batch_results import _recoverable_gap, _render_report_header

    # Judge auth failed for the whole round: inference landed, scoring did not.
    judge_down = {"task_key": "A/open", "has_summary": True, "status": "incomplete_scoring",
                  "num_total": 5, "num_scored": 0, "num_evaluation_errors": 5}
    assert _recoverable_gap(judge_down) == "5 evaluation errors"
    # Singular reads as singular.
    assert _recoverable_gap({**judge_down, "num_evaluation_errors": 1}) == "1 evaluation error"

    # Nothing a re-run changes: the samples were scored, or there were none, or every one of
    # them was unscorable for lack of a usable reference.
    assert _recoverable_gap({"task_key": "B/x", "has_summary": True, "num_scored": 7}) is None
    assert _recoverable_gap({"task_key": "C/x", "has_summary": True, "num_total": 0,
                             "num_scored": 0}) is None
    assert _recoverable_gap({"task_key": "D/x", "has_summary": True, "num_total": 5,
                             "num_scored": 0, "num_unscorable": 5}) is None
    # Already reported in their own section with their own reason.
    assert _recoverable_gap({"task_key": "E/x", "status": "not_run", "num_scored": 0}) is None
    assert _recoverable_gap({"task_key": "F/x", "status": "skipped", "num_scored": 0}) is None

    header = "\n".join(_render_report_header([judge_down,
                                              {"task_key": "B/x", "has_summary": True,
                                               "status": "completed", "num_scored": 7}]))
    assert "recoverable by re-running: 1" in header
    assert "--retry-failed" in header
    assert "A/open: 5 evaluation errors" in header
    assert "B/x:" not in header

    # Silent when there is nothing to recover, so the section means something when present.
    clean = "\n".join(_render_report_header([
        {"task_key": "B/x", "has_summary": True, "status": "completed", "num_scored": 7},
    ]))
    assert "recoverable by re-running" not in clean


def test_evaluator_version_line_says_unrecorded_rather_than_blank():
    """Summaries written before versions were recorded must not render a dangling label."""
    from healthcorebench.aggregation.batch_results import _render_report_header

    header = "\n".join(_render_report_header([
        {"task_key": "A/x", "has_summary": True, "status": "completed", "num_scored": 1,
         "metrics_by_evaluator": {"multiple_choice": {}}},
    ]))
    assert "Evaluator version(s): multiple_choice unrecorded" in header
