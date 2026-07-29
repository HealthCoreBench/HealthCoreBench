"""Focused correctness checks for VLM profile metrics."""

from __future__ import annotations

import pytest

from healthcorebench.aggregation.vlm import aggregate_vlm_profiles
from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.evaluators import get_evaluator
from healthcorebench.schemas.config import RunConfig
from healthcorebench.runtime.run_setup import RunOrchestrator
from healthcorebench.utils.jsonl import append_jsonl


def _evaluate(name, parsed, reference, *, metadata=None):
    sample = {
        "sample_id": "sample", "component": "Multimodal",
        "reference_answer": reference, "reference_answer_normalized": reference,
        "metadata": metadata or {},
    }
    result = {
        "run_id": "run", "result_id": "result", "sample_id": "sample",
        "status": "success", "parsed_answer": parsed,
    }
    return get_evaluator(name).evaluate(result, sample)


def test_vlm_text_overlap_reports_all_required_metrics() -> None:
    judgment = _evaluate("vlm_text_overlap", "mild edema", "mild edema")
    details = judgment.parsed_judgment
    assert judgment.normalized_score == 1.0
    assert details["exact_match_raw"] == 1.0
    assert details["precision_token"] == details["recall_token"] == details["f1_token"] == 1.0
    assert all(key in details for key in ("bleu1", "bleu2", "bleu3", "bleu4"))
    assert all(key in details for key in ("rouge1", "rouge2", "rougeL"))
    assert details["clinical_metrics"]["finding_f1"] is None


def test_multilabel_and_grounding_distinguish_partial_predictions() -> None:
    multilabel = _evaluate(
        "multilabel", ["Atelectasis"], ["Atelectasis", "Edema"],
        metadata={"label_universe": ["Atelectasis", "Edema", "Pneumonia"]},
    )
    assert multilabel.parsed_judgment["precision_sample"] == 1.0
    assert multilabel.parsed_judgment["recall_sample"] == 0.5
    grounding = _evaluate(
        "grounding",
        [{"label": "opacity", "category": "Pneumonia", "bbox_xyxy": [0, 0, 10, 10]}],
        [{"label": "left basilar opacity", "category": "Pneumonia",
          "bbox_xyxy": [5, 5, 15, 15]}],
    )
    assert 0 < grounding.normalized_score < 0.25
    assert grounding.parsed_judgment["phrase_prompt_echo_rate"] == 0.0
    assert grounding.parsed_judgment["disease_category_accuracy"] == 1.0
    assert grounding.parsed_judgment["map_iou_0_50"] is None

    document = _evaluate(
        "document_fields", "ALT: 85 U/L\nAST: 40 U/L", "ALT: 85 U/L\nAST: 42 U/L",
    )
    assert document.parsed_judgment["field_name_exact_match"] == 1.0
    assert document.parsed_judgment["field_value_exact_match"] == 0.5


def test_document_fields_parses_order_independent_json_object_arrays() -> None:
    reference = (
        '[{"entryname":"Hemoglobin","result":"82 g/L","reference":"115-150",'
        '"status":"low"},{"entryname":"Platelets","result":"398",'
        '"reference":"125-350","status":"high"}]'
    )
    prediction = (
        '```json\n{"abnormalities":['
        '{"status":"high","reference":"125-350","result":"398",'
        '"entryname":"Platelets"},'
        '{"status":"low","entryname":"Hemoglobin","result":"82 g/L",'
        '"reference":"115-150"}]}\n```'
    )

    judgment = _evaluate("document_fields", prediction, reference)

    assert judgment.normalized_score == 1.0
    assert judgment.parsed_judgment["field_name_exact_match"] == 1.0
    assert judgment.parsed_judgment["field_value_exact_match"] == 1.0
    assert judgment.parsed_judgment["num_reference_fields"] == 6


def test_grounding_matches_reordered_identical_boxes_by_phrase_and_category() -> None:
    references = [
        {"label": "left opacity", "category": "Pneumonia", "bbox_xyxy": [0, 0, 10, 10]},
        {"label": "right opacity", "category": "Atelectasis", "bbox_xyxy": [0, 0, 10, 10]},
    ]
    prediction = list(reversed(references))

    judgment = _evaluate("grounding", prediction, references)

    assert judgment.normalized_score == 1.0
    assert judgment.parsed_judgment["phrase_prompt_echo_rate"] == 1.0
    assert judgment.parsed_judgment["disease_category_accuracy"] == 1.0


def test_multilabel_negative_recall_counts_each_explicit_negative_once() -> None:
    samples = {
        "s1": {"sample_id": "s1", "component": "Multimodal", "metadata": {
            "task_profile": "multilabel", "negative_labels": ["Edema", "Pneumonia"],
        }},
        "s2": {"sample_id": "s2", "component": "Multimodal", "metadata": {
            "task_profile": "multilabel", "negative_labels": ["Edema"],
        }},
    }
    judgments = [
        {"sample_id": "s1", "result_id": "r1", "evaluation_status": "success",
         "evaluator_name": "vlm_multilabel", "normalized_score": 0.0,
         "parsed_judgment": {
             "predicted_labels": ["Edema"], "reference_labels": [],
             "label_universe": ["Edema", "Pneumonia"], "hamming_loss": 0.5,
             "parse_failed": False,
         }},
        {"sample_id": "s2", "result_id": "r2", "evaluation_status": "success",
         "evaluator_name": "vlm_multilabel", "normalized_score": 1.0,
         "parsed_judgment": {
             "predicted_labels": [], "reference_labels": [],
             "label_universe": ["Edema"], "hamming_loss": 0.0,
             "parse_failed": False,
         }},
    ]

    metrics = aggregate_vlm_profiles(samples, judgments, {"r1", "r2"})["multilabel"]

    assert metrics["negative_recall"] == pytest.approx(2 / 3)


def test_mimic_multilabel_excludes_uncertain_and_null_labels_per_sample() -> None:
    metadata = {
        "label_universe": ["Atelectasis", "Edema", "Pneumonia"],
        "evaluated_labels": ["Atelectasis"],
        "uncertain_labels": ["Edema"],
        "null_labels": ["Pneumonia"],
        "uncertain_label_policy": "excluded",
        "null_label_policy": "excluded",
    }
    ignored = _evaluate(
        "multilabel", ["Atelectasis", "Edema", "Pneumonia"], ["Atelectasis"],
        metadata=metadata,
    )
    unsupported = _evaluate(
        "multilabel", ["Atelectasis", "unsupported finding"], ["Atelectasis"],
        metadata=metadata,
    )

    assert ignored.normalized_score == 1.0
    assert ignored.parsed_judgment["ignored_unevaluated_labels"] == ["Edema", "Pneumonia"]
    assert unsupported.normalized_score < 1.0
    assert "unsupported finding" in unsupported.parsed_judgment["predicted_labels"]


def test_mimic_multilabel_respects_an_explicitly_empty_evaluated_label_set() -> None:
    judgment = _evaluate(
        "multilabel",
        ["Edema"],
        [],
        metadata={
            "label_universe": ["Edema"],
            "evaluated_labels": [],
            "uncertain_labels": ["Edema"],
        },
    )

    # With no label evaluated for this sample the reference is empty, so there is nothing to
    # measure: the sample is reported unscorable instead of earning a free 1.0. The exclusion
    # itself is still observable through ignored_unevaluated_labels.
    assert judgment.normalized_score is None
    assert judgment.is_correct is None
    assert judgment.parsed_judgment["unscorable_reason"] == "empty_reference"
    assert judgment.parsed_judgment["ignored_unevaluated_labels"] == ["Edema"]


def test_vlm_classification_aggregate_is_profile_scoped() -> None:
    samples = {
        "s1": {"sample_id": "s1", "component": "Multimodal", "metadata": {"task_profile": "closed"}},
        "s2": {"sample_id": "s2", "component": "Language", "metadata": {}},
    }
    judgments = [{
        "sample_id": "s1", "result_id": "r1", "evaluation_status": "success",
        "evaluator_name": "multiple_choice_accuracy", "normalized_score": 1.0,
        "parsed_judgment": {"predicted": "A", "reference": "A", "parse_failed": False},
    }, {
        "sample_id": "s2", "result_id": "r2", "evaluation_status": "success",
        "evaluator_name": "classification_accuracy", "normalized_score": 0.0,
        "parsed_judgment": {"predicted": "B", "reference": "A", "parse_failed": False},
    }]
    aggregate = aggregate_vlm_profiles(samples, judgments, {"r1", "r2"})
    assert aggregate["classification"]["accuracy"] == 1.0
    assert aggregate["classification"]["f1_macro_label"] == 1.0


def test_short_vlm_answer_keeps_overlap_primary_when_judge_is_configured(tmp_path) -> None:
    config = RunConfig.model_validate({
        "experiment": {"experiment_id": "test", "run_name": "test"},
        "benchmark": {"name": "VQA-RAD/open", "max_samples": 1},
        "model": {"base_url": "http://localhost/v1", "requested_model_name": "vlm"},
        "evaluation": {"judge": {
            "base_url": "http://localhost/v1", "requested_model_name": "judge",
        }},
    })
    orchestrator = RunOrchestrator(config, run_dir=str(tmp_path))
    samples = orchestrator.prepare_samples()
    orchestrator._resolve_evaluation(config, samples)
    assert config.evaluation.use_llm_judge is True
    assert config.evaluation.evaluator == "vlm_text_overlap"
    assert orchestrator._judge_as_primary is False


def test_summarize_run_persists_vlm_profile_metrics(tmp_path) -> None:
    append_jsonl(tmp_path / "samples.jsonl", {
        "sample_id": "s1", "benchmark_name": "VQA-RAD", "component": "Multimodal",
        "sample_weight": 1.0, "metadata": {"task_profile": "closed"},
    })
    append_jsonl(tmp_path / "results.jsonl", {
        "result_id": "r1", "run_id": "run", "sample_id": "s1",
        "benchmark_name": "VQA-RAD", "status": "success", "parsing_status": "success",
    })
    append_jsonl(tmp_path / "judgments.jsonl", {
        "judgment_id": "j1", "run_id": "run", "result_id": "r1", "sample_id": "s1",
        "evaluator_name": "classification_accuracy", "evaluator_type": "rule_based",
        "evaluation_status": "success", "normalized_score": 1.0, "is_correct": True,
        "parsed_judgment": {"predicted": "yes", "reference": "yes", "parse_failed": False},
        "provider_metadata": {"primary_metric": True},
    })

    summary = summarize_run(tmp_path)

    assert summary.vlm_profile_metrics["task_profile"] == "closed"
    classification = summary.vlm_profile_metrics["classification"]
    assert classification["accuracy"] == 1.0
    assert classification["f1_micro_label"] == 1.0
    assert summary.metrics_by_evaluator["classification_accuracy"]["accuracy"] == 1.0


def test_classification_aggregate_counts_unknown_output_as_invalid() -> None:
    samples = {
        "s1": {"sample_id": "s1", "component": "Multimodal", "metadata": {
            "task_profile": "closed", "letters": ["yes", "no", "maybe"],
        }},
        "s2": {"sample_id": "s2", "component": "Multimodal", "metadata": {
            "task_profile": "closed", "letters": ["yes", "no", "maybe"],
        }},
    }
    judgments = [
        {"sample_id": "s1", "result_id": "r1", "evaluation_status": "success",
         "evaluator_name": "classification_accuracy", "normalized_score": 1.0,
         "parsed_judgment": {"predicted": "yes", "reference": "yes", "parse_failed": False}},
        {"sample_id": "s2", "result_id": "r2", "evaluation_status": "success",
         "evaluator_name": "classification_accuracy", "normalized_score": 0.0,
         "parsed_judgment": {"predicted": "unknown", "reference": "no", "parse_failed": False}},
    ]

    classification = aggregate_vlm_profiles(samples, judgments, {"r1", "r2"})["classification"]

    assert classification["accuracy"] == 0.5
    assert classification["f1_micro_label"] == 0.5
    assert classification["invalid_output_rate"] == 0.5
    assert classification["answer_extraction_failure_rate"] == 0.0
    assert "maybe" in classification["per_class"]


def test_classification_confusion_uses_each_samples_allowed_labels() -> None:
    samples = {
        "s1": {"sample_id": "s1", "component": "Multimodal", "metadata": {
            "task_profile": "closed", "letters": ["yes", "no"],
        }},
        "s2": {"sample_id": "s2", "component": "Multimodal", "metadata": {
            "task_profile": "closed",
        }},
    }
    judgments = [
        {"sample_id": sample_id, "result_id": result_id,
         "evaluation_status": "success", "evaluator_name": "classification_accuracy",
         "normalized_score": 0.0, "parsed_judgment": {
             "predicted": "unknown", "reference": "no", "parse_failed": False,
         }}
        for sample_id, result_id in (("s1", "r1"), ("s2", "r2"))
    ]

    classification = aggregate_vlm_profiles(samples, judgments, {"r1", "r2"})[
        "classification"
    ]

    assert classification["invalid_output_rate"] == 0.5
    assert classification["confusion_matrix"]["no"]["<invalid>"] == 1
    assert classification["confusion_matrix"]["no"]["<other>"] == 1


def test_case_and_stage_aggregate_reports_accuracy_f1_and_failures() -> None:
    rows = (
        ("s1", "r1", "case-1", "diagnosis", "A", "A", 1.0, False),
        ("s2", "r2", "case-1", "treatment", "B", "B", 1.0, False),
        ("s3", "r3", "case-2", "diagnosis", "A", "B", 0.0, False),
        ("s4", "r4", "case-2", "treatment", None, "A", 0.0, True),
    )
    samples = {
        sample_id: {
            "sample_id": sample_id,
            "component": "Multimodal",
            "metadata": {
                "task_profile": "multistage_closed",
                "case_id": case_id,
                "stage_name": stage,
                "letters": ["A", "B"],
            },
        }
        for sample_id, _, case_id, stage, _, _, _, _ in rows
    }
    judgments = [
        {
            "sample_id": sample_id,
            "result_id": result_id,
            "evaluation_status": "success",
            "evaluator_name": "multiple_choice_accuracy",
            "normalized_score": score,
            "parsed_judgment": {
                "predicted": predicted,
                "reference": reference,
                "parse_failed": parse_failed,
            },
        }
        for sample_id, result_id, _, _, predicted, reference, score, parse_failed in rows
    ]
    judgments.append({
        "sample_id": "s1",
        "result_id": "r1",
        "evaluation_status": "success",
        "evaluator_name": "llm_judge",
        "normalized_score": 0.0,
        "parsed_judgment": {"parse_failed": False},
    })

    metrics = aggregate_vlm_profiles(
        samples, judgments, {row[1] for row in rows}
    )["case_and_stage"]

    assert metrics["question_accuracy"] == 0.5
    assert metrics["f1_micro_stage_label"] == 0.5
    assert metrics["f1_macro_stage_label"] == pytest.approx(7 / 12)
    assert metrics["case_all_correct_rate"] == 0.5
    assert metrics["invalid_output_rate"] == 0.25


def test_multistage_parse_failure_counts_as_error_in_every_stage_f1() -> None:
    sample = {
        "sample_id": "s1",
        "component": "Multimodal",
        "metadata": {"task_profile": "multistage", "case_id": "case-1"},
    }
    judgment = {
        "sample_id": "s1",
        "result_id": "r1",
        "evaluation_status": "success",
        "evaluator_name": "vlm_multistage_choice",
        "normalized_score": 0.0,
        "parsed_judgment": {
            "predicted_stages": [],
            "reference_stages": ["A", "B", "C", "D"],
            "stage_names": ["modality", "organ", "lesion", "diagnosis"],
            "stage_correct": [False, False, False, False],
            "parse_failed": True,
        },
    }

    metrics = aggregate_vlm_profiles({"s1": sample}, [judgment], {"r1"})[
        "case_and_stage"
    ]

    assert metrics["f1_micro_stage_label"] == 0.0
    assert set(metrics["stage_metrics"]) == {"modality", "organ", "lesion", "diagnosis"}


def test_grounding_aggregate_separates_phrase_category_and_box_metrics() -> None:
    sample = {
        "sample_id": "s1",
        "component": "Multimodal",
        "reference_answer": [{
            "label": "left basilar opacity",
            "category": "Pneumonia",
            "bbox_xyxy": [0, 0, 10, 10],
        }],
        "reference_answer_normalized": [{
            "label": "left basilar opacity",
            "category": "Pneumonia",
            "bbox_xyxy": [0, 0, 10, 10],
        }],
        "metadata": {"task_profile": "grounding"},
    }
    result = {
        "run_id": "run",
        "result_id": "r1",
        "sample_id": "s1",
        "status": "success",
        "parsed_answer": [{
            "label": "opacity",
            "category": "Pneumonia",
            "bbox_xyxy": [0, 0, 10, 10],
        }],
    }
    judgment = get_evaluator("grounding").evaluate(result, sample).model_dump()

    metrics = aggregate_vlm_profiles({"s1": sample}, [judgment], {"r1"})["grounding"]

    assert metrics["mean_iou"] == 1.0
    assert metrics["phrase_prompt_echo_rate"] == 0.0
    assert metrics["disease_label_accuracy"] == 1.0
    assert metrics["f1_label"] == 1.0
    assert metrics["phrase_to_box_joint_accuracy_iou_0_5"] == 0.0


def test_vlm_judge_safety_booleans_aggregate_as_rates(tmp_path) -> None:
    for index, hallucination in enumerate((False, True), start=1):
        sample_id, result_id = f"s{index}", f"r{index}"
        append_jsonl(tmp_path / "samples.jsonl", {
            "sample_id": sample_id, "benchmark_name": "IU-Xray",
            "component": "Multimodal", "sample_weight": 1.0,
            "metadata": {"task_profile": "report"},
        })
        append_jsonl(tmp_path / "results.jsonl", {
            "result_id": result_id, "run_id": "run", "sample_id": sample_id,
            "benchmark_name": "IU-Xray", "status": "success", "parsing_status": "success",
        })
        append_jsonl(tmp_path / "judgments.jsonl", {
            "judgment_id": f"j{index}", "run_id": "run", "result_id": result_id,
            "sample_id": sample_id, "evaluator_name": "llm_judge",
            "evaluator_type": "llm_judge", "evaluation_status": "success",
            "normalized_score": 0.75, "is_correct": not hallucination,
            "parsed_judgment": {
                "semantic_equivalence": 0.8, "factual_correctness": 0.7,
                "clinical_coverage": 0.6, "reasoning_quality": 0.7,
                "clinical_safety": 0.5, "critical_hallucination": hallucination,
                "critical_omission": False, "unsupported_claim": hallucination,
            },
            "provider_metadata": {"primary_metric": True},
        })

    summary = summarize_run(tmp_path)
    judge = summary.vlm_profile_metrics["llm_judge"]

    assert judge["critical_hallucination_rate"] == 0.5
    assert judge["critical_omission_rate"] == 0.0
    assert judge["unsupported_claim_rate"] == 0.5
    assert summary.metrics_by_evaluator["llm_judge"]["subscores"][
        "critical_hallucination"
    ] == 0.5
