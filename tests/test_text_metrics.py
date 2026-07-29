"""Unit tests for the generation/short-answer metrics: text_f1_em, rouge, bleu; the
auto-selection of these metrics; per-task secondary evaluators; and the summary-layer
metrics-by-evaluator aggregation."""

import json

from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.evaluators.numeric_tolerance import _to_float
from healthcorebench.evaluators import (
    get_evaluator,
    select_evaluator_name,
    default_extra_evaluators,
)


# --------------------------------------------------------------------------- #
# text_f1_em
# --------------------------------------------------------------------------- #
def _score(ev_name, pred, sample):
    ev = get_evaluator(ev_name)
    return ev.score(ev.normalize(pred, sample), sample)


def test_text_f1_em_exact_and_partial():
    raw, norm, correct, parsed = _score("text_f1_em", "Myasthenia Gravis",
                                        {"reference_answer": "myasthenia gravis"})
    assert correct is True and norm == 1.0 and parsed["em"] == 1.0

    # extra words + dropped article -> EM false but partial token-F1
    raw, norm, correct, parsed = _score("text_f1_em", "the myasthenia gravis disease",
                                        {"reference_answer": "myasthenia gravis"})
    assert correct is False and 0.0 < norm < 1.0 and parsed["em"] == 0.0


def test_text_f1_em_aliases_and_no_slash_autosplit():
    # explicit alias list is honored (RareBench-style alternates provided by the adapter)
    _, _, correct, _ = _score("text_f1_em", "heart attack",
                              {"reference_answer": "myocardial infarction",
                               "reference_aliases": ["heart attack", "MI"]})
    assert correct is True
    # a list-valued reference is honored
    _, _, correct, _ = _score("text_f1_em", "flu", {"reference_answer": ["influenza", "flu"]})
    assert correct is True
    # "/" in a reference is NOT auto-split — "mg/dL" must not accept "dL" (units/ratios/dates)
    _, _, correct, _ = _score("text_f1_em", "dL", {"reference_answer": "mg/dL"})
    assert correct is False


def test_numeric_parser_preserves_large_scientific_and_fractional_values():
    assert _to_float("result: 1234.5 mg") == 1234.5
    assert _to_float("result: 12,345") == 12345.0
    assert _to_float("result: -1.2e-3") == -0.0012
    assert _to_float("result: .75") == 0.75
    _, _, correct, _ = _score("text_f1_em", "80", {"reference_answer": "120/80"})
    assert correct is False


def test_text_f1_em_non_latin_scripts():
    # Cyrillic / Arabic must tokenize (not collapse to empty → false 1.0 for any two strings).
    _, f1_wrong, em, _ = _score("text_f1_em", "печень", {"reference_answer": "сердце"})
    assert em is False and f1_wrong == 0.0          # different words -> no credit
    _, f1_same, em, _ = _score("text_f1_em", "сердце", {"reference_answer": "сердце"})
    assert em is True and f1_same == 1.0
    # accented Latin is preserved (not truncated to "caf")
    _, _, em, _ = _score("text_f1_em", "café", {"reference_answer": "café"})
    assert em is True


def test_rouge_bleu_flag_parse_failure():
    # a None parsed answer (genuine parse failure) must be flagged, not silently scored.
    for ev in ("rouge", "bleu"):
        e = get_evaluator(ev)
        *_, parsed = e.score(e.normalize(None, {}), {"reference_answer": "a b c"})
        assert parsed["parse_failed"] is True


def test_text_f1_em_chinese_and_parse_failure():
    _, norm, correct, _ = _score("text_f1_em", "右侧肩胛骨骨折",
                                 {"reference_answer": "右侧肩胛骨骨折"})
    assert correct is True and norm == 1.0
    # partial Chinese overlap
    _, norm, correct, _ = _score("text_f1_em", "肩胛骨骨折", {"reference_answer": "右侧肩胛骨骨折"})
    assert correct is False and 0.0 < norm < 1.0
    # parse failure (None) -> unscorable, flagged, never crashes. Scoring it 0.0 would put a
    # response that produced no answer span in the same bucket as a confidently wrong one.
    raw, norm, correct, parsed = _score("text_f1_em", None, {"reference_answer": "foo"})
    assert (raw, norm, correct) == (None, None, None)
    assert parsed["unscorable_reason"] == "unparsed_answer"
    assert parsed["parse_failed"] is True


# --------------------------------------------------------------------------- #
# rouge / bleu
# --------------------------------------------------------------------------- #
def test_rouge_scores_and_no_binary_correct():
    raw, norm, correct, parsed = _score(
        "rouge", "The patient has shortness of breath and cough.",
        {"reference_answer": "Patient presents with shortness of breath and a cough."})
    assert correct is None                      # ROUGE has no binary notion
    assert 0.0 < norm <= 1.0
    for k in ("rouge1", "rouge2", "rougeL"):
        assert set(parsed[k]) == {"precision", "recall", "fmeasure"}


def test_rouge_chinese_segments_with_jieba():
    raw, norm, correct, _ = _score("rouge", "患者出现呼吸急促和咳嗽症状。",
                                   {"reference_answer": "患者主诉呼吸急促伴咳嗽。"})
    # jieba segmentation yields real overlap (character-blind splitting would be ~0 or nonsense)
    assert norm > 0.3


def test_bleu_normalized_range_and_zh_tokenizer():
    raw, norm, correct, parsed = _score(
        "bleu", "The patient has shortness of breath.",
        {"reference_answer": "The patient presents with shortness of breath."})
    assert correct is None and 0.0 <= norm <= 1.0 and parsed["tokenize"] == "13a"
    assert set(parsed) >= {"bleu1", "bleu2", "bleu3", "bleu4"}
    assert all(0.0 <= parsed[f"bleu{order}"] <= 100.0 for order in range(1, 5))
    assert raw == parsed["bleu4"]
    assert norm == round(parsed["bleu4"] / 100.0, 6)
    _, _, _, parsed_zh = _score("bleu", "患者呼吸急促", {"reference_answer": "患者呼吸急促伴咳嗽"})
    assert parsed_zh["tokenize"] == "zh"


# --------------------------------------------------------------------------- #
# auto-selection + per-task secondary defaults
# --------------------------------------------------------------------------- #
def test_auto_select_new_metrics():
    assert select_evaluator_name("text_f1", "short_answer") == "text_f1_em"
    assert select_evaluator_name("rouge", "summary") == "rouge"
    assert select_evaluator_name("bleu", "free_text") == "bleu"
    assert select_evaluator_name("llm_judge", "free_text") is None


def test_default_extra_evaluators_mapping():
    assert default_extra_evaluators("MeQSum/summarization") == ["bleu"]
    assert default_extra_evaluators("ACI-Bench_HF/summarization") == ["bleu"]
    assert default_extra_evaluators("ClinicBench/hospitalization") == ["bleu"]
    assert default_extra_evaluators("RareBench/diagnosis") == ["text_f1_em"]
    assert default_extra_evaluators("MedicationQA/open") == ["rouge"]
    assert default_extra_evaluators("MMLU/mcqa") == []          # no extras for plain MCQA
    assert default_extra_evaluators(None) == []


# --------------------------------------------------------------------------- #
# summary aggregation: accuracy only for classification + metrics-by-evaluator
# --------------------------------------------------------------------------- #
def _write_run(tmp_path, samples, results, judgments):
    for name, rows in (("samples.jsonl", samples), ("results.jsonl", results),
                       ("judgments.jsonl", judgments)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_summary_classification_reports_accuracy_only(tmp_path):
    labels = [("yes", "yes"), ("no", "no"), ("yes", "yes"), ("maybe", "no"), ("maybe", "yes")]
    samples = [{"run_id": "r", "sample_id": f"s{i}", "benchmark_name": "PubMedQA",
                "answer_format": "yes_no_maybe"} for i in range(5)]
    results = [{"run_id": "r", "result_id": f"res{i}", "sample_id": f"s{i}",
                "status": "success", "total_tokens": 5} for i in range(5)]
    judgments = [{"run_id": "r", "result_id": f"res{i}", "sample_id": f"s{i}",
                  "evaluator_name": "classification_accuracy", "evaluator_type": "rule_based",
                  "evaluation_status": "success", "normalized_score": 1.0 if p == g else 0.0,
                  "is_correct": p == g, "parsed_judgment": {"predicted": p, "reference": g},
                  "provider_metadata": {"primary_metric": True}}
                 for i, (p, g) in enumerate(labels)]
    _write_run(tmp_path, samples, results, judgments)

    s = summarize_run(tmp_path)
    assert abs(s.metrics.score - 0.6) < 1e-9                   # accuracy 3/5
    metric_fields = s.metrics.model_dump()
    assert "macro_f1" not in metric_fields
    assert "per_class_f1" not in metric_fields
    assert "classification_accuracy" in s.metrics_by_evaluator
    assert s.metrics_by_evaluator["classification_accuracy"]["accuracy"] == 0.6


def test_summary_judge_primary_beats_secondary(tmp_path):
    """A judge-primary task with a rule-based secondary: the headline must stay the judge's."""
    samples = [{"run_id": "r", "sample_id": f"s{i}", "benchmark_name": "RareBench",
                "answer_format": "free_text"} for i in range(4)]
    results = [{"run_id": "r", "result_id": f"res{i}", "sample_id": f"s{i}",
                "status": "success", "total_tokens": 5} for i in range(4)]
    judge_correct = [True, True, False, True]                  # judge score = 3/4
    f1_vals = [1.0, 0.5, 0.0, 0.8]                             # secondary differs
    judgments = []
    for i in range(4):
        judgments.append({"run_id": "r", "result_id": f"res{i}", "sample_id": f"s{i}",
                          "evaluator_name": "llm_judge", "evaluator_type": "llm_judge",
                          "evaluation_status": "success",
                          "normalized_score": 1.0 if judge_correct[i] else 0.0,
                          "is_correct": judge_correct[i],
                          "provider_metadata": {"primary_metric": True}})
        judgments.append({"run_id": "r", "result_id": f"res{i}", "sample_id": f"s{i}",
                          "evaluator_name": "text_f1_em", "evaluator_type": "rule_based",
                          "evaluation_status": "success", "normalized_score": f1_vals[i],
                          "is_correct": f1_vals[i] == 1.0,
                          "parsed_judgment": {"predicted": "x", "reference": "y"}})
    _write_run(tmp_path, samples, results, judgments)

    s = summarize_run(tmp_path)
    assert abs(s.metrics.score - 0.75) < 1e-9                  # judge, not the F1 mean (0.575)
    assert s.metrics_by_evaluator["llm_judge"]["mean_score"] == 0.75
    assert s.metrics_by_evaluator["text_f1_em"]["n"] == 4


def test_summary_honors_sample_weights(tmp_path):
    samples = [
        {"run_id": "r", "sample_id": "s1", "benchmark_name": "B", "sample_weight": 1.0},
        {"run_id": "r", "sample_id": "s2", "benchmark_name": "B", "sample_weight": 9.0},
    ]
    results = [
        {"run_id": "r", "result_id": "r1", "sample_id": "s1", "status": "success"},
        {"run_id": "r", "result_id": "r2", "sample_id": "s2", "status": "success"},
    ]
    judgments = [
        {"run_id": "r", "result_id": "r1", "sample_id": "s1",
         "evaluator_name": "multiple_choice_accuracy", "evaluator_type": "rule_based",
         "evaluation_status": "success", "normalized_score": 1.0, "is_correct": True,
         "provider_metadata": {"primary_metric": True}},
        {"run_id": "r", "result_id": "r2", "sample_id": "s2",
         "evaluator_name": "multiple_choice_accuracy", "evaluator_type": "rule_based",
         "evaluation_status": "success", "normalized_score": 0.0, "is_correct": False,
         "provider_metadata": {"primary_metric": True}},
    ]
    _write_run(tmp_path, samples, results, judgments)
    summary = summarize_run(tmp_path)
    assert summary.metrics.score == 0.1
    assert summary.metrics.sample_weight_sum == 10.0
    assert summary.metrics.confidence_interval_method.startswith("weighted_case_bootstrap")
    assert summary.metrics_by_evaluator["multiple_choice_accuracy"]["mean_score"] == 0.1


def test_summary_clusters_repeated_responses_for_confidence_interval(tmp_path):
    samples = [
        {"run_id": "r", "sample_id": "s1", "benchmark_name": "B"},
        {"run_id": "r", "sample_id": "s2", "benchmark_name": "B"},
    ]
    outcomes = [("s1", 0, 1.0), ("s1", 1, 0.0), ("s2", 0, 1.0), ("s2", 1, 1.0)]
    results = [
        {"run_id": "r", "result_id": f"r{i}", "sample_id": sample_id,
         "sample_repeat_index": repeat_index, "status": "success"}
        for i, (sample_id, repeat_index, _) in enumerate(outcomes)
    ]
    judgments = [
        {"run_id": "r", "result_id": f"r{i}", "sample_id": sample_id,
         "evaluator_name": "multiple_choice_accuracy", "evaluator_type": "rule_based",
         "evaluation_status": "success", "normalized_score": score, "is_correct": bool(score),
         "provider_metadata": {"primary_metric": True}}
        for i, (sample_id, _, score) in enumerate(outcomes)
    ]
    _write_run(tmp_path, samples, results, judgments)

    summary = summarize_run(tmp_path)

    assert summary.metrics.score == 0.75
    assert summary.metrics.confidence_interval_method == (
        "clustered_sample_bootstrap_95_seed_20260721"
    )
