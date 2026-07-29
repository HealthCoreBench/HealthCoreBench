"""Cross-task result artifacts for a batch evaluation.

JSON and CSV retain one detailed row per task. Markdown separates tasks into homogeneous metric
tables so every table contains only a benchmark key and the metrics applicable to its rows.

Every table also discloses what its scores exclude — failed, truncated and unparsed samples —
and the Markdown carries a header stating how many configured tasks are represented, the score
denominator policy, and which confidence-interval estimators the ``95% CI`` column mixes.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from healthcorebench.utils.jsonl import atomic_write_json


def _overlap_note(task_key: str) -> str | None:
    """The registry's content-overlap note for a task, if it has one.

    Imported lazily: the registry pulls in the adapter modules, and the aggregation layer is used
    on its own (``batch-report`` over existing summaries) where that cost buys nothing.
    """
    try:
        from healthcorebench.benchmarks.registry import get_registry

        entry = get_registry().get(task_key)
    except Exception:
        return None
    return getattr(entry, "overlap_note", None) or None


_EXPORT_COLUMNS = [
    ("task_key", "Benchmark"),
    ("status", "Status"),
    ("samples", "OK/Total"),
    ("scored", "Scored/Total"),
    ("errors", "Err I/P/E/L/M"),
    ("num_total", "Total"),
    ("num_failed", "Failed"),
    ("num_context_truncated", "Truncated"),
    ("num_parsing_errors", "Unparsed"),
    ("num_evaluation_skipped", "Evaluation Skipped"),
    ("num_unscorable", "Unscorable"),
    ("unscorable_reasons_display", "Unscorable Reasons"),
    ("overlap_note", "Content Overlap"),
    ("num_source_records_dropped", "Source Records Dropped"),
    ("source_record_drop_reasons_display", "Source Drop Reasons"),
    ("num_max_length", "Max Length"),
    ("num_missing_scoring", "Missing Scoring"),
    ("score_coverage", "Score Coverage"),
    ("effective_max_tokens_display", "Effective Max Tokens"),
    ("adaptive_recovered_results", "Adaptive Recovered"),
    ("score_denominator_policy", "Score Denominator Policy"),
    ("sample_weight_sum", "Sample Weight Sum"),
    ("confidence_interval_lower", "95% CI Lower"),
    ("confidence_interval_upper", "95% CI Upper"),
    ("confidence_interval_method", "95% CI Method"),
    ("bleu_scale", "BLEU Scale"),
    ("primary_evaluator", "Primary Method"),
    ("score_display", "Score Display"),
    ("primary_score", "Core Score"),
    ("vlm_primary_score", "VLM Primary Score"),
    ("accuracy", "Acc"),
    ("judge_score", "Judge"),
    ("exact_match", "EM"),
    ("token_f1", "Token-F1"),
    ("rouge_1_precision", "ROUGE-1 Precision"),
    ("rouge_1_recall", "ROUGE-1 Recall"),
    ("rouge_1_f1", "ROUGE-1 F1"),
    ("rouge_2_precision", "ROUGE-2 Precision"),
    ("rouge_2_recall", "ROUGE-2 Recall"),
    ("rouge_2_f1", "ROUGE-2 F1"),
    ("rouge_l_precision", "ROUGE-L Precision"),
    ("rouge_l_recall", "ROUGE-L Recall"),
    ("rouge_l_f1", "ROUGE-L F1"),
    ("bleu_1", "BLEU-1 (%)"),
    ("bleu_2", "BLEU-2 (%)"),
    ("bleu_3", "BLEU-3 (%)"),
    ("bleu_4", "BLEU-4 (%)"),
    ("numeric_score", "Numeric"),
    ("likert_score", "Likert"),
    ("multilabel_subset_accuracy", "Multilabel Subset Accuracy"),
    ("multilabel_precision", "Multilabel Sample Precision"),
    ("multilabel_recall", "Multilabel Sample Recall"),
    ("multilabel_f1", "Multilabel Sample F1"),
    ("multilabel_hamming_loss", "Multilabel Hamming Loss"),
    ("document_field_pair_em", "Document Field Pair EM"),
    ("document_field_precision", "Document Field Precision"),
    ("document_field_recall", "Document Field Recall"),
    ("document_field_f1", "Document Field F1"),
]

_ROUGE_COLUMNS = (
    ("rouge_1_precision", "ROUGE-1 Precision"),
    ("rouge_1_recall", "ROUGE-1 Recall"),
    ("rouge_1_f1", "ROUGE-1 F1"),
    ("rouge_2_precision", "ROUGE-2 Precision"),
    ("rouge_2_recall", "ROUGE-2 Recall"),
    ("rouge_2_f1", "ROUGE-2 F1"),
    ("rouge_l_precision", "ROUGE-L Precision"),
    ("rouge_l_recall", "ROUGE-L Recall"),
    ("rouge_l_f1", "ROUGE-L F1"),
)
_BLEU_COLUMNS = (
    ("bleu_1", "BLEU-1 (%)"),
    ("bleu_2", "BLEU-2 (%)"),
    ("bleu_3", "BLEU-3 (%)"),
    ("bleu_4", "BLEU-4 (%)"),
)
_VLM_CLASSIFICATION_COLUMNS = (
    ("vlm_accuracy", "Accuracy"),
    ("vlm_precision_macro_label", "Macro Precision"),
    ("vlm_recall_macro_label", "Macro Recall"),
    ("vlm_f1_macro_label", "Macro F1"),
    ("vlm_precision_micro_label", "Micro Precision"),
    ("vlm_recall_micro_label", "Micro Recall"),
    ("vlm_f1_micro_label", "Micro F1"),
    ("vlm_precision_weighted_label", "Weighted Precision"),
    ("vlm_recall_weighted_label", "Weighted Recall"),
    ("vlm_f1_weighted_label", "Weighted F1"),
    ("vlm_invalid_output_rate", "Invalid Output Rate"),
)
_VLM_MULTILABEL_COLUMNS = (
    ("vlm_subset_accuracy", "Subset Accuracy"),
    ("vlm_hamming_loss", "Hamming Loss"),
    ("vlm_precision_macro_label", "Macro Precision"),
    ("vlm_recall_macro_label", "Macro Recall"),
    ("vlm_f1_macro_label", "Macro F1"),
    ("vlm_precision_micro_label", "Micro Precision"),
    ("vlm_recall_micro_label", "Micro Recall"),
    ("vlm_f1_micro_label", "Micro F1"),
    ("vlm_precision_weighted_label", "Weighted Precision"),
    ("vlm_recall_weighted_label", "Weighted Recall"),
    ("vlm_f1_weighted_label", "Weighted F1"),
    ("vlm_positive_recall", "Positive Recall"),
    ("vlm_negative_recall", "Negative Recall"),
)
_VLM_GROUNDING_COLUMNS = (
    # These come from the grounding *profile* aggregation, which is a different estimator over a
    # different unit than the headline per-sample grounding score. Labels say so explicitly so
    # neither can be read as the other.
    ("vlm_mean_iou", "Mean IoU (profile)"), ("vlm_median_iou", "Median IoU (profile)"),
    ("vlm_precision_iou_0_25", "Precision@0.25"),
    ("vlm_recall_iou_0_25", "Recall@0.25"),
    ("vlm_precision_iou_0_5", "Precision@0.50"),
    ("vlm_recall_iou_0_5", "Recall@0.50"),
    ("vlm_precision_iou_0_75", "Precision@0.75"),
    ("vlm_recall_iou_0_75", "Recall@0.75"),
    # "Phrase EM" used to sit here. MS-CXR hands the query phrase to the model and asks it to
    # repeat the string back, so the column scored recitation, not grounding: a stub echoing
    # the prompt with the box [0,0,1,1] reads 1.000. It is a format-compliance diagnostic and
    # is kept in the metric payload as vlm_phrase_prompt_echo_rate, out of the main table.
    ("vlm_grounding_label_f1", "Label F1"),
    ("vlm_joint_accuracy", "Recall@0.50 (phrase intact)"),
)
_VLM_CASE_COLUMNS = (
    ("vlm_question_accuracy", "Question Accuracy"),
    ("vlm_stage_f1_macro", "Stage Macro F1"),
    ("vlm_stage_f1_micro", "Stage Micro F1"),
    ("vlm_stage_f1_weighted", "Stage Weighted F1"),
    ("vlm_case_mean_accuracy", "Case Mean Accuracy"),
    ("vlm_case_median_accuracy", "Case Median Accuracy"),
    ("vlm_case_all_correct_rate", "Case All-Correct Rate"),
    ("vlm_case_success_rate", "Case Success Rate"),
    ("vlm_invalid_output_rate", "Invalid Output Rate"),
)
_VLM_DOCUMENT_COLUMNS = (
    ("vlm_field_name_exact_match", "Field Name EM"),
    ("vlm_field_value_exact_match", "Field Value EM"),
    # Distinct from the evaluator's "Field Pair EM": same quantity, different aggregation.
    ("vlm_field_pair_exact_match", "Field Pair EM (profile)"),
    ("vlm_precision_micro_field", "Micro Field Precision"),
    ("vlm_recall_micro_field", "Micro Field Recall"),
    ("vlm_f1_micro_field", "Micro Field F1"),
    ("vlm_precision_macro_field", "Macro Field Precision"),
    ("vlm_recall_macro_field", "Macro Field Recall"),
    ("vlm_f1_macro_field", "Macro Field F1"),
    ("vlm_missing_field_rate", "Missing Field Rate"),
    ("vlm_invented_field_rate", "Invented Field Rate"),
)
_VLM_JUDGE_COLUMNS = (
    ("judge_score", "Judge Overall"),
    ("vlm_judge_semantic_equivalence", "Semantic Equivalence"),
    ("vlm_judge_factual_correctness", "Factual Correctness"),
    ("vlm_judge_clinical_coverage", "Clinical Coverage"),
    ("vlm_judge_reasoning_quality", "Reasoning Quality"),
    ("vlm_judge_clinical_safety", "Clinical Safety"),
    ("vlm_judge_critical_hallucination_rate", "Critical Hallucination Rate"),
    ("vlm_judge_critical_omission_rate", "Critical Omission Rate"),
    ("vlm_judge_unsupported_claim_rate", "Unsupported Claim Rate"),
)
_EVALUATOR_METRICS = {
    "multiple_choice": ("Accuracy", (("accuracy", "Accuracy"),)),
    "classification": ("Accuracy", (("accuracy", "Accuracy"),)),
    "multiple_answer": ("Set Match", (("accuracy", "Set Match"),)),
    "exact_match": ("Exact Match", (("accuracy", "Exact Match"),)),
    "llm_judge": ("Judge", (("judge_score", "Judge Score"),)),
    "text_f1_em": (
        "Token-F1 / EM",
        (("exact_match", "Exact Match"), ("token_f1", "Token-F1")),
    ),
    "rouge": ("ROUGE", _ROUGE_COLUMNS),
    "bleu": ("BLEU", _BLEU_COLUMNS),
    "numeric_tolerance": ("Numeric Tolerance", (("numeric_score", "Numeric Score"),)),
    "likert_credit": ("Likert Credit", (("likert_score", "Likert Score"),)),
    "vlm_text_overlap": (
        "VLM Text Overlap",
        (
            ("vlm_em_raw", "EM Raw"), ("vlm_em_normalized", "EM Normalized"),
            ("vlm_token_precision", "Token Precision"),
            ("vlm_token_recall", "Token Recall"), ("vlm_token_f1", "Token F1"),
            ("vlm_bleu_1", "BLEU-1 (%)"), ("vlm_bleu_2", "BLEU-2 (%)"),
            ("vlm_bleu_3", "BLEU-3 (%)"), ("vlm_bleu_4", "BLEU-4 (%)"),
            ("vlm_rouge_1", "ROUGE-1"), ("vlm_rouge_2", "ROUGE-2"),
            ("vlm_rouge_l", "ROUGE-L"),
        ),
    ),
    "multilabel": (
        "Multilabel",
        (
            ("multilabel_subset_accuracy", "Subset Accuracy"),
            ("multilabel_precision", "Sample Precision"),
            ("multilabel_recall", "Sample Recall"),
            ("multilabel_f1", "Sample F1"),
            ("multilabel_hamming_loss", "Hamming Loss"),
        ),
    ),
    # ``vlm_primary_score`` is the headline ``metrics.score``: the mean per-sample evaluator
    # score whose 95% CI is the one printed. It is kept distinct from the profile aggregates
    # below, which summarize the same task along other units.
    "multistage_choice": (
        "VLM Case/Stage", (("vlm_primary_score", "Stage Accuracy (primary score)"),),
    ),
    "grounding": (
        "VLM Grounding", (("vlm_primary_score", "Grounding Score (primary score)"),),
    ),
    "document_fields": (
        "Document Fields",
        (
            ("document_field_pair_em", "Field Pair EM"),
            ("document_field_precision", "Field Precision"),
            ("document_field_recall", "Field Recall"),
            ("document_field_f1", "Field F1"),
        ),
    ),
    "any_of": ("Any Accepted Answer", (("accuracy", "Accuracy"),)),
}
_JUDGMENT_EVALUATOR_NAMES = {
    "multiple_choice_accuracy": "multiple_choice",
    "classification_accuracy": "classification",
    "multiple_answer_set_match": "multiple_answer",
    "exact_match": "exact_match",
    "llm_judge": "llm_judge",
    "text_f1_em": "text_f1_em",
    "rouge": "rouge",
    "bleu": "bleu",
    "numeric_tolerance": "numeric_tolerance",
    "likert_credit": "likert_credit",
    "vlm_text_overlap": "vlm_text_overlap",
    "vlm_multilabel": "multilabel",
    "multilabel": "multilabel",
    "vlm_multistage_choice": "multistage_choice",
    "vlm_grounding": "grounding",
    "vlm_document_fields": "document_fields",
    "any_of_match": "any_of",
}
_FAMILY_ORDER = {
    "Accuracy": 0,
    "Set Match": 1,
    "Exact Match": 2,
    "Judge": 3,
    "Token-F1 / EM": 4,
    "ROUGE": 5,
    "BLEU": 6,
    "Numeric Tolerance": 7,
    "Likert Credit": 8,
    "Any Accepted Answer": 9,
    "Multilabel": 10,
    "Document Fields": 11,
    "VLM Classification": 20,
    "VLM Multilabel": 21,
    "VLM Grounding": 22,
    "VLM Case/Stage": 23,
    "VLM Document Fields": 24,
}
_GROUP_ORDER = {
    "Accuracy": 0,
    "Set Match": 1,
    "Exact Match": 2,
    "Token-F1 / EM": 3,
    "Judge": 4,
    "Judge + Token-F1 / EM": 5,
    "Judge + ROUGE": 6,
    "ROUGE": 7,
    "ROUGE + BLEU": 8,
    "BLEU": 9,
    "Numeric Tolerance": 10,
    "Likert Credit": 11,
    "Any Accepted Answer": 12,
    "Multilabel": 13,
    "Document Fields": 14,
}
# A task losing at least this fraction of its samples to inference failures is reported as
# ``heavily_failed``: its score is computed over a minority of the benchmark and must not be
# read as a benchmark result. 20% is the point at which the surviving subset stops being
# representative of the configured sample selection.
_HEAVY_FAILURE_FRACTION = 0.2


def _read_json(path: Path) -> dict:
    """Read a run-directory JSON file, treating an absent file as empty.

    A task directory can hold a manifest without a summary (killed before aggregation), and a
    rebuilt report must disclose that task rather than fail on it.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean(metrics: dict, evaluator: str) -> float | None:
    value = (metrics.get(evaluator) or {}).get("mean_score")
    return float(value) if isinstance(value, (int, float)) else None


def _format_counter(counter: Any) -> str:
    """Flatten a ``{reason: count}`` map into one CSV-safe cell, largest bucket first.

    The structured map still reaches JSON consumers; this is only so the CSV and Markdown
    views can show *why* records went missing without a reader having to open summary.json.
    """
    if not isinstance(counter, dict) or not counter:
        return "-"
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return "; ".join(f"{key}={value}" for key, value in items)


def _subscore(metrics: dict, evaluator: str, name: str) -> float | None:
    value = ((metrics.get(evaluator) or {}).get("subscores") or {}).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _nested_metric(payload: dict, *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def build_batch_result_rows(
    run_dirs: list[str | Path],
    *,
    configured_task_keys: list[str] | None = None,
    skipped_task_reasons: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build one normalized display/export row per benchmark task.

    ``configured_task_keys`` names every task the batch was configured to run. Any configured
    key without a row of its own is emitted as ``not_run`` so an interrupted batch reports its
    missing tasks instead of silently shrinking to the ones that finished.

    ``skipped_task_reasons`` maps a configured key to why the runner declined to start it, so
    a deliberate skip is distinguishable from a task the batch never reached.
    """
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        manifest = _read_json(run_dir / "manifest.json")
        summary = _read_json(run_dir / "summary.json")
        task_key = manifest.get("benchmark", {}).get("registry_key") or summary.get("benchmark_name", "")
        benchmark, _, task = task_key.partition("/")
        if not summary:
            # Started but never aggregated: report the task with no metrics rather than
            # dropping it or raising on the missing summary.
            rows.append(_placeholder_row(
                task_key or run_dir.name, status="no_summary",
                run_status=manifest.get("run_status", "unknown"), run_dir=run_dir,
            ))
            continue
        counts = summary.get("counts") or {}

        evaluators = summary.get("metrics_by_evaluator") or {}
        evaluation = (manifest.get("full_config", {}).get("evaluation") or {})
        vlm_profile = ((summary.get("vlm_profile_metrics") or {}).get("task_profile"))
        vlm_metrics = summary.get("vlm_profile_metrics") or {}
        vlm_classification = vlm_metrics.get("classification") or {}
        vlm_multilabel = vlm_metrics.get("multilabel") or {}
        vlm_grounding = vlm_metrics.get("grounding") or {}
        vlm_cases = vlm_metrics.get("case_and_stage") or {}
        vlm_documents = vlm_metrics.get("document_fields") or {}
        vlm_judge = vlm_metrics.get("llm_judge") or {}
        # Keep export semantics aligned with RunOrchestrator._resolve_evaluation. Short-answer
        # and deterministic document-field tasks retain their automatic metric as the headline;
        # their judge result is an auxiliary semantic view.
        judge_as_primary = vlm_profile not in {
            "short_open", "document_qa", "document_parse",
        }
        primary = ("llm_judge" if evaluation.get("use_llm_judge") and judge_as_primary
                   else evaluation.get("evaluator"))
        metrics_summary = summary.get("metrics") or {}
        tokens_summary = summary.get("tokens") or {}
        num_total = counts.get("num_total", 0)
        successful = counts.get("num_successful", 0)
        scored = counts.get("num_scored", 0)
        failed = counts.get("num_failed", 0)
        # Coverage is measured against every configured sample, not only the ones that came
        # back: a task whose inferences mostly failed has the same scored/successful ratio as a
        # clean task, so comparing against num_total is what makes the gap visible.
        incomplete_scoring = scored < successful or scored < num_total
        failed_fraction = (failed / num_total) if num_total else 0.0
        heavily_failed = failed_fraction >= _HEAVY_FAILURE_FRACTION
        primary_score = metrics_summary.get("score")
        interval = metrics_summary.get("confidence_interval") or []
        run_status = manifest.get("run_status", "unknown")
        display_status = (
            "heavily_failed" if heavily_failed
            else "incomplete_scoring" if incomplete_scoring
            else run_status
        )


        accuracy = None
        for name in ("multiple_choice_accuracy", "classification_accuracy",
                     "multiple_answer_set_match", "exact_match", "any_of_match"):
            if name in evaluators:
                accuracy = (evaluators[name] or {}).get("accuracy")
                if accuracy is None:
                    accuracy = (evaluators[name] or {}).get("mean_score")
                break

        rouge_l_f1 = _subscore(evaluators, "rouge", "rougeL.fmeasure")
        if rouge_l_f1 is None:
            rouge_l_f1 = _mean(evaluators, "rouge")
        row = {
            "benchmark": benchmark,
            "task": task or "-",
            "task_key": task_key,
            "status": display_status,
            "run_status": run_status,
            "run_dir": str(run_dir.resolve()),
            "has_summary": True,
            # Recorded on the redundant task by the registry. Carried into the row because a
            # reader comparing two scores cannot otherwise tell that they are largely the same
            # questions asked twice -- MMedBench is 61% MedQA_USMLE/MCMLE/FrenchMedMCQA.
            "overlap_note": _overlap_note(task_key),
            "num_total": counts.get("num_total", 0),
            "num_logical_responses": counts.get("num_logical_responses") or counts.get("num_total", 0),
            "num_successful": counts.get("num_successful", 0),
            "num_failed": counts.get("num_failed", 0),
            "num_scored": counts.get("num_scored", 0),
            "num_parsing_errors": counts.get("num_parsing_errors", 0),
            "num_evaluation_errors": counts.get("num_evaluation_errors", 0),
            "num_evaluation_skipped": counts.get("num_evaluation_skipped", 0),
            "num_unscorable": counts.get("num_unscorable", 0),
            "unscorable_reasons": counts.get("unscorable_reasons") or {},
            "unscorable_reasons_display": _format_counter(counts.get("unscorable_reasons")),
            "num_context_truncated": counts.get("num_context_truncated", 0),
            "num_source_records_dropped": counts.get("num_source_records_dropped"),
            "source_record_drop_reasons": counts.get("source_record_drop_reasons") or {},
            "source_record_drop_reasons_display": _format_counter(
                counts.get("source_record_drop_reasons")
            ),
            "num_max_length": counts.get("num_max_length", 0),
            "num_missing_scoring": counts.get("num_missing_scoring", 0),
            "samples": (f"{counts.get('num_successful', 0)}/"
                        f"{counts.get('num_logical_responses') or counts.get('num_total', 0)}"),
            # Coverage of the whole task, not of the responses that happened to arrive.
            "scored": f"{scored}/{num_total}",

            "errors": (f"{counts.get('num_failed', 0)}/"
                       f"{counts.get('num_parsing_errors', 0)}/"
                       f"{counts.get('num_evaluation_errors', 0)}/"
                       f"{counts.get('num_max_length', 0)}/"
                       f"{counts.get('num_missing_scoring', 0)}"),
            "score_denominator_policy": metrics_summary.get("score_denominator_policy"),
            "sample_weight_sum": metrics_summary.get("sample_weight_sum"),
            "confidence_interval": interval or None,
            "confidence_interval_lower": interval[0] if len(interval) == 2 else None,
            "confidence_interval_upper": interval[1] if len(interval) == 2 else None,
            "confidence_interval_method": metrics_summary.get("confidence_interval_method"),
            "bleu_scale": "0-100" if "bleu" in evaluators or "vlm_text_overlap" in evaluators else None,
            "incomplete_scoring": incomplete_scoring,
            "heavily_failed": heavily_failed,
            "failed_fraction": round(failed_fraction, 6),

            "primary_evaluator": primary or "unknown",
            "extra_evaluators": list(evaluation.get("extra_evaluators") or []),
            "primary_score": primary_score,
            # A partial score remains informative when its coverage is displayed alongside it.
            # Reserve N/A for runs with no successfully scored responses at all.
            "score_display": primary_score if scored > 0 and primary_score is not None else "N/A",
            # The coverage the score was actually taken over. ``score_display`` stays a bare
            # number so it remains machine-comparable; this is the companion that says how much
            # of the task it speaks for, and it sorts, so a 0.33-coverage task is findable
            # instead of depending on a reader cross-checking the Scored column by eye.
            "score_coverage": round(scored / num_total, 6) if num_total else None,
            # The token ladder is only auditable if the budgets it actually settled on are
            # reported: a task that silently finished at 512 tokens is not comparable with one
            # that ran the whole way at 8192.
            "effective_max_tokens_distribution": (
                tokens_summary.get("effective_max_tokens_distribution") or {}
            ),
            "effective_max_tokens_display": _format_counter(
                tokens_summary.get("effective_max_tokens_distribution")
            ),
            "adaptive_recovered_results": tokens_summary.get("adaptive_recovered_results", 0),
            "accuracy": accuracy,
            "judge_score": _mean(evaluators, "llm_judge"),
            "exact_match": _subscore(evaluators, "text_f1_em", "em"),
            "token_f1": _subscore(evaluators, "text_f1_em", "f1"),
            "rouge_1_precision": _subscore(evaluators, "rouge", "rouge1.precision"),
            "rouge_1_recall": _subscore(evaluators, "rouge", "rouge1.recall"),
            "rouge_1_f1": _subscore(evaluators, "rouge", "rouge1.fmeasure"),
            "rouge_2_precision": _subscore(evaluators, "rouge", "rouge2.precision"),
            "rouge_2_recall": _subscore(evaluators, "rouge", "rouge2.recall"),
            "rouge_2_f1": _subscore(evaluators, "rouge", "rouge2.fmeasure"),
            "rouge_l_precision": _subscore(evaluators, "rouge", "rougeL.precision"),
            "rouge_l_recall": _subscore(evaluators, "rouge", "rougeL.recall"),
            "rouge_l_f1": rouge_l_f1,
            # BLEU-N uses the conventional 0..100 display scale.
            "bleu_1": _subscore(evaluators, "bleu", "bleu1"),
            "bleu_2": _subscore(evaluators, "bleu", "bleu2"),
            "bleu_3": _subscore(evaluators, "bleu", "bleu3"),
            "bleu_4": _subscore(evaluators, "bleu", "bleu4"),
            "numeric_score": _mean(evaluators, "numeric_tolerance"),
            "likert_score": _mean(evaluators, "likert_credit"),
            "multilabel_subset_accuracy": _subscore(
                evaluators, "vlm_multilabel", "subset_accuracy"
            ),
            "multilabel_precision": _subscore(
                evaluators, "vlm_multilabel", "precision_sample"
            ),
            "multilabel_recall": _subscore(
                evaluators, "vlm_multilabel", "recall_sample"
            ),
            "multilabel_f1": _subscore(evaluators, "vlm_multilabel", "f1_sample"),
            "multilabel_hamming_loss": _subscore(
                evaluators, "vlm_multilabel", "hamming_loss"
            ),
            "document_field_pair_em": _subscore(
                evaluators, "vlm_document_fields", "field_pair_exact_match"
            ),
            "document_field_precision": _subscore(
                evaluators, "vlm_document_fields", "precision_field"
            ),
            "document_field_recall": _subscore(
                evaluators, "vlm_document_fields", "recall_field"
            ),
            "document_field_f1": _subscore(
                evaluators, "vlm_document_fields", "f1_field"
            ),
            "vlm_primary_score": next((
                _mean(evaluators, name) for name in (
                    "vlm_multilabel", "vlm_multistage_choice", "vlm_grounding",
                    "vlm_document_fields",
                ) if _mean(evaluators, name) is not None
            ), metrics_summary.get("score") if primary in {"multilabel", "document_fields"} else None),
            "vlm_em_raw": _subscore(evaluators, "vlm_text_overlap", "exact_match_raw"),
            "vlm_em_normalized": _subscore(
                evaluators, "vlm_text_overlap", "exact_match_normalized"
            ),
            "vlm_token_precision": _subscore(evaluators, "vlm_text_overlap", "precision_token"),
            "vlm_token_recall": _subscore(evaluators, "vlm_text_overlap", "recall_token"),
            "vlm_token_f1": _subscore(evaluators, "vlm_text_overlap", "f1_token"),
            "vlm_bleu_1": _subscore(evaluators, "vlm_text_overlap", "bleu1"),
            "vlm_bleu_2": _subscore(evaluators, "vlm_text_overlap", "bleu2"),
            "vlm_bleu_3": _subscore(evaluators, "vlm_text_overlap", "bleu3"),
            "vlm_bleu_4": _subscore(evaluators, "vlm_text_overlap", "bleu4"),
            "vlm_rouge_1": _subscore(
                evaluators, "vlm_text_overlap", "rouge1.fmeasure"
            ),
            "vlm_rouge_2": _subscore(
                evaluators, "vlm_text_overlap", "rouge2.fmeasure"
            ),
            "vlm_rouge_l": _subscore(
                evaluators, "vlm_text_overlap", "rougeL.fmeasure"
            ),
            "vlm_profile_metrics": vlm_metrics,
            "vlm_accuracy": _nested_metric(vlm_classification, "accuracy"),
            "vlm_subset_accuracy": _nested_metric(vlm_multilabel, "subset_accuracy"),
            "vlm_hamming_loss": _nested_metric(vlm_multilabel, "hamming_loss"),
            "vlm_precision_macro_label": _nested_metric(
                vlm_classification or vlm_multilabel, "precision_macro_label"
            ),
            "vlm_recall_macro_label": _nested_metric(
                vlm_classification or vlm_multilabel, "recall_macro_label"
            ),
            "vlm_f1_macro_label": _nested_metric(
                vlm_classification or vlm_multilabel, "f1_macro_label"
            ),
            "vlm_precision_micro_label": _nested_metric(
                vlm_classification or vlm_multilabel, "precision_micro_label"
            ),
            "vlm_recall_micro_label": _nested_metric(
                vlm_classification or vlm_multilabel, "recall_micro_label"
            ),
            "vlm_f1_micro_label": _nested_metric(
                vlm_classification or vlm_multilabel, "f1_micro_label"
            ),
            "vlm_precision_weighted_label": _nested_metric(
                vlm_classification or vlm_multilabel, "precision_weighted_label"
            ),
            "vlm_recall_weighted_label": _nested_metric(
                vlm_classification or vlm_multilabel, "recall_weighted_label"
            ),
            "vlm_f1_weighted_label": _nested_metric(
                vlm_classification or vlm_multilabel, "f1_weighted_label"
            ),
            "vlm_positive_recall": _nested_metric(vlm_multilabel, "positive_recall"),
            "vlm_negative_recall": _nested_metric(vlm_multilabel, "negative_recall"),
            "vlm_invalid_output_rate": _nested_metric(
                vlm_classification or vlm_multilabel or vlm_cases,
                "invalid_output_rate",
            ),
            "vlm_mean_iou": _nested_metric(vlm_grounding, "mean_iou"),
            "vlm_median_iou": _nested_metric(vlm_grounding, "median_iou"),
            "vlm_precision_iou_0_25": _nested_metric(vlm_grounding, "precision_iou_0_25"),
            "vlm_recall_iou_0_25": _nested_metric(vlm_grounding, "recall_iou_0_25"),
            "vlm_precision_iou_0_5": _nested_metric(vlm_grounding, "precision_iou_0_5"),
            "vlm_recall_iou_0_5": _nested_metric(vlm_grounding, "recall_iou_0_5"),
            "vlm_precision_iou_0_75": _nested_metric(vlm_grounding, "precision_iou_0_75"),
            "vlm_recall_iou_0_75": _nested_metric(vlm_grounding, "recall_iou_0_75"),
            "vlm_phrase_prompt_echo_rate": _nested_metric(
                vlm_grounding, "phrase_prompt_echo_rate"
            ),
            "vlm_grounding_label_f1": _nested_metric(vlm_grounding, "f1_label"),
            "vlm_joint_accuracy": _nested_metric(
                vlm_grounding, "phrase_to_box_joint_accuracy_iou_0_5"
            ),
            "vlm_question_accuracy": _nested_metric(vlm_cases, "question_accuracy"),
            "vlm_stage_f1_macro": _nested_metric(vlm_cases, "f1_macro_stage_label"),
            "vlm_stage_f1_micro": _nested_metric(vlm_cases, "f1_micro_stage_label"),
            "vlm_stage_f1_weighted": _nested_metric(vlm_cases, "f1_weighted_stage_label"),
            "vlm_case_mean_accuracy": _nested_metric(vlm_cases, "case_mean_accuracy"),
            "vlm_case_median_accuracy": _nested_metric(vlm_cases, "case_median_accuracy"),
            "vlm_case_all_correct_rate": _nested_metric(vlm_cases, "case_all_correct_rate"),
            "vlm_case_success_rate": _nested_metric(vlm_cases, "case_success_rate"),
            "vlm_field_name_exact_match": _nested_metric(
                vlm_documents, "field_name_exact_match"
            ),
            "vlm_field_value_exact_match": _nested_metric(
                vlm_documents, "field_value_exact_match"
            ),
            "vlm_field_pair_exact_match": _nested_metric(
                vlm_documents, "field_pair_exact_match"
            ),
            "vlm_precision_micro_field": _nested_metric(
                vlm_documents, "precision_micro_field"
            ),
            "vlm_recall_micro_field": _nested_metric(vlm_documents, "recall_micro_field"),
            "vlm_f1_micro_field": _nested_metric(vlm_documents, "f1_micro_field"),
            "vlm_precision_macro_field": _nested_metric(
                vlm_documents, "precision_macro_field"
            ),
            "vlm_recall_macro_field": _nested_metric(vlm_documents, "recall_macro_field"),
            "vlm_f1_macro_field": _nested_metric(vlm_documents, "f1_macro_field"),
            "vlm_missing_field_rate": _nested_metric(
                vlm_documents, "missing_critical_field_rate"
            ),
            "vlm_invented_field_rate": _nested_metric(vlm_documents, "invented_field_rate"),
            "vlm_judge_semantic_equivalence": _nested_metric(
                vlm_judge, "semantic_equivalence"
            ),
            "vlm_judge_factual_correctness": _nested_metric(
                vlm_judge, "factual_correctness"
            ),
            "vlm_judge_clinical_coverage": _nested_metric(vlm_judge, "clinical_coverage"),
            "vlm_judge_reasoning_quality": _nested_metric(vlm_judge, "reasoning_quality"),
            "vlm_judge_clinical_safety": _nested_metric(vlm_judge, "clinical_safety"),
            "vlm_judge_critical_hallucination_rate": _nested_metric(
                vlm_judge, "critical_hallucination_rate"
            ),
            "vlm_judge_critical_omission_rate": _nested_metric(
                vlm_judge, "critical_omission_rate"
            ),
            "vlm_judge_unsupported_claim_rate": _nested_metric(
                vlm_judge, "unsupported_claim_rate"
            ),
            "metrics_by_evaluator": evaluators,
        }
        rows.append(row)
    known_keys = {row["task_key"] for row in rows}
    for task_key in configured_task_keys or []:
        if task_key not in known_keys:
            # A task the runner refused to start (a data-protection refusal, a missing
            # corpus) is reported with the refusal itself, not as a bare ``not_run``: the
            # difference between "the batch was interrupted before reaching this" and "this
            # was deliberately skipped, here is why" is the whole point of the row.
            reason = (skipped_task_reasons or {}).get(task_key)
            rows.append(_placeholder_row(
                task_key,
                status="skipped" if reason else "not_run",
                run_status="skipped" if reason else "not_run",
                note=reason,
            ))
            known_keys.add(task_key)
    return sorted(rows, key=lambda r: (r["benchmark"].lower(), r["task"].lower()))


def _placeholder_row(task_key: str, *, status: str, run_status: str,
                     run_dir: str | Path | None = None,
                     note: str | None = None) -> dict[str, Any]:
    """Row for a task with no summary — configured but unrun, or run but never aggregated."""
    benchmark, _, task = task_key.partition("/")
    return {
        "benchmark": benchmark,
        "task": task or "-",
        "task_key": task_key,
        "status": status,
        "run_status": run_status,
        "note": note,
        "run_dir": str(Path(run_dir).resolve()) if run_dir is not None else None,
        "has_summary": False,
        "overlap_note": _overlap_note(task_key),
        "num_total": 0,
        "num_logical_responses": 0,
        "num_successful": 0,
        "num_failed": 0,
        "num_scored": 0,
        "num_parsing_errors": 0,
        "num_evaluation_errors": 0,
        "num_evaluation_skipped": 0,
        "num_unscorable": 0,
        "unscorable_reasons": {},
        "unscorable_reasons_display": "-",
        "num_context_truncated": 0,
        "num_source_records_dropped": None,
        "source_record_drop_reasons": {},
        "source_record_drop_reasons_display": "-",
        "num_max_length": 0,
        "num_missing_scoring": 0,
        "samples": "0/0",
        "scored": "0/0",
        "errors": "0/0/0/0/0",
        "score_denominator_policy": None,
        "sample_weight_sum": None,
        "confidence_interval": None,
        "confidence_interval_lower": None,
        "confidence_interval_upper": None,
        "confidence_interval_method": None,
        "bleu_scale": None,
        "incomplete_scoring": False,
        "heavily_failed": False,
        "failed_fraction": 0.0,
        "primary_evaluator": "unknown",
        "extra_evaluators": [],
        "primary_score": None,
        "score_display": "N/A",
        "score_coverage": None,
        "effective_max_tokens_distribution": {},
        "effective_max_tokens_display": "-",
        "adaptive_recovered_results": 0,
        "vlm_profile_metrics": {},
        "metrics_by_evaluator": {},
    }


def discover_task_run_dirs(run_root: str | Path) -> list[Path]:
    """Every task directory under ``run_root`` that holds a manifest or a summary.

    Used to rebuild a cross-task report from whatever a batch actually left on disk, including
    tasks whose own aggregation never completed.
    """
    run_root = Path(run_root)
    found = {
        path.parent
        for name in ("summary.json", "manifest.json")
        for path in run_root.rglob(name)
    }
    return sorted(found)



def _format_metric(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        # BLEU conventionally uses 0..100; other normalized metrics use 0..1.
        return f"{value:.2f}" if key.startswith("bleu_") else f"{value:.3f}"
    return str(value)


def _merge_family(
    families: dict[str, tuple[tuple[str, str], ...]],
    family: str,
    columns: tuple[tuple[str, str], ...],
) -> None:
    """Add profile columns to a family without displacing the ones already registered.

    The evaluator that owns a family contributes the headline ``metrics.score`` column; the
    task profile contributes further aggregates under the same heading. Assigning instead of
    merging dropped the headline column, so ``metrics.score`` — and the 95% CI computed from
    it — appeared under no column at all.
    """
    merged = list(families.get(family, ()))
    for column in columns:
        if column not in merged:
            merged.append(column)
    families[family] = tuple(merged)


def _metric_profile(row: dict) -> tuple[str, tuple[tuple[str, str], ...]]:
    evaluator_names = [row.get("primary_evaluator"), *(row.get("extra_evaluators") or [])]
    evaluator_names.extend(
        _JUDGMENT_EVALUATOR_NAMES.get(name)
        for name in (row.get("metrics_by_evaluator") or {})
    )

    families: dict[str, tuple[tuple[str, str], ...]] = {}
    for evaluator_name in evaluator_names:
        spec = _EVALUATOR_METRICS.get(evaluator_name)
        if spec is None:
            continue
        family, evaluator_columns = spec
        families.setdefault(family, evaluator_columns)

    vlm_metrics = row.get("vlm_profile_metrics") or {}
    if vlm_metrics.get("classification"):
        families["VLM Classification"] = _VLM_CLASSIFICATION_COLUMNS
        families.pop("Accuracy", None)
    if vlm_metrics.get("multilabel"):
        _merge_family(families, "VLM Multilabel", _VLM_MULTILABEL_COLUMNS)
    if vlm_metrics.get("grounding"):
        _merge_family(families, "VLM Grounding", _VLM_GROUNDING_COLUMNS)
    if vlm_metrics.get("case_and_stage"):
        _merge_family(families, "VLM Case/Stage", _VLM_CASE_COLUMNS)
    if vlm_metrics.get("document_fields"):
        _merge_family(families, "VLM Document Fields", _VLM_DOCUMENT_COLUMNS)
    if vlm_metrics.get("llm_judge"):
        _merge_family(families, "Judge", _VLM_JUDGE_COLUMNS)

    ordered_families = sorted(
        families,
        key=lambda family: (_FAMILY_ORDER.get(family, 100), family),
    )
    columns: list[tuple[str, str]] = []
    for family in ordered_families:
        evaluator_columns = families[family]
        for column in evaluator_columns:
            if column not in columns:
                columns.append(column)

    if not columns:
        return "Core Score", (("primary_score", "Core Score"),)
    return " + ".join(ordered_families), tuple(columns)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _recoverable_gap(row: dict) -> str | None:
    """Why this row shows no score, when a re-run could still produce one.

    Every cell in a metric column reads ``N/A`` when a task scored nothing, and the reader
    cannot tell from that whether the endpoint was down for ten minutes or whether the task is
    structurally unscorable. The first is worth re-running and the second is not, so the
    distinction belongs in the report rather than in whoever remembers the run.

    Returns ``None`` when the row is fine, or when nothing a re-run does would change it.
    """
    if row.get("status") in {"not_run", "skipped"}:
        return None  # already named in their own section, with their own reason
    if row.get("num_scored", 0) > 0:
        return None
    if not row.get("has_summary"):
        return "ran but produced no summary"

    total = row.get("num_total", 0) or 0
    if total == 0:
        return None  # nothing was ever selected; a re-run changes nothing
    failed = row.get("num_failed", 0) or 0
    missing = row.get("num_missing_scoring", 0) or 0
    errors = row.get("num_evaluation_errors", 0) or 0
    # An unscorable sample gave the evaluator no usable reference, and a truncated one was
    # deliberately excluded from scoring; re-requesting either produces the same outcome.
    if failed or missing or errors:
        def plural(count: int, noun: str) -> str:
            return f"{count} {noun}{'' if count == 1 else 's'}"

        parts = [plural(failed, "inference failure") if failed else "",
                 plural(missing, "unscored result") if missing else "",
                 plural(errors, "evaluation error") if errors else ""]
        return ", ".join(part for part in parts if part)
    return None


def _render_recoverable_gaps(rows: list[dict]) -> list[str]:
    """Name the zero-score tasks a re-run would fill, and give the command that fills them."""
    gaps = {row["task_key"]: reason for row in rows if (reason := _recoverable_gap(row))}
    if not gaps:
        return []
    lines = [
        f"- Tasks scored nothing but are recoverable by re-running: {len(gaps)}. "
        "Re-run the same config with `runtime.resume: true` and `--retry-failed`: completed "
        "samples are skipped, failed ones are re-requested, and results whose judgment never "
        "landed are re-scored without new inference.",
    ]
    lines.extend(f"  - {key}: {reason}" for key, reason in sorted(gaps.items()))
    return lines


def _render_report_header(rows: list[dict]) -> list[str]:
    """Header stating what the tables cover and how their numbers were derived.

    Without it a reader cannot tell how many configured tasks are missing, over which
    denominator a score was taken, or that one ``95% CI`` column mixes estimators.
    """
    reported = [row for row in rows if row.get("has_summary")]
    not_run = [row for row in rows if row.get("status") == "not_run"]
    # Deliberately declined, with the refusal attached — kept apart from ``not_run`` so the
    # header does not read as "the batch never got there" for a task the runner turned down.
    skipped = [row for row in rows if row.get("status") == "skipped"]
    no_summary = [row for row in rows if not row.get("has_summary")
                  and row.get("status") not in {"not_run", "skipped"}]
    policies = sorted({row.get("score_denominator_policy") for row in reported
                       if row.get("score_denominator_policy")})
    methods: dict[str, int] = {}
    for row in reported:
        method = row.get("confidence_interval_method")
        if method:
            methods[method] = methods.get(method, 0) + 1

    lines = ["# Evaluation Results", ""]
    lines.append(f"- Tasks configured in this report: {len(rows)}")
    lines.append(f"- Tasks completed with a summary: {len(reported)}")
    if not_run:
        lines.append(f"- Tasks configured but not run: {len(not_run)} "
                     f"({', '.join(sorted(row['task_key'] for row in not_run))})")
    if skipped:
        lines.append(f"- Tasks declined before running: {len(skipped)} "
                     f"({', '.join(sorted(row['task_key'] for row in skipped))}) "
                     "— see the reason column under Tasks Without Results")
    if no_summary:
        lines.append(f"- Tasks run without a summary: {len(no_summary)} "
                     f"({', '.join(sorted(row['task_key'] for row in no_summary))})")
    lines.append("- Score denominator policy: "
                 + (", ".join(policies) if policies else "unknown"))
    lines.append("- 95% CI method(s): " + (
        ", ".join(f"{method} ({count} task{'s' if count != 1 else ''})"
                  for method, count in sorted(methods.items()))
        if methods else "none"))
    lines.append(f"- Scored is scored/configured samples; a task losing >= "
                 f"{_HEAVY_FAILURE_FRACTION:.0%} of its samples to inference failures is "
                 f"reported as heavily_failed.")
    dropped = {row["task_key"]: row.get("num_source_records_dropped") for row in reported
               if row.get("num_source_records_dropped")}
    if dropped:
        lines.append("- Source records dropped before sampling: " + ", ".join(
            f"{key} ({count})" for key, count in sorted(dropped.items())))
    # Unscorable responses are neither correct nor an error: the sample gave the evaluator
    # nothing to compare against. Naming the tasks here is what stops a score taken over a
    # third of its samples from being read as a score over all of them.
    unscorable = {row["task_key"]: (row.get("num_unscorable"), row.get("score_coverage"))
                  for row in reported if row.get("num_unscorable")}
    if unscorable:
        lines.append("- Responses unscorable for lack of a usable reference: " + ", ".join(
            f"{key} ({count}"
            + (f", score covers {coverage:.0%}" if isinstance(coverage, float) else "")
            + ")"
            for key, (count, coverage) in sorted(unscorable.items())))
    # Two tasks in this report can be largely the same questions. Averaging across them
    # double-weights the shared items, and a per-task comparison reads as two independent
    # measurements when it is one measurement scored twice.
    overlaps = {row["task_key"]: row.get("overlap_note") for row in reported
                if row.get("overlap_note")}
    if overlaps:
        lines.append("- Tasks whose content overlaps another task in this suite:")
        for key, note in sorted(overlaps.items()):
            lines.append(f"  - {key} overlaps {note}")
    lines.extend(_render_recoverable_gaps(rows))
    # Two tasks scored by different versions of the same evaluator are not comparable, and the
    # difference does not show up anywhere in the score columns. Evaluator 1.1 stopped scoring an
    # unparsed answer as a hard zero, which moved IgakuQA from 0.4897 to 0.5234 with the model
    # unchanged -- read as a model improvement if the version is not on the page.
    versions: dict[str, set[str]] = {}
    for row in reported:
        for name, block in (row.get("metrics_by_evaluator") or {}).items():
            if isinstance(block, dict):
                versions.setdefault(name, set()).update(block.get("evaluator_versions") or [])
    if versions:
        lines.append("- Evaluator version(s): " + ", ".join(
            f"{name} {'/'.join(sorted(seen)) or 'unrecorded'}"
            for name, seen in sorted(versions.items())))
        mixed = sorted(name for name, seen in versions.items() if len(seen) > 1)
        if mixed:
            lines.append(f"  - WARNING: {', '.join(mixed)} "
                         f"{'appears' if len(mixed) == 1 else 'appear'} at more than one version "
                         "in this report; those means span two scoring definitions and are not "
                         "comparable. Re-score the older tasks.")
    lines.append("")
    return lines


def _render_not_run_table(rows: list[dict]) -> list[str]:
    lines = ["## Tasks Without Results", ""]
    headers = ["Task", "Status", "Reason", "Run Directory"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in sorted(rows, key=lambda value: value["task_key"].lower()):
        lines.append("| " + " | ".join(_escape_markdown(value) for value in (
            row["task_key"], str(row.get("status") or "unknown"),
            str(row.get("note") or "-"),
            str(row.get("run_dir") or "-"),
        )) + " |")
    lines.append("")
    return lines


def _render_grouped_markdown(rows: list[dict]) -> str:
    scored_rows = [row for row in rows if row.get("has_summary")]
    unreported_rows = [row for row in rows if not row.get("has_summary")]
    grouped: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict]] = {}
    for row in scored_rows:
        title, columns = _metric_profile(row)
        grouped.setdefault((title, columns), []).append(row)

    def group_sort_key(item):
        (title, _), _rows = item
        return (_GROUP_ORDER.get(title, 100), title)

    lines = _render_report_header(rows)
    for (title, columns), group_rows in sorted(grouped.items(), key=group_sort_key):
        headers = [
            "Benchmark",
            *(label for _, label in columns),
            "Status", "Scored", "Max Length", "Missing Scoring",
            # Excluded samples: without these a 2/10 task renders exactly like a 10/10 one.
            "Unscorable", "Failed", "Truncated", "Unparsed", "95% CI",
        ]
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| " + " | ".join(map(_escape_markdown, headers)) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in sorted(group_rows, key=lambda value: value["task_key"].lower()):
            has_scored_results = row.get("num_scored", 0) > 0
            interval = row.get("confidence_interval")
            interval_text = (
                f"[{interval[0]:.3f}, {interval[1]:.3f}]"
                if has_scored_results and isinstance(interval, list) and len(interval) == 2 else "-"
            )
            values = [
                row["task_key"],
                *(_format_metric(key, row.get(key)) if has_scored_results else "N/A"
                  for key, _ in columns),
                str(row.get("status") or "unknown"),
                str(row.get("scored") or "0/0"),
                str(row.get("num_max_length", 0)),
                str(row.get("num_missing_scoring", 0)),
                str(row.get("num_unscorable", 0)),
                str(row.get("num_failed", 0)),
                str(row.get("num_context_truncated", 0)),
                str(row.get("num_parsing_errors", 0)),
                interval_text,
            ]
            lines.append("| " + " | ".join(_escape_markdown(value) for value in values) + " |")
        lines.append("")
    if unreported_rows:
        lines.extend(_render_not_run_table(unreported_rows))
    return "\n".join(lines)


def write_batch_result_files(rows: list[dict], output_dir: str | Path) -> dict[str, str]:
    """Save detailed JSON/CSV plus metric-homogeneous Markdown tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "all_tasks_results.json"
    csv_path = output_dir / "all_tasks_results.csv"
    md_path = output_dir / "all_tasks_results.md"
    atomic_write_json(json_path, {"num_tasks": len(rows), "rows": rows})

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=[key for key, _ in _EXPORT_COLUMNS])
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key, _ in _EXPORT_COLUMNS})
    csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8")
    md_path.write_text(_render_grouped_markdown(rows), encoding="utf-8")
    return {"json": str(json_path.resolve()), "csv": str(csv_path.resolve()),
            "markdown": str(md_path.resolve())}


def batch_output_dir(run_dirs: list[str | Path]) -> Path:
    """Return the nearest shared directory containing all task result directories."""
    resolved = [str(Path(p).resolve()) for p in run_dirs]
    return Path(os.path.commonpath(resolved))
