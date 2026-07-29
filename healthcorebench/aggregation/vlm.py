"""Profile-aware aggregate metrics for multimodal benchmark runs."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _is_unscorable(judgment: dict) -> bool:
    """Did the evaluator decline to score this record?

    ``evaluation_status`` stays ``"success"`` for an unscorable judgment — nothing went wrong at
    evaluation time — so the profile filter in ``aggregate_vlm_profiles`` lets it through. Its
    details deliberately omit the metric families (IoU, field rates, label sets), and every
    aggregator below reads those with a ``0.0`` default, so leaving these rows in the means would
    silently drag them toward zero: a record removed from the headline denominator would still be
    depressing the profile metrics. The ``parse_failed`` counters keep counting them.
    """
    return bool((judgment.get("parsed_judgment") or {}).get("unscorable_reason"))


def _scorable(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not _is_unscorable(row["judgment"])]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _classification(rows: list[dict]) -> dict | None:
    rows = [row for row in rows if row["judgment"].get("evaluator_name") in {
        "multiple_choice_accuracy", "classification_accuracy",
    }]
    if not rows:
        return None
    pairs = []
    parse_failures = 0
    for row in rows:
        parsed = row["judgment"].get("parsed_judgment") or {}
        predicted, reference = parsed.get("predicted"), parsed.get("reference")
        parse_failed = parsed.get("parse_failed") is True
        parse_failures += int(parse_failed)
        if reference is not None:
            metadata = row["sample"].get("metadata") or {}
            allowed = metadata.get("letters") or metadata.get("labels") or []
            pairs.append((predicted, reference, allowed, parse_failed))
    reference_labels = {str(reference) for _, reference, _, _ in pairs}
    allowed_labels = {
        str(label)
        for row in rows
        for label in (
            (row["sample"].get("metadata") or {}).get("letters")
            or (row["sample"].get("metadata") or {}).get("labels")
            or []
        )
    }
    labels = sorted(reference_labels | allowed_labels)
    invalid = sum(
        parse_failed or bool(allowed) and predicted not in allowed
        for predicted, _, allowed, parse_failed in pairs
    )
    per_class = {}
    confusion = {
        reference: {
            predicted: 0 for predicted in labels + ["<other>", "<invalid>"]
        }
        for reference in labels
    }
    weighted = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    total_support = 0
    for label in labels:
        tp = sum(
            predicted == label and reference == label
            for predicted, reference, _, _ in pairs
        )
        fp = sum(
            predicted == label and reference != label
            for predicted, reference, _, _ in pairs
        )
        fn = sum(
            predicted != label and reference == label
            for predicted, reference, _, _ in pairs
        )
        support = sum(reference == label for _, reference, _, _ in pairs)
        scores = _prf(tp, fp, fn)
        per_class[label] = {**scores, "support": support}
        total_support += support
        for metric in weighted:
            weighted[metric] += scores[metric] * support
    for predicted, reference, allowed, parse_failed in pairs:
        predicted_key = str(predicted) if predicted is not None else "<invalid>"
        if predicted_key not in labels:
            predicted_key = (
                "<invalid>"
                if parse_failed or predicted is None or allowed and predicted not in allowed
                else "<other>"
            )
        confusion[str(reference)][predicted_key] += 1
    macro = {
        metric: statistics.fmean(item[metric] for item in per_class.values()) if per_class else 0.0
        for metric in ("precision", "recall", "f1")
    }
    num_correct = sum(
        predicted == reference for predicted, reference, _, _ in pairs
    )
    # Each single-label error contributes one false positive and one false negative,
    # including an unparsable or out-of-vocabulary prediction.
    micro = _prf(num_correct, len(pairs) - num_correct, len(pairs) - num_correct)
    weighted = {
        metric: _safe_div(value, total_support) for metric, value in weighted.items()
    }
    output: dict[str, Any] = {
        "accuracy": _safe_div(num_correct, len(pairs)),
        "exact_match": _safe_div(num_correct, len(pairs)),
        "precision_macro_label": macro["precision"],
        "recall_macro_label": macro["recall"],
        "f1_macro_label": macro["f1"],
        "precision_micro_label": micro["precision"],
        "recall_micro_label": micro["recall"],
        "f1_micro_label": micro["f1"],
        "precision_weighted_label": weighted["precision"],
        "recall_weighted_label": weighted["recall"],
        "f1_weighted_label": weighted["f1"],
        "per_class": per_class,
        "confusion_matrix": confusion,
        "invalid_output_rate": _safe_div(invalid, len(rows)),
        "answer_extraction_failure_rate": _safe_div(parse_failures, len(rows)),
    }
    if set(label.casefold() for label in labels) <= {"yes", "no"} and labels:
        positive = next((label for label in labels if label.casefold() == "yes"), "yes")
        negative = next((label for label in labels if label.casefold() == "no"), "no")
        pos = per_class.get(positive, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        neg = per_class.get(negative, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        output["binary_metrics"] = {
            "balanced_accuracy": (pos["recall"] + neg["recall"]) / 2,
            "sensitivity": pos["recall"],
            "specificity": neg["recall"],
            "positive_predictive_value": pos["precision"],
            "negative_predictive_value": neg["precision"],
            "positive_class_f1": pos["f1"],
            "negative_class_f1": neg["f1"],
        }
    return output


def _multilabel(rows: list[dict]) -> dict | None:
    rows = [row for row in rows if row["judgment"].get("evaluator_name") == "vlm_multilabel"]
    if not rows:
        return None
    labels: set[str] = set()
    sets = []
    parse_failures = 0
    scorable = []
    for row in rows:
        parsed = row["judgment"].get("parsed_judgment") or {}
        parse_failures += int(parsed.get("parse_failed") is True)
        # The failure *rate* keeps the full denominator below; only the label metrics drop the
        # unscorable rows, whose details carry no predicted label set to measure.
        if _is_unscorable(row["judgment"]):
            continue
        scorable.append(row)
        prediction = set(parsed.get("predicted_labels") or [])
        reference = set(parsed.get("reference_labels") or [])
        parse_failed = parsed.get("parse_failed") is True
        labels.update(parsed.get("label_universe") or [])
        labels.update(prediction | reference)
        sets.append((prediction, reference, parse_failed))
    per_label = {}
    totals = Counter()
    for label in sorted(labels):
        tp = sum(label in prediction and label in reference for prediction, reference, _ in sets)
        fp = sum(label in prediction and label not in reference for prediction, reference, _ in sets)
        fn = sum(label not in prediction and label in reference for prediction, reference, _ in sets)
        support = sum(label in reference for _, reference, _ in sets)
        per_label[label] = {**_prf(tp, fp, fn), "support": support}
        totals.update(tp=tp, fp=fp, fn=fn, support=support)
    macro = {
        metric: statistics.fmean(item[metric] for item in per_label.values()) if per_label else 0.0
        for metric in ("precision", "recall", "f1")
    }
    micro = _prf(totals["tp"], totals["fp"], totals["fn"])
    weighted = {
        metric: _safe_div(
            sum(item[metric] * item["support"] for item in per_label.values()), totals["support"]
        ) for metric in ("precision", "recall", "f1")
    }
    explicit_negatives = [
        label
        for row in scorable
        for label in (row["sample"].get("metadata") or {}).get("negative_labels") or []
    ]
    false_positive_negatives = sum(
        label in prediction
        for row, (prediction, _, _) in zip(scorable, sets)
        for label in (row["sample"].get("metadata") or {}).get("negative_labels") or []
    )
    no_finding = per_label.get("No Finding")
    return {
        "subset_accuracy": _mean([
            float(not parse_failed and prediction == reference)
            for prediction, reference, parse_failed in sets
        ]),
        "exact_set_match": _mean([
            float(not parse_failed and prediction == reference)
            for prediction, reference, parse_failed in sets
        ]),
        "hamming_loss": _mean([
            (row["judgment"].get("parsed_judgment") or {}).get("hamming_loss", 0.0)
            for row in scorable
        ]),
        "precision_micro_label": micro["precision"],
        "recall_micro_label": micro["recall"],
        "f1_micro_label": micro["f1"],
        "precision_macro_label": macro["precision"],
        "recall_macro_label": macro["recall"],
        "f1_macro_label": macro["f1"],
        "precision_weighted_label": weighted["precision"],
        "recall_weighted_label": weighted["recall"],
        "f1_weighted_label": weighted["f1"],
        "per_label": per_label,
        "positive_recall": micro["recall"],
        "negative_recall": (
            1.0 - _safe_div(false_positive_negatives, len(explicit_negatives))
            if explicit_negatives else None
        ),
        "no_finding_accuracy": no_finding["recall"] if no_finding else None,
        "no_finding_f1": no_finding["f1"] if no_finding else None,
        "invalid_output_rate": _safe_div(parse_failures, len(rows)),
        "answer_extraction_failure_rate": _safe_div(parse_failures, len(rows)),
        "auroc": None,
        "auprc": None,
        "brier_score": None,
        "ece": None,
        "probability_metrics_availability_reason": "No class probabilities were requested or recorded.",
        "uncertain_label_policy": next((
            (row["judgment"].get("parsed_judgment") or {}).get("uncertain_label_policy")
            for row in rows if (row["judgment"].get("parsed_judgment") or {}).get("uncertain_label_policy")
        ), None),
        "null_label_policy": next((
            (row["judgment"].get("parsed_judgment") or {}).get("null_label_policy")
            for row in rows if (row["judgment"].get("parsed_judgment") or {}).get("null_label_policy")
        ), None),
    }


def _grounding(rows: list[dict]) -> dict | None:
    parsed = [
        row["judgment"].get("parsed_judgment") or {} for row in rows
        if row["judgment"].get("evaluator_name") == "vlm_grounding"
        and not _is_unscorable(row["judgment"])
    ]
    if not parsed:
        return None
    ious = [match.get("iou", 0.0) for item in parsed for match in item.get("matches") or []]
    output = {
        "mean_iou": statistics.fmean(ious) if ious else 0.0,
        "median_iou": statistics.median(ious) if ious else 0.0,
        # Format compliance, not accuracy: the query phrase is supplied in the prompt and the
        # model is asked to repeat it in "label". See evaluators/grounding.py for the stub
        # experiment that scores 1.0 here on 0.0 IoU. Deliberately not a report column.
        # Judgments written before the rename carry the count under ``phrase_exact_match``;
        # the fallback lets this re-aggregate their judgments.jsonl without a migration.
        "phrase_prompt_echo_rate": statistics.fmean(
            item.get("phrase_prompt_echo_rate", item.get("phrase_exact_match", 0.0))
            for item in parsed
        ),
    }
    label_hits = sum(
        match.get("category_match") is True
        for item in parsed for match in item.get("matches") or []
    )
    total_predictions = sum(item.get("num_predictions", 0) for item in parsed)
    total_references = sum(item.get("num_references", 0) for item in parsed)
    label_precision = _safe_div(label_hits, total_predictions)
    label_recall = _safe_div(label_hits, total_references)
    output.update({
        "disease_label_accuracy": _safe_div(label_hits, total_references),
        "precision_label": label_precision,
        "recall_label": label_recall,
        "f1_label": _safe_div(2 * label_precision * label_recall, label_precision + label_recall),
        # The phrase half of this conjunction is free (see phrase_prompt_echo_rate), so in
        # practice this is recall at IoU 0.5 over the regions whose query phrase came back
        # intact. It is not an independent phrase-accuracy signal.
        "phrase_to_box_joint_accuracy_iou_0_5": _safe_div(sum(
            match.get("phrase_match") is True and match.get("iou", 0.0) >= 0.5
            for item in parsed for match in item.get("matches") or []
        ), total_references),
    })
    for threshold in ("0_25", "0_5", "0_75"):
        output[f"precision_iou_{threshold}"] = statistics.fmean(
            item.get(f"precision_iou_{threshold}", 0.0) for item in parsed
        )
        output[f"recall_iou_{threshold}"] = statistics.fmean(
            item.get(f"recall_iou_{threshold}", 0.0) for item in parsed
        )
    output.update({
        "map_iou_0_50": None,
        "map_iou_0_75": None,
        "map_iou_0_50_0_95": None,
        "map_availability_reason": "Predictions do not provide ranked confidence scores.",
    })
    return output


def _cases(rows: list[dict]) -> dict | None:
    rows = [
        row
        for row in rows
        if row["judgment"].get("evaluator_name")
        in {
            "classification_accuracy",
            "multiple_choice_accuracy",
            "vlm_multistage_choice",
        }
    ]
    if not rows:
        return None
    case_scores: dict[str, list[float]] = defaultdict(list)
    stage_scores: dict[str, list[float]] = defaultdict(list)
    stage_pairs: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    parse_failures = 0
    for row in rows:
        metadata = row["sample"].get("metadata") or {}
        case_id = metadata.get("case_id")
        stage = metadata.get("stage_name")
        score = row["judgment"].get("normalized_score")
        if case_id is not None and isinstance(score, (int, float)):
            case_scores[str(case_id)].append(float(score))
        if stage is not None and isinstance(score, (int, float)):
            stage_scores[str(stage)].append(float(score))
        parsed = row["judgment"].get("parsed_judgment") or {}
        parse_failures += int(parsed.get("parse_failed") is True)
        # The failure *rate* keeps the full denominator; an unscorable record contributes no
        # stage pairs, or its absent prediction would be scored as a wrong label at every stage.
        if _is_unscorable(row["judgment"]):
            continue
        if stage is not None and parsed.get("reference") is not None:
            stage_pairs[str(stage)].append((parsed.get("predicted"), parsed.get("reference")))
        for stage_name, correct in zip(
            parsed.get("stage_names") or [], parsed.get("stage_correct") or []
        ):
            stage_scores[str(stage_name)].append(1.0 if correct else 0.0)
        stage_names = parsed.get("stage_names") or []
        predictions = parsed.get("predicted_stages") or []
        references = parsed.get("reference_stages") or []
        for index, (stage_name, reference) in enumerate(zip(stage_names, references)):
            predicted = predictions[index] if index < len(predictions) else None
            stage_pairs[str(stage_name)].append((predicted, reference))
    if not case_scores:
        return None
    means = [statistics.fmean(scores) for scores in case_scores.values()]
    all_stage_pairs = [pair for pairs in stage_pairs.values() for pair in pairs]
    stage_label_metrics = _label_metrics(all_stage_pairs)
    return {
        "case_mean_accuracy": statistics.fmean(means),
        "case_median_accuracy": statistics.median(means),
        "case_all_correct_rate": statistics.fmean(all(score == 1.0 for score in scores) for scores in case_scores.values()),
        "case_success_rate": statistics.fmean(any(score == 1.0 for score in scores) for scores in case_scores.values()),
        "stage_accuracy": {
            stage: statistics.fmean(scores) for stage, scores in sorted(stage_scores.items())
        },
        "stage_metrics": {
            stage: {
                "accuracy": statistics.fmean(stage_scores.get(stage) or [0.0]),
                **_label_metrics(pairs),
            }
            for stage, pairs in sorted(stage_pairs.items())
        },
        "precision_macro_stage_label": stage_label_metrics["precision_macro_label"],
        "recall_macro_stage_label": stage_label_metrics["recall_macro_label"],
        "f1_macro_stage_label": stage_label_metrics["f1_macro_label"],
        "precision_micro_stage_label": stage_label_metrics["precision_micro_label"],
        "recall_micro_stage_label": stage_label_metrics["recall_micro_label"],
        "f1_micro_stage_label": stage_label_metrics["f1_micro_label"],
        "precision_weighted_stage_label": stage_label_metrics["precision_weighted_label"],
        "recall_weighted_stage_label": stage_label_metrics["recall_weighted_label"],
        "f1_weighted_stage_label": stage_label_metrics["f1_weighted_label"],
        "question_accuracy": statistics.fmean(
            score for scores in case_scores.values() for score in scores
        ),
        "question_exact_match": statistics.fmean(
            score == 1.0 for scores in case_scores.values() for score in scores
        ),
        "critical_stage_accuracy": None,
        "critical_stage_failure_rate": None,
        "critical_stage_availability_reason": (
            "The source data does not identify a canonical set of critical stages."
        ),
        "invalid_output_rate": _safe_div(parse_failures, len(rows)),
        "num_cases": len(case_scores),
    }


def _label_metrics(pairs: list[tuple[Any, Any]]) -> dict[str, float]:
    labels = sorted({str(reference) for _, reference in pairs if reference is not None})
    per_label = {}
    total_support = 0
    for label in labels:
        tp = sum(predicted == label and reference == label for predicted, reference in pairs)
        fp = sum(predicted == label and reference != label for predicted, reference in pairs)
        fn = sum(predicted != label and reference == label for predicted, reference in pairs)
        support = sum(reference == label for _, reference in pairs)
        per_label[label] = {**_prf(tp, fp, fn), "support": support}
        total_support += support
    correct = sum(predicted == reference for predicted, reference in pairs)
    errors = len(pairs) - correct
    micro = _prf(correct, errors, errors)
    return {
        "precision_macro_label": statistics.fmean(
            item["precision"] for item in per_label.values()
        ) if per_label else 0.0,
        "recall_macro_label": statistics.fmean(
            item["recall"] for item in per_label.values()
        ) if per_label else 0.0,
        "f1_macro_label": statistics.fmean(
            item["f1"] for item in per_label.values()
        ) if per_label else 0.0,
        "precision_micro_label": micro["precision"],
        "recall_micro_label": micro["recall"],
        "f1_micro_label": micro["f1"],
        "precision_weighted_label": _safe_div(sum(
            item["precision"] * item["support"] for item in per_label.values()
        ), total_support),
        "recall_weighted_label": _safe_div(sum(
            item["recall"] * item["support"] for item in per_label.values()
        ), total_support),
        "f1_weighted_label": _safe_div(sum(
            item["f1"] * item["support"] for item in per_label.values()
        ), total_support),
    }


def _documents(rows: list[dict]) -> dict | None:
    parsed = [
        row["judgment"].get("parsed_judgment") or {} for row in rows
        if row["judgment"].get("evaluator_name") == "vlm_document_fields"
        and not _is_unscorable(row["judgment"])
    ]
    if not parsed:
        return None
    matched = sum(item.get("num_matched_fields", 0) for item in parsed)
    predicted = sum(item.get("num_predicted_fields", 0) for item in parsed)
    reference = sum(item.get("num_reference_fields", 0) for item in parsed)
    precision, recall = _safe_div(matched, predicted), _safe_div(matched, reference)
    return {
        "field_pair_exact_match": statistics.fmean(item.get("field_pair_exact_match", 0.0) for item in parsed),
        "precision_micro_field": precision,
        "recall_micro_field": recall,
        "f1_micro_field": _safe_div(2 * precision * recall, precision + recall),
        "precision_macro_field": statistics.fmean(item.get("precision_field", 0.0) for item in parsed),
        "recall_macro_field": statistics.fmean(item.get("recall_field", 0.0) for item in parsed),
        "f1_macro_field": statistics.fmean(item.get("f1_field", 0.0) for item in parsed),
        "missing_critical_field_rate": statistics.fmean(item.get("missing_field_rate", 0.0) for item in parsed),
        "invented_field_rate": statistics.fmean(item.get("invented_field_rate", 0.0) for item in parsed),
        "field_name_exact_match": statistics.fmean(
            item.get("field_name_exact_match", 0.0) for item in parsed
        ),
        "field_value_exact_match": statistics.fmean(
            item.get("field_value_exact_match", 0.0) for item in parsed
        ),
        "numeric_exact_match": None,
        "numeric_tolerance_accuracy": None,
        "direction_match_accuracy": None,
        "numeric_metrics_availability_reason": (
            "No dataset-specific numeric tolerance, unit, reference-range, or direction "
            "annotation is supplied."
        ),
    }


def _judge(rows: list[dict]) -> dict | None:
    judgments = [
        row["judgment"] for row in rows
        if row["judgment"].get("evaluator_name") == "llm_judge"
    ]
    if not judgments:
        return None

    def mean_field(key: str) -> float | None:
        values = [
            (judgment.get("parsed_judgment") or {}).get(key)
            for judgment in judgments
        ]
        values = [float(value) for value in values if isinstance(value, (int, float))]
        return statistics.fmean(values) if values else None

    scores = [
        float(judgment["normalized_score"]) for judgment in judgments
        if isinstance(judgment.get("normalized_score"), (int, float))
    ]
    return {
        "overall": statistics.fmean(scores) if scores else None,
        "semantic_equivalence": mean_field("semantic_equivalence"),
        "factual_correctness": mean_field("factual_correctness"),
        "clinical_coverage": mean_field("clinical_coverage"),
        "reasoning_quality": mean_field("reasoning_quality"),
        "clinical_safety": mean_field("clinical_safety"),
        "critical_hallucination_rate": mean_field("critical_hallucination"),
        "critical_omission_rate": mean_field("critical_omission"),
        "unsupported_claim_rate": mean_field("unsupported_claim"),
    }


def aggregate_vlm_profiles(samples_by_id: dict, judgments: list[dict], valid_result_ids: set) -> dict | None:
    """Aggregate only effective multimodal judgments; language summaries stay unchanged."""
    effective = []
    for judgment in judgments:
        sample = samples_by_id.get(judgment.get("sample_id")) or {}
        if sample.get("component") != "Multimodal":
            continue
        if judgment.get("evaluation_status") != "success" or judgment.get("result_id") not in valid_result_ids:
            continue
        effective.append({"sample": sample, "judgment": judgment})
    if not effective:
        return None
    profile = next((
        (row["sample"].get("metadata") or {}).get("task_profile") for row in effective
    ), None)
    return {
        "task_profile": profile,
        "classification": _classification(effective),
        "multilabel": _multilabel(effective),
        "grounding": _grounding(effective),
        "case_and_stage": _cases(effective),
        "document_fields": _documents(effective),
        "llm_judge": _judge(effective),
        "medical_semantic_metrics": {
            "precision_entity": None, "recall_entity": None, "f1_entity": None,
            "precision_finding": None, "recall_finding": None, "f1_finding": None,
            "precision_diagnosis": None, "recall_diagnosis": None,
            "f1_diagnosis": None, "f1_anatomy": None, "f1_negation": None,
            "availability_reason": "No validated medical fact extractor is configured.",
        },
        "probability_metrics": {
            "auroc": None, "auprc": None, "brier_score": None, "ece": None,
            "availability_reason": "No class probabilities were requested or recorded.",
        },
    }
