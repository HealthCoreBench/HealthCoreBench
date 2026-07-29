"""Bounding-box and phrase scoring for medical visual grounding.

A parsed answer of ``None`` is unscorable rather than a zero-IoU miss. ``normalize`` passes the
parsed answer through untouched, so ``None`` means only that the adapter's extractor recovered no
box list at all; an explicitly empty list stays ``[]`` and is scored as "the model localized
nothing", which is a real answer. Folding both into ``predictions = normalized_answer or []``
reported ``mean_iou=0.0`` under ``evaluation_status="success"``, which counted the record in
``num_parsing_errors`` *and* ``num_scored`` — a dead extractor and a model that never hits a box
summarized identically. ``aggregation/vlm.py`` skips unscorable judgments so the same record does
not silently drag ``phrase_prompt_echo_rate`` / ``precision_iou_*`` / ``recall_iou_*`` toward 0.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators._text_util import normalized_string
from healthcorebench.evaluators.base import BaseEvaluator, unscorable


def _box(item: dict) -> list[float] | None:
    value = item.get("bbox_xyxy") or item.get("box") or item.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(number) for number in value]
    except (TypeError, ValueError):
        return None


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


class GroundingEvaluator(BaseEvaluator):
    evaluator_name = "vlm_grounding"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        return parsed_answer

    def score(self, normalized_answer: Any, sample: dict):
        references = sample.get("reference_answer_normalized") or sample.get("reference_answer") or []
        if normalized_answer is None:
            # Only the IoU-agnostic counts are reported: emitting num_predictions / the
            # precision_iou_* family here would let a leaked row pull those means toward 0.
            # ``parse_failed`` is kept so the parsing-error count and the per-benchmark report
            # still see this response as an extraction failure.
            return unscorable(
                "unparsed_answer",
                parse_failed=True,
                num_references=len([
                    item for item in references
                    if isinstance(item, dict) and _box(item) is not None
                ]),
            )
        predictions = normalized_answer
        valid_predictions = [(item, _box(item)) for item in predictions if isinstance(item, dict)]
        valid_predictions = [(item, box) for item, box in valid_predictions if box is not None]
        valid_references = [(item, _box(item)) for item in references if isinstance(item, dict)]
        valid_references = [(item, box) for item, box in valid_references if box is not None]

        used: set[int] = set()
        matches = []
        for reference, reference_box in valid_references:
            phrase = normalized_string(reference.get("label") or "")
            category = normalized_string(reference.get("category") or "")
            candidates = []
            for index, (prediction, prediction_box) in enumerate(valid_predictions):
                if index in used:
                    continue
                predicted_phrase = normalized_string(prediction.get("label") or "")
                predicted_category = normalized_string(prediction.get("category") or "")
                phrase_match = phrase == predicted_phrase if phrase and predicted_phrase else None
                category_match = (
                    category == predicted_category if category and predicted_category else None
                )
                candidates.append((
                    phrase_match is True,
                    category_match is True,
                    _iou(reference_box, prediction_box),
                    -index,
                    index,
                    phrase_match,
                    category_match,
                ))
            if candidates:
                _, _, iou, _, index, phrase_match, category_match = max(candidates)
                used.add(index)
                matches.append({
                    "iou": iou,
                    "phrase_match": phrase_match,
                    "category_match": category_match,
                    "label_match": (
                        category_match if category_match is not None else phrase_match
                    ),
                })
            else:
                matches.append({
                    "iou": 0.0, "phrase_match": False,
                    "category_match": False, "label_match": False,
                })

        mean_iou = sum(item["iou"] for item in matches) / len(matches) if matches else 0.0
        details: dict[str, Any] = {
            "matches": matches,
            "mean_iou": round(mean_iou, 6),
            "num_predictions": len(valid_predictions),
            "num_references": len(valid_references),
            # NOT a measure of grounding ability. MS-CXR is *phrase* grounding: the query
            # phrase is handed to the model in the prompt and it is asked to echo it back in
            # "label", so this counts how often the model copied a string it was given. A stub
            # that returns every prompt phrase with the box [0,0,1,1] scores 1.0 here while
            # scoring 0.0 on mean_iou and on disease_category_accuracy (measured on
            # runs/multimodal_benchmarks_v3/MX-CXR). Read it as a format-compliance check:
            # below 1.0 means the model mangled or dropped query phrases. Use mean_iou,
            # recall_iou_*, and disease_category_accuracy for actual performance.
            "phrase_prompt_echo_rate": (
                sum(item["phrase_match"] is True for item in matches) / len(matches)
                if matches else 0.0
            ),
            "disease_category_accuracy": (
                sum(item["category_match"] is True for item in matches) / len(matches)
                if matches else 0.0
            ),
            "parse_failed": False,
        }
        for threshold in (0.25, 0.5, 0.75):
            hits = sum(item["iou"] >= threshold for item in matches)
            suffix = str(threshold).replace(".", "_")
            details[f"precision_iou_{suffix}"] = hits / len(valid_predictions) if valid_predictions else 0.0
            details[f"recall_iou_{suffix}"] = hits / len(valid_references) if valid_references else 0.0
        details["map_iou_0_50"] = None
        details["map_iou_0_75"] = None
        details["map_iou_0_50_0_95"] = None
        details["map_availability_reason"] = "Predictions do not provide ranked confidence scores."
        return round(mean_iou, 6), round(mean_iou, 6), None, details
