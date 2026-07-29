"""Ordered stage scoring for multi-step VLM reasoning tasks.

A parsed answer of ``None`` is unscorable rather than a run of wrong stages. ``normalize``
returns ``None`` only when the parsed answer itself is ``None`` (an empty stage list stays
``[]`` and is scored as zero correct stages), so ``None`` means exactly one thing: the adapter's
extractor produced nothing. Folding it into ``prediction = normalized_answer or []`` scored it
0.0 under ``evaluation_status="success"``, which counted the record in ``num_parsing_errors``
*and* ``num_scored`` — a dead extractor and a model that fails every stage summarized
identically. ``parse_failed`` is kept in the details so the parsing-error count is unaffected.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators.base import BaseEvaluator, unscorable


class MultistageChoiceEvaluator(BaseEvaluator):
    evaluator_name = "vlm_multistage_choice"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        if parsed_answer is None:
            return None
        return [str(value).strip().upper() for value in parsed_answer]

    def score(self, normalized_answer: Any, sample: dict):
        reference = sample.get("reference_answer_normalized") or sample.get("reference_answer") or []
        reference = [str(value).strip().upper() for value in reference]
        if normalized_answer is None:
            return unscorable(
                "unparsed_answer",
                predicted_stages=None,
                reference_stages=reference,
                stage_names=(sample.get("metadata") or {}).get("stage_names"),
                parse_failed=True,
            )
        prediction = normalized_answer
        correct = [index < len(prediction) and prediction[index] == value for index, value in enumerate(reference)]
        # Denominator spans both sequences: scoring over ``len(reference)`` alone made surplus
        # predicted stages free, so a 4-stage answer to a 2-stage case still scored 1.0.
        stages = max(len(reference), len(prediction))
        accuracy = sum(correct) / stages if stages else 0.0
        all_correct = bool(reference) and len(prediction) == len(reference) and all(correct)
        return accuracy, accuracy, all_correct, {
            "predicted_stages": prediction,
            "reference_stages": reference,
            "stage_names": sample.get("metadata", {}).get("stage_names"),
            "stage_correct": correct,
            "num_scored_stages": stages,
            "stage_accuracy": round(accuracy, 6),
            "case_all_correct": 1.0 if all_correct else 0.0,
            "parse_failed": False,
        }
