"""Multiple-choice accuracy evaluator.

Compares the parsed option letter to the reference letter. A parsed answer of ``None`` (the
parser could not decide) is *not* scored: it is returned through ``unscorable`` so it leaves
the score denominator instead of entering it as a zero. Scoring it 0.0 made a run whose parser
never fired read as "100% scored, accuracy 0.000" — a broken extractor and a model that
answers every question wrong produced byte-identical summaries.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators.base import BaseEvaluator, unscorable


class MultipleChoiceEvaluator(BaseEvaluator):
    evaluator_name = "multiple_choice_accuracy"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        if parsed_answer is None:
            return None
        return str(parsed_answer).strip().upper()

    def score(self, normalized_answer: Any, sample: dict):
        reference = sample.get("reference_answer_normalized") or sample.get("reference_answer")
        reference = str(reference).strip().upper() if reference is not None else None
        if normalized_answer is None:
            # ``parse_failed`` is kept in the details so the parsing-error count and the
            # per-benchmark report still see this response as an extraction failure.
            return unscorable(
                "unparsed_answer", predicted=None, reference=reference, parse_failed=True,
            )
        is_correct = reference is not None and normalized_answer == reference
        raw = 1.0 if is_correct else 0.0
        parsed = {"predicted": normalized_answer, "reference": reference, "parse_failed": False}
        return raw, raw, bool(is_correct), parsed
