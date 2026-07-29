"""Deterministic scoring against any accepted short answer.

``score`` used to fold two different failures into one ``parse_failed`` flag: a parsed answer of
``None`` (the adapter's extractor produced nothing) and a blank/whitespace answer (the model
replied, with nothing in it). They are now split. ``None`` is returned through ``unscorable`` so
the record leaves the score denominator — scoring it 0.0 under ``evaluation_status="success"``
counted it in ``num_parsing_errors`` *and* ``num_scored``, making a dead extractor
indistinguishable from a model that never names an accepted term. A blank answer stays a scored
zero: the model did answer, it just named nothing. ``parse_failed`` is kept on both paths so the
parsing-error count is unaffected.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators._text_util import normalized_string, reference_candidates, word_tokens
from healthcorebench.evaluators.base import BaseEvaluator, unscorable


def _contains_token_phrase(prediction: list[str], accepted: list[str]) -> bool:
    """Match a complete accepted phrase, never a substring inside an unrelated word."""
    if not accepted or len(accepted) > len(prediction):
        return False
    width = len(accepted)
    return any(prediction[index:index + width] == accepted
               for index in range(len(prediction) - width + 1))


class AnyOfMatchEvaluator(BaseEvaluator):
    """Give full credit when the answer names at least one accepted term."""

    evaluator_name = "any_of_match"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        return parsed_answer

    def score(self, normalized_answer: Any, sample: dict):
        references = reference_candidates(sample)
        unextracted = normalized_answer is None
        # A present-but-blank answer is still an answer; only ``None`` means "nothing extracted".
        parse_failed = unextracted or not str(normalized_answer).strip()
        if not references:
            # Reference defects are reported first and are now tagged, so summarize.py books them
            # as num_unscorable instead of num_evaluation_errors.
            return unscorable(
                "missing_reference",
                predicted=None if unextracted else normalized_string(normalized_answer),
                references=[],
                matched_reference=None,
                parse_failed=parse_failed,
                reference_missing=True,
            )
        if unextracted:
            # ``parse_failed`` is kept in the details so the parsing-error count and the
            # per-benchmark report still see this response as an extraction failure.
            return unscorable(
                "unparsed_answer",
                predicted=None,
                references=references,
                matched_reference=None,
                parse_failed=True,
                reference_missing=False,
            )

        prediction_tokens = word_tokens(normalized_answer) if not parse_failed else []
        matched = next(
            (
                reference
                for reference in references
                if _contains_token_phrase(prediction_tokens, word_tokens(reference))
            ),
            None,
        )
        is_correct = matched is not None
        score = 1.0 if is_correct else 0.0
        return score, score, is_correct, {
            "predicted": normalized_string(normalized_answer) if not parse_failed else None,
            "references": references,
            "matched_reference": matched,
            "parse_failed": parse_failed,
            "reference_missing": False,
        }
