"""Complete deterministic text-overlap metrics for free-text VLM outputs.

A reference that tokenizes to nothing (``"-"``, punctuation only) is unscorable rather than
trivially satisfied: with an empty gold token bag there is no overlap to measure, and scoring
two empty bags as a perfect match credited even an unparsed prediction with 1.0.

A prediction of ``None`` is unscorable for the mirror-image reason. ``normalize`` passes the
parsed answer through untouched, so ``None`` means only that the adapter's extractor produced
nothing — an empty *response* still arrives as ``""`` and is scored against the gold. Returning
0.0 under ``evaluation_status="success"`` counted the record in ``num_parsing_errors`` *and*
``num_scored``, so a broken extractor and a model whose captions never overlap the reference
produced byte-identical summaries.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from healthcorebench.evaluators._text_util import (
    normalized_string,
    reference_candidates,
    word_tokens,
)
from healthcorebench.evaluators.base import BaseEvaluator, unscorable
from healthcorebench.evaluators.bleu import BleuEvaluator
from healthcorebench.evaluators.rouge import RougeEvaluator


def _token_scores(prediction: str, reference: str) -> tuple[float, float, float] | None:
    predicted = word_tokens(prediction)
    expected = word_tokens(reference)
    if not expected:
        return None
    if not predicted:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


class VLMTextOverlapEvaluator(BaseEvaluator):
    """EM, token P/R/F1, BLEU-1..4, and ROUGE-1/2/L in one judgment."""

    evaluator_name = "vlm_text_overlap"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        return parsed_answer

    def score(self, normalized_answer: Any, sample: dict):
        prediction = "" if normalized_answer is None else str(normalized_answer)
        references = reference_candidates(sample)
        if not references:
            return unscorable(
                "missing_reference",
                parse_failed=normalized_answer is None,
                availability={"text_overlap": "missing_reference"},
            )

        candidates = [
            (*scores, reference)
            for reference in references
            if (scores := _token_scores(prediction, reference)) is not None
        ]
        if not candidates:
            return unscorable(
                "empty_reference",
                parse_failed=normalized_answer is None,
                availability={"text_overlap": "empty_reference"},
            )
        if normalized_answer is None:
            # Reference defects stay ahead of this check (they describe the task, not the
            # response). ``parse_failed`` is kept in the details so the parsing-error count and
            # the per-benchmark report still see this response as an extraction failure.
            return unscorable(
                "unparsed_answer",
                parse_failed=True,
                num_references=len(references),
                availability={"text_overlap": "unparsed_answer"},
            )
        best_precision, best_recall, best_f1, best_reference = max(
            candidates, key=lambda item: item[2]
        )
        raw_em = any(prediction == str(reference) for reference in references)
        normalized_prediction = normalized_string(prediction)
        normalized_em = any(
            normalized_prediction == normalized_string(reference) for reference in references
        )

        _, _, _, rouge = RougeEvaluator().score(prediction, sample)
        _, _, _, bleu = BleuEvaluator().score(prediction, sample)
        parsed = {
            "exact_match_raw": 1.0 if raw_em else 0.0,
            "exact_match_normalized": 1.0 if normalized_em else 0.0,
            "answer_variant_exact_match": 1.0 if normalized_em else 0.0,
            "precision_token": round(best_precision, 6),
            "recall_token": round(best_recall, 6),
            "f1_token": round(best_f1, 6),
            "bleu1": bleu["bleu1"],
            "bleu2": bleu["bleu2"],
            "bleu3": bleu["bleu3"],
            "bleu4": bleu["bleu4"],
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "reference": best_reference,
            "num_references": len(references),
            "parse_failed": normalized_answer is None,
            "clinical_metrics": {
                "medical_concept_f1": None,
                "finding_f1": None,
                "diagnosis_f1": None,
                "anatomy_f1": None,
                "negation_f1": None,
                "critical_hallucination_rate": None,
                "critical_omission_rate": None,
                "availability_reason": "No validated medical fact extractor is configured.",
            },
        }
        # Token F1 is the deterministic primary score; EM remains the strict correctness flag.
        return round(best_f1, 6), round(best_f1, 6), bool(normalized_em), parsed
