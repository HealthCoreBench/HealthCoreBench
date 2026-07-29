"""ROUGE evaluator (ROUGE-1 / ROUGE-2 / ROUGE-L) for summarization-style tasks.

The literature-standard automatic metric for medical summarization (MeQSum, ACI-Bench). We
report all three; ROUGE-L F-measure is the headline (``normalized_score``) and ROUGE-1/2 land
in ``parsed_judgment``. There is no binary notion of correctness, so ``is_correct`` is None.

Tokenization is CJK-aware (jieba for Chinese, word-splitting for Latin) via a custom tokenizer
so Chinese references are not collapsed to a single token. Stemming is off for determinism and
cross-language consistency. Multiple references are supported: the best-matching reference (by
ROUGE-L) is reported.

A prediction of ``None`` is unscorable rather than a 0.0 overlap. ``normalize`` keeps the parsed
answer as-is, so ``None`` means only that the adapter's extractor produced nothing; an empty or
useless *response* still arrives as a string and is scored honestly against the gold. Returning
0.0 under ``evaluation_status="success"`` counted the record in ``num_parsing_errors`` *and*
``num_scored``, so a dead extractor and a model that generates nothing usable summarized
identically. The missing-reference exit is now tagged too: an untagged ``(None, None, None)`` is
booked as ``num_evaluation_errors`` by ``aggregation/summarize.py``, which is the wrong bucket for
a task-definition defect.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators._text_util import reference_candidates, rouge_tokens
from healthcorebench.evaluators.base import BaseEvaluator, unscorable

_scorer = None


class _CjkAwareTokenizer:
    """rouge_score tokenizer protocol: ``.tokenize(text) -> list[str]``."""

    def tokenize(self, text):
        return rouge_tokens(text)


def _get_scorer():
    global _scorer
    if _scorer is None:
        from rouge_score import rouge_scorer  # lazy import
        _scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], tokenizer=_CjkAwareTokenizer()
        )
    return _scorer


class RougeEvaluator(BaseEvaluator):
    evaluator_name = "rouge"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        # keep None so a genuine parse failure is detectable in score(); coercing to "" here
        # would make parse_failed always False.
        return parsed_answer

    def score(self, normalized_answer: Any, sample: dict):
        prediction = "" if normalized_answer is None else str(normalized_answer)
        references = reference_candidates(sample)
        parse_failed = normalized_answer is None
        if not references:
            # Reference defects are reported first: they describe the task, not the response.
            return unscorable("missing_reference", parse_failed=parse_failed, reference=None)
        if parse_failed:
            # ``parse_failed`` is kept in the details so the parsing-error count and the
            # per-benchmark report still see this response as an extraction failure.
            return unscorable("unparsed_answer", parse_failed=True, reference=None,
                              num_references=len(references))

        scorer = _get_scorer()
        best = None
        best_ref = None
        for ref in references:
            s = scorer.score(ref, prediction)  # rouge_score signature is (target, prediction)
            if best is None or s["rougeL"].fmeasure > best["rougeL"].fmeasure:
                best = s
                best_ref = ref

        def _trip(m):
            return {"precision": round(m.precision, 6), "recall": round(m.recall, 6),
                    "fmeasure": round(m.fmeasure, 6)}

        rouge_l = best["rougeL"].fmeasure
        parsed = {
            "rouge1": _trip(best["rouge1"]),
            "rouge2": _trip(best["rouge2"]),
            "rougeL": _trip(best["rougeL"]),
            "reference": best_ref,
            "num_references": len(references),
            "parse_failed": normalized_answer is None,
        }
        # ROUGE has no binary correct/incorrect; headline is ROUGE-L F.
        return round(rouge_l, 6), round(rouge_l, 6), None, parsed
