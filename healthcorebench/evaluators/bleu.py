"""BLEU-1/2/3/4 evaluator (sentence-BLEU via sacrebleu) for generated text.

Each BLEU-N is the cumulative sentence BLEU through order N, using sacrebleu's exponential
smoothing and effective order. All four values use the conventional 0..100 BLEU scale in
``parsed_judgment``. ``normalized_score`` is BLEU-4/100 in [0,1], although BLEU is normally a
secondary metric beside ROUGE for summarization tasks. Chinese uses sacrebleu's ``zh``
tokenizer; otherwise the default ``13a``. Multiple references are passed to sacrebleu natively.

A prediction of ``None`` is unscorable rather than a 0.0 BLEU. ``normalize`` keeps the parsed
answer as-is, so ``None`` means only that the adapter's extractor produced nothing; an empty or
useless *response* still arrives as a string and is scored honestly. Returning 0.0 under
``evaluation_status="success"`` counted the record in ``num_parsing_errors`` *and* ``num_scored``,
so a dead extractor and a model that generates nothing usable summarized identically. The
missing-reference exit is tagged too: an untagged ``(None, None, None)`` is booked as
``num_evaluation_errors`` by ``aggregation/summarize.py``, the wrong bucket for a task defect.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators._text_util import has_cjk, reference_candidates
from healthcorebench.evaluators.base import BaseEvaluator, unscorable


class BleuEvaluator(BaseEvaluator):
    evaluator_name = "bleu"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        # keep None so a genuine parse failure is detectable in score().
        return parsed_answer

    def score(self, normalized_answer: Any, sample: dict):
        from sacrebleu.metrics import BLEU  # lazy import

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

        tokenize = "zh" if (has_cjk(prediction) or any(has_cjk(r) for r in references)) else "13a"
        scores = {}
        for order in range(1, 5):
            metric = BLEU(
                tokenize=tokenize,
                smooth_method="exp",
                max_ngram_order=order,
                effective_order=True,
            )
            scores[f"bleu{order}"] = round(
                metric.sentence_score(prediction, references).score,
                4,
            )
        norm = scores["bleu4"] / 100.0
        parsed = {
            **scores,
            "tokenize": tokenize,
            "num_references": len(references),
            "parse_failed": normalized_answer is None,
        }
        return scores["bleu4"], round(norm, 6), None, parsed
