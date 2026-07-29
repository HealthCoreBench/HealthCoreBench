"""Exact-match + token-F1 evaluator for short free-form answers.

For benchmarks whose reference is a short canonical string — an entity, a disease name, a
factoid, an ID — where a rule-based comparison is meaningful and an LLM judge is overkill.
Reports two SQuAD-style scores: **EM** (normalized strings equal) and **token-F1** (bag-of-
token overlap, forgiving of extra words / word order). Both are taken as the best match over
all accepted reference forms (aliases / slash-alternates), so a correct-but-aliased answer is
not marked wrong.

``normalized_score`` / ``raw_score`` carry the token-F1 (the headline number); ``is_correct``
carries EM, so accuracy-style confidence intervals reflect strict exact match while the mean
score reflects partial-credit F1.

A reference that tokenizes to nothing (``"-"``, ``"n/a"``, punctuation only) is unscorable, not
trivially satisfied: with an empty gold token bag there is no overlap to measure, and treating
it as a perfect match credited even an unparsed prediction with 1.0.

A prediction of ``None`` is unscorable too. ``normalize`` passes the parsed answer straight
through, so ``None`` means only that the adapter's extractor produced nothing — an empty reply
still arrives as ``""`` and is scored. Scoring the extraction failure 0.0 under
``evaluation_status="success"`` put the record in ``num_parsing_errors`` *and* ``num_scored``,
so "the extractor never fired" and "the model answered every question wrong" produced
byte-identical summaries. ``parse_failed`` stays in the details so the parsing-error count holds.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators._text_util import (
    normalized_string,
    reference_candidates,
    token_f1,
    word_tokens,
)
from healthcorebench.evaluators.base import BaseEvaluator, unscorable


class TextF1EMEvaluator(BaseEvaluator):
    evaluator_name = "text_f1_em"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        # keep the raw parsed answer; tokenization happens in score against each reference.
        return parsed_answer

    def score(self, normalized_answer: Any, sample: dict):
        references = reference_candidates(sample)
        parse_failed = normalized_answer is None
        pred_norm = normalized_string(normalized_answer) if normalized_answer is not None else ""
        pred_tokens = word_tokens(normalized_answer)

        if not references:
            return unscorable("missing_reference", predicted=pred_norm, references=[],
                              parse_failed=parse_failed)

        scored = [(ref, token_f1(pred_tokens, word_tokens(ref))) for ref in references]
        if all(f1 is None for _, f1 in scored):
            return unscorable("empty_reference", predicted=pred_norm, references=references,
                              parse_failed=parse_failed)

        if parse_failed:
            # Reference defects are reported first (they describe the task, not the response);
            # only once the gold is usable is an unextracted answer the reason we cannot score.
            # ``parse_failed`` is kept in the details so the parsing-error count and the
            # per-benchmark report still see this response as an extraction failure.
            return unscorable("unparsed_answer", predicted=None, references=references,
                              num_references=len(references), parse_failed=True)

        best_em = False
        best_f1 = 0.0
        best_ref = None
        for ref, f1 in scored:
            if pred_norm and pred_norm == normalized_string(ref):
                best_em = True
            if f1 is not None and f1 > best_f1:
                best_f1 = f1
                best_ref = ref

        em_score = 1.0 if best_em else 0.0
        parsed = {
            "predicted": pred_norm,
            "reference": best_ref,
            "num_references": len(references),
            "em": em_score,
            "f1": round(best_f1, 6),
            "parse_failed": parse_failed,
        }
        # headline = token-F1 (partial credit); is_correct = strict EM.
        return round(best_f1, 6), round(best_f1, 6), bool(best_em), parsed
