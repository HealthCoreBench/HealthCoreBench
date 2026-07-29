"""Exact-match evaluator for short free-form answers.

Normalizes by lowercasing, stripping, and collapsing whitespace/punctuation edges. Suitable
for close-ended answers where a canonical string match is meaningful.

A reference that normalizes to the empty string is unscorable rather than matchable: stripping
edge punctuation reduces ``"?"`` and ``"."`` alike to ``""``, so a punctuation-only gold matched a
punctuation-only prediction and scored 1.0.

A prediction of ``None`` is likewise unscorable rather than wrong. ``_norm`` returns ``None`` only
for a ``None`` input (an empty or punctuation-only response still yields ``""``), so ``None`` here
means exactly one thing: the adapter's extractor produced no answer. Scoring that 0.0 under
``evaluation_status="success"`` put the record in ``num_parsing_errors`` *and* ``num_scored``, which
made "the extractor never fired" and "the model got every question wrong" produce byte-identical
summaries. ``parse_failed`` stays in the details so the parsing-error count is unaffected.
"""

from __future__ import annotations

import re
from typing import Any

from healthcorebench.evaluators.base import BaseEvaluator, unscorable


def _norm(text: Any) -> str | None:
    if text is None:
        return None
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;!?\"'")
    return s


class ExactMatchEvaluator(BaseEvaluator):
    evaluator_name = "exact_match"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        return _norm(parsed_answer)

    def score(self, normalized_answer: Any, sample: dict):
        reference = _norm(sample.get("reference_answer_normalized") or sample.get("reference_answer"))
        parse_failed = normalized_answer is None
        # Reference defects are reported first: they describe the task, not the response, and
        # keeping their order stable leaves existing ``unscorable_reasons`` histograms comparable.
        if reference is None:
            return unscorable("missing_reference", predicted=normalized_answer, reference=None,
                              parse_failed=parse_failed)
        if not reference:
            return unscorable("empty_reference", predicted=normalized_answer, reference=reference,
                              parse_failed=parse_failed)
        if parse_failed:
            # ``parse_failed`` is kept in the details so the parsing-error count and the
            # per-benchmark report still see this response as an extraction failure.
            return unscorable("unparsed_answer", predicted=None, reference=reference,
                              parse_failed=True)
        is_correct = normalized_answer == reference
        raw = 1.0 if is_correct else 0.0
        return raw, raw, bool(is_correct), {"predicted": normalized_answer,
                                            "reference": reference, "parse_failed": False}
