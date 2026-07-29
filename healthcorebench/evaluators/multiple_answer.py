"""Multiple-answer (one-or-more correct) evaluator.

For questions where the reference is a *set* of option letters. The parsed answer (a list of
letters, or ``None`` when the parser could not decide) is compared to the reference set with
exact set equality: a prediction is correct only if it names every correct letter and no extra
ones. ``raw_score``/``normalized_score`` are 1.0 on an exact set match, else 0.0.

A parsed answer of ``None`` (the parser could not decide) is *not* scored: it is returned
through ``unscorable`` so it leaves the score denominator instead of entering it as a zero.
Scoring it 0.0 made "the extractor never fired" and "the model got every question wrong"
produce byte-identical summaries; one measured IgakuQA run carried 128 such records as
successful zeros. This evaluator is the one the CMB / CMExam / IgakuQA / MMedBench /
GlobalDentBench / FrenchMedMCQA tasks use, so the conflation was not hypothetical.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators.base import BaseEvaluator, unscorable


def _to_letter_set(value: Any) -> frozenset[str] | None:
    """Coerce a list/str reference-or-prediction into a set of upper-cased letters.

    Returns ``None`` for a ``None`` input. Accepts a list of letters, or a string like
    "B,D" / "B and D" / "BD".
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        items = value
    else:
        # split a string on any non-letter separator, keep single letters.
        import re
        items = re.findall(r"[A-Za-z]+", str(value))
        # "BD" -> ["B","D"]; "B" -> ["B"]; leave multi-char tokens as-is (unexpected).
        expanded: list[str] = []
        for tok in items:
            expanded.extend(list(tok) if len(tok) > 1 else [tok])
        items = expanded
    out = {str(x).strip().upper() for x in items if str(x).strip()}
    return frozenset(out)


class MultipleAnswerEvaluator(BaseEvaluator):
    evaluator_name = "multiple_answer_set_match"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        return _to_letter_set(parsed_answer)

    def score(self, normalized_answer: Any, sample: dict):
        reference = _to_letter_set(
            sample.get("reference_answer_normalized")
            if sample.get("reference_answer_normalized") is not None
            else sample.get("reference_answer")
        )
        if normalized_answer is None:
            # ``parse_failed`` is kept in the details so the parsing-error count and the
            # per-benchmark report still see this response as an extraction failure.
            return unscorable(
                "unparsed_answer",
                predicted=None,
                reference=sorted(reference) if reference is not None else None,
                parse_failed=True,
            )
        is_correct = reference is not None and normalized_answer == reference
        raw = 1.0 if is_correct else 0.0
        parsed = {
            "predicted": sorted(normalized_answer),
            "reference": sorted(reference) if reference is not None else None,
            "parse_failed": False,
        }
        return raw, raw, bool(is_correct), parsed
