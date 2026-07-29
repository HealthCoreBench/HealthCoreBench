"""Likert partial-credit evaluator (e.g. Script Concordance Test / SCTPublic).

Script Concordance Tests do not have a single "correct" option. Each question carries an expert
credit distribution over the Likert options (-2, -1, 0, +1, +2): the fraction of the expert panel
that chose each option, normalized so the modal option scores 1.0. The model's score for a question
is the (normalized) credit of the option it picked — a real value in [0, 1], not a 0/1 correctness.

The parsed answer is the chosen Likert integer (or ``None`` if unparseable). The credit vector is
read from ``sample.metadata["credit"]`` as a mapping ``{"-2": w, "-1": w, "0": w, "1": w, "2": w}``.
``raw_score`` and ``normalized_score`` are the earned credit; ``is_correct`` is True when the model
picked an option with the maximum credit (the modal expert answer), else False.

A missing or all-zero credit vector is a broken task definition, not a wrong answer: it is raised
so the base class records ``evaluation_status="error"`` instead of scoring every sample 0.0 under
``evaluation_status="success"``.

``normalize`` distinguishes the two ways an answer can be unusable, because they mean opposite
things. A parsed answer of ``None`` is an *extraction* failure — nothing was recovered from the
response — and is returned through ``unscorable`` so the record leaves the score denominator;
scoring it 0.0 under ``evaluation_status="success"`` counted it in ``num_parsing_errors`` *and*
``num_scored``, making a dead extractor indistinguishable from a model that picks badly every
time. A value that *was* recovered but is not one of the five Likert levels (a stray "7", "yes")
is a model failure: the model answered, off the scale, and keeps its scored 0.0 credit.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators.base import BaseEvaluator, unscorable

_LEVELS = ("-2", "-1", "0", "1", "2")
#: Returned by ``normalize`` for an answer that was recovered but is not a Likert level. It is a
#: sentinel rather than ``None`` so ``score`` can tell "nothing was extracted" (unscorable) from
#: "the model chose something off the scale" (a scored zero). It never reaches a record: the base
#: class serializes only ``score``'s details dict, not the normalized answer.
_OFF_SCALE = object()


class LikertCreditEvaluator(BaseEvaluator):
    evaluator_name = "likert_credit"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        if parsed_answer is None:
            return None
        try:
            v = int(parsed_answer)
        except (TypeError, ValueError):
            return _OFF_SCALE
        return v if v in (-2, -1, 0, 1, 2) else _OFF_SCALE

    def score(self, normalized_answer: Any, sample: dict):
        meta = sample.get("metadata") or {}
        credit = meta.get("credit") or {}
        # normalize the credit vector so the max option is 1.0 (matches official sct_score).
        try:
            weights = {k: float(credit.get(k, 0.0)) for k in _LEVELS}
        except (TypeError, ValueError):
            weights = {k: 0.0 for k in _LEVELS}
        max_w = max(weights.values()) if weights else 0.0
        if max_w <= 0:
            # Without a credit vector every option would earn 0.0 while still reporting
            # evaluation_status="success" — an entire task silently scoring 0. The base class
            # turns this into evaluation_status="error", which is the honest outcome. This stays
            # ahead of the unparsed check: a broken task definition is not scorable either way,
            # and "error" is the louder, more accurate signal.
            raise ValueError(
                "likert_credit requires a positive metadata.credit distribution over "
                f"{_LEVELS}; got {credit!r}"
            )
        weights = {k: v / max_w for k, v in weights.items()}

        if normalized_answer is None:
            # ``credit`` is deliberately omitted: ``summarize._metrics_by_evaluator`` averages
            # every numeric leaf of ``parsed_judgment``, so reporting 0.0 here would put the
            # record back into the credit denominator it was just removed from.
            # ``parse_failed`` is kept so the parsing-error count and the per-benchmark report
            # still see this response as an extraction failure.
            return unscorable("unparsed_answer", predicted=None, parse_failed=True)
        if normalized_answer is _OFF_SCALE:
            # An answer was recovered, it just is not on the Likert scale. The model answered and
            # earns no expert credit, so this remains a scored zero inside the denominator.
            return 0.0, 0.0, False, {"predicted": None, "parse_failed": True, "credit": 0.0,
                                     "off_scale": True}

        key = str(normalized_answer)
        earned = weights.get(key, 0.0)
        # modal option(s): those with normalized credit == 1.0
        is_modal = earned >= 1.0
        parsed = {"predicted": normalized_answer, "credit": earned,
                  "parse_failed": False, "is_modal": is_modal}
        return earned, earned, bool(is_modal), parsed
