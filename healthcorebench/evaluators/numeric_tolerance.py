"""Numeric-tolerance evaluator (e.g. MedCalc-Bench).

Scores a numeric prediction as correct when it falls within an accepted range. The range is
taken from the sample's ``metadata`` (``lower_limit`` / ``upper_limit``); if no range is given it
falls back to a relative tolerance around ``reference_answer`` (``metadata.rel_tol``, default 5%).

A response that *is* present but contains no recoverable number scores 0 and is flagged
``parse_failed``: the model answered, it just did not answer with a value. A parsed answer of
``None`` is different — the extractor produced nothing — and is returned through ``unscorable`` so
it leaves the score denominator instead of entering it as a zero. Scoring it 0.0 under
``evaluation_status="success"`` counted the record in ``num_parsing_errors`` *and* ``num_scored``,
so a dead extractor and a model that miscalculates every case summarized identically.

The adapter must isolate the final answer before evaluation; this evaluator parses the resulting
numeric token. Date-typed answers (``metadata.output_type == "date"``) normalize calendar dates and
MedCalc's gestational-age tuple format before exact comparison.

Numeric extraction prefers the number following an explicit answer marker and otherwise takes the
last number on the last non-empty line: taking the *first* number anywhere read intermediate
working ("Step 1: 120 mg, so the answer is 5 mg" → 1.0) as the answer. A percentage prediction is
reconciled against a fractional gold ("5%" vs 0.05) and an interval prediction ("1.5-2.0") is
accepted when the gold falls inside it.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, NamedTuple

from healthcorebench.evaluators.base import BaseEvaluator, unscorable

# One complete numeric token. The comma form requires at least one comma, so a plain four-digit
# value cannot be prematurely matched as its first three digits.
_NUM = (
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"|[-+]?\.\d+(?:[eE][-+]?\d+)?"
)
_NUM_RE = re.compile(r"(?<![\w.])(?:" + _NUM + r")")
# A numeric token plus the trailing ``%`` and the ``a - b`` interval form that change its meaning.
_MEASUREMENT_RE = re.compile(
    r"(?<![\w.])(?P<low>" + _NUM + r")\s*(?P<low_pct>%)?"
    r"(?:\s*(?:-|–|to|~)\s*(?P<high>" + _NUM + r")\s*(?P<high_pct>%)?)?"
)
# Explicit final-answer markers. Anchoring on these keeps reasoning digits out of the answer;
# MedCalc's own prompt demands ``Answer: <value>``, which this matches directly.
_ANSWER_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer|result|value|total|score|最终答案|答案|结果|计算结果)"
    r"\s*(?:\*\*)?\s*(?:is|are|=|:|：)\s*",
    re.IGNORECASE,
)


def _float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except (TypeError, ValueError):
        return None


class _Measurement(NamedTuple):
    """A parsed numeric answer: point value, whether it was a percentage, and its interval."""

    value: float
    percent: bool
    low: float
    high: float


def _measurement_at(text: str, search_from_end: bool) -> _Measurement | None:
    matches = list(_MEASUREMENT_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1] if search_from_end else matches[0]
    low = _float(m.group("low"))
    if low is None:
        return None
    high = _float(m.group("high")) if m.group("high") else None
    percent = bool(m.group("low_pct") or m.group("high_pct"))
    if high is None:
        return _Measurement(low, percent, low, low)
    lo, hi = min(low, high), max(low, high)
    # An interval's point value is its midpoint, so relative-tolerance comparison still works.
    return _Measurement((lo + hi) / 2, percent, lo, hi)


def _parse_measurement(value: Any) -> _Measurement | None:
    """Extract the answer measurement, preferring an explicit answer marker.

    Falls back to the last number on the last non-empty line — models put the conclusion last,
    so scanning from the end is far safer than taking the first number in the response.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _Measurement(float(value), False, float(value), float(value))
    text = str(value)
    marker = list(_ANSWER_MARKER_RE.finditer(text))
    if marker:
        tail = text[marker[-1].end():]
        # Only the marker's own line: a following sentence is explanation, not the answer.
        found = _measurement_at(tail.splitlines()[0] if tail.splitlines() else "", False)
        if found is not None:
            return found
        found = _measurement_at(tail, False)
        if found is not None:
            return found
    for line in reversed([line for line in text.splitlines() if line.strip()]):
        found = _measurement_at(line, True)
        if found is not None:
            return found
    return None


def _to_float(value: Any) -> float | None:
    measurement = _parse_measurement(value)
    return measurement.value if measurement else None


def _percent_variants(pred: _Measurement, ref_percent: bool) -> list[_Measurement]:
    """Scale variants to try so ``5%`` reconciles with a gold of ``0.05`` (and back).

    Only the mismatched-notation direction is generated, so an unambiguous pair is never
    loosened into a 100x-wide acceptance window.
    """
    variants = [pred]
    if pred.percent and not ref_percent:
        variants.append(_Measurement(pred.value / 100, True, pred.low / 100, pred.high / 100))
    elif ref_percent and not pred.percent:
        variants.append(_Measurement(pred.value * 100, False, pred.low * 100, pred.high * 100))
    return variants


class NumericToleranceEvaluator(BaseEvaluator):
    evaluator_name = "numeric_tolerance"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        # Keep the raw parsed answer; numeric extraction happens in score() so date-typed
        # samples can compare strings instead.
        return parsed_answer

    def score(self, normalized_answer: Any, sample: dict):
        meta = sample.get("metadata") or {}
        output_type = meta.get("output_type")
        reference = sample.get("reference_answer_normalized")
        if reference is None:
            reference = sample.get("reference_answer")

        if normalized_answer is None:
            # ``normalize`` passes the parsed answer straight through, so ``None`` here means one
            # thing only: the adapter's extractor produced no answer. That is not a wrong
            # calculation, so the record leaves the score denominator. An answer that is present
            # but holds no recoverable number still scores 0.0 below — the model did reply.
            # ``parse_failed`` is kept in the details so the parsing-error count and the
            # per-benchmark report still see this response as an extraction failure.
            return unscorable(
                "unparsed_answer", predicted=None, reference=reference, parse_failed=True,
                mode="date" if output_type == "date" else (
                    "range" if _to_float(meta.get("lower_limit")) is not None else "rel_tol"),
            )

        # Date answers: exact normalized string match.
        if output_type == "date":
            pred = None if normalized_answer is None else str(normalized_answer).strip()
            ref = None if reference is None else str(reference).strip()
            parse_failed = pred is None or pred == ""
            pred_date = _parse_mmddyyyy(pred)
            ref_date = _parse_mmddyyyy(ref)
            pred_gestational = _parse_gestational_age(pred)
            ref_gestational = _parse_gestational_age(ref)
            if pred and re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", pred) and pred_date is None:
                parse_failed = True
            if pred and re.search(r"\b(?:weeks?|days?)\b", pred, re.IGNORECASE) and pred_gestational is None:
                parse_failed = True
            if pred_date is not None and ref_date is not None:
                is_correct = pred_date == ref_date
                normalized_pred = pred_date.strftime("%m/%d/%Y")
                normalized_ref = ref_date.strftime("%m/%d/%Y")
            elif pred_gestational is not None and ref_gestational is not None:
                is_correct = pred_gestational == ref_gestational
                normalized_pred = f"({pred_gestational[0]} weeks, {pred_gestational[1]} days)"
                normalized_ref = f"({ref_gestational[0]} weeks, {ref_gestational[1]} days)"
            else:
                # MedCalc also labels gestational-age tuples as dates. Preserve exact matching
                # for those non-calendar values while validating real MM/DD/YYYY answers.
                is_correct = (not parse_failed) and ref is not None and pred == ref
                normalized_pred, normalized_ref = pred, ref
            raw = 1.0 if is_correct else 0.0
            return raw, raw, bool(is_correct), {"predicted": normalized_pred,
                                                "reference": normalized_ref,
                                                "parse_failed": parse_failed, "mode": "date"}

        pred_measurement = _parse_measurement(normalized_answer)
        pred = pred_measurement.value if pred_measurement else None
        parse_failed = pred is None
        lower = _to_float(meta.get("lower_limit"))
        upper = _to_float(meta.get("upper_limit"))
        ref_measurement = _parse_measurement(reference)
        ref = ref_measurement.value if ref_measurement else None
        ref_percent = bool(ref_measurement and ref_measurement.percent) or "%" in str(
            meta.get("unit") or ""
        )

        def _accepts(candidate: _Measurement) -> bool:
            if lower is not None and upper is not None:
                lo, hi = min(lower, upper), max(lower, upper)
                # An interval prediction is accepted when it overlaps the accepted range.
                return candidate.low <= hi and candidate.high >= lo
            if ref is not None:
                rel_tol = float(meta.get("rel_tol", 0.05))
                margin = abs(ref) * rel_tol
                # Accept a bracketing interval, or a point within tolerance of the gold.
                return (candidate.low - margin <= ref <= candidate.high + margin
                        or abs(candidate.value - ref) <= margin)
            return False

        candidates = _percent_variants(pred_measurement, ref_percent) if pred_measurement else []
        is_correct = any(_accepts(candidate) for candidate in candidates)

        raw = 1.0 if is_correct else 0.0
        parsed = {"predicted": pred, "reference": ref, "lower": lower, "upper": upper,
                  "predicted_interval": (
                      [pred_measurement.low, pred_measurement.high]
                      if pred_measurement and pred_measurement.low != pred_measurement.high
                      else None
                  ),
                  "predicted_is_percent": bool(pred_measurement and pred_measurement.percent),
                  "reference_is_percent": ref_percent,
                  "parse_failed": parse_failed,
                  "mode": "range" if lower is not None else "rel_tol"}
        return raw, raw, bool(is_correct), parsed


def _parse_mmddyyyy(value: str | None) -> datetime | None:
    if value is None or not re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", value.strip()):
        return None
    try:
        return datetime.strptime(value.strip(), "%m/%d/%Y")
    except ValueError:
        return None


def _parse_gestational_age(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(
        r"\(\s*['\"]?(\d+)\s+weeks?['\"]?\s*,\s*['\"]?(\d+)\s+days?['\"]?\s*\)",
        value.strip(),
        re.IGNORECASE,
    )
    return (int(match.group(1)), int(match.group(2))) if match else None
