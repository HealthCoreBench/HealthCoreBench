"""Field-pair extraction metrics for medical document parsing outputs.

Reference and prediction are both reduced to order-independent ``(field, key, value)`` triples.
Two shapes of noise are handled explicitly because both silently corrupted scores:

* A model preamble ("Based on the provided clinical text, here is the extracted information:")
  ends in a colon and used to be counted as an invented field, capping precision (and therefore
  F1) on otherwise perfect extractions. Only a contiguous block of field-shaped lines counts.
* Lab tables key rows by test name and the model's key names rarely match the gold's verbatim
  (``检验项目名称`` vs ``entryname``, ``结果`` vs ``result``), and gold values carry units the model
  omits (``418 U/L`` vs ``418``). Identity keys, value keys and measurement units are canonicalized
  so a correct extraction is not scored 0 for cosmetic key/unit differences.

A reference from which no field pair can be extracted is unscorable rather than trivially
satisfied: with nothing to compare against, the ``precision = 1.0 if not reference`` branch used
to award a full 1.0 to any prediction, including a parse failure.

A prediction of ``None`` is unscorable too, and for a reason the ``empty_reference`` branch does
not cover: with a perfectly good reference, ``_field_pairs(None)`` yields an empty set and the
record scored ``f1=0.0 / exact=False`` under ``evaluation_status="success"``, so it was counted in
``num_parsing_errors`` *and* ``num_scored`` — a dead extractor read exactly like a model that
extracts no correct field. An empty *response* still reaches ``_field_pairs`` as ``""`` and keeps
its scored zero; only ``None`` (nothing extracted at all) leaves the denominator.
"""

from __future__ import annotations

import json
import re
from typing import Any

from healthcorebench.evaluators._text_util import normalized_string
from healthcorebench.evaluators.base import BaseEvaluator, unscorable

# Mirrors ``benchmarks.answer_parsing.final_answer_region``: the answer a model emits after a
# closed reasoning block supersedes the reasoning itself. Reimplemented locally to keep the
# evaluator layer independent of the adapter layer.
_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)
_CODE_FENCE = re.compile(r"\s*```(?:json|markdown|md)?\s*(.*?)\s*```\s*", re.I | re.S)

_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")
_EMPHASIS = re.compile(r"\*\*|__|[*`]")
_SENTENCE_END = re.compile(r"[.!?。！？;；]")
_COMMA = re.compile(r"[,，、]")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Keys naming *which* row a field belongs to (the lab test / field label). Compared in
# canonical form so ``entry_name`` / ``Entry Name`` / ``entryname`` are one key.
_IDENTITY_KEYS = frozenset({
    "entryname", "name", "field", "fieldname", "item", "itemname", "test", "testname",
    "检验项目", "检验项目名称", "项目名称", "项目",
})

# Value keys carry the same meaning across the Chinese and English emissions of a task; fold
# both directions onto one canonical token so gold ``result`` matches predicted ``结果``.
_VALUE_KEY_ALIASES = {
    "结果": "result", "result": "result", "值": "result", "value": "result", "测定值": "result",
    "参考范围": "reference", "参考值": "reference", "reference": "reference",
    "referencerange": "reference", "refrange": "reference", "参考区间": "reference",
    "异常状态": "status", "状态": "status", "status": "status",
    "abnormalstatus": "status", "abnormality": "status", "异常": "status",
    "单位": "unit", "unit": "unit", "units": "unit",
}

# A bare measurement: a number (or numeric range) optionally followed by a unit. The unit is not
# the quantity under test, so it is dropped before comparison.
_MEASUREMENT = re.compile(
    r"\s*(?P<number>[-+]?\d+(?:[.,]\d+)?(?:\s*[-–~〜至]\s*[-+]?\d+(?:[.,]\d+)?)?)"
    r"(?:\s*[A-Za-z%‰µμΩ°/·^\d.*()\[\]\s]+)?\s*"
)


def _answer_region(text: str) -> str:
    """Prefer the answer emitted after a model's closed reasoning block."""
    parts = _THINK_CLOSE.split(text)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def _canonical_key(key: Any) -> str:
    """Punctuation-free lowercase form of a key, for identity/alias lookup."""
    return re.sub(r"[\s_\-]+", "", str(key).strip().lower())


def _value_key(key: Any) -> str:
    canonical = _canonical_key(key)
    return _VALUE_KEY_ALIASES.get(canonical, canonical)


def _comparable_value(value: Any) -> str:
    """Normalized value with any measurement unit removed."""
    text = str(value or "").strip()
    match = _MEASUREMENT.fullmatch(text)
    return normalized_string(match.group("number") if match else text)


def _field_name_like(key: str) -> bool:
    """Is the text left of a colon a field label rather than a sentence?

    Field labels are short and unpunctuated; a preamble sentence ending in a colon is not one.
    """
    if not key or _SENTENCE_END.search(key) or _COMMA.search(key):
        return False
    return len(key) <= 48 and len(_WORD.findall(key)) <= 6


def _split_field_line(line: str) -> tuple[str, str] | None:
    """Split ``key: value`` when the key looks like a field name, else ``None``."""
    if ":" not in line and "：" not in line:
        return None
    key, value = re.split(r"[:：]", line, maxsplit=1)
    key = _EMPHASIS.sub("", _BULLET.sub("", key)).strip()
    return (key, value) if _field_name_like(key) else None


def _json_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


def _json_field_pairs(payload: Any) -> set[tuple[str, str, str]] | None:
    """Convert common structured extraction JSON into order-independent field triples."""
    if isinstance(payload, dict):
        nested = next((
            payload[key]
            for key in ("fields", "items", "results", "abnormalities")
            if isinstance(payload.get(key), list)
        ), None)
        if nested is not None:
            payload = nested
        else:
            return {
                (normalized_string(key), "value", _comparable_value(_json_value(val)))
                for key, val in payload.items()
            }
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        return None

    pairs = set()
    for index, item in enumerate(payload):
        identity_key = next(
            (key for key in item
             if _canonical_key(key) in _IDENTITY_KEYS and item.get(key) not in (None, "")),
            None,
        )
        field_name = normalized_string(
            item.get(identity_key) if identity_key else f"row {index + 1}"
        )
        values = [(key, val) for key, val in item.items() if key != identity_key]
        if not values:
            values = [(identity_key or "value", item.get(identity_key))]
        for key, val in values:
            pairs.add((field_name, _value_key(key), _comparable_value(_json_value(val))))
    return pairs


def _table_field_pairs(text: str) -> set[tuple[str, str, str]] | None:
    """Field triples from a Markdown pipe table, or ``None`` when the text is not one."""
    rows = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
            rows.append(cells)
    if len(rows) < 2:
        return None
    header = rows[0]
    pairs = set()
    for row in rows[1:]:
        field_name = normalized_string(row[0]) if row else ""
        for index, value in enumerate(row[1:], start=1):
            column = _value_key(header[index]) if index < len(header) else f"column {index}"
            pairs.add((field_name, column, _comparable_value(value)))
    return pairs


def _line_field_pairs(text: str) -> set[tuple[str, str, str]]:
    """Field triples from ``key: value`` lines, restricted to one contiguous labelled block.

    Leading prose is skipped and the block ends at the first non-field line, so neither a
    preamble nor a trailing explanation can be counted as an extracted (invented) field.
    """
    pairs: set[tuple[str, str, str]] = set()
    started = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue  # blank lines separate fields within a block; they do not end it
        field = _split_field_line(stripped)
        if field is None:
            if started:
                break  # commentary after the block
            continue  # preamble before the block
        started = True
        key, value = field
        pairs.add((normalized_string(key), "value", _comparable_value(value)))
    return pairs


def _field_pairs(value: Any) -> set[tuple[str, str, str]]:
    text = _answer_region(str(value or "").replace("\\n", "\n"))
    if not text:
        return set()
    fenced = _CODE_FENCE.fullmatch(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    structured = _json_field_pairs(payload)
    if structured is not None:
        return structured
    table = _table_field_pairs(text)
    if table is not None:
        return table
    return _line_field_pairs(text)


class DocumentFieldsEvaluator(BaseEvaluator):
    evaluator_name = "vlm_document_fields"
    evaluator_type = "rule_based"
    evaluator_version = "1.2"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        return parsed_answer

    def score(self, normalized_answer: Any, sample: dict):
        prediction = _field_pairs(normalized_answer)
        reference = _field_pairs(
            sample.get("reference_answer_normalized") or sample.get("reference_answer")
        )
        if not reference:
            return unscorable(
                "empty_reference",
                parse_failed=normalized_answer is None,
                num_predicted_fields=len(prediction),
            )
        if normalized_answer is None:
            # Reference defects stay ahead of this check (they describe the task, not the
            # response). Only ``num_reference_fields`` is reported: emitting the rate family here
            # would let a leaked row pull ``missing_field_rate`` and friends toward 0.
            # ``parse_failed`` is kept so the parsing-error count and the per-benchmark report
            # still see this response as an extraction failure.
            return unscorable(
                "unparsed_answer",
                parse_failed=True,
                num_reference_fields=len(reference),
                num_predicted_fields=None,
            )
        overlap = prediction & reference
        predicted_names = {item[0] for item in prediction}
        reference_names = {item[0] for item in reference}
        predicted_values = {(item[0], item[1]): item[2] for item in prediction}
        reference_values = {(item[0], item[1]): item[2] for item in reference}
        comparable_values = set(predicted_values) | set(reference_values)
        matching_values = sum(
            predicted_values.get(key) == reference_values.get(key) for key in comparable_values
        )
        precision = len(overlap) / len(prediction) if prediction else 0.0
        recall = len(overlap) / len(reference)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact = prediction == reference
        return f1, f1, exact, {
            "field_pair_exact_match": 1.0 if exact else 0.0,
            "field_name_exact_match": 1.0 if predicted_names == reference_names else 0.0,
            "field_value_exact_match": (
                matching_values / len(comparable_values) if comparable_values else 1.0
            ),
            "precision_field": round(precision, 6),
            "recall_field": round(recall, 6),
            "f1_field": round(f1, 6),
            "missing_field_rate": len(reference - prediction) / len(reference),
            "invented_field_rate": len(prediction - reference) / len(prediction) if prediction else 0.0,
            "num_reference_fields": len(reference),
            "num_predicted_fields": len(prediction),
            "num_matched_fields": len(overlap),
            "parse_failed": False,
            "numeric_tolerance_accuracy": None,
            "direction_match_accuracy": None,
            "unavailable_reason": "No dataset-specific numeric tolerance or direction annotation is supplied.",
        }
