"""Group per-sample scores by metadata dimensions (difficulty, capability, ...)."""

from __future__ import annotations

from collections import defaultdict


def group_scores(rows: list[dict], dimension: str) -> dict:
    """Group ``rows`` (each {group_value, is_correct/score}) by a dimension value.

    ``rows`` are dicts with keys ``value`` (the group key) and ``score`` (float) plus
    ``correct`` (bool|None). Returns ``{value: {"n": int, "score": float, "num_correct": int}}``.
    Rows with a ``None`` group value are bucketed under ``"__unspecified__"``.
    """
    buckets: dict = defaultdict(lambda: {
        "n": 0, "score_sum": 0.0, "weight_sum": 0.0,
        "num_correct": 0, "weighted_correct": 0.0, "num_scored": 0,
    })
    for r in rows:
        key = r.get("value")
        key = key if key is not None else "__unspecified__"
        b = buckets[key]
        b["n"] += 1
        weight = float(r.get("weight", 1.0))
        if weight <= 0:
            raise ValueError("group score weights must be positive")
        score = r.get("score")
        if score is not None:
            b["score_sum"] += score * weight
            b["weight_sum"] += weight
            b["num_scored"] += 1
        if r.get("correct") is True:
            b["num_correct"] += 1
            b["weighted_correct"] += weight

    out = {}
    for key, b in buckets.items():
        score = (b["score_sum"] / b["weight_sum"]) if b["weight_sum"] else None
        out[key] = {"n": b["n"], "num_scored": b["num_scored"],
                    "num_correct": b["num_correct"], "score": score,
                    "weight_sum": b["weight_sum"],
                    "weighted_num_correct": b["weighted_correct"]}
    return out
