"""Base evaluator: normalize a parsed answer and produce a JudgmentRecord.

The three-layer pipeline (parse → normalize → evaluate) is honoured: the adapter already
parsed the raw response into ``parsed_answer``; the evaluator normalizes and scores. An
evaluation error is recorded as a judgment with ``evaluation_status="error"`` — never
silently turned into ``is_correct=False``.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from healthcorebench.schemas.judgment import JudgmentRecord
from healthcorebench.utils.timestamps import utc_now_iso


def unscorable(reason: str, **details: Any) -> tuple[None, None, None, dict]:
    """Build the established "not scorable" signal, tagged with a machine-readable reason.

    ``score()`` returning ``(None, None, None, parsed)`` keeps a sample out of the score
    denominator instead of crediting or penalizing it. The reason is recorded in
    ``parsed_judgment`` so a dropped sample is auditable rather than silently missing.
    """
    return None, None, None, {"unscorable_reason": reason, **details}


class BaseEvaluator(ABC):
    evaluator_name: str = "base"
    evaluator_type: str = "rule_based"
    evaluator_version: str = "1.0"

    @abstractmethod
    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        """Map a parsed answer to a canonical comparable form."""

    @abstractmethod
    def score(self, normalized_answer: Any, sample: dict) -> tuple[float | None, float | None, bool | None, dict]:
        """Return ``(raw_score, normalized_score, is_correct, parsed_judgment)``."""

    def evaluate(self, result: dict, sample: dict) -> JudgmentRecord:
        """Produce a judgment for one result record (dict form)."""
        jid = f"jdg_{uuid.uuid4().hex[:16]}"
        base = dict(
            judgment_id=jid, run_id=result.get("run_id"), result_id=result.get("result_id"),
            sample_id=result.get("sample_id"), evaluator_type=self.evaluator_type,
            evaluator_name=self.evaluator_name, evaluator_version=self.evaluator_version,
            timestamp=utc_now_iso(),
        )
        # Failed inference is not scorable: emit a skipped judgment, never is_correct=False.
        if result.get("status") != "success":
            return JudgmentRecord(**base, evaluation_status="skipped",
                                  evaluation_error="inference_failed", is_correct=None)
        try:
            normalized = self.normalize(result.get("parsed_answer"), sample)
            raw_score, norm_score, is_correct, parsed = self.score(normalized, sample)
            return JudgmentRecord(
                **base, raw_score=raw_score, normalized_score=norm_score, is_correct=is_correct,
                parsed_judgment=parsed, evaluation_status="success",
            )
        except Exception as exc:
            return JudgmentRecord(**base, evaluation_status="error",
                                  evaluation_error=str(exc)[:500], is_correct=None)
