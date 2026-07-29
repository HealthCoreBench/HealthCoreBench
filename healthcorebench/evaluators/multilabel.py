"""Deterministic set scoring for VLM multi-label findings and concepts.

An empty reference label set is reported as *unscorable* rather than trivially satisfied: set
F1 is undefined with no positive gold label, and the degenerate ``precision = 1.0 if not
reference`` branch used to hand a full 1.0 to any prediction (including a parse failure).

A parsed answer of ``None`` is unscorable for the mirror-image reason. ``_label_set`` returns
``None`` only for a ``None`` input — an empty list or empty string both become an empty
``frozenset``, i.e. "the model named no findings", which is a real and scorable answer. Scoring
the extraction failure 0.0 under ``evaluation_status="success"`` counted the record in
``num_parsing_errors`` *and* ``num_scored``, so a dead extractor was indistinguishable from a
model that never names a correct finding. ``parse_failed`` stays in the details so the
parsing-error count is unaffected.
"""

from __future__ import annotations

from typing import Any

from healthcorebench.evaluators.base import BaseEvaluator, unscorable
from healthcorebench.evaluators._text_util import normalized_string


def _label_set(value: Any, label_universe: list[str] | None = None) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        values = str(value).replace(";", ",").split(",")
    canonical = {
        normalized_string(label): str(label).strip()
        for label in label_universe or []
        if normalized_string(label)
    }
    normalized = (normalized_string(item) for item in values)
    return frozenset(canonical.get(item, item) for item in normalized if item)


class MultilabelEvaluator(BaseEvaluator):
    evaluator_name = "vlm_multilabel"
    evaluator_type = "rule_based"
    evaluator_version = "1.1"

    def normalize(self, parsed_answer: Any, sample: dict) -> Any:
        return _label_set(parsed_answer, (sample.get("metadata") or {}).get("label_universe"))

    def score(self, normalized_answer: Any, sample: dict):
        metadata = sample.get("metadata") or {}
        label_universe = metadata.get("label_universe")
        reference = _label_set(
            sample.get("reference_answer_normalized")
            if sample.get("reference_answer_normalized") is not None
            else sample.get("reference_answer"),
            label_universe,
        )
        if reference is None:
            return unscorable("missing_reference", parse_failed=normalized_answer is None)
        parse_failed = normalized_answer is None
        prediction = normalized_answer or frozenset()
        ignored_labels: set[str] = set()
        if "evaluated_labels" in metadata:
            global_labels = set(label_universe or [])
            evaluated_labels = set(metadata.get("evaluated_labels") or [])
            # Known labels marked uncertain/null for this sample are excluded. Preserve
            # out-of-vocabulary predictions so unsupported findings remain false positives.
            ignored_labels = set(prediction) & (global_labels - evaluated_labels)
            prediction = frozenset(
                label for label in prediction
                if label in evaluated_labels or label not in global_labels
            )
            reference = frozenset(label for label in reference if label in evaluated_labels)
        # Checked after per-sample exclusion: a gold set that is empty (either as given, or once
        # uncertain/null labels are removed) carries no positive label to measure against.
        if not reference:
            return unscorable(
                "empty_reference",
                parse_failed=parse_failed,
                predicted_labels=sorted(prediction),
                ignored_unevaluated_labels=sorted(ignored_labels),
            )
        if parse_failed:
            # Reference defects stay ahead of this check (they describe the task, not the
            # response). ``parse_failed`` is kept in the details so the parsing-error count and
            # the per-benchmark report still see this response as an extraction failure.
            return unscorable(
                "unparsed_answer",
                parse_failed=True,
                predicted_labels=None,
                reference_labels=sorted(reference),
                ignored_unevaluated_labels=sorted(ignored_labels),
            )
        true_positive = len(prediction & reference)
        precision = true_positive / len(prediction) if prediction else 0.0
        recall = true_positive / len(reference)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact = prediction == reference
        scoring_universe = (
            metadata.get("evaluated_labels")
            if "evaluated_labels" in metadata
            else label_universe
        )
        # Hamming loss needs a fixed denominator to be comparable across samples. Without a
        # declared universe it would degrade to |ref ∪ pred|, which changes shape per sample
        # and cannot be averaged — report it as unavailable instead of an incomparable number.
        universe = (
            set(scoring_universe) | set(reference) | set(prediction)
            if scoring_universe is not None else None
        )
        hamming = (
            len(prediction ^ reference) / len(universe)
            if universe else None
        )
        parsed = {
            "predicted_labels": sorted(prediction),
            "reference_labels": sorted(reference),
            "label_universe": sorted(universe) if universe is not None else None,
            "ignored_unevaluated_labels": sorted(ignored_labels),
            "subset_accuracy": 1.0 if exact else 0.0,
            "exact_set_match": 1.0 if exact else 0.0,
            "hamming_loss": round(hamming, 6) if hamming is not None else None,
            "hamming_loss_unavailable_reason": (
                None if hamming is not None else "No label_universe supplied for this sample."
            ),
            "precision_sample": round(precision, 6),
            "recall_sample": round(recall, 6),
            "f1_sample": round(f1, 6),
            "parse_failed": False,
            "uncertain_label_policy": metadata.get("uncertain_label_policy"),
            "null_label_policy": metadata.get("null_label_policy"),
        }
        score = round(f1, 6)
        return score, score, exact, parsed
