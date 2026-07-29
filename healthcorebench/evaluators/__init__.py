"""Evaluators: turn a result's parsed answer into scored judgments.

Evaluators are pure scoring — they do not deploy or call the evaluated model (LLM judges do
call a separate judge model via the same client, but with independent config and token
accounting). Each judgment distinguishes ``raw_score`` (benchmark-native), ``normalized_score``
([0,1]) and ``is_correct`` (bool or None).
"""

from healthcorebench.evaluators.base import BaseEvaluator
from healthcorebench.evaluators.multiple_choice import MultipleChoiceEvaluator
from healthcorebench.evaluators.multiple_answer import MultipleAnswerEvaluator
from healthcorebench.evaluators.exact_match import ExactMatchEvaluator
from healthcorebench.evaluators.classification import ClassificationEvaluator
from healthcorebench.evaluators.numeric_tolerance import NumericToleranceEvaluator
from healthcorebench.evaluators.likert_credit import LikertCreditEvaluator
from healthcorebench.evaluators.text_f1_em import TextF1EMEvaluator
from healthcorebench.evaluators.rouge import RougeEvaluator
from healthcorebench.evaluators.bleu import BleuEvaluator
from healthcorebench.evaluators.vlm_text_overlap import VLMTextOverlapEvaluator
from healthcorebench.evaluators.multilabel import MultilabelEvaluator
from healthcorebench.evaluators.multistage_choice import MultistageChoiceEvaluator
from healthcorebench.evaluators.grounding import GroundingEvaluator
from healthcorebench.evaluators.document_fields import DocumentFieldsEvaluator
from healthcorebench.evaluators.any_of import AnyOfMatchEvaluator

EVALUATOR_REGISTRY = {
    "multiple_choice": MultipleChoiceEvaluator,
    "multiple_answer": MultipleAnswerEvaluator,
    "exact_match": ExactMatchEvaluator,
    "classification": ClassificationEvaluator,
    "numeric_tolerance": NumericToleranceEvaluator,
    "likert_credit": LikertCreditEvaluator,
    "text_f1_em": TextF1EMEvaluator,
    "rouge": RougeEvaluator,
    "bleu": BleuEvaluator,
    "vlm_text_overlap": VLMTextOverlapEvaluator,
    "multilabel": MultilabelEvaluator,
    "multistage_choice": MultistageChoiceEvaluator,
    "grounding": GroundingEvaluator,
    "document_fields": DocumentFieldsEvaluator,
    "any_of": AnyOfMatchEvaluator,
}


def get_evaluator(name: str) -> BaseEvaluator:
    if name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Unknown evaluator '{name}'. Known: {sorted(EVALUATOR_REGISTRY)}")
    return EVALUATOR_REGISTRY[name]()


# Maps a sample's declared ``evaluation_metric`` to the rule-based evaluator name. ``llm_judge``
# is intentionally absent: it is not a rule-based evaluator and is handled via the judge path.
_METRIC_TO_EVALUATOR = {
    "accuracy": "multiple_choice",      # single_choice letters OR classification labels (both do exact-match)
    "set_match": "multiple_answer",     # one-or-more correct letters
    "numeric_tolerance": "numeric_tolerance",
    "likert_credit": "likert_credit",
    "exact_match": "exact_match",
    "text_f1": "text_f1_em",            # short free-form answers: EM + token-F1
    "rouge": "rouge",                   # summarization-style overlap
    "bleu": "bleu",
    "vlm_text_overlap": "vlm_text_overlap",
    "multilabel": "multilabel",
    "multistage_choice": "multistage_choice",
    "grounding": "grounding",
    "document_fields": "document_fields",
    "any_of_match": "any_of",
}


def select_evaluator_name(evaluation_metric: str | None, answer_format: str | None = None) -> str | None:
    """Pick the rule-based evaluator for a benchmark from its sample's declared metric.

    Returns the evaluator registry name, or ``None`` when the metric is ``llm_judge`` (scored by
    the judge path, not a rule-based evaluator). ``accuracy`` maps to ``multiple_choice`` for
    lettered choices and to ``classification`` for label / yes-no formats (both are exact-match,
    but ``classification`` lower-cases while ``multiple_choice`` upper-cases — pick per format).

    An unknown metric raises. Silently defaulting to ``multiple_choice`` scored a free-text
    benchmark with upper-cased letter exact-match, collapsing it to ~0 while still reporting
    ``evaluation_status="success"`` — a mis-declared metric (``"f1"``, ``"Accuracy"``) must fail
    loudly rather than produce a plausible-looking wrong number.
    """
    if evaluation_metric in (None, "llm_judge"):
        return None
    if evaluation_metric == "accuracy":
        # ``nli`` is not an answer_format (MedNLI declares ``label``); it is only a prompt
        # template name, so it is deliberately not listed here.
        if answer_format in ("label", "yes_no", "yes_no_maybe"):
            return "classification"
        return "multiple_choice"
    if evaluation_metric not in _METRIC_TO_EVALUATOR:
        raise ValueError(
            f"Unknown evaluation_metric '{evaluation_metric}'. Known: "
            f"{sorted({'accuracy', 'llm_judge', *_METRIC_TO_EVALUATOR})}"
        )
    return _METRIC_TO_EVALUATOR[evaluation_metric]


# Per-task default *secondary* metrics, reported alongside the primary score (never replacing
# it). Keyed by registry task key ("<bench>/<task>"). These are the tasks where the primary
# score stays the LLM judge (long-form QA / diagnosis) or ROUGE (summarization), but a cheap
# rule-based cross-check adds signal: token-F1/EM for short-ish entity answers, ROUGE-L for
# long-form text, BLEU for summaries. Users override via config.evaluation.extra_evaluators.
DEFAULT_EXTRA_EVALUATORS = {
    # Generated clinical text: ROUGE-L is primary; BLEU-1/2/3/4 are secondary metrics.
    "MeQSum/summarization": ["bleu"],
    "ACI-Bench_HF/summarization": ["bleu"],
    "ClinicBench/hospitalization": ["bleu"],
    # diagnosis / short-ish free answers: judge stays primary, token-F1+EM as a rule cross-check.
    # This cross-check is only meaningful when the reference is a short span *and* the prompt gives
    # the answer a locatable home; each adapter below therefore asks for a final "Answer:" line and
    # scores that line rather than the whole reply. Without both halves the metric degenerates into
    # a prediction/reference length ratio (measured 0.02-0.08 across these tasks) that looks like a
    # capability number but is not one.
    "AgentClinic/diagnosis": ["text_f1_em"],
    "RareBench/diagnosis": ["text_f1_em"],
    "VivaBench/diagnosis": ["text_f1_em"],
    "MedCaseReasoning/open": ["text_f1_em"],
    "MedThink-Bench/open": ["text_f1_em"],
    "MedChain/diagnosis": ["text_f1_em"],
    # MedR-Bench/diagnosis deliberately has no rule-based cross-check: its reference is a prose
    # paragraph (measured over 200 cases: median 27.5 words / 220 characters, up to 71 words) and
    # its prompt is MedR-Bench's official ``oracle_diagnose.txt`` verbatim, which asks for a
    # differential and for further tests when data is insufficient. Shortening the answer would
    # break protocol fidelity, and against a paragraph reference token-F1 (like ROUGE-L) measures
    # length overlap, not diagnostic correctness. The LLM judge remains the primary score.
    #
    # GeneTuring needs no prompt change: its prompt already says "Return only the final answer …
    # without explanation" and the recorded replies obey it (5-12 characters: "LMP10",
    # "Chromosome 1"). Short-answer extraction would in fact *corrupt* them — a genomic coordinate
    # like "chrX:12345-99999" reads as a labelled value and would be truncated to "12345-99999".
    "GeneTuring/open": ["text_f1_em"],
    # long-form open QA: judge primary, ROUGE-L as a cheap secondary signal.
    "CareQA/open": ["rouge"],
    "MedicationQA/open": ["rouge"],
    "MedQuAD/open": ["rouge"],
    "LiveQA/open": ["rouge"],
    "webMedQA/open": ["rouge"],
    "LLMEval-Med/open": ["rouge"],
    "RJUA-QA/open": ["rouge"],
    "MedQA-CS/open": ["rouge"],
    "MedS-Bench/task18": ["rouge"],
    "MedS-Bench/task46": ["rouge"],
    "MedS-Bench/task50": ["rouge"],
    "MedS-Bench/task100": ["rouge"],
}


def default_extra_evaluators(task_key: str | None) -> list[str]:
    """Benchmark-default secondary metrics for a task key (empty when none defined)."""
    return list(DEFAULT_EXTRA_EVALUATORS.get(task_key or "", []))


__all__ = [
    "BaseEvaluator",
    "MultipleChoiceEvaluator",
    "MultipleAnswerEvaluator",
    "ExactMatchEvaluator",
    "ClassificationEvaluator",
    "NumericToleranceEvaluator",
    "LikertCreditEvaluator",
    "TextF1EMEvaluator",
    "RougeEvaluator",
    "BleuEvaluator",
    "VLMTextOverlapEvaluator",
    "MultilabelEvaluator",
    "MultistageChoiceEvaluator",
    "GroundingEvaluator",
    "DocumentFieldsEvaluator",
    "AnyOfMatchEvaluator",
    "EVALUATOR_REGISTRY",
    "get_evaluator",
    "select_evaluator_name",
    "DEFAULT_EXTRA_EVALUATORS",
    "default_extra_evaluators",
]
