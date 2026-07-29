"""Contract tests between what a prompt *asks for* and what the scorer *compares against*.

The 302 tests that shipped before this file all passed while `MedDocBench/ltr_abnormality_qa`
scored exactly 0.000 in three consecutive runs, because its prompt asked for JSON with Chinese
keys while its reference used English ones. No unit test can catch that: each half is
self-consistent. What catches it is asserting the *contract* between the two halves.

Three contracts are checked across every registered task:

1. When the answer space is closed and scoring is exact matching, the admissible answers must be
   shown to the model. Otherwise the task is a guessing game (`EHRBench/decision` shipped 8-102
   controlled-vocabulary candidates per record and never put them in the prompt).
2. When the reference is a handful of tokens and scoring is exact-match or token-F1, the prompt
   must ask for a short answer. Otherwise the metric measures output length, not correctness
   (13 VLM tasks and 2 text tasks had `exact_match` identically 0.000 for this reason).
3. When scoring compares document fields, the field names in the reference must be the field
   names the prompt requests.

A task may only be exempted through the allowlists below, and each entry must state why the data
genuinely behaves that way. An exemption is a documented limitation, not a silenced test.
"""

from __future__ import annotations

import itertools
import json
import re

import pytest

from healthcorebench.benchmarks.registry import get_adapter, get_registry
from healthcorebench.evaluators import default_extra_evaluators
from healthcorebench.evaluators._text_util import normalized_string

# Metrics that require the model to reproduce a value exactly (or as a set), so the admissible
# values have to be visible in the prompt.
_CLOSED_SPACE_METRICS = {"accuracy", "set_match", "exact_match", "multilabel", "any_of_match"}
# Metrics scored by string overlap against a short gold, where a verbose answer cannot win.
_BREVITY_METRICS = {"exact_match", "text_f1", "vlm_text_overlap"}
# The same property, named as evaluators rather than metrics. A task can reach a length-sensitive
# metric as a *secondary* evaluator while its primary metric is an LLM judge that does not care
# about length -- seven tasks run ``text_f1_em`` that way. Checking only the primary metric leaves
# those secondary columns free to report ~0 for correct-but-verbose answers.
_BREVITY_EVALUATORS = {"exact_match", "text_f1_em", "vlm_text_overlap"}
_MAX_SHORT_REFERENCE_TOKENS = 5

# Cues that tell the model to answer briefly, in every language the collection prompts in.
_BREVITY_CUES = (
    "no other text", "nothing else", "and no other", "only the", "exactly one",
    "one of the following", "answer with", "return only", "output only",
    "short", "brief", "concise", "specific entity", "single word", "one word",
    "不要", "只输出", "仅输出", "直接输出", "简要", "简短",
    "だけ", "のみ", "簡潔", "短く",
    "만 ", "만.", "간단", "짧",
    "فقط", "دون أي نص", "بإيجاز",
    "seulement", "uniquement", "sans aucun autre", "brièvement",
    "únicamente", "solamente", "sin ningún otro", "brevemente",
    "endast", "kortfattat",
    "только", "кратко",
    "soltanto", "unicamente", "brevemente",
)

# Tasks whose answer space is closed in the data but deliberately not enumerated in the prompt,
# because the task *is* open recall and the closed set is only a scoring aid.
_OPEN_RECALL_BY_DESIGN: dict[str, str] = {
    # RareBench/GeneTuring style tasks: the model must name the disease/gene, and the reference
    # set exists only so synonyms can be accepted. Showing the list would give the answer away.
    "RareBench/diagnosis": "free-text diagnosis; reference aliases are a scoring aid only",
    "GeneTuring/open": "free-text gene/genome answer; the gold list is a scoring aid only",
    "BioHopR/single": "free-text entity recall; accepted answers are a scoring aid only",
    "BioHopR/multi": "free-text entity recall; accepted answers are a scoring aid only",
    "MedS-Bench/task29": "free-text drug/dose extraction; accepted forms are a scoring aid only",
    "3MDBench/diagnosis": "free-text dermatology diagnosis; the 34-label list is a scoring aid",
}

# Reference distributions that really are degenerate in the shipped data. Each of these is a
# property of the source dataset, not a framework defect, and each must stay visible.
_DEGENERATE_BY_DATA: dict[str, str] = {
    "MedS-Bench/task12": "all 650 shipped instances have output ['no'] — a constant predictor "
                         "scores 1.000, so this task cannot discriminate models",
}


def _implemented_task_keys() -> list[str]:
    return sorted(key for key, entry in get_registry().items() if entry.adapter_dotted)


def _load(key: str, limit: int, *, stride: int = 1, window: int | None = None):
    """Yield ``(adapter, sample)`` for records of a task.

    ``source_file_manifest`` is skipped on purpose: it hashes every source file (hundreds of MB
    for some benchmarks) and only affects ``sample_id`` stability, which these tests do not assert.

    ``stride`` samples across the file instead of taking the first N. Several benchmarks ship
    label-sorted records — ``ClinicBench/drug_interaction`` is 100 "yes" followed by 100 "no",
    ``EHRBench/risk`` opens with a long run of "no" — so a first-N sample looks single-class even
    when the dataset is balanced. The stride is prime to avoid aliasing with periodic label blocks.
    """
    adapter = get_adapter(key)
    files = adapter.discover_source_files()
    stop = window if window is not None else limit * stride
    records = itertools.islice(adapter.load_raw_samples(files), 0, stop, stride)
    for index, raw in enumerate(itertools.islice(records, limit)):
        yield adapter, adapter.normalize_sample(raw, index)


def _prompt_text(adapter, sample) -> str:
    """Flatten the built messages to the text the model actually sees."""
    parts: list[str] = []
    for message in adapter.build_messages(sample):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for piece in content:
                if isinstance(piece, dict) and piece.get("type") == "text":
                    parts.append(str(piece.get("text") or ""))
    return "\n".join(parts)


_GENERIC_LABEL_VOCABULARIES = (
    {"yes", "no"}, {"yes", "no", "maybe"}, {"true", "false"},
)
# Above this size a label vocabulary is a scoring aid, not something a prompt can enumerate.
_MAX_ENUMERABLE_LABELS = 20


def _closed_answer_space(sample) -> list[str] | None:
    """The admissible answers a sample declares, when the prompt is obliged to show them.

    ``candidates`` is a per-record controlled vocabulary: the record itself says which values are
    admissible, exact matching is the official metric, and the model cannot possibly guess between
    near-identical options (``EU OBSERVATION`` vs ``DIRECT OBSERVATION`` vs ``OBSERVATION ADMIT``),
    so withholding it makes the task unanswerable.

    A global ``labels`` vocabulary is different. When it is small it is a genuine closed-set
    classification task and the prompt should enumerate it; when it is large, or when it is just
    yes/no, it is a scoring aid for free-text recall and enumerating it would give the answer away.
    """
    metadata = sample.metadata or {}
    candidates = metadata.get("candidates")
    if isinstance(candidates, (list, tuple)) and len(candidates) >= 2:
        return [str(value) for value in candidates]
    if sample.answer_format != "label":
        return None
    for field in ("labels", "label_universe", "accepted_labels"):
        values = metadata.get(field)
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            continue
        if len(values) > _MAX_ENUMERABLE_LABELS:
            return None
        lowered = {str(value).strip().lower() for value in values}
        if any(lowered <= generic for generic in _GENERIC_LABEL_VOCABULARIES):
            return None
        return [str(value) for value in values]
    return None


def _reference_strings(sample) -> list[str]:
    reference = (
        sample.reference_answer_normalized
        if sample.reference_answer_normalized is not None
        else sample.reference_answer
    )
    if reference is None:
        return []
    if isinstance(reference, (list, tuple, set, frozenset)):
        return [str(item) for item in reference]
    return [str(reference)]


# --------------------------------------------------------------------------- #
# Contract 1: a closed answer space must be shown to the model.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task_key", _implemented_task_keys())
def test_closed_answer_space_is_visible_in_the_prompt(task_key: str) -> None:
    if task_key in _OPEN_RECALL_BY_DESIGN:
        pytest.skip(_OPEN_RECALL_BY_DESIGN[task_key])
    for adapter, sample in _load(task_key, 3):
        if sample.evaluation_metric not in _CLOSED_SPACE_METRICS:
            continue
        if _closed_answer_space(sample) is None:
            continue
        prompt = normalized_string(_prompt_text(adapter, sample))
        for reference in _reference_strings(sample):
            normalized = normalized_string(reference)
            if not normalized:
                continue
            assert normalized in prompt, (
                f"{task_key}: the gold answer {reference!r} is scored by exact matching against a "
                f"closed answer space, but it never appears in the prompt, so the model cannot "
                f"know it is admissible. Show the candidate list in build_messages()."
            )


# --------------------------------------------------------------------------- #
# Contract 2: a short gold requires the prompt to ask for a short answer.
# --------------------------------------------------------------------------- #
def _scores_only_a_signposted_span(adapter, sample, reference: str) -> bool:
    """Whether the adapter extracts a short answer out of a long, signposted response.

    This is the other way to keep a length-sensitive metric honest, and the stronger one: rather
    than asking the model to be terse, let it reason freely and score only the span behind a
    declared marker. Six tasks (AgentClinic, MedCaseReasoning, MedChain, MedThink-Bench,
    RareBench, VivaBench) run ``text_f1_em`` as a secondary metric on exactly this contract, so
    a cue-phrase check alone would flag them while their scored span is already short.
    """
    padding = "Let me work through the presentation step by step. " * 40
    for marker in ("Answer:", "Final Answer:", "最终答案:", "答案:"):
        if marker.lower() not in _prompt_text(adapter, sample).lower():
            continue
        try:
            parsed = adapter.parse_response(sample, f"{padding}\n{marker} {reference}")
        except Exception:
            return False
        if parsed is None:
            continue
        text = " ".join(str(parsed).split())
        if text and len(text.split()) <= max(len(reference.split()) + 2,
                                             _MAX_SHORT_REFERENCE_TOKENS):
            return True
    return False


@pytest.mark.parametrize("task_key", _implemented_task_keys())
def test_short_reference_metrics_ask_for_a_short_answer(task_key: str) -> None:
    secondary = sorted(_BREVITY_EVALUATORS.intersection(default_extra_evaluators(task_key)))
    for adapter, sample in _load(task_key, 3):
        if sample.evaluation_metric not in _BREVITY_METRICS and not secondary:
            continue
        scoring = (repr(sample.evaluation_metric) if sample.evaluation_metric in _BREVITY_METRICS
                   else f"secondary evaluator(s) {', '.join(secondary)}")
        references = _reference_strings(sample)
        if not references:
            continue
        longest = max(len(str(reference).split()) for reference in references)
        if longest > _MAX_SHORT_REFERENCE_TOKENS:
            continue
        prompt = _prompt_text(adapter, sample).lower()
        if any(cue in prompt for cue in _BREVITY_CUES):
            continue
        assert _scores_only_a_signposted_span(adapter, sample, str(references[0])), (
            f"{task_key}: the gold answer is {longest} token(s) and scoring is "
            f"{scoring}, which divides by the prediction length — but the prompt never asks for "
            f"a short answer and the adapter does not extract a signposted span either, so a "
            f"correct verbose reply scores ~0. Either add a brevity instruction (see "
            f"bioasq_factoid_open) or a final-answer marker plus extraction (see "
            f"agentclinic_diagnosis_open)."
        )


# --------------------------------------------------------------------------- #
# Contract 3: document-field names must be the ones the prompt requests.
# --------------------------------------------------------------------------- #
def _reference_field_names(reference: str) -> set[str]:
    text = str(reference or "").strip()
    if not text:
        return set()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return set()
    records = payload if isinstance(payload, list) else [payload]
    names: set[str] = set()
    for record in records:
        if isinstance(record, dict):
            names.update(str(key) for key in record)
    return names


@pytest.mark.parametrize("task_key", _implemented_task_keys())
def test_document_field_names_are_requested_by_the_prompt(task_key: str) -> None:
    for adapter, sample in _load(task_key, 3):
        if sample.evaluation_metric != "document_fields":
            continue
        prompt = normalized_string(_prompt_text(adapter, sample))
        for reference in _reference_strings(sample):
            for field in _reference_field_names(reference):
                normalized = normalized_string(field)
                if not normalized:
                    continue
                assert normalized in prompt, (
                    f"{task_key}: the reference is keyed by {field!r} but the prompt never asks "
                    f"for that field name, so field matching can never intersect and the task "
                    f"scores 0 regardless of the answer's quality."
                )


# --------------------------------------------------------------------------- #
# Contract 4: a degenerate reference distribution must be declared, not silent.
# --------------------------------------------------------------------------- #
_DISCRIMINATIVE_METRICS = {"accuracy", "set_match", "multilabel", "exact_match"}


@pytest.mark.parametrize("task_key", _implemented_task_keys())
def test_reference_distribution_is_not_silently_degenerate(task_key: str) -> None:
    samples = list(_load(task_key, 135, stride=37, window=5_000))
    if not samples:
        pytest.skip("no samples")
    if samples[0][1].evaluation_metric not in _DISCRIMINATIVE_METRICS:
        pytest.skip("not a discriminative metric")
    references = [tuple(sorted(_reference_strings(sample))) for _, sample in samples]
    distinct = set(references)
    reason = _DEGENERATE_BY_DATA.get(task_key)
    degenerate = len(references) >= 10 and len(distinct) == 1
    if degenerate:
        # The strided sample is a probe, not a verdict. A small, heavily imbalanced file gets
        # read only 14 records deep at stride 37, and at ClinicalBench/mortality's 96.6% base
        # rate an all-zero draw is the *expected* outcome — which says nothing about whether
        # the reference extraction works. Confirm against a contiguous read before failing, so
        # "degenerate" means the labels really do not vary rather than that we undersampled.
        references = [tuple(sorted(_reference_strings(sample)))
                      for _, sample in _load(task_key, 5_000, window=5_000)]
        distinct = set(references)
        degenerate = len(references) >= 10 and len(distinct) == 1
    if reason:
        assert degenerate, (
            f"{task_key} is allowlisted as degenerate but its references now vary "
            f"({len(distinct)} distinct in {len(references)}). Remove the allowlist entry."
        )
        return
    assert not degenerate, (
        f"{task_key}: every one of {len(references)} sampled references is {references[0]!r}, so a "
        f"constant predictor scores 1.000 and the metric cannot rank models. Either the reference "
        f"extraction is wrong, or the data is genuinely single-class and belongs in "
        f"_DEGENERATE_BY_DATA with a reason."
    )
