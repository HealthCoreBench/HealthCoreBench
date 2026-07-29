"""The token-F1 cross-check must score an answer, not a length ratio.

Eight tasks carried ``text_f1_em`` as a secondary metric while their prompts asked an open
question and their ``parse_response`` returned the whole reply. SQuAD-style token-F1 between a
1,500-character essay and a two-word diagnosis is a prediction/reference length ratio: measured
0.024 (AgentClinic), 0.013 (MedCaseReasoning), 0.031 (MedThink-Bench) on the recorded run, with
exact-match structurally pinned at 0.000.

The fix is a *pair* — the prompt has to name a locatable place for the answer and the parser has
to read exactly that place. Neither half works alone: without the marker,
``extract_short_answer``'s last-line fallback lands on the closing caveat sentence. These tests
therefore assert both halves together, and pin the two deliberate non-changes (MedR-Bench,
GeneTuring) so they are not "fixed" by symmetry later.

Reasoning is never suppressed: for all of these tasks the *primary* metric is the LLM judge,
which reads ``raw_response`` in full, so ``parse_response`` only affects the secondary metric.
"""

from __future__ import annotations

import itertools
import statistics

import pytest

from healthcorebench.benchmarks.registry import get_adapter
from healthcorebench.evaluators import DEFAULT_EXTRA_EVALUATORS, get_evaluator

# Tasks whose prompt now prescribes a final "Answer:" line and whose parser reads it.
_MARKED_TASKS = [
    "AgentClinic/diagnosis",
    "RareBench/diagnosis",
    "VivaBench/diagnosis",
    "MedCaseReasoning/open",
    "MedThink-Bench/open",
    "MedChain/diagnosis",
]

# Stand-in for the reasoning the models actually emit, so the test proves the parser skips the
# prose rather than merely handling a one-line reply.
_REASONING = (
    "Let me work through this step by step. The presentation, the examination findings and the\n"
    "laboratory results point in one direction, and the alternatives fit the time course less\n"
    "well. Note that further testing would be needed to confirm this before treatment.\n\n"
)


def _first_sample(task_key: str):
    adapter = get_adapter(task_key)
    files = adapter.discover_source_files()
    if not all(path.exists() for path in files):
        pytest.skip(f"{task_key}: benchmark data not present")
    raw = next(iter(itertools.islice(adapter.load_raw_samples(files), 0, 1)))
    return adapter, adapter.normalize_sample(raw, 0)


def _prompt_text(adapter, sample) -> str:
    return "\n".join(str(m.get("content") or "") for m in adapter.build_messages(sample))


def _f1_em(sample, predicted) -> tuple[float, bool]:
    evaluator = get_evaluator("text_f1_em")
    payload = sample.model_dump()
    _, normalized, is_correct, _ = evaluator.score(
        evaluator.normalize(predicted, payload), payload)
    return (normalized or 0.0), bool(is_correct)


@pytest.mark.parametrize("task_key", _MARKED_TASKS)
def test_prompt_names_the_place_the_cross_check_reads(task_key: str) -> None:
    """A rule-based metric may only score a span the prompt actually asked for."""
    adapter, sample = _first_sample(task_key)

    prompt = _prompt_text(adapter, sample)

    # ASCII marker on purpose: the shared extractor's regex accepts English labels and an ASCII
    # ":"/"=" only, so the Chinese prompt must prescribe the same literal form.
    assert "Answer:" in prompt, f"{task_key} prompt does not request a final 'Answer:' line"
    assert "text_f1_em" in DEFAULT_EXTRA_EVALUATORS[task_key]


@pytest.mark.parametrize("task_key", _MARKED_TASKS)
def test_marked_answer_is_scored_instead_of_the_reasoning(task_key: str) -> None:
    _, sample = _first_sample(task_key)
    adapter = get_adapter(task_key)
    reference = sample.reference_answer.splitlines()[0].strip()
    reply = f"{_REASONING}Answer: {reference}"

    predicted = adapter.parse_response(sample, reply)

    assert predicted is not None
    assert len(predicted) < len(reply) / 2, "the whole reply is still being scored"
    f1, exact = _f1_em(sample, predicted)
    assert f1 == pytest.approx(1.0)
    assert exact is True
    # And the unmarked essay must not be passed through wholesale either.
    assert len(adapter.parse_response(sample, _REASONING) or "") < len(_REASONING)


def test_chinese_task_tolerates_a_full_width_colon() -> None:
    """A Chinese IME types "："; the extractor's marker regex is ASCII-only."""
    _, sample = _first_sample("MedChain/diagnosis")
    adapter = get_adapter("MedChain/diagnosis")
    reference = sample.reference_answer.splitlines()[0].strip()

    predicted = adapter.parse_response(sample, f"{_REASONING}Answer：{reference}")

    assert predicted == reference
    assert _f1_em(sample, predicted) == (pytest.approx(1.0), True)


def test_medrbench_keeps_no_rule_based_cross_check() -> None:
    """Its reference is a paragraph, so token overlap measures length, not correctness.

    MedR-Bench's prompt is the benchmark's own ``oracle_diagnose.txt`` verbatim — it asks for a
    differential and for further tests when the data is insufficient — so a brevity instruction
    would break protocol fidelity. Removing the metric is the honest option; the LLM judge stays
    the primary score.
    """
    assert "MedR-Bench/diagnosis" not in DEFAULT_EXTRA_EVALUATORS

    adapter = get_adapter("MedR-Bench/diagnosis")
    files = adapter.discover_source_files()
    if not all(path.exists() for path in files):
        pytest.skip("MedR-Bench data not present")
    lengths = [
        len(adapter.normalize_sample(raw, i).reference_answer.split())
        for i, raw in enumerate(itertools.islice(adapter.load_raw_samples(files), 0, 40))
    ]
    # Sanity-check the premise of the removal rather than trusting the comment.
    assert statistics.median(lengths) > 10


def test_geneturing_answers_are_not_truncated_by_extraction() -> None:
    """GeneTuring is the one task where extraction would be the regression.

    Its prompt already demands the bare answer and the recorded replies obey it (5-12
    characters). Running them through ``extract_short_answer`` would read "chrX:" as a label and
    return "12345-99999", silently corrupting a genomic coordinate.
    """
    assert DEFAULT_EXTRA_EVALUATORS["GeneTuring/open"] == ["text_f1_em"]

    adapter = get_adapter("GeneTuring/open")
    files = adapter.discover_source_files()
    if not all(path.exists() for path in files):
        pytest.skip("GeneTuring data not present")
    # The nucleotide-sequence modules have their own parser; a coordinate belongs to the others.
    sample = next(
        s for s in (
            adapter.normalize_sample(raw, i)
            for i, raw in enumerate(itertools.islice(adapter.load_raw_samples(files), 0, 400))
        )
        if (s.metadata or {}).get("gene_output_kind") != "sequence"
    )

    assert adapter.parse_response(sample, "chrX:12345-99999") == "chrX:12345-99999"
