"""Every task must be able to score its own gold answer as correct.

A benchmark whose prompt asks for one format, whose parser looks for a second and whose
evaluator compares against a third produces a run that is green everywhere and zero on the
scoreboard. ``MedDocBench/ltr_abnormality_qa`` scored 0.000 three runs in a row while all 302
tests passed, because nothing in the suite ever connected ``build_messages`` →
``parse_response`` → evaluator end to end. Each link was unit-tested; the chain was not.

This closes that: the reference answer the adapter recorded is handed back to
``parse_response`` as if a model had emitted it, scored by the same evaluator production
would auto-select, and required to come out correct. A task that cannot recognise its own
gold cannot score a real model either, whatever its unit tests say — it is a per-task
non-degenerate floor, the thing the audit found missing.

Wiring mirrors ``run_setup._parse_onto`` exactly (a ``SimpleNamespace`` shim over the sample
dict, a ``status="success"`` result record) so a pass here means the production path works,
not that a convenient test-only path does.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.evaluators import get_evaluator, select_evaluator_name
from tests.test_prompt_reference_contract import (
    _implemented_task_keys,
    _load,
    _reference_strings,
)

# A continuous metric never sets is_correct; an exact reproduction of the gold has to land at
# the top of its range instead. Below 1.0 only to absorb float noise in BLEU/ROUGE smoothing.
_CONTINUOUS_FLOOR = 0.99


def _renderings(references: list[str], raw_reference=None) -> list[str]:
    """Ways an obedient model could write the gold, given the formats the prompts ask for.

    Requiring the *bare* reference to parse would fail tasks whose prompt legitimately demands
    a wrapper (``\\boxed{}``, ``Answer:``), and requiring the wrapper would fail the ones that
    ask for a bare token. So the assertion is existential: at least one obedient rendering must
    score. Both the per-item and the joined forms are offered because set-valued metrics need
    every gold at once while any-of metrics need exactly one.

    A structured gold (grounding boxes, field dicts) gets a JSON rendering too. ``str()`` on a
    dict is a Python repr with single quotes, which no prompt asks for and no parser should
    accept; scoring a task on that would be testing the test.
    """
    candidates: list[str] = []
    if raw_reference is not None and not isinstance(raw_reference, str):
        try:
            candidates.append(json.dumps(raw_reference, ensure_ascii=False))
        except (TypeError, ValueError):
            pass
    singles = [reference for reference in references if str(reference).strip()]
    if not singles:
        return candidates
    joined = [", ".join(singles), "; ".join(singles), "\n".join(singles)] if len(singles) > 1 else []
    for body in [*singles, *joined]:
        candidates.extend([
            body,
            f"Answer: {body}",
            f"The answer is {body}.",
            f"\\boxed{{{body}}}",
            f"<answer>{body}</answer>",
            f"最终答案：{body}",
        ])
    return candidates


def _scores_correct(judgment) -> bool:
    if judgment.evaluation_status != "success":
        return False
    if judgment.is_correct is True:
        return True
    score = judgment.normalized_score
    return isinstance(score, (int, float)) and score >= _CONTINUOUS_FLOOR


@pytest.mark.parametrize("task_key", _implemented_task_keys())
def test_task_scores_its_own_gold_answer_as_correct(task_key: str) -> None:
    try:
        loaded = list(_load(task_key, 2))
    except (FileNotFoundError, BenchmarkSplitNotFoundError) as exc:
        pytest.skip(f"source data unavailable: {exc}")
    if not loaded:
        pytest.skip("no samples")

    checked = 0
    for adapter, sample in loaded:
        evaluator_name = select_evaluator_name(sample.evaluation_metric, sample.answer_format)
        if evaluator_name is None:
            pytest.skip(
                f"no rule-based evaluator for evaluation_metric="
                f"{sample.evaluation_metric!r}/answer_format={sample.answer_format!r}; "
                "this task is scored by the LLM judge, which cannot run offline"
            )
        evaluator = get_evaluator(evaluator_name)
        references = _reference_strings(sample)
        if not references:
            # An empty gold is the ``unscorable`` disposition, not a contract breach.
            continue

        sample_dict = sample.model_dump()
        shim = SimpleNamespace(**sample_dict)
        # JudgmentRecord requires the identity fields a real result record carries; only their
        # presence matters here, nothing asserted below reads them.
        result_ids = {
            "run_id": "gold-roundtrip",
            "result_id": f"res_{task_key}",
            "sample_id": str(sample.sample_id),
            "status": "success",
        }
        attempts: list[str] = []
        raw_reference = (
            sample.reference_answer_normalized
            if sample.reference_answer_normalized is not None
            else sample.reference_answer
        )
        for rendering in _renderings(references, raw_reference):
            parsed = adapter.parse_response(shim, rendering)
            judgment = evaluator.evaluate({**result_ids, "parsed_answer": parsed}, sample_dict)
            if _scores_correct(judgment):
                break
            attempts.append(f"{rendering!r} -> parsed={parsed!r} score={judgment.normalized_score!r}")
        else:
            pytest.fail(
                f"{task_key}: no obedient rendering of the gold answer {references!r} scores as "
                f"correct under evaluator {evaluator_name!r} "
                f"(evaluation_metric={sample.evaluation_metric!r}, "
                f"answer_format={sample.answer_format!r}). The prompt, the parser and the "
                f"evaluator disagree about the answer format, so this task scores ~0 for every "
                f"model regardless of how good it is. Tried:\n  " + "\n  ".join(attempts[:6])
            )
        checked += 1

    if not checked:
        pytest.skip("every loaded sample has an empty reference")
