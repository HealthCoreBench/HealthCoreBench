"""An unparsed answer is not a wrong answer.

``multiple_answer_set_match`` and ``classification_accuracy`` used to return
``raw=0.0 / is_correct=False / evaluation_status="success"`` when the extractor produced
``None``. The record was then counted in ``num_parsing_errors`` *and* in ``num_scored``, so "the
extractor never fired" and "the model got every question wrong" produced byte-identical
summaries — a dead parser reported as "100% scored, accuracy 0.000".

Both now return ``unscorable("unparsed_answer", ..., parse_failed=True)``, matching
``multiple_choice_accuracy``: the record leaves the score denominator (``num_unscorable``) while
``parse_failed`` keeps ``aggregation/summarize.py``'s legacy branch counting it as a parsing
error. Measured over the recorded runs, 661 judgments across 36 task-runs were affected —
IgakuQA/mcqa alone carried 128 of them (accuracy 0.4897 -> 0.5234).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.evaluators import get_evaluator

# (registry key, a parsed answer that matches the reference, one that does not)
_CASES = [
    ("multiple_answer", ["B", "D"], ["A"]),
    ("classification", "yes", "no"),
]
_REFERENCES = {
    "multiple_answer": ["B", "D"],
    "classification": "yes",
}


def _sample(evaluator_name: str) -> dict:
    reference = _REFERENCES[evaluator_name]
    return {"sample_id": "s1", "reference_answer": reference,
            "reference_answer_normalized": reference}


def _judge(evaluator_name: str, parsed_answer):
    evaluator = get_evaluator(evaluator_name)
    result = {"run_id": "r", "result_id": "res1", "sample_id": "s1",
              "status": "success", "parsed_answer": parsed_answer}
    return evaluator.evaluate(result, _sample(evaluator_name))


@pytest.mark.parametrize("evaluator_name,right,wrong", _CASES)
def test_unparsed_answer_leaves_the_score_denominator(
        evaluator_name: str, right, wrong) -> None:
    judgment = _judge(evaluator_name, None)

    # Still a successful evaluation — nothing went wrong *at evaluation time*.
    assert judgment.evaluation_status == "success"
    assert judgment.raw_score is None
    assert judgment.normalized_score is None
    assert judgment.is_correct is None
    assert judgment.parsed_judgment["unscorable_reason"] == "unparsed_answer"
    # summarize.py's legacy branch reads this to keep the parsing-error count honest.
    assert judgment.parsed_judgment["parse_failed"] is True


@pytest.mark.parametrize("evaluator_name,right,wrong", _CASES)
def test_a_wrong_answer_is_still_a_scored_zero(evaluator_name: str, right, wrong) -> None:
    """The fix must not make genuine mistakes disappear from the denominator."""
    judgment = _judge(evaluator_name, wrong)

    assert judgment.normalized_score == 0.0
    assert judgment.is_correct is False
    assert judgment.parsed_judgment["parse_failed"] is False
    assert "unscorable_reason" not in judgment.parsed_judgment


@pytest.mark.parametrize("evaluator_name,right,wrong", _CASES)
def test_a_right_answer_still_scores_one(evaluator_name: str, right, wrong) -> None:
    judgment = _judge(evaluator_name, right)

    assert judgment.normalized_score == 1.0
    assert judgment.is_correct is True
    assert judgment.parsed_judgment["parse_failed"] is False


@pytest.mark.parametrize("evaluator_name,right,wrong", _CASES)
def test_version_is_bumped_so_old_judgments_stay_identifiable(
        evaluator_name: str, right, wrong) -> None:
    """The scoring semantics changed; a judgment recorded under 1.0 is not comparable."""
    evaluator = get_evaluator(evaluator_name)
    assert evaluator.evaluator_version == "1.1"
    # These are the two evaluators the fix targets; pin the recorded names too.
    assert evaluator.evaluator_name in {"multiple_answer_set_match", "classification_accuracy"}


def _write_run(run_dir: Path, evaluator_name: str, parsed_answers: list) -> None:
    """Materialize the smallest run ``summarize_run`` will accept."""
    evaluator = get_evaluator(evaluator_name)
    sample = _sample(evaluator_name)
    results, judgments, samples = [], [], []
    for index, parsed in enumerate(parsed_answers):
        sid, rid = f"s{index}", f"res{index}"
        samples.append({**sample, "sample_id": sid, "run_id": "run1",
                        "benchmark_name": "SyntheticBench"})
        results.append({"run_id": "run1", "result_id": rid, "sample_id": sid,
                        "sample_repeat_index": 0, "status": "success",
                        "finish_reason": "stop",
                        "parsing_status": "success" if parsed is not None else "error",
                        "parsed_answer": parsed, "benchmark_name": "SyntheticBench"})
        judgment = evaluator.evaluate(
            {"run_id": "run1", "result_id": rid, "sample_id": sid,
             "status": "success", "parsed_answer": parsed},
            {**sample, "sample_id": sid})
        judgments.append(json.loads(judgment.model_dump_json()))
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("results.jsonl", results), ("judgments.jsonl", judgments),
                       ("samples.jsonl", samples), ("attempts.jsonl", [])):
        (run_dir / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run1", "benchmark": {"name": "SyntheticBench"}}), encoding="utf-8")


@pytest.mark.parametrize("evaluator_name,right,wrong", _CASES)
def test_dead_parser_and_wrong_model_no_longer_summarize_identically(
        evaluator_name: str, right, wrong, tmp_path: Path) -> None:
    """The regression this fix exists for, asserted at the summary level."""
    _write_run(tmp_path / "dead_parser", evaluator_name, [None] * 4)
    _write_run(tmp_path / "wrong_model", evaluator_name, [wrong] * 4)

    dead = summarize_run(tmp_path / "dead_parser").counts
    missed = summarize_run(tmp_path / "wrong_model").counts

    assert (dead.num_scored, dead.num_unscorable, dead.num_parsing_errors) == (0, 4, 4)
    assert dead.unscorable_reasons == {"unparsed_answer": 4}
    assert (missed.num_scored, missed.num_unscorable, missed.num_parsing_errors) == (4, 0, 0)
    # ...and the two are now distinguishable, which is the whole point.
    assert dead.model_dump() != missed.model_dump()
    # The accounting invariant still holds on both sides of the split.
    for counts in (dead, missed):
        assert counts.num_successful == (
            counts.num_scored + counts.num_missing_scoring + counts.num_evaluation_errors
            + counts.num_evaluation_skipped + counts.num_unscorable)
