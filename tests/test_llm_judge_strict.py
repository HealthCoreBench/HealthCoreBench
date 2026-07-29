"""Strict LLM-judge behavior: no silent zero scores and complete attempt logs."""

from __future__ import annotations

import pytest

from healthcorebench.aggregation.summarize import _flatten_numeric
from healthcorebench.clients.openai_client import ModelResponse
from healthcorebench.evaluators.llm_judge import (
    DEFAULT_JUDGE_PROMPT,
    LLMJudgeEvaluator,
    _prompt_declared_scale,
)
from healthcorebench.runtime.recorder import Recorder
from healthcorebench.schemas.config import RunConfig
from healthcorebench.utils.jsonl import read_jsonl
from healthcorebench.utils.timestamps import utc_now_iso


class _JudgeClient:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = 0

    async def chat_completion(self, messages, **kwargs):
        self.calls += 1
        content = next(self.contents)
        now = utc_now_iso()
        return ModelResponse(
            content=content, model_requested="judge", model_returned="judge-returned",
            provider_request_id=f"req-{self.calls}", finish_reason="stop",
            prompt_tokens=10, completion_tokens=4, total_tokens=14,
            request_start_time=now, request_end_time=now, latency_seconds=0.01,
            raw_response={"id": f"req-{self.calls}", "content": content},
        )


def _result():
    return {
        "run_id": "run", "result_id": "result", "sample_id": "sample",
        "sample_repeat_index": 0, "successful_attempt_id": "model-attempt",
        "status": "success", "formatted_prompt": [{"role": "user", "content": "case"}],
        "reference_answer": "answer", "raw_response": "prediction",
    }


async def test_empty_judge_response_retries_then_records_success(tmp_path):
    recorder = Recorder(tmp_path)
    client = _JudgeClient([None, '{"correct": true, "rationale": "matches"}'])
    evaluator = LLMJudgeEvaluator(
        client=client, judge_model="judge", recorder=recorder, run_id="run", max_retries=1,
    )
    evaluator.retry_policy.initial_seconds = 0
    evaluator.retry_policy.max_seconds = 0

    judgment = await evaluator.evaluate_async(_result(), {"source_content": {}})

    assert judgment.evaluation_status == "success"
    assert judgment.is_correct is True
    assert judgment.judge_raw_response is not None
    assert judgment.judge_prompt_tokens == 10
    assert judgment.judge_completion_tokens == 4
    assert judgment.judge_total_tokens == 14
    assert judgment.judge_attempt_id is not None
    attempts = read_jsonl(tmp_path / "attempts.jsonl")
    assert [a["status"] for a in attempts] == ["error", "success"]
    assert all(a["request_purpose"] == "evaluation_judge" for a in attempts)
    assert attempts[0]["error_type"] == "empty_response"


async def test_unparseable_judge_response_is_error_not_zero_score(tmp_path):
    recorder = Recorder(tmp_path)
    evaluator = LLMJudgeEvaluator(
        client=_JudgeClient(["I cannot decide."]), judge_model="judge",
        recorder=recorder, run_id="run", max_retries=0,
    )

    judgment = await evaluator.evaluate_async(_result(), {"source_content": {}})

    assert judgment.evaluation_status == "error"
    assert judgment.is_correct is None
    assert judgment.raw_score is None
    assert judgment.normalized_score is None
    assert judgment.judge_raw_response == "I cannot decide."
    assert judgment.judge_total_tokens == 14
    assert "parseable" in judgment.evaluation_error
    attempt = read_jsonl(tmp_path / "attempts.jsonl")[0]
    assert attempt["status"] == "error"
    assert attempt["error_type"] == "judge_error"
    assert attempt["raw_response"] is not None


# Verdict wordings that a substring test used to read backwards. "correct: true" is a substring
# of "incorrect: true", so an explicitly rejected answer scored 1.0; the Chinese forms have the
# same trap in both directions (不正确 contains 正确, and 正确:否 negates in the *value*).
@pytest.mark.parametrize(("reply", "expected"), [
    ("incorrect: true — the model confused metformin with metoprolol", False),
    ("Incorrect: TRUE", False),
    ("inaccurate: yes", False),
    ("not correct: true", False),
    ("wrong: true", False),
    ("correct: true", True),
    ("Correct: FALSE", False),
    ("**correct**: true", True),
    ("Yes, but the answer is incorrect.", None),
    ("Yes. The answer matches the reference.", True),
    ("No. The answer contradicts the reference.", False),
    ("不正确:是", False),
    ("正确：否", False),
    ("正确：是", True),
    ("不正确：否", True),
    ("回答不正确，与参考答案矛盾。", False),
    ("回答正确，与参考答案一致。", True),
    ("模型回答错误，正确答案是二甲双胍。", None),
])
def test_leading_verdict_reads_negations_as_negations(reply, expected):
    assert LLMJudgeEvaluator._leading_verdict(reply) is expected


async def test_negated_verdict_field_is_not_scored_correct(tmp_path):
    """End to end: "incorrect: true" must not come back as a 1.0 success."""
    evaluator = LLMJudgeEvaluator(
        client=_JudgeClient(["incorrect: true — the model named the wrong drug"]),
        judge_model="judge", recorder=Recorder(tmp_path), run_id="run", max_retries=0,
    )

    judgment = await evaluator.evaluate_async(_result(), {"source_content": {}})

    assert judgment.evaluation_status == "success"
    assert judgment.is_correct is False
    assert judgment.normalized_score == 0.0


def test_abstention_in_chinese_is_not_a_wrong_answer():
    correct, score, _, dimensions = LLMJudgeEvaluator._parse_judge_details(
        "无法判断：病例信息不足，缺少检验结果。")

    assert correct is None and score is None
    assert dimensions == {"judge_abstained": True}


# --- scale handling ------------------------------------------------------------------------- #

_UNIT = _prompt_declared_scale(DEFAULT_JUDGE_PROMPT)
_LIKERT = _prompt_declared_scale("Rate the answer on a scale of 1 to 5.")


def test_builtin_judge_prompt_declares_the_unit_interval():
    assert _UNIT == (0.0, 1.0)
    assert _LIKERT == (1.0, 4.0)


@pytest.mark.parametrize(("payload", "prompt_scale", "expected"), [
    # The scale the prompt asked for is evidence; a compliant answer needs no heuristic.
    ({"correct": True, "score": 0.75, "rationale": "r"}, _UNIT, 0.75),
    ({"correct": True, "score": 4, "rationale": "r"}, _LIKERT, 0.75),
    # The judge's own declaration outranks the prompt's.
    ({"correct": True, "score": 4, "score_scale": 5, "rationale": "r"}, _UNIT, 0.75),
    ({"correct": True, "score": 70, "out_of": 100, "rationale": "r"}, _UNIT, 0.7),
    # max_score 5 with a sub-1 answer is a 0-5 scale, not a 1-5 one that would reject it.
    ({"correct": False, "score": 0, "max_score": 5, "rationale": "r"}, _UNIT, 0.0),
])
def test_declared_scales_are_used_verbatim(payload, prompt_scale, expected):
    import json

    correct, score, _, _ = LLMJudgeEvaluator._parse_judge_details(
        json.dumps(payload), prompt_scale)

    assert correct is not None
    assert score == pytest.approx(expected)


@pytest.mark.parametrize(("score", "reason"), [
    (4, "ambiguous_judge_score_scale"),    # 4/5, 4/10 and 4/100 are equally plausible
    (7, "ambiguous_judge_score_scale"),
    (88, "ambiguous_judge_score_scale"),
    (-1, "judge_score_outside_scale"),
])
def test_undeclared_out_of_range_score_is_unscorable_not_guessed(score, reason):
    import json

    correct, normalized, _, dimensions = LLMJudgeEvaluator._parse_judge_details(
        json.dumps({"correct": True, "score": score, "rationale": "r"}), _UNIT)

    assert correct is None and normalized is None
    assert dimensions["judge_unscorable"] is True
    assert dimensions["unscorable_reason"] == reason


async def test_unscorable_scale_errors_instead_of_collapsing_to_one(tmp_path):
    """The old code fell through to the yes/no regex, turning a graded 4 into a flat 1.0."""
    evaluator = LLMJudgeEvaluator(
        client=_JudgeClient(['{"correct": true, "score": 4, "rationale": "mostly right"}']),
        judge_model="judge", recorder=Recorder(tmp_path), run_id="run", max_retries=0,
    )

    judgment = await evaluator.evaluate_async(_result(), {"source_content": {}})

    assert judgment.evaluation_status == "error"
    assert judgment.is_correct is None
    assert judgment.normalized_score is None
    assert "not scorable" in judgment.evaluation_error
    # The rejected judgement stays auditable rather than being discarded.
    assert judgment.parsed_judgment["unscorable_reason"] == "ambiguous_judge_score_scale"
    assert judgment.parsed_judgment["rationale"] == "mostly right"


async def test_abstention_keeps_its_parsed_judgment(tmp_path):
    evaluator = LLMJudgeEvaluator(
        client=_JudgeClient(['{"rationale": "Not enough information to judge."}']),
        judge_model="judge", recorder=Recorder(tmp_path), run_id="run", max_retries=0,
    )

    judgment = await evaluator.evaluate_async(_result(), {"source_content": {}})

    assert judgment.evaluation_status == "error"
    assert judgment.parsed_judgment["judge_abstained"] is True
    assert judgment.parsed_judgment["rationale"] == "Not enough information to judge."
    assert judgment.judge_rationale == "Not enough information to judge."
    assert judgment.parsed_judgment["score"] is None


def test_scale_provenance_does_not_leak_into_reported_subscores():
    """``summarize._flatten_numeric`` publishes every numeric leaf — nested dicts included."""
    import json

    _, _, _, dimensions = LLMJudgeEvaluator._parse_judge_details(
        json.dumps({"correct": True, "score": 4, "score_scale": 5,
                    "clinical_coverage": 3, "rationale": "r"}), _UNIT)

    assert dimensions["judge_score_scale"] == "4 on 1-5 (judge_declared_scale)"
    flattened = _flatten_numeric({"score": 0.75, **dimensions})
    assert flattened == {"score": 0.75, "clinical_coverage": 0.5}


def test_full_run_persists_model_and_judge_provenance(tmp_path, monkeypatch):
    import json
    import healthcorebench.runtime.run_setup as run_setup

    class _RunClient:
        def __init__(self, requested_model_name, **kwargs):
            self.requested_model_name = requested_model_name
            self.base_url = kwargs.get("base_url")

        async def list_models(self):
            return [self.requested_model_name]

        async def chat_completion(self, messages, **kwargs):
            now = utc_now_iso()
            is_judge = (kwargs.get("model") or self.requested_model_name) == "judge"
            content = ('{"correct": true, "rationale": "valid"}' if is_judge
                       else "The answer is A.")
            return ModelResponse(
                content=content, model_requested=self.requested_model_name,
                model_returned=self.requested_model_name, provider_request_id="req",
                finish_reason="stop", prompt_tokens=10, completion_tokens=4, total_tokens=14,
                request_start_time=now, request_end_time=now, latency_seconds=0.01,
                raw_response={"model": self.requested_model_name, "content": content},
            )

        async def aclose(self):
            pass

    monkeypatch.setattr(run_setup, "OpenAICompatibleClient", _RunClient)
    cfg = RunConfig(
        experiment={"experiment_id": "judge-e2e", "run_name": "judge-e2e"},
        benchmark={"name": "MMLU/mcqa", "max_samples": 1},
        model={"base_url": "http://model/v1", "requested_model_name": "model"},
        generation={"max_tokens": 32}, runtime={"max_retries": 0},
        output={"root_dir": str(tmp_path)},
        evaluation={
            "use_llm_judge": True,
            "judge": {"base_url": "http://judge/v1", "requested_model_name": "judge",
                      "max_retries": 0},
        },
    )
    run_dir = tmp_path / "run"

    run_setup.RunOrchestrator(cfg, run_dir=str(run_dir)).run()

    attempts = read_jsonl(run_dir / "attempts.jsonl")
    assert [a["request_purpose"] for a in attempts] == ["model_inference", "evaluation_judge"]
    judgment = read_jsonl(run_dir / "judgments.jsonl")[0]
    assert judgment["evaluation_status"] == "success"
    assert judgment["judge_raw_response"] is not None
    assert judgment["judge_prompt_tokens"] == 10
    assert judgment["judge_completion_tokens"] == 4
    assert judgment["judge_total_tokens"] == 14
    assert judgment["judge_attempt_id"] == attempts[1]["attempt_id"]
    result = read_jsonl(run_dir / "results.jsonl")[0]
    assert result["parsed_answer"] == "A"
    assert result["parsing_status"] == "success"
    assert result["evaluation_status"] == "success"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["metrics"]["score"] == 1.0
    assert summary["tokens"]["evaluation_tokens"] == 14
