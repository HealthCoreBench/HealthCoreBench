"""Regression tests for resume safety, adaptive budgets, and audit completeness."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import healthcorebench.runtime.run_setup as run_setup
from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.benchmarks.context_window import (
    ContextOverflowError,
    UnbudgetedContextWarning,
    fit_context_to_window,
)
from healthcorebench.clients.errors import ClientError, ErrorType
from healthcorebench.evaluators.llm_judge import LLMJudgeEvaluator
from healthcorebench.runtime.executor import Executor
from healthcorebench.runtime.recorder import Recorder
from healthcorebench.runtime.resume import ResumeIndex
from healthcorebench.runtime.retry import RetryPolicy
from healthcorebench.runtime.runner import Runner
from healthcorebench.runtime.run_setup import _stratified_select
from healthcorebench.schemas.config import (
    GenerationConfig, MediaConfig, OutputConfig, RunConfig,
)
from healthcorebench.schemas.judgment import JudgmentRecord
from healthcorebench.utils.jsonl import append_jsonl, atomic_write_json, read_jsonl
from healthcorebench.tools.validate import validate_run
from tests.mock_client import MockClient
from tests.test_e2e_mmlu import _AnswerKeyClient


SAMPLE = {
    "sample_id": "urn:s1", "sample_index": 0, "benchmark_name": "MMLU",
    "benchmark_split": "test", "reference_answer": "A",
    "logical_messages": [{"role": "user", "content": "Q?"}],
}
# The output-budget ladder shared by every shipped config; its first entry is also their
# generation.max_tokens. Tiers above 8192 were dropped because the served endpoints answer
# them with "max_tokens cannot be greater than max_model_len" or time out.
TOKEN_BUDGET_LADDER = [8192, 4096, 2048, 1024, 512, 256, 128, 64]


def test_stratified_selection_balances_sources_languages_specialties_and_labels():
    def sample(index, source, language, specialty, reference):
        return SimpleNamespace(
            source_file=source,
            language=language,
            specialty=specialty,
            answer_format="label",
            reference_answer_normalized=reference,
            reference_answer=reference,
            metadata={},
            sample_id=f"s{index}",
        )

    candidates = [
        sample(0, "a.json", "en", "first", "A"),
        sample(1, "a.json", "en", "first", "A"),
        sample(2, "a.json", "en", "second", "B"),
        sample(3, "b.json", "zh", "third", "C"),
    ]

    selected = _stratified_select(candidates, 3)

    assert [item.sample_id for item in selected] == ["s0", "s3", "s2"]


def _executor(
    tmp_path,
    client,
    generation,
    *,
    same_budget_error_retries=2,
    same_budget_timeout_retries=1,
):
    return Executor(
        client=client, run_id="run", provider="mock", generation=generation,
        media=MediaConfig(), output=OutputConfig(),
        retry_policy=RetryPolicy(max_retries=0, initial_seconds=0, max_seconds=0),
        recorder=Recorder(tmp_path),
        same_budget_error_retries=same_budget_error_retries,
        same_budget_timeout_retries=same_budget_timeout_retries,
    )


async def test_context_error_follows_configured_budget_ladder_one_step_at_a_time(tmp_path):
    error = ClientError(
        ErrorType.INVALID_REQUEST,
        "This model's maximum context length is 16384 tokens. However, your request has "
        "11800 input tokens.", http_status=400,
    )
    client = MockClient(behaviours=[error, error, error, {"content": "A"}])
    generation = GenerationConfig(
        max_tokens=8192,
        max_tokens_candidates=[8192, 6400, 4096, 2048],
    )
    result = await _executor(tmp_path, client, generation).execute(
        SAMPLE, SAMPLE["logical_messages"], 0,
    )

    # A context overflow is deterministic: repeating the same budget cannot fix it, so every
    # error moves exactly one tier down the ladder and burns no same-budget retry.
    assert result.status == "success"
    assert result.effective_max_tokens == 2048
    assert result.max_tokens_adjustment_reason == "context_error"
    assert result.adaptive_retry_count == 3
    attempts = read_jsonl(tmp_path / "attempts.jsonl")
    assert [a["effective_max_tokens"] for a in attempts] == [8192, 6400, 4096, 2048]
    assert attempts[0]["provider_metadata"]["server_reported_input_tokens"] == 11800
    assert attempts[0]["request_start_time"] and attempts[0]["latency_seconds"] is not None


async def test_configured_single_same_budget_retry_then_reduces(tmp_path):
    timeout = ClientError(ErrorType.API_TIMEOUT, "timed out")
    client = MockClient(behaviours=[timeout, timeout, {"content": "A"}])
    generation = GenerationConfig(max_tokens=8192, max_tokens_candidates=[8192, 4096])
    result = await _executor(
        tmp_path, client, generation, same_budget_error_retries=1,
    ).execute(
        SAMPLE, SAMPLE["logical_messages"], 0,
    )
    assert result.status == "success" and result.effective_max_tokens == 4096
    attempts = read_jsonl(tmp_path / "attempts.jsonl")
    assert [a["effective_max_tokens"] for a in attempts] == [8192, 8192, 4096]
    assert all(a["request_start_time"] and a["request_end_time"] for a in attempts)


async def test_error_immediately_follows_next_budget_when_same_budget_retries_disabled(tmp_path):
    timeout = ClientError(ErrorType.API_TIMEOUT, "timed out")
    client = MockClient(behaviours=[timeout, {"content": "A"}])
    generation = GenerationConfig(
        max_tokens=TOKEN_BUDGET_LADDER[0],
        max_tokens_candidates=TOKEN_BUDGET_LADDER,
    )
    result = await _executor(
        tmp_path,
        client,
        generation,
        same_budget_error_retries=0,
    ).execute(SAMPLE, SAMPLE["logical_messages"], 0)

    assert result.status == "success"
    assert result.effective_max_tokens == TOKEN_BUDGET_LADDER[1]
    attempts = read_jsonl(tmp_path / "attempts.jsonl")
    assert [attempt["effective_max_tokens"] for attempt in attempts] == TOKEN_BUDGET_LADDER[:2]


async def test_ladder_is_walked_for_retryable_errors_and_skipped_for_deterministic_ones(tmp_path):
    """The ladder is a remedy, not a ritual.

    Walking a tier costs a full request, so it is only worth doing when a smaller output
    budget could plausibly change the outcome. That is true for retryable failures (an
    overloaded or slow server may well answer a smaller request) and false for deterministic
    rejections: a 401/403/400 is the server refusing this request as written, and re-asking
    with fewer output tokens returns the identical error. Those must fail on the first
    attempt instead of spending the whole ladder.
    """
    deterministic = [
        ClientError(ErrorType.AUTHENTICATION_ERROR, "invalid token", http_status=401),
        ClientError(ErrorType.PERMISSION_ERROR, "forbidden", http_status=403),
        ClientError(ErrorType.INVALID_REQUEST, "invalid request", http_status=400),
    ]
    for index, error in enumerate(deterministic):
        client = MockClient(behaviours=[error] * 32)
        generation = GenerationConfig(
            max_tokens=TOKEN_BUDGET_LADDER[0],
            max_tokens_candidates=TOKEN_BUDGET_LADDER,
        )
        run_dir = tmp_path / f"deterministic_{index}"
        result = await _executor(run_dir, client, generation).execute(
            SAMPLE, SAMPLE["logical_messages"], 0,
        )

        assert result.status == "error", error.error_type
        assert client.calls == 1, error.error_type
        assert result.retry_count == 0
        assert result.adaptive_retry_count == 0
        assert result.effective_max_tokens == TOKEN_BUDGET_LADDER[0]
        attempts = read_jsonl(run_dir / "attempts.jsonl")
        assert [a["effective_max_tokens"] for a in attempts] == TOKEN_BUDGET_LADDER[:1]

    # A retryable error does walk the ladder: two same-budget retries per tier, then one tier
    # down, for at most transient_error_ladder_steps=2 tiers. Nine attempts is the ceiling.
    retryable = ClientError(ErrorType.SERVER_ERROR, "server failed", http_status=500)
    client = MockClient(behaviours=[*([retryable] * 8), {"content": "A"}])
    generation = GenerationConfig(
        max_tokens=TOKEN_BUDGET_LADDER[0],
        max_tokens_candidates=TOKEN_BUDGET_LADDER,
    )
    run_dir = tmp_path / "retryable"
    result = await _executor(run_dir, client, generation).execute(
        SAMPLE, SAMPLE["logical_messages"], 0,
    )

    assert result.status == "success"
    assert result.effective_max_tokens == TOKEN_BUDGET_LADDER[2]
    assert result.adaptive_retry_count == 2
    attempts = read_jsonl(run_dir / "attempts.jsonl")
    assert [attempt["effective_max_tokens"] for attempt in attempts] == [
        budget for budget in TOKEN_BUDGET_LADDER[:3] for _ in range(3)
    ]


async def test_second_same_budget_retry_can_succeed_without_reducing(tmp_path):
    error = ClientError(ErrorType.SERVER_ERROR, "server failed", http_status=500)
    client = MockClient(behaviours=[error, error, {"content": "A"}])
    generation = GenerationConfig(
        max_tokens=TOKEN_BUDGET_LADDER[0],
        max_tokens_candidates=TOKEN_BUDGET_LADDER,
    )

    result = await _executor(tmp_path, client, generation).execute(
        SAMPLE, SAMPLE["logical_messages"], 0,
    )

    assert result.status == "success"
    assert result.effective_max_tokens == TOKEN_BUDGET_LADDER[0]
    assert result.adaptive_retry_count == 0
    attempts = read_jsonl(tmp_path / "attempts.jsonl")
    assert [attempt["effective_max_tokens"] for attempt in attempts] == [
        TOKEN_BUDGET_LADDER[0]
    ] * 3


async def test_persistent_transient_error_stops_after_two_ladder_steps(tmp_path):
    """A persistently failing server must not be handed all eight tiers.

    runtime.transient_error_ladder_steps=2 caps how far a timeout/5xx may descend, so the
    run gives up at the third tier instead of burning 24 doomed requests per sample. The
    ladder's remaining tiers stay reserved for context errors, which are unlimited.
    """
    error = ClientError(ErrorType.SERVER_ERROR, "server failed", http_status=500)
    client = MockClient(behaviours=[error] * 32)
    generation = GenerationConfig(
        max_tokens=TOKEN_BUDGET_LADDER[0],
        max_tokens_candidates=TOKEN_BUDGET_LADDER,
    )

    result = await _executor(tmp_path, client, generation).execute(
        SAMPLE, SAMPLE["logical_messages"], 0,
    )

    assert result.status == "error"
    assert result.effective_max_tokens == TOKEN_BUDGET_LADDER[2]
    # 3 tiers x (1 attempt + 2 same-budget retries) = 9 attempts, i.e. 8 retries.
    assert result.retry_count == 8
    assert result.adaptive_retry_count == 2
    attempts = read_jsonl(tmp_path / "attempts.jsonl")
    assert [attempt["effective_max_tokens"] for attempt in attempts] == [
        budget for budget in TOKEN_BUDGET_LADDER[:3] for _ in range(3)
    ]


async def test_sample_output_budget_caps_request_attempt_and_result(tmp_path):
    client = MockClient(behaviours=[{"content": "A"}])
    generation = GenerationConfig(max_tokens=8192, max_tokens_candidates=[8192, 4096, 64])
    sample = {**SAMPLE, "metadata": {"request_max_tokens": 64}}

    result = await _executor(tmp_path, client, generation).execute(
        sample, sample["logical_messages"], 0,
    )

    assert result.requested_max_tokens == 64
    assert result.effective_max_tokens == 64
    attempt = read_jsonl(tmp_path / "attempts.jsonl")[0]
    assert attempt["requested_max_tokens"] == 64
    assert attempt["effective_max_tokens"] == 64


async def test_adaptive_attempt_does_not_consume_transient_retry_budget(tmp_path):
    context = ClientError(
        ErrorType.INVALID_REQUEST,
        "maximum context length is 16384 tokens; request has 11800 input tokens",
        http_status=400,
    )
    transient = ClientError(ErrorType.SERVER_ERROR, "temporary", http_status=500)
    client = MockClient(behaviours=[context, transient, {"content": "A"}])
    generation = GenerationConfig(max_tokens=8192, max_tokens_candidates=[8192, 4096])
    executor = _executor(tmp_path, client, generation)
    executor.retry_policy.max_retries = 1
    result = await executor.execute(SAMPLE, SAMPLE["logical_messages"], 0)
    assert result.status == "success" and client.calls == 3


async def test_resume_backfills_missing_judgment_without_inference(tmp_path):
    recorder = Recorder(tmp_path)
    recorder.record_result({
        "result_id": "r1", "run_id": "run", "sample_id": "urn:s1",
        "sample_repeat_index": 0, "benchmark_name": "MMLU", "status": "success",
        "raw_response": "A", "reference_answer": "A",
    })
    index = ResumeIndex.from_run_dir(tmp_path)

    def score(result, sample, only_evaluators=None):
        assert only_evaluators == {"multiple_choice_accuracy"}
        result.parsed_answer = "A"
        result.parsing_status = "success"
        return [JudgmentRecord(
            judgment_id="j1", run_id="run", result_id=result.result_id,
            sample_id=result.sample_id, evaluator_name="multiple_choice_accuracy",
            normalized_score=1.0, raw_score=1.0, is_correct=True,
            provider_metadata={"primary_metric": True},
        )]

    client = MockClient()
    executor = _executor(tmp_path, client, GenerationConfig(max_tokens=64))
    runner = Runner(
        executor=executor, recorder=recorder, concurrency=1, resume_index=index,
        score_fn=score, expected_evaluators=["multiple_choice_accuracy"],
    )
    report = await runner.run([SAMPLE])
    assert client.calls == 0 and report["counts"]["skipped"] == 1
    assert len(read_jsonl(tmp_path / "judgments.jsonl")) == 1
    assert read_jsonl(tmp_path / "results.jsonl")[-1]["evaluation_status"] == "success"


def test_resume_config_mismatch_does_not_change_any_jsonl(tmp_path, monkeypatch):
    cfg = RunConfig(
        experiment={"experiment_id": "resume", "run_name": "resume"},
        benchmark={"name": "MMLU/mcqa", "max_samples": 1},
        model={"base_url": "http://mock/v1", "requested_model_name": "mock"},
        generation={"max_tokens": 64}, runtime={"max_retries": 0},
        output={"root_dir": str(tmp_path)},
        evaluation={"evaluator": "multiple_choice"},
    )
    run_dir = tmp_path / "run"
    probe = run_setup.RunOrchestrator(cfg, run_dir=str(run_dir))
    samples = probe.prepare_samples()
    key = {samples[0]["source_content"]["question"]: (samples[0]["reference_answer"], samples[0]["sample_id"])}
    monkeypatch.setattr(run_setup, "OpenAICompatibleClient", lambda **kw: _AnswerKeyClient(key))
    probe.run()
    before = {p.name: p.read_bytes() for p in run_dir.glob("*.jsonl")}

    changed = cfg.model_copy(deep=True)
    changed.generation.max_tokens = 32
    with pytest.raises(run_setup.RunSetupError, match="config changed"):
        run_setup.RunOrchestrator(changed, run_dir=str(run_dir)).run()
    assert {p.name: p.read_bytes() for p in run_dir.glob("*.jsonl")} == before


def _context_budget_cfg(tmp_path, **overrides):
    generation = {"max_tokens": 64}
    generation.update(overrides.pop("generation", {}))
    return RunConfig(
        experiment={"experiment_id": "ctx", "run_name": "ctx"},
        benchmark={"name": "MMLU/mcqa", "max_samples": 1},
        model={"base_url": "http://mock/v1", "requested_model_name": "mock"},
        generation=generation, runtime={"max_retries": 0},
        output={"root_dir": str(tmp_path)},
        evaluation={"evaluator": "multiple_choice"},
        **overrides,
    )


def test_head_tail_without_a_window_is_refused_before_any_request(tmp_path):
    """head_tail with no max_model_len asks for trimming and silently gets none.

    No client is installed: reaching the network at all would mean the gate ran too late,
    so a connection error here is a real failure rather than an artifact of the test.
    """
    cfg = _context_budget_cfg(tmp_path, generation={"context_overflow_policy": "head_tail"})
    with pytest.raises(run_setup.RunSetupError, match="head_tail"):
        run_setup.RunOrchestrator(cfg, run_dir=str(tmp_path / "run")).run()


def test_output_budget_larger_than_the_window_is_refused_when_no_rung_fits(tmp_path):
    """max_tokens=262144 against a 16k window makes every request invalid.

    With the ladder off there is no lower rung to fall back to, so the whole run can only
    produce 400s — the measured 95.7% invalid-request rate. Refuse it at setup.
    """
    cfg = _context_budget_cfg(tmp_path, generation={
        "max_tokens": 262144, "adaptive_max_tokens": False,
    }, hardware={"max_model_len": 16384})
    with pytest.raises(run_setup.RunSetupError, match="No usable context budget"):
        run_setup.RunOrchestrator(cfg, run_dir=str(tmp_path / "run")).run()


def test_output_budget_the_ladder_can_rescue_only_warns(tmp_path, monkeypatch):
    """Same mismatch, but a reachable rung fits: wasteful, not fatal, so the run proceeds."""
    cfg = _context_budget_cfg(tmp_path, generation={
        "max_tokens": 262144, "adaptive_max_tokens": True,
        "max_tokens_candidates": [8192, 4096, 64],
    }, hardware={"max_model_len": 16384})
    probe = run_setup.RunOrchestrator(cfg, run_dir=str(tmp_path / "run"))
    samples = probe.prepare_samples()
    key = {samples[0]["source_content"]["question"]: (samples[0]["reference_answer"], samples[0]["sample_id"])}
    monkeypatch.setattr(run_setup, "OpenAICompatibleClient", lambda **kw: _AnswerKeyClient(key))
    with pytest.warns(UnbudgetedContextWarning, match="adaptive ladder drops to 64"):
        result = probe.run()
    assert result["status"] in {"completed", "completed_with_errors"}


class _ConcurrentJudgeClient:
    def __init__(self):
        self.active = 0
        self.peak = 0

    async def chat_completion(self, messages, **kwargs):
        from healthcorebench.clients.openai_client import ModelResponse
        from healthcorebench.utils.timestamps import utc_now_iso
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        now = utc_now_iso()
        return ModelResponse(
            content='{"correct": true, "rationale": "a \\"quoted\\" reason"}',
            model_requested="judge", model_returned="judge", finish_reason="stop",
            request_start_time=now, request_end_time=now, latency_seconds=0.01,
            prompt_tokens=1, completion_tokens=1, total_tokens=2, raw_response={},
        )


async def test_judge_concurrency_and_exact_prompt_audit():
    client = _ConcurrentJudgeClient()
    evaluator = LLMJudgeEvaluator(client=client, judge_model="judge", concurrency=2, max_retries=0)
    result = {
        "run_id": "run", "result_id": "r", "sample_id": "s", "status": "success",
        "formatted_prompt": "q", "reference_answer": "a", "raw_response": "a",
    }
    judgments = await asyncio.gather(*(evaluator.evaluate_async(
        {**result, "result_id": f"r{i}", "sample_id": f"s{i}"}, {},
    ) for i in range(6)))
    assert client.peak <= 2
    assert all(j.evaluation_status == "success" for j in judgments)
    assert judgments[0].judge_rationale == 'a "quoted" reason'
    assert judgments[0].evaluator_prompt and judgments[0].evaluator_prompt_hash
    assert judgments[0].evaluator_request_hash
    assert judgments[0].provider_metadata["judge_parse_mode"] == "strict_json"


def test_judge_prompt_includes_aliases_and_task_rubric():
    evaluator = LLMJudgeEvaluator(client=_ConcurrentJudgeClient(), judge_model="judge")
    messages = evaluator._build_messages(
        {"formatted_prompt": "q", "reference_answer": "primary", "raw_response": "alternate"},
        {"reference_aliases": ["alternate"], "metadata": {"scoring_points": ["key point"]}},
    )
    prompt = messages[0]["content"]
    assert "primary" in prompt and "alternate" in prompt and "key point" in prompt


def test_judge_supports_bounded_partial_credit():
    correct, score, rationale = LLMJudgeEvaluator._parse_judge(
        '{"score": 0.5, "correct": false, "rationale": "partially covered"}'
    )
    assert correct is False and score == 0.5 and rationale == "partially covered"


def test_continuous_scores_use_bootstrap_and_judge_tokens_are_latest_wins(tmp_path):
    append_jsonl(tmp_path / "samples.jsonl", {"sample_id": "s", "benchmark_name": "B", "answer_format": "text"})
    append_jsonl(tmp_path / "results.jsonl", {
        "result_id": "r", "run_id": "run", "sample_id": "s", "benchmark_name": "B",
        "status": "success", "parsing_status": "success",
    })
    for score, tokens in ((0.1, 10), (0.8, 20)):
        append_jsonl(tmp_path / "judgments.jsonl", {
            "judgment_id": f"j{tokens}", "run_id": "run", "result_id": "r", "sample_id": "s",
            "evaluator_type": "llm_judge", "evaluator_name": "llm_judge",
            "evaluation_status": "success", "normalized_score": score,
            "judge_total_tokens": tokens, "provider_metadata": {"primary_metric": True},
        })
    summary = summarize_run(tmp_path)
    assert summary.metrics.score == 0.8
    assert summary.metrics.confidence_interval_method.startswith("percentile_bootstrap")
    assert summary.tokens["current_effective_evaluation_tokens"] == 20


def test_context_error_and_multiline_final_answer():
    assert parse_multiple_choice_letter(
        "Therefore, the correct answer is:\n\nC", list("ABCD")
    ) == "C"
    with pytest.raises(ContextOverflowError):
        fit_context_to_window(
            "context", fixed_prompt="q", max_model_len=100, max_output_tokens=100,
            reserve_tokens=10, policy="head_tail",
        )
    with pytest.raises(ContextOverflowError):
        fit_context_to_window(
            "x" * 1000, fixed_prompt="q", max_model_len=100,
            max_output_tokens=10, reserve_tokens=10, policy="error",
        )


async def test_length_finish_is_visible_and_not_scored(tmp_path):
    recorder = Recorder(tmp_path)
    client = MockClient(behaviours=[{"content": "A", "finish_reason": "length"}])
    executor = _executor(tmp_path, client, GenerationConfig(max_tokens=64))

    def score(result, sample):
        raise AssertionError("mark_incomplete must not score a truncated response")

    def parse(result, sample):
        result.parsed_answer = "A"
        result.parsing_status = "success"

    score.parse_result = parse
    runner = Runner(
        executor=executor, recorder=recorder, concurrency=1, score_fn=score,
        expected_evaluators=["multiple_choice_accuracy"],
        length_finish_policy="mark_incomplete",
    )
    report = await runner.run([SAMPLE])
    assert report["counts"]["max_length"] == 1
    assert report["counts"]["missing_scoring"] == 1
    result = read_jsonl(tmp_path / "results.jsonl")[0]
    assert result["raw_response"] == "A" and result["parsed_answer"] == "A"
    assert result["evaluation_status"] == "skipped"
    assert result["evaluation_skip_reason"] == "max_length"


def test_validate_allows_completed_zero_success_task(tmp_path):
    append_jsonl(tmp_path / "samples.jsonl", {"sample_id": "s", "benchmark_name": "B"})
    append_jsonl(tmp_path / "results.jsonl", {
        "result_id": "r", "run_id": "run", "sample_id": "s", "benchmark_name": "B",
        "status": "error", "error_type": "api_timeout",
    })
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_status": "completed_with_errors", "generation": {"n": 1},
        "full_config": {"evaluation": {"evaluator": "multiple_choice"}},
    }))
    atomic_write_json(tmp_path / "summary.json", summarize_run(tmp_path).model_dump())
    assert validate_run(tmp_path)["valid"] is True


def test_validate_reports_legacy_summary_as_warning_not_corruption(tmp_path):
    append_jsonl(tmp_path / "samples.jsonl", {
        "sample_id": "s", "benchmark_name": "B", "reference_answer": "A",
    })
    append_jsonl(tmp_path / "results.jsonl", {
        "run_id": "r", "result_id": "r", "sample_id": "s", "benchmark_name": "B",
        "status": "error", "error_type": "api_timeout",
    })
    atomic_write_json(tmp_path / "manifest.json", {
        "run_id": "r", "run_status": "completed_with_errors", "benchmark": {"name": "B"},
        "generation": {"n": 1}, "full_config": {"evaluation": {}},
    })
    atomic_write_json(tmp_path / "summary.json", {
        "summary_code_version": "0.9", "counts": {"num_total": 1},
    })

    report = validate_run(tmp_path)

    assert report["valid"] is True
    assert len(report["warnings"]) == 1
    assert "Re-run summarize" in report["warnings"][0]


def test_all_benchmark_configs_validate():
    from healthcorebench.config import load_config
    text_config = load_config("configs/run_all_benchmarks_text.yaml")
    multimodal_config = load_config("configs/run_all_benchmarks_multimodal.yaml")

    assert text_config.evaluation.judge is not None
    assert text_config.model.api_key is not None
    assert text_config.evaluation.judge.api_key is not None
    assert text_config.generation.max_tokens_candidates == TOKEN_BUDGET_LADDER
    assert multimodal_config.generation.max_tokens_candidates == TOKEN_BUDGET_LADDER
    assert len(multimodal_config.benchmark.name.split(",")) == 56
    assert multimodal_config.evaluation.judge is not None
    assert multimodal_config.evaluation.judge.api_key is not None
    assert multimodal_config.media.max_images == 64
    assert multimodal_config.media.max_video_frames == 32


def test_example_text_uses_configured_judge_only_for_judged_tasks(tmp_path):
    from healthcorebench.config import load_config

    for benchmark, expected_judge, expected_evaluator in (
        ("MMLU/mcqa", False, "multiple_choice"),
        ("MedQuAD/open", True, None),
    ):
        config = load_config("configs/example_text.yaml", {
            "benchmark.name": benchmark,
            "benchmark.max_samples": 1,
        })
        orchestrator = run_setup.RunOrchestrator(
            config, run_dir=str(tmp_path / benchmark.replace("/", "_")),
        )
        samples = orchestrator.prepare_samples()
        orchestrator._resolve_evaluation(config, samples)

        assert config.evaluation.judge is not None
        assert config.evaluation.use_llm_judge is expected_judge
        assert config.evaluation.evaluator == expected_evaluator
