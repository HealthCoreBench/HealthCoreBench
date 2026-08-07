"""Integration tests for the Runner: concurrency, per-sample flush, resume, scoring."""

import asyncio

import pytest

from healthcorebench.clients.openai_client import ModelResponse
from healthcorebench.clients.errors import ErrorType
from healthcorebench.runtime.executor import Executor
from healthcorebench.runtime.recorder import Recorder
from healthcorebench.runtime.retry import RetryPolicy
from healthcorebench.runtime.runner import Runner
from healthcorebench.runtime.resume import ResumeIndex
from healthcorebench.schemas.config import GenerationConfig, MediaConfig, OutputConfig
from healthcorebench.schemas.judgment import JudgmentRecord
from healthcorebench.utils.jsonl import read_jsonl
from healthcorebench.utils.timestamps import utc_now_iso
from tests.mock_client import MockClient, err


def _make(tmp_path, client, score_fn=None, resume_index=None, concurrency=4):
    rec = Recorder(tmp_path)
    ex = Executor(client=client, run_id="run1", provider="mock",
                  generation=GenerationConfig(), media=MediaConfig(), output=OutputConfig(),
                  retry_policy=RetryPolicy(max_retries=1, initial_seconds=0.001, max_seconds=0.01),
                  recorder=rec)
    runner = Runner(executor=ex, recorder=rec, concurrency=concurrency, score_fn=score_fn,
                    resume_index=resume_index or ResumeIndex())
    return runner, rec


def _samples(n):
    return [
        {"sample_id": f"urn:s{i}", "sample_index": i, "benchmark_name": "MMLU",
         "benchmark_split": "test", "reference_answer": "A", "logical_messages": [{"role": "user", "content": f"Q{i}"}]}
        for i in range(n)
    ]


class _ConcurrencyProbeClient:
    """Keep calls open briefly so the Runner's real in-flight request count is observable."""

    requested_model_name = "probe-model"

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def chat_completion(self, messages, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.02)
        finally:
            self.active -= 1
        now = utc_now_iso()
        return ModelResponse(
            content="A", model_requested=self.requested_model_name,
            model_returned=self.requested_model_name, finish_reason="stop",
            request_start_time=now, request_end_time=now, latency_seconds=0.02,
            raw_response={"choices": [{"message": {"content": "A"}}]},
        )


class _MessageRecordingClient(MockClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = []

    async def chat_completion(self, messages, **kwargs):
        self.messages.append(messages)
        return await super().chat_completion(messages, **kwargs)


def _liveclin_samples(*, include_second_case: bool = True):
    samples = []
    for case_index, case_id in enumerate(("case-a", "case-b")):
        if case_index == 1 and not include_second_case:
            break
        for stage_index in range(2):
            samples.append({
                "sample_id": f"urn:{case_id}:q{stage_index}",
                "sample_index": len(samples),
                "benchmark_name": "LiveClin",
                "benchmark_split": "test",
                "source_file": "liveclin.json",
                "source_record_index": case_index,
                "metadata": {
                    "conversation_mode": "case_sequential",
                    "case_id": case_id,
                    "stage_index": stage_index,
                },
                "reference_answer": "A",
                "logical_messages": [{
                    "role": "user", "content": f"{case_id}:q{stage_index}",
                }],
            })
    return samples


@pytest.mark.parametrize(("concurrency", "expected_peak"), [(1, 1), (3, 3)])
async def test_runtime_concurrency_controls_in_flight_model_requests(
    tmp_path, concurrency, expected_peak,
):
    client = _ConcurrencyProbeClient()
    runner, _ = _make(tmp_path, client, concurrency=concurrency)

    report = await runner.run(_samples(6))

    assert report["counts"]["succeeded"] == 6
    assert client.peak == expected_peak


async def test_case_sequential_protocol_carries_history_and_isolates_cases(tmp_path):
    client = _MessageRecordingClient(
        behaviours=[
            {"content": "A0"}, {"content": "A1"},
            {"content": "B0"}, {"content": "B1"},
        ]
    )
    runner, _ = _make(tmp_path, client, concurrency=1)

    report = await runner.run(_liveclin_samples())

    assert report["counts"]["succeeded"] == 4
    assert [[message["role"] for message in request] for request in client.messages] == [
        ["user"], ["user", "assistant", "user"],
        ["user"], ["user", "assistant", "user"],
    ]
    assert [message["content"] for message in client.messages[1]] == [
        "case-a:q0", "A0", "case-a:q1",
    ]
    assert [message["content"] for message in client.messages[3]] == [
        "case-b:q0", "B0", "case-b:q1",
    ]


async def test_case_sequential_resume_rebuilds_history_from_persisted_response(tmp_path):
    first_client = _MessageRecordingClient(behaviours=[{"content": "A0"}])
    first_runner, recorder = _make(tmp_path, first_client, concurrency=1)
    await first_runner.run(_liveclin_samples(include_second_case=False)[:1])

    resume_index = ResumeIndex.from_run_dir(tmp_path)
    second_client = _MessageRecordingClient(behaviours=[{"content": "A1"}])
    second_runner, _ = _make(
        tmp_path, second_client, concurrency=1, resume_index=resume_index,
    )
    await second_runner.run(_liveclin_samples(include_second_case=False))

    assert len(second_client.messages) == 1
    assert [message["content"] for message in second_client.messages[0]] == [
        "case-a:q0", "A0", "case-a:q1",
    ]
    assert len(read_jsonl(recorder.path("results.jsonl"))) == 2


async def test_case_sequential_does_not_request_next_stage_without_previous_response(tmp_path):
    client = _MessageRecordingClient(
        behaviours=[err(ErrorType.AUTHENTICATION_ERROR, http_status=401)],
    )
    runner, recorder = _make(tmp_path, client, concurrency=1)

    report = await runner.run(_liveclin_samples(include_second_case=False))

    assert report["counts"]["failed"] == 1
    assert client.calls == 1
    assert len(client.messages) == 1
    assert [message["content"] for message in client.messages[0]] == ["case-a:q0"]
    assert len(read_jsonl(recorder.path("results.jsonl"))) == 1


async def test_runner_runs_all_and_flushes(tmp_path):
    runner, rec = _make(tmp_path, MockClient(behaviours=[{"content": "A"}]))
    report = await runner.run(_samples(5))
    assert report["counts"]["succeeded"] == 5
    results = read_jsonl(rec.path("results.jsonl"))
    assert len(results) == 5
    # every result flushed with success + raw response present
    assert all(r["status"] == "success" and r["raw_response"] == "A" for r in results)


async def test_runner_scoring_callback(tmp_path):
    def score_fn(result, sample):
        correct = (result.raw_response or "").strip().upper().startswith(sample["reference_answer"])
        return [JudgmentRecord(judgment_id=f"j_{result.result_id}", run_id="run1",
                               result_id=result.result_id, sample_id=result.sample_id,
                               evaluator_name="mc_accuracy", raw_score=1.0 if correct else 0.0,
                               normalized_score=1.0 if correct else 0.0, is_correct=correct)]
    runner, rec = _make(tmp_path, MockClient(behaviours=[{"content": "A"}]), score_fn=score_fn)
    await runner.run(_samples(3))
    judgments = read_jsonl(rec.path("judgments.jsonl"))
    assert len(judgments) == 3
    assert all(j["is_correct"] for j in judgments)


async def test_runner_surfaces_primary_evaluation_failure(tmp_path):
    def score_fn(result, sample):
        return [JudgmentRecord(
            judgment_id=f"j_{result.result_id}", run_id="run1",
            result_id=result.result_id, sample_id=result.sample_id,
            evaluator_type="llm_judge", evaluator_name="llm_judge",
            evaluation_status="error", evaluation_error="invalid judge response",
            provider_metadata={"primary_metric": True},
        )]

    runner, rec = _make(tmp_path, MockClient(behaviours=[{"content": "A"}]), score_fn=score_fn)
    report = await runner.run(_samples(1))

    assert report["counts"]["scored"] == 0
    assert report["counts"]["evaluation_failed"] == 1
    result = read_jsonl(rec.path("results.jsonl"))[0]
    assert result["evaluation_status"] == "error"


async def test_runner_resume_skips_completed(tmp_path):
    # first run
    runner, rec = _make(tmp_path, MockClient(behaviours=[{"content": "A"}]))
    await runner.run(_samples(3))
    assert len(read_jsonl(rec.path("results.jsonl"))) == 3
    # second run resumes: index sees all done -> no new results
    idx = ResumeIndex.from_run_dir(tmp_path)
    runner2, rec2 = _make(tmp_path, MockClient(behaviours=[{"content": "A"}]), resume_index=idx)
    report = await runner2.run(_samples(3))
    assert report["counts"]["skipped"] == 3
    assert report["counts"]["attempted"] == 0
    # results file unchanged (still 3, no duplicates)
    assert len(read_jsonl(rec2.path("results.jsonl"))) == 3
