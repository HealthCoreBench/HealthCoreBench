"""Integration tests for the Runner: concurrency, per-sample flush, resume, scoring."""

from healthcorebench.runtime.executor import Executor
from healthcorebench.runtime.recorder import Recorder
from healthcorebench.runtime.retry import RetryPolicy
from healthcorebench.runtime.runner import Runner
from healthcorebench.runtime.resume import ResumeIndex
from healthcorebench.schemas.config import GenerationConfig, MediaConfig, OutputConfig
from healthcorebench.schemas.judgment import JudgmentRecord
from healthcorebench.utils.jsonl import read_jsonl
from tests.mock_client import MockClient


def _make(tmp_path, client, score_fn=None, resume_index=None):
    rec = Recorder(tmp_path)
    ex = Executor(client=client, run_id="run1", provider="mock",
                  generation=GenerationConfig(), media=MediaConfig(), output=OutputConfig(),
                  retry_policy=RetryPolicy(max_retries=1, initial_seconds=0.001, max_seconds=0.01),
                  recorder=rec)
    runner = Runner(executor=ex, recorder=rec, concurrency=4, score_fn=score_fn,
                    resume_index=resume_index or ResumeIndex())
    return runner, rec


def _samples(n):
    return [
        {"sample_id": f"urn:s{i}", "sample_index": i, "benchmark_name": "MMLU",
         "benchmark_split": "test", "reference_answer": "A", "logical_messages": [{"role": "user", "content": f"Q{i}"}]}
        for i in range(n)
    ]


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
