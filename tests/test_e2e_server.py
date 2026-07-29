"""End-to-end test against a real mock HTTP server (exercises the true AsyncOpenAI client).

Covers: real HTTP request path, /v1/models identity probe, server-side 500 then retry
success, and the full tool chain reparse → rescore → summarize → validate → export-parquet.
"""

import json
from pathlib import Path

from healthcorebench.runtime.run_setup import RunOrchestrator
from healthcorebench.schemas.config import RunConfig
from healthcorebench.tools import reparse_run, rescore_run, validate_run, export_parquet
from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.utils.jsonl import read_jsonl
from tests.mock_server import MockServer, MockServerState


def _cfg(tmp_path, base_url, max_retries=3):
    return RunConfig(
        experiment={"experiment_id": "srv_exp", "run_name": "mmlu_srv"},
        benchmark={"name": "MMLU/mcqa", "split": "test", "max_samples": 6},
        model={"base_url": base_url, "requested_model_name": "req-model", "provider": "local_vllm"},
        generation={"temperature": 0.0, "max_tokens": 32, "n": 1},
        runtime={"concurrency": 3, "max_retries": max_retries,
                 "retry_backoff_initial_seconds": 0.001, "retry_backoff_max_seconds": 0.01, "resume": True},
        output={"root_dir": str(tmp_path)},
        evaluation={"evaluator": "multiple_choice"},
    )


def test_e2e_against_real_http_server(tmp_path):
    state = MockServerState()
    state.answer_letter = "A"
    state.fail_first_n = 2  # first two chat calls 500, then succeed (tests retry over HTTP)
    with MockServer(state) as server:
        cfg = _cfg(tmp_path, server.base_url)
        orch = RunOrchestrator(cfg, run_dir=str(tmp_path / "run"))
        result = orch.run()
        run_dir = Path(result["run_dir"])

    # manifest captured served model via /v1/models
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["model"]["returned_model_names"] == ["mock-served-model"]
    assert manifest["run_status"] in ("completed", "completed_with_errors")

    results = read_jsonl(run_dir / "results.jsonl")
    attempts = read_jsonl(run_dir / "attempts.jsonl")
    assert len(results) == 6
    assert all(r["status"] == "success" for r in results)
    # the two injected failures were recorded as separate attempts
    assert sum(1 for a in attempts if a["status"] == "error") == 2
    assert sum(1 for a in attempts if a["status"] == "success") == 6

    # tool chain works offline
    rep = reparse_run(run_dir)
    assert rep["reparsed"] == 6
    res = rescore_run(run_dir)
    assert res["rescored"] == 6
    summary = summarize_run(run_dir)
    assert summary.counts.num_scored >= 6
    report = validate_run(run_dir)
    assert report["valid"], report["issues"]
    exp = export_parquet(run_dir)
    assert "results" in exp["written"]
    assert (run_dir / "parquet" / "results.parquet").exists()


def test_missing_usage_is_null(tmp_path):
    state = MockServerState()
    state.return_usage = False
    with MockServer(state) as server:
        cfg = _cfg(tmp_path, server.base_url, max_retries=1)
        orch = RunOrchestrator(cfg, run_dir=str(tmp_path / "run"))
        orch.run()
        run_dir = Path(tmp_path / "run")
    results = read_jsonl(run_dir / "results.jsonl")
    assert all(r["prompt_tokens"] is None and r["total_tokens"] is None for r in results)
