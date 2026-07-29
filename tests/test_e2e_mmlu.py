"""End-to-end test: full MMLU run against a mock client, then summary from JSONL.

Patches the client so each request returns the sample's correct letter (extracted from the
prompt's reference via a side channel), exercising: sample loading, manifest, attempts,
results, judgments, summary recomputation, resume (no duplicate work), and standalone
summarize equivalence.
"""

import json
from pathlib import Path

import pytest

import healthcorebench.runtime.run_setup as run_setup
from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.clients.openai_client import ModelResponse
from healthcorebench.schemas.config import RunConfig
from healthcorebench.runtime.run_setup import RunSetupError
from healthcorebench.utils.jsonl import read_jsonl
from healthcorebench.utils.timestamps import utc_now_iso


class _AnswerKeyClient:
    """Mock client that answers each question with the reference letter.

    It maps the prompt text back to the reference by matching the question stem against a
    provided answer key keyed on a substring of the user message.
    """

    def __init__(self, answer_key: dict, requested_model_name="mock", wrong_ids=None):
        self.answer_key = answer_key
        self.requested_model_name = requested_model_name
        self.base_url = "http://mock/v1"
        self.wrong_ids = wrong_ids or set()
        self.calls = 0

    async def list_models(self):
        return ["mock-served"]

    async def chat_completion(self, messages, **kwargs):
        self.calls += 1
        user = messages[-1]["content"]
        text = user if isinstance(user, str) else " ".join(p.get("text", "") for p in user)
        letter = "A"
        matched_id = None
        # pick the most specific (longest) matching question to avoid stem collisions
        best_len = -1
        for key, (ref, sid) in self.answer_key.items():
            if key in text and len(key) > best_len:
                best_len = len(key)
                letter = ref
                matched_id = sid
        if matched_id in self.wrong_ids:
            # deliberately answer wrong
            letter = "Z"
        start = utc_now_iso()
        return ModelResponse(
            content=f"The answer is {letter}.", model_requested=self.requested_model_name,
            model_returned="mock-served", system_fingerprint="fp", provider_request_id="req",
            finish_reason="stop", prompt_tokens=12, completion_tokens=4, total_tokens=16,
            request_start_time=start, request_end_time=start, latency_seconds=0.001,
            raw_response={"id": "req", "model": "mock-served"},
        )

    async def aclose(self):
        pass


def _build_answer_key(adapter, samples):
    key = {}
    for s in samples:
        # use the question stem (first 40 chars) as the match key
        q = s.source_content.get("question", "")
        key[q[:40]] = (s.reference_answer, s.sample_id)
    return key


def _cfg(tmp_path, max_samples=8):
    return RunConfig(
        experiment={"experiment_id": "test_exp", "run_name": "mmlu_mock"},
        benchmark={"name": "MMLU/mcqa", "split": "test", "max_samples": max_samples},
        model={"base_url": "http://mock/v1", "requested_model_name": "mock"},
        generation={"temperature": 0.0, "max_tokens": 64, "n": 1},
        runtime={"concurrency": 4, "max_retries": 1, "resume": True},
        output={"root_dir": str(tmp_path)},
        evaluation={"evaluator": "multiple_choice"},
    )


def test_full_mmlu_run_and_summary(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, max_samples=8)
    orch = run_setup.RunOrchestrator(cfg, run_dir=str(tmp_path / "run"))

    # Prepare samples to build an answer key, then patch the client.
    samples = orch.prepare_samples()
    key = {}
    for d in samples:
        q = d["source_content"].get("question", "")
        key[q] = (d["reference_answer"], d["sample_id"])
    # answer the 2nd sample wrong to exercise a non-perfect score
    wrong = {samples[1]["sample_id"]}

    monkeypatch.setattr(run_setup, "OpenAICompatibleClient",
                        lambda **kw: _AnswerKeyClient(key, wrong_ids=wrong))

    result = orch.run()
    run_dir = Path(result["run_dir"])

    # artifacts exist
    for fn in ("manifest.json", "samples.jsonl", "attempts.jsonl", "results.jsonl",
               "judgments.jsonl", "summary.json", "events.jsonl"):
        assert (run_dir / fn).exists(), f"missing {fn}"
    assert not (run_dir / "prices.json").exists()

    results = read_jsonl(run_dir / "results.jsonl")
    judgments = read_jsonl(run_dir / "judgments.jsonl")
    assert len(results) == 8
    assert all(r["status"] == "success" for r in results)
    assert sum(r["parsed_answer"] is not None for r in results) == 7
    assert sum(r["normalized_answer"] is not None for r in results) == 7
    assert all(r["parser_name"] == "MMLUAdapter" for r in results)
    assert sum(r["parsing_status"] == "success" for r in results) == 7
    assert sum(r["parsing_status"] == "error" for r in results) == 1
    assert all(r["evaluation_status"] == "success" for r in results)
    assert len(judgments) == 8

    # summary: 7 of the 8 responses are scorable and all 7 are correct. The 8th is the
    # deliberately unparsable one; it leaves the denominator as ``unscorable`` rather than
    # scoring 0, so "the parser broke" and "the model was wrong" stay distinguishable. The
    # count identity has to hold across that split.
    summary = json.loads((run_dir / "summary.json").read_text())
    counts = summary["counts"]
    assert counts["num_scored"] == 7
    assert counts["num_unscorable"] == 1
    assert counts["unscorable_reasons"] == {"unparsed_answer": 1}
    assert counts["num_parsing_errors"] == 1
    assert counts["num_successful"] == (
        counts["num_scored"] + counts["num_missing_scoring"] + counts["num_evaluation_errors"]
        + counts["num_evaluation_skipped"] + counts["num_unscorable"]
    )
    assert abs(summary["metrics"]["score"] - 1.0) < 1e-9
    assert summary["metrics"]["confidence_interval"] is not None
    assert "cost" not in summary
    assert all("cost" not in attempt for attempt in read_jsonl(run_dir / "attempts.jsonl"))
    assert all(not any("cost" in key for key in row) for row in results)

    # manifest completed, model identity captured
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_status"] in ("completed", "completed_with_errors")
    assert manifest["model"]["returned_model_names"] == ["mock-served"]
    assert manifest["benchmark"]["selected_num_samples"] == 8
    assert manifest["execution_identity"]["adapter_name"] == "MMLUAdapter"
    assert manifest["execution_identity"]["selected_samples_hash"]
    assert "pricing" not in manifest
    # no API key anywhere in manifest
    assert "sk-" not in json.dumps(manifest)

    # standalone summarize matches embedded summary metric
    recomputed = summarize_run(run_dir)
    assert abs(recomputed.metrics.score - summary["metrics"]["score"]) < 1e-12


def test_resume_no_duplicate_requests(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, max_samples=5)
    run_dir = str(tmp_path / "run")

    orch = run_setup.RunOrchestrator(cfg, run_dir=run_dir)
    samples = orch.prepare_samples()
    key = {d["source_content"]["question"]: (d["reference_answer"], d["sample_id"]) for d in samples}

    client_holder = {}
    def make_client(**kw):
        c = _AnswerKeyClient(key)
        client_holder["last"] = c
        return c
    monkeypatch.setattr(run_setup, "OpenAICompatibleClient", make_client)

    orch.run()
    first_calls = client_holder["last"].calls
    assert first_calls == 5

    # second run resumes; should make zero new requests
    orch2 = run_setup.RunOrchestrator(cfg, run_dir=run_dir)
    orch2.run()
    assert client_holder["last"].calls == 0
    # results file still has exactly 5 (no duplicates)
    assert len(read_jsonl(Path(run_dir) / "results.jsonl")) == 5


def test_resume_rejects_changed_adapter_identity(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, max_samples=2)
    run_dir = str(tmp_path / "run")
    first = run_setup.RunOrchestrator(cfg, run_dir=run_dir)
    samples = first.prepare_samples()
    key = {d["source_content"]["question"]: (d["reference_answer"], d["sample_id"])
           for d in samples}
    monkeypatch.setattr(run_setup, "OpenAICompatibleClient", lambda **kw: _AnswerKeyClient(key))
    first.run()

    resumed = run_setup.RunOrchestrator(cfg, run_dir=run_dir)
    resumed.adapter.adapter_version = "test-incompatible-version"
    with pytest.raises(RunSetupError, match="execution identity"):
        resumed.run()


def test_save_source_content_false_redacts_persisted_samples(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, max_samples=2)
    cfg.output.save_source_content = False
    orchestrator = run_setup.RunOrchestrator(cfg, run_dir=str(tmp_path / "run"))
    samples = orchestrator.prepare_samples()
    key = {d["source_content"]["question"]: (d["reference_answer"], d["sample_id"])
           for d in samples}
    monkeypatch.setattr(run_setup, "OpenAICompatibleClient", lambda **kw: _AnswerKeyClient(key))

    orchestrator.run()

    persisted = read_jsonl(tmp_path / "run" / "samples.jsonl")
    assert len(persisted) == 2
    assert all(sample["source_content"] == {} for sample in persisted)


class _FlakyClient(_AnswerKeyClient):
    """Answers correctly, except that a chosen sample fails until ``heal`` is set.

    Models the case the report's re-run guidance is about: the endpoint was unavailable for
    part of a run, so some samples have a recorded failure and no score.
    """

    def __init__(self, answer_key, fail_stem: str, **kw):
        super().__init__(answer_key, **kw)
        self.fail_stem = fail_stem
        self.heal = False

    async def chat_completion(self, messages, **kwargs):
        user = messages[-1]["content"]
        text = user if isinstance(user, str) else " ".join(p.get("text", "") for p in user)
        if self.fail_stem in text and not self.heal:
            from healthcorebench.clients.errors import ClientError, ErrorType

            # Non-retryable so the sample fails terminally in one attempt: the point of the
            # test is the *next* run, not the within-run retry ladder.
            raise ClientError(
                ErrorType.SERVICE_UNAVAILABLE, "upstream unavailable",
                http_status=503, exception_class="ClientError", retryable=False,
            )
        return await super().chat_completion(messages, **kwargs)


def _run_once(cfg, run_dir, client):
    import healthcorebench.runtime.run_setup as rs

    orch = rs.RunOrchestrator(cfg, run_dir=str(run_dir))
    samples = orch.prepare_samples()
    key = {d["source_content"].get("question", ""): (d["reference_answer"], d["sample_id"])
           for d in samples}
    client.answer_key = key
    return orch, samples


def test_retry_failed_fills_the_gap_a_previous_run_left(tmp_path, monkeypatch):
    """A failed sample stays unscored forever unless ``retry_failed`` re-requests it.

    This is the mechanism the batch report points operators at, so it needs a test that a
    second run actually converts the failure into a score — and that without the flag it does
    not, which is why the shipped configs set it.
    """
    run_dir = tmp_path / "run"
    cfg = _cfg(tmp_path, max_samples=4)

    probe = run_setup.RunOrchestrator(cfg, run_dir=str(tmp_path / "probe"))
    stems = [d["source_content"].get("question", "") for d in probe.prepare_samples()]
    doomed = stems[1]

    client = _FlakyClient({}, fail_stem=doomed)
    monkeypatch.setattr(run_setup, "OpenAICompatibleClient", lambda **kw: client)

    orch, _ = _run_once(cfg, run_dir, client)
    first = orch.run()
    assert first["summary_metrics"] is not None
    results = read_jsonl(Path(first["run_dir"]) / "results.jsonl")
    assert sum(1 for r in results if r["status"] != "success") == 1, "expected one failure"
    scored_before = json.loads((Path(first["run_dir"]) / "summary.json")
                               .read_text(encoding="utf-8"))["counts"]["num_scored"]
    assert scored_before == 3

    # The endpoint recovers. A plain resume leaves the failure alone: that is the behaviour
    # that made one wobble permanent in the report.
    client.heal = True
    orch2, _ = _run_once(cfg, run_dir, client)
    orch2.run()
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")
                      )["counts"]["num_scored"] == 3

    # With retry_failed the same second run picks it up, and does not redo the other three.
    healed = cfg.model_copy(deep=True)
    healed.runtime.retry_failed = True
    client.calls = 0
    orch3, _ = _run_once(healed, run_dir, client)
    orch3.run()
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")
                      )["counts"]["num_scored"] == 4
    assert client.calls == 1, f"only the failed sample should be re-requested, got {client.calls}"
