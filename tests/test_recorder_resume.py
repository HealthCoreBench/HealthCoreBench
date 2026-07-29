"""Unit tests: JSONL append/read, recorder, resume indexing, duplicate detection."""

from healthcorebench.utils.jsonl import append_jsonl, read_jsonl, count_jsonl, atomic_write_json
from healthcorebench.runtime import Recorder, ResumeIndex


def test_append_and_read_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"a": 2})
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]
    assert count_jsonl(p) == 2


def test_jsonl_skips_torn_trailing_line(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, {"a": 1})
    with open(p, "a") as f:
        f.write('{"a": 2, "b":')  # torn line, no newline
    assert read_jsonl(p) == [{"a": 1}]


def test_atomic_write_json(tmp_path):
    p = tmp_path / "m.json"
    atomic_write_json(p, {"k": "v"})
    import json
    assert json.loads(p.read_text()) == {"k": "v"}
    assert not (tmp_path / "m.json.tmp").exists()


def test_recorder_and_resume_roundtrip(tmp_path):
    rec = Recorder(tmp_path)
    rec.record_sample({"sample_id": "urn:s1", "sample_index": 0, "benchmark_name": "B", "benchmark_split": "test"})
    rec.record_result({"result_id": "r1", "run_id": "run", "sample_id": "urn:s1",
                       "sample_repeat_index": 0, "benchmark_name": "B", "status": "success"})
    rec.record_result({"result_id": "r2", "run_id": "run", "sample_id": "urn:s2",
                       "sample_repeat_index": 0, "benchmark_name": "B", "status": "error"})
    rec.record_judgment({"judgment_id": "j1", "run_id": "run", "result_id": "r1",
                         "sample_id": "urn:s1", "evaluator_name": "mc", "evaluation_status": "success"})

    idx = ResumeIndex.from_run_dir(tmp_path)
    assert idx.has_success("urn:s1", 0)
    assert idx.has_failure("urn:s2", 0)
    assert idx.is_judged("r1", "mc")
    assert not idx.is_judged("r1", "exact_match")
    assert idx.sample_recorded("urn:s1")


def test_manifest_status_update_preserves_fields(tmp_path):
    rec = Recorder(tmp_path)
    rec.write_manifest({"run_id": "run", "run_status": "running", "keep": "me"})
    rec.update_manifest_status("completed", extra={"end_time": "2026-01-01T00:00:00Z"})
    m = rec.read_manifest()
    assert m["run_status"] == "completed"
    assert m["keep"] == "me"  # not wiped
    assert m["end_time"] == "2026-01-01T00:00:00Z"


def test_success_supersedes_prior_failure(tmp_path):
    rec = Recorder(tmp_path)
    rec.record_result({"result_id": "rA", "run_id": "run", "sample_id": "urn:s", "sample_repeat_index": 0,
                       "benchmark_name": "B", "status": "error"})
    rec.record_result({"result_id": "rB", "run_id": "run", "sample_id": "urn:s", "sample_repeat_index": 0,
                       "benchmark_name": "B", "status": "success"})
    idx = ResumeIndex.from_run_dir(tmp_path)
    assert idx.has_success("urn:s", 0)
    assert not idx.has_failure("urn:s", 0)
