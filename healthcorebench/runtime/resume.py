"""Resume index: which (sample_id, repeat) are already done, at each stage.

Built by scanning existing JSONL on startup. The logical key is
``(sample_id, sample_repeat_index)``. We track, per key:

* whether a successful result exists (inference done),
* whether a failed result exists (final failure recorded),
* which evaluator judgments already exist (so scoring isn't repeated).

This lets the runner skip completed inference, only backfill scoring for results that lack
it, and — under ``--retry-failed`` — re-request only failed samples. Append-only files are
never rewritten; resume is purely additive.
"""

from __future__ import annotations

from pathlib import Path

from healthcorebench.utils.jsonl import iter_jsonl


class ResumeIndex:
    def __init__(self) -> None:
        self.successful_results: dict[tuple[str, int], str] = {}   # key -> result_id
        self.failed_results: dict[tuple[str, int], str] = {}       # key -> result_id
        self.results_by_id: dict[str, dict] = {}                   # latest persisted result
        self.judged: dict[str, set[str]] = {}                      # result_id -> {evaluator_name}
        self.recorded_samples: set[str] = set()

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "ResumeIndex":
        idx = cls()
        run_dir = Path(run_dir)

        for rec in iter_jsonl(run_dir / "samples.jsonl"):
            sid = rec.get("sample_id")
            if sid:
                idx.recorded_samples.add(sid)

        for rec in iter_jsonl(run_dir / "results.jsonl"):
            sid = rec.get("sample_id")
            rep = rec.get("sample_repeat_index", 0)
            rid = rec.get("result_id")
            if sid is None or rid is None:
                continue
            key = (sid, rep)
            idx.results_by_id[rid] = rec
            if rec.get("status") == "success":
                idx.successful_results[key] = rid
                idx.failed_results.pop(key, None)
            else:
                # only mark failed if not already superseded by a success
                if key not in idx.successful_results:
                    idx.failed_results[key] = rid

        for rec in iter_jsonl(run_dir / "judgments.jsonl"):
            rid = rec.get("result_id")
            name = rec.get("evaluator_name")
            if rid and name and rec.get("evaluation_status") == "success":
                idx.judged.setdefault(rid, set()).add(name)

        return idx

    def has_success(self, sample_id: str, repeat: int) -> bool:
        return (sample_id, repeat) in self.successful_results

    def has_failure(self, sample_id: str, repeat: int) -> bool:
        return (sample_id, repeat) in self.failed_results

    def result_id_for(self, sample_id: str, repeat: int) -> str | None:
        key = (sample_id, repeat)
        return self.successful_results.get(key) or self.failed_results.get(key)

    def is_judged(self, result_id: str, evaluator_name: str) -> bool:
        return evaluator_name in self.judged.get(result_id, set())

    def result_for(self, sample_id: str, repeat: int) -> dict | None:
        result_id = self.result_id_for(sample_id, repeat)
        return self.results_by_id.get(result_id) if result_id else None

    def sample_recorded(self, sample_id: str) -> bool:
        return sample_id in self.recorded_samples
