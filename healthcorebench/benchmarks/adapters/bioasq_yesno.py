"""BioASQ adapter (biomedical QA — yes/no subset, English).

Fixed data: ``43_BioASQ/*/*.json`` — each file is ``{"questions": [...]}`` where a question has
``body``, ``type`` in {yesno, factoid, list, summary}, ``exact_answer`` and ``ideal_answer``.

This adapter exposes only the ``yesno`` questions as a two-way classification task (the
factoid/list/summary types need free-text/LLM-judge scoring and are handled separately). The
reference is ``exact_answer`` normalized to "yes"/"no"; scored with the ``classification``
evaluator. Questions are de-duplicated by ``id`` (the golden files overlap across Task years).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.answer_parsing import parse_yes_no_maybe
from healthcorebench.schemas.sample import EvaluationSample


def _norm_yesno(v: Any) -> str | None:
    s = (v[0] if isinstance(v, list) and v else v)
    s = str(s or "").strip().lower()
    if s in ("yes", "no"):
        return s
    return None


class BioASQYesNoAdapter(BaseBenchmarkAdapter):
    benchmark_name = "BioASQ"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "yes_no"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"BioASQ (yesno) provides only 'test'; requested '{self.split}'.")
        files = sorted(p for p in directory.rglob("*.json") if p.is_file())
        return files

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        seen: set[str] = set()
        for f in files:
            rel = self.rel_path(f)
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict) or "questions" not in data:
                continue
            for i, q in enumerate(data.get("questions", [])):
                if q.get("type") != "yesno":
                    continue
                label = _norm_yesno(q.get("exact_answer"))
                if label is None or not str(q.get("body") or "").strip():
                    continue
                qid = str(q.get("id") or f"{rel}:{i}")
                if qid in seen:
                    continue
                seen.add(qid)
                yield {"question": q, "label": label, "qid": qid,
                       "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        q = raw_sample["question"]
        label: str = raw_sample["label"]
        qid: str = raw_sample["qid"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        body = str(q["body"]).strip()
        content_hash = self.input_hash({"q": body})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=qid, content_hash=content_hash)

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=qid,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=rec_index,
            source_record_hash=self.input_hash(q),
            input_hash=self.input_hash({"body": body}),
            reference_hash=self.reference_hash(label),
            input_type="text",
            task_type="classification",
            component="Language",
            capability="Knowledge",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="yes_no",
            evaluation_metric="accuracy",
            source_content={"body": body},
            reference_answer=label,
            reference_answer_normalized=label,
            metadata={"labels": ["yes", "no"]},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        body = sample.source_content["body"]
        prompt = f"{body}\nAnswer with exactly one word: yes or no."
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        v = parse_yes_no_maybe(raw_response)
        # this subset is strictly binary; a "maybe" parse counts as undecided.
        return v if v in ("yes", "no") else None
