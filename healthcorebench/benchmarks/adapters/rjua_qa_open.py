"""RJUA-QA adapter (Chinese urology clinical QA).

Fixed data: ``57_RJUA-QA/rjua_test.json`` — a JSON list of records::

    {"id": str, "question": str, "context": str (clinical background/guidelines),
     "answer": str, "disease": str, "advice": str}

Task: open-ended free-text answering, Chinese. The ``context`` (clinical background) is
provided to the model along with the question; an LLM judge scores the answer against the
reference ``answer``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


class RJUAQAOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "RJUA-QA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"RJUA-QA provides only 'test'; requested '{self.split}'.")
        return [directory / "rjua_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            if not str(rec.get("answer") or "").strip():
                continue
            yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = rec["question"]
        context = str(rec.get("context") or "").strip()
        reference = str(rec["answer"]).strip()

        source_id = str(rec.get("id") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"q": question, "ctx": context})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=source_id, content_hash=content_hash)

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=source_id,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=rec_index,
            source_record_hash=self.input_hash(rec),
            input_hash=self.input_hash({"question": question, "context": context}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("disease"),
            language="zh",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"question": question, "context": context},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"disease": rec.get("disease"), "advice": rec.get("advice")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        ctx = c.get("context")
        if ctx:
            content = f"参考资料：\n{ctx}\n\n问题：{c['question']}"
        else:
            content = c["question"]
        return [{"role": "user", "content": content}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
