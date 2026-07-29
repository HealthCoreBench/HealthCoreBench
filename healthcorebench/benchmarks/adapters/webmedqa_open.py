"""webMedQA adapter (Chinese consumer medical QA).

Fixed data: ``58_webMedQA/webmedqa_test.json`` — a JSON list of records::

    {"question_id": str, "department": str, "question": str,
     "candidates": [{"answer": str, "label": int}, ...]}

Task: open-ended free-text answering, Chinese. Each question has exactly one best-answer
candidate (``label == 1``), used as the reference; an LLM judge scores the model's answer
against it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


class WebMedQAOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "webMedQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"webMedQA provides only 'test'; requested '{self.split}'.")
        return [directory / "webmedqa_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            best = [c for c in rec.get("candidates", []) if c.get("label") == 1]
            if not best or not str(best[0].get("answer") or "").strip():
                continue  # no usable reference answer
            yield {"record": rec, "reference": str(best[0]["answer"]).strip(),
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = rec["question"]
        reference = raw_sample["reference"]

        source_id = str(rec.get("question_id") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"q": question})
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
            input_hash=self.input_hash({"question": question}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Knowledge",
            specialty=rec.get("department"),
            language="zh",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"question": question},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"department": rec.get("department")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": sample.source_content["question"]}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
