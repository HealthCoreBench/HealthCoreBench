"""MedicationQA adapter (open-ended medication question answering).

Fixed data: ``26_MedicationQA/medicationqa_test.json`` — a JSON list of records::

    {"Question": str, "Focus (Drug)": str, "Question Type": str, "Answer": str,
     "Section Title": str, "URL": str}

Task: open-ended free-text answering, scored by an LLM judge against the reference
``Answer``. ``parse_response`` returns the model's raw text (no extraction); the judge
compares it to the reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


class MedicationQAOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedicationQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedicationQA provides only 'test'; requested '{self.split}'.")
        return [directory / "medicationqa_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            # Skip records with an empty/blank reference answer (unscorable by the judge).
            if not str(rec.get("Answer") or "").strip():
                continue
            yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = rec["Question"]
        reference = str(rec["Answer"]).strip()

        source_id = f"{rel}:{rec_index}"
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
            specialty=rec.get("Question Type"),
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"question": question, "focus_drug": rec.get("Focus (Drug)")},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"section_title": rec.get("Section Title"), "url": rec.get("URL")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        q = sample.source_content["question"]
        return [{"role": "user", "content": q}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        # Open-ended: the answer is the raw model text; the judge scores it.
        return (raw_response or "").strip()
