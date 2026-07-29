"""MEDEC medical-error correction task for the 311 erroneous test notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


class MEDECCorrectionAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MEDEC"
    benchmark_version = "1.0"
    prompt_template_name = "error_correction"

    def discover_source_files(self) -> list[Path]:
        if self.split != "test":
            raise BenchmarkSplitNotFoundError("MEDEC correction provides only 'test'.")
        return [self.get_benchmark_directory() / "medec_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        path = files[0]
        rel = self.rel_path(path)
        for index, record in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            text = str(record.get("Text") or "").strip()
            corrected = str(record.get("Corrected Text") or "").strip()
            if record.get("Error Flag") == 1 and text and corrected:
                yield {"record": record, "text": text, "corrected": corrected,
                       "source_file_rel": rel, "source_record_index": index}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        record = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        index = raw_sample["source_record_index"]
        source_id = str(record.get("Text ID") or f"{rel}:{index}")
        return EvaluationSample(
            sample_id=self.make_sample_id(source_file_rel=rel, source_sample_id=source_id,
                                          content_hash=self.input_hash(raw_sample["text"])),
            source_sample_id=source_id, sample_index=sample_index, benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version, benchmark_split=self.split, source_file=rel,
            source_record_index=index, source_record_hash=self.input_hash(record),
            input_hash=self.input_hash(raw_sample["text"]),
            reference_hash=self.reference_hash(raw_sample["corrected"]), task_type="error_correction",
            component="Language", capability="Reasoning", specialty=record.get("Error Type"),
            language="en", modality="Text", answer_format="free_text", evaluation_metric="llm_judge",
            source_content={"text": raw_sample["text"]}, reference_answer=raw_sample["corrected"],
            reference_answer_normalized=raw_sample["corrected"],
            metadata={"error_sentence_id": record.get("Error Sentence ID"),
                      "corrected_sentence": record.get("Corrected Sentence")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        prompt = (
            "The clinical note below contains a medical error. Correct the erroneous medical "
            "statement while preserving the rest of the note. Return the corrected note.\n\n"
            f"{sample.source_content['text']}"
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
