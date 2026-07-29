"""MeQSum adapter (consumer health question summarization).

Fixed data: ``37_MeQSum/meqsum_test.json`` — a JSON list of records::

    {"File": str, "CHQ": str (long consumer health question), "Summary": str (short summary)}

Task: summarization. The model summarizes the long CHQ; an LLM judge scores the produced
summary against the reference ``Summary``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

_INSTRUCTION = "Summarize the following patient health question into a single concise question:\n\n"


class MeQSumSummarizationAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MeQSum"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "summarization"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MeQSum provides only 'test'; requested '{self.split}'.")
        return [directory / "meqsum_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            if not str(rec.get("Summary") or "").strip():
                continue
            yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        chq = str(rec["CHQ"]).strip()
        reference = str(rec["Summary"]).strip()

        source_id = str(rec.get("File") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"chq": chq})
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
            input_hash=self.input_hash({"chq": chq}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="summarization",
            component="Language",
            capability="Summarization",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="summary",
            # summarization — ROUGE-1/2/L is the standard metric (BLEU added as a secondary).
            evaluation_metric="rouge",
            source_content={"question": chq},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"file": rec.get("File")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        chq = sample.source_content["question"]
        return [{"role": "user", "content": _INSTRUCTION + chq}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
