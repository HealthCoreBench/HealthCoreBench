"""DiagnosisArena adapter.

Fixed data: ``51_DiagnosisArena/diagnosisarena_test.json`` — a JSON list of records::

    {"id": int, "Case Information": str, "Physical Examination": str,
     "Diagnostic Tests": str, "Final Diagnosis": str,
     "Options": {"A": str, "B": str, "C": str, "D": str}, "Right Option": "A".."D"}

Task: single-choice (A-D). The clinical vignette (case info + physical exam + diagnostic
tests) forms the question stem; the model picks the correct diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample


def _build_stem(rec: dict) -> str:
    parts = []
    for label, key in (("Case Information", "Case Information"),
                       ("Physical Examination", "Physical Examination"),
                       ("Diagnostic Tests", "Diagnostic Tests")):
        val = rec.get(key)
        if val:
            parts.append(f"{label}: {val}")
    parts.append("What is the most likely diagnosis?")
    return "\n\n".join(parts)


class DiagnosisArenaAdapter(BaseBenchmarkAdapter):
    benchmark_name = "DiagnosisArena"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"DiagnosisArena provides only 'test'; requested '{self.split}'.")
        return [directory / "diagnosisarena_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = _build_stem(rec)
        options = rec["Options"]
        letters = sorted(options.keys())
        block = "\n".join(f"{l}. {options[l]}" for l in letters)
        answer_letter = str(rec["Right Option"]).strip().upper()

        source_id = str(rec.get("id", f"{rel}:{rec_index}"))
        content_hash = self.input_hash({"q": question, "o": options})
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
            input_hash=self.input_hash({"question": question, "choices_block": block}),
            reference_hash=self.reference_hash(answer_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Reasoning",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "options": options, "letters": letters},
            reference_answer=answer_letter,
            reference_answer_normalized=answer_letter,
            metadata={"letters": letters, "final_diagnosis": rec.get("Final Diagnosis")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        letters = c["letters"]
        block = "\n".join(f"{l}. {c['options'][l]}" for l in letters)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
