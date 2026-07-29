"""MMLU (medical subjects) adapter.

Fixed data: ``benchmarks/medical_llm_benchmarks/1_MMLU/`` holds per-subject ``*_test.json``
files, each a JSON list of records::

    {"question": str, "subject": str, "choices": [str, ...], "answer": int}

``answer`` is the 0-based index into ``choices``. The ``test`` split maps to the six
medical/biology subject files (matching the merged ``mmlu-med-bio_test.json`` of 1,089
records). The merged file itself is excluded from discovery to avoid double-counting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

# The six medical/biology MMLU subjects that constitute the medical test split.
_MEDICAL_SUBJECT_FILES = [
    "anatomy_test.json",
    "clinical_knowledge_test.json",
    "college_biology_test.json",
    "college_medicine_test.json",
    "medical_genetics_test.json",
    "professional_medicine_test.json",
]


class MMLUAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MMLU"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(
                f"MMLU only provides a 'test' split; requested '{self.split}'."
            )
        files = [directory / name for name in _MEDICAL_SUBJECT_FILES]
        # Deterministic order = the fixed list above (sorted by subject file name is also stable).
        return files

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
            rel = self.rel_path(f)
            with open(f, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            for i, rec in enumerate(records):
                yield {
                    "record": rec,
                    "source_file_rel": rel,
                    "source_record_index": i,
                }

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = rec["question"]
        choices = [str(c) for c in rec["choices"]]
        answer_index = int(rec["answer"])
        block, letters = format_lettered_choices(choices)
        correct_letter = letters[answer_index]
        subject = rec.get("subject")

        # source id: subject + within-file index (stable within a fixed file)
        source_sample_id = f"{subject}:{rec_index}"
        content_hash = self.input_hash({"question": question, "choices": choices})
        sample_id = self.make_sample_id(
            source_file_rel=rel, source_sample_id=source_sample_id, content_hash=content_hash
        )

        # input_hash is over the normalized model-facing content (question + lettered choices),
        # independent of prompt template wording.
        input_payload = {"question": question, "choices_block": block}

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=source_sample_id,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=rec_index,
            source_record_hash=self.input_hash(rec),
            input_hash=self.input_hash(input_payload),
            reference_hash=self.reference_hash(correct_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=subject,
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices, "subject": subject},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "answer_index": answer_index},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        content = sample.source_content
        block, _ = format_lettered_choices([str(c) for c in content["choices"]])
        prompt = multiple_choice_prompt(content["question"], block, lang="en")
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        letters = sample.metadata.get("letters") or ["A", "B", "C", "D"]
        return parse_multiple_choice_letter(raw_response, letters)
