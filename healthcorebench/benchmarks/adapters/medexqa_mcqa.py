"""MedExQA adapter.

Fixed data: ``25_MedExQA/<specialty>_test.json`` — five specialties, each a JSON list::

    {"question": str, "options": {"A": str, "B": str, "C": str, "D": str},
     "explanation_1": str, "explanation_2": str, "answer": "A".."D", "specialty": str}

Task: single-choice (A-D), English. The five specialties are exposed as splits:
biomedical_engineer | clinical_laboratory_scientist | clinical_psychologist |
occupational_therapist | speech_pathologist. Default split ``test`` maps to the first.
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

_SPECIALTIES = (
    "biomedical_engineer",
    "clinical_laboratory_scientist",
    "clinical_psychologist",
    "occupational_therapist",
    "speech_pathologist",
)


class MedExQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedExQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _specialties(self) -> tuple[str, ...]:
        # "test" (default) aggregates all five specialties; a named split takes one.
        if self.split == "test":
            return _SPECIALTIES
        if self.split in _SPECIALTIES:
            return (self.split,)
        raise BenchmarkSplitNotFoundError(
            f"MedExQA split must be one of {_SPECIALTIES} (or 'test'=all); got '{self.split}'."
        )

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / f"{s}_test.json" for s in self._specialties()]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
            rel = self.rel_path(f)
            # one file per specialty, so the file name is an exact fallback for the field.
            specialty = f.name.removesuffix("_test.json")
            with open(f, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            for i, rec in enumerate(records):
                yield {"record": rec, "source_file_rel": rel, "source_record_index": i,
                       "specialty": specialty}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = rec["question"]
        options = rec["options"]
        letters = sorted(options.keys())
        block = "\n".join(f"{l}. {options[l]}" for l in letters)
        answer_letter = str(rec["answer"]).strip().upper()

        source_id = f"{rel}:{rec_index}"
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
            capability="Knowledge",
            specialty=rec.get("specialty") or raw_sample.get("specialty"),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "options": options, "letters": letters},
            reference_answer=answer_letter,
            reference_answer_normalized=answer_letter,
            metadata={"letters": letters},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        letters = c["letters"]
        block = "\n".join(f"{l}. {c['options'][l]}" for l in letters)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
