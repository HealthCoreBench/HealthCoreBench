"""KorMedMCQA adapter (Korean medical licensing exams).

Fixed data: ``38_KorMedMCQA/kormedmcqa_<subject>_test.json`` — four subjects
(doctor, nurse, pharm, dentist), each a JSON list of records::

    {"subject": str, "year": int, "period": int, "q_number": int, "question": str,
     "A": str, "B": str, "C": str, "D": str, "E": str, "answer": int (1-based), "cot": str}

Task: single-choice (A-E), Korean. ``answer`` is 1-based into A-E.
The four subjects are exposed as splits: ``doctor`` | ``nurse`` | ``pharm`` | ``dentist``.
Default split ``test`` maps to ``doctor``.
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

_LETTERS = ["A", "B", "C", "D", "E"]
_SUBJECTS = ("doctor", "nurse", "pharm", "dentist")


class KorMedMCQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "KorMedMCQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _subjects(self) -> tuple[str, ...]:
        # "test" (default) aggregates all four subjects; a named split takes one.
        if self.split == "test":
            return _SUBJECTS
        if self.split in _SUBJECTS:
            return (self.split,)
        raise BenchmarkSplitNotFoundError(
            f"KorMedMCQA split must be one of {_SUBJECTS} (or 'test'=all); got '{self.split}'."
        )

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / f"kormedmcqa_{s}_test.json" for s in self._subjects()]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
            rel = self.rel_path(f)
            with open(f, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            for i, rec in enumerate(records):
                yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = rec["question"]
        choices = [rec[l] for l in _LETTERS]
        block, letters = format_lettered_choices([str(c) for c in choices], _LETTERS)
        correct_letter = _LETTERS[int(rec["answer"]) - 1]  # answer is 1-based

        source_id = f"{rec.get('year')}-{rec.get('period')}-{rec.get('q_number')}"
        content_hash = self.input_hash({"q": question, "c": choices})
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
            reference_hash=self.reference_hash(correct_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=rec.get("subject"),
            language="ko",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "year": rec.get("year")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]], _LETTERS)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or _LETTERS)
