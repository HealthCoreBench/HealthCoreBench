"""MedMCQA adapter.

Fixed data: ``3_MedMCQA/medmcqa_test.json`` — a JSON list of records::

    {"id": str, "question": str, "opa": str, "opb": str, "opc": str, "opd": str,
     "cop": int (0-3, correct option index), "choice_type": "single"|"multi",
     "exp": str|None, "subject_name": str, "topic_name": str|None}

Although ``choice_type`` may be "multi", the dataset stores a single ``cop`` index, so it is
scored as single-choice (A-D) here. ``choice_type`` is preserved in metadata.
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

_LETTERS = ["A", "B", "C", "D"]


class MedMCQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedMCQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedMCQA provides only 'test'; requested '{self.split}'.")
        return [directory / "medmcqa_test.json"]

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

        question = rec["question"]
        choices = [rec["opa"], rec["opb"], rec["opc"], rec["opd"]]
        cop = int(rec["cop"])
        correct_letter = _LETTERS[cop]
        block, letters = format_lettered_choices([str(c) for c in choices], _LETTERS)

        source_id = rec.get("id") or f"{rel}:{rec_index}"
        content_hash = self.input_hash({"q": question, "c": choices})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=str(source_id), content_hash=content_hash)

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=str(source_id),
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
            specialty=rec.get("subject_name"),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "choice_type": rec.get("choice_type"),
                      "topic_name": rec.get("topic_name")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]], _LETTERS)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        letters = sample.metadata.get("letters") or _LETTERS
        return parse_multiple_choice_letter(raw_response, letters)
