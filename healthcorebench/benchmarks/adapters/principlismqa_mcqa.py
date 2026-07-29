"""PrinciplismQA adapter (medical-ethics knowledge MCQA, English).

Fixed data: ``67_PrinciplismQA/data/knowledge-mcqa.json`` — a JSON list of records::

    {"id": int, "question_id": int, "question": str,
     "options": {"A": str, "B": str, "C": str, "D": str}, "correct_answer": str (letter),
     "explanation": str, "principlism": str}

Task: single-choice medical-ethics questions. ``options`` is a letter->text dict and
``correct_answer`` is the correct letter. Options are re-lettered locally in sorted key order
for a canonical prompt. (The benchmark's open-ended split is a separate task.)
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


class PrinciplismQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "PrinciplismQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"PrinciplismQA MCQA provides only 'test'; requested '{self.split}'.")
        return [directory / "data" / "knowledge-mcqa.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            opts = rec.get("options") or {}
            ca = str(rec.get("correct_answer", "")).strip().upper()
            if len(opts) < 2 or ca not in {str(k).upper() for k in opts}:
                continue
            yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(rec["question"]).strip()
        src_keys = sorted(rec["options"], key=lambda k: str(k).upper())
        choices = [str(rec["options"][k]) for k in src_keys]
        block, letters = format_lettered_choices(choices)
        correct_pos = [str(k).upper() for k in src_keys].index(str(rec["correct_answer"]).strip().upper())
        correct_letter = letters[correct_pos]

        source_id = str(rec.get("id") or rec.get("question_id") or f"{rel}:{rec_index}")
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
            capability="Reasoning",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "principlism": rec.get("principlism")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
