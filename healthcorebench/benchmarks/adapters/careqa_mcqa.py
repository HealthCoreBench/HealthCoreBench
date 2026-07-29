"""CareQA adapter (Spanish MIR medical exam QA, English/Spanish).

Fixed data: ``10_CareQA/CareQA_<lang>.json`` — a JSON list of records::

    {"exam_id": int, "question": str, "op1": str, "op2": str, "op3": str, "op4": str,
     "cop": int (correct option, 1-based), "year": int, "category": str, "unique_id": str}

Task: single-choice, four options. ``cop`` is the correct option number (1-4). Languages are
exposed as splits (``en`` default, ``es``); ``test`` maps to en. The separate
``CareQA_<lang>_open.json`` free-text variant is a different task and handled elsewhere.
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

_LANGS = {"test": "en", "en": "en", "es": "es"}


class CareQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "CareQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _lang(self) -> str:
        if self.split not in _LANGS:
            raise BenchmarkSplitNotFoundError(f"CareQA split must be one of {sorted(_LANGS)}; got '{self.split}'.")
        return _LANGS[self.split]

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / f"CareQA_{self._lang()}.json"]

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
        lang = self._lang()

        question = rec["question"]
        choices = [str(rec[f"op{n}"]) for n in (1, 2, 3, 4)]
        block, letters = format_lettered_choices(choices)
        correct_pos = int(rec["cop"]) - 1  # cop is 1-based
        correct_letter = letters[correct_pos]

        source_id = str(rec.get("unique_id") or rec.get("exam_id") or f"{rel}:{rec_index}")
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
            specialty=rec.get("category"),
            language=lang,
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
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
