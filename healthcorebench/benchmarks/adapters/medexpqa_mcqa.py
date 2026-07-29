"""MedExpQA adapter (multilingual medical exam QA with explanations).

Fixed data: ``42_MedExpQA/medexpqa_<lang>_test.jsonl`` — four languages (en, es, fr, it),
one JSON object per line::

    {"id": str, "lang": str, "full_question": str, "options": {"1".."5": str},
     "correct_option": int, "explanations": ..., "type": str, "year": int, ...}

Task: single-choice. ``options`` keys are numeric strings ("1".."5") and ``correct_option``
is the numeric key; both are mapped to letters (A, B, ...) by numeric order. Languages are
exposed as splits; default ``test`` maps to en.
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

_LANGS = {"test": "en", "en": "en", "es": "es", "fr": "fr", "it": "it"}


class MedExpQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedExpQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _lang(self) -> str:
        if self.split not in _LANGS:
            raise BenchmarkSplitNotFoundError(f"MedExpQA split must be one of {sorted(_LANGS)}; got '{self.split}'.")
        return _LANGS[self.split]

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split == "test":
            return [directory / f"medexpqa_{lang}_test.jsonl" for lang in ("en", "es", "fr", "it")]
        return [directory / f"medexpqa_{self._lang()}_test.jsonl"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
          rel = self.rel_path(f)
          with open(f, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                yield {"record": json.loads(line), "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        lang = Path(rel).stem.split("_")[1]

        question = rec.get("full_question") or rec.get("question", "")
        # numeric-keyed options -> ordered choices + letters
        num_keys = sorted(rec["options"].keys(), key=lambda k: int(k))
        choices = [str(rec["options"][k]) for k in num_keys]
        block, letters = format_lettered_choices(choices)
        correct_pos = num_keys.index(str(rec["correct_option"]))
        correct_letter = letters[correct_pos]

        source_id = str(rec.get("id", f"{rel}:{rec_index}"))
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
            specialty=rec.get("type"),
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
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
