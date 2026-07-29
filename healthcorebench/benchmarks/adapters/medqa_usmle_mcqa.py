"""MedQA-USMLE (4-options) adapter.

Fixed data: ``69_MedQA-USMLE/phrases_no_exclude_test.jsonl`` — one JSON object per line::

    {"question": str, "answer": str, "options": {"A": str, ...},
     "answer_idx": "A".."D", "meta_info": str, "metamap_phrases": [...]}

Task: single-choice (A-D), English USMLE-style.
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


class MedQAUSMLEAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedQA_USMLE"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"
    _source_file = "phrases_no_exclude_test.jsonl"
    _lang = "en"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"{self.benchmark_name} provides only 'test'; requested '{self.split}'.")
        return [directory / self._source_file]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                yield {"record": json.loads(line), "source_file_rel": rel, "source_record_index": i}

    def _ordered_options(self, options: dict) -> tuple[list[str], list[str]]:
        letters = sorted(options.keys())
        texts = [str(options[k]) for k in letters]
        return letters, texts

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = rec["question"]
        options = rec["options"]
        letters, texts = self._ordered_options(options)
        answer_letter = str(rec["answer_idx"]).strip().upper()
        block = "\n".join(f"{l}. {t}" for l, t in zip(letters, texts))

        content_hash = self.input_hash({"q": question, "o": options})
        source_id = f"{rel}:{rec_index}"
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
            specialty=rec.get("meta_info"),
            language=raw_sample.get("language", self._lang),
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
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
