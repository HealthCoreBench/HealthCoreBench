"""Meta-MedQA adapter.

Fixed data: ``24_Meta-MedQA/meta_medqa_test.json`` — a JSON list of records::

    {"question": str, "options": {"A": str, ... up to "F"}, "answer_idx": "A"..,
     "answer": str, "meta_info": str, "kind": str, "metamap_phrases": str}

Task: single-choice with a variable number of options (A-F). ``answer_idx`` is the letter.
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


class MetaMedQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "Meta-MedQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"Meta-MedQA provides only 'test'; requested '{self.split}'.")
        return [directory / "meta_medqa_test.json"]

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
        options = rec["options"]
        letters = sorted(options.keys())
        block = "\n".join(f"{l}. {options[l]}" for l in letters)
        answer_letter = str(rec["answer_idx"]).strip().upper()

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
            specialty=rec.get("meta_info"),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "options": options, "letters": letters},
            reference_answer=answer_letter,
            reference_answer_normalized=answer_letter,
            metadata={"letters": letters, "kind": rec.get("kind")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        letters = c["letters"]
        block = "\n".join(f"{l}. {c['options'][l]}" for l in letters)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
