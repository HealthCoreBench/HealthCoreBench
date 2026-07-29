"""AfriMedQA multiple-answer MCQ adapter (``AfriMedQA_v2/multiple_answer``).

306 of the 3,910 ``question_type == "mcq"`` records in ``afrimedqa_v2_test.json`` carry a
comma-separated ``correct_answer`` ("option2,option4"): 114 with two correct options, 126 with
three, 56 with four and 13 with five. The single-answer adapter's ``correct not in kept`` guard
silently discarded all of them, so 7.8% of the MCQ set was neither scored nor reported.

They are scored here as a set-match task, mirroring ``globaldentbench_multiple_answer.py``.
Records are partitioned by ``afrimedqa_mcqa.classify_mcq_record``, the same function the
single-answer adapter uses, so the two filters cannot drift apart: this task keeps exactly the
records that one returns ``"multiple"`` for and reports every other record as a drop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.adapters.afrimedqa_mcqa import (
    SINGLE_ANSWER_TASK_REASON,
    classify_mcq_record,
    ordered_option_keys,
)
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letters
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import format_lettered_choices, multiple_answer_prompt
from healthcorebench.schemas.sample import EvaluationSample


class AfriMedQAMultipleAnswerAdapter(BaseBenchmarkAdapter):
    benchmark_name = "AfriMedQA_v2"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_answer"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(
                f"AfriMedQA_v2 provides only 'test'; requested '{self.split}'."
            )
        return [self.get_benchmark_directory() / "afrimedqa_v2_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            kind, kept, gold_keys = classify_mcq_record(rec)
            if kind != "multiple":
                self.drop_source_record(SINGLE_ANSWER_TASK_REASON if kind == "single" else kind)
                continue
            yield {"record": rec, "kept": kept, "gold_keys": gold_keys,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        kept: dict[str, str] = raw_sample["kept"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(rec.get("question_clean") or rec["question"]).strip()
        ordered_keys = ordered_option_keys(kept)
        choices = [kept[k] for k in ordered_keys]
        block, letters = format_lettered_choices(choices)
        correct = sorted({letters[ordered_keys.index(k)] for k in raw_sample["gold_keys"]})
        reference = ",".join(correct)

        source_id = str(rec.get("sample_id") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"q": question, "c": choices})

        return EvaluationSample(
            sample_id=self.make_sample_id(
                source_file_rel=rel, source_sample_id=source_id, content_hash=content_hash
            ),
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
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=rec.get("specialty"),
            language="en",
            modality="Text",
            answer_format="multi_choice",
            evaluation_metric="set_match",
            source_content={"question": question, "choices": choices},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"letters": letters, "country": rec.get("country"), "tier": rec.get("tier"),
                      "num_correct_options": len(correct)},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_answer_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letters(
            raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"]
        )
