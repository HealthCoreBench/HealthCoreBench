"""GlobalDentBench adapter (MCQ task).

Fixed data: ``71_GlobalDentBench/GlobalDentBench-OA.json`` — a dict grouping records by task
type: ``{"MCQ": [...], "SAQ": [...], "CBQ": [...]}``. This adapter handles only the ``MCQ``
group; SAQ/CBQ (short-answer / case-based) are open-ended and belong to a separate adapter.

Each MCQ record::

    {"id": str, "from": str, "question": str, "options": {"A": str, ...},
     "answer": "A".., "reason": str, "tags": [...], "country_regions": ..., "continents": ...}

Task: single-choice, English (international dental exams).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample


class GlobalDentBenchMCQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "GlobalDentBench"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"GlobalDentBench provides only 'test'; requested '{self.split}'.")
        return [directory / "GlobalDentBench-OA.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # The MCQ group contains a small number of multi-answer ("A, B") and empty-answer
        # records. This adapter scores single-choice only, so those are skipped rather than
        # guessed or mis-scored. source_record_index stays aligned to the original MCQ order.
        for i, rec in enumerate(data.get("MCQ", [])):
            ans = str(rec.get("answer", "")).replace(" ", "")
            opts = rec.get("options") or {}
            # Valid single-choice item: exactly one answer letter that exists among *at least
            # two* options. Skips multi-answer ("A, B"), empty answers, answers pointing
            # outside the options, and the 162 records whose option set collapsed to a single
            # entry — those are unanswerable-by-construction and were all scored correct.
            if len(ans) != 1 or ans not in opts or len(opts) < 2:
                # Multi-answer records are not lost: GlobalDentBench/multiple_answer picks
                # up every one whose letters all exist among the options, so counting them
                # as dropped here would overstate the loss. Match that adapter's own letter
                # extraction — ``ans`` still carries the separators ("A,B").
                letters = re.findall(r"[A-Z]", ans.upper())
                routed = len(letters) > 1 and set(letters) <= set(opts)
                if not routed:
                    if not ans:
                        self.drop_source_record("empty_answer")
                    elif len(opts) < 2:
                        self.drop_source_record("fewer_than_two_options")
                    else:
                        self.drop_source_record("answer_not_among_options")
                continue
            yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = rec["question"]
        options = {k: str(v) for k, v in rec["options"].items()}
        letters = sorted(options.keys())
        block = "\n".join(f"{l}. {options[l]}" for l in letters)
        answer_letter = str(rec["answer"]).strip().upper()

        source_id = str(rec.get("id", f"{rel}:{rec_index}"))
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
            specialty="dentistry",
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "options": options, "letters": letters},
            reference_answer=answer_letter,
            reference_answer_normalized=answer_letter,
            metadata={"letters": letters, "from": rec.get("from")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        letters = c["letters"]
        block = "\n".join(f"{l}. {c['options'][l]}" for l in letters)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
