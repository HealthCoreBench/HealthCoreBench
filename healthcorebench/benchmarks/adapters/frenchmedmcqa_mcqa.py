"""FrenchMedMCQA adapter (French pharmacy exam, one-or-more correct).

Fixed data: ``18_FrenchMedMCQA/frenchmedmcqa_test.json`` — a JSON list of records::

    {"id": str, "question": str, "answers": {"a": str, "b": str, ..., "e": str},
     "correct_answers": [str, ...] (letters, lower-case), "subject_name": str,
     "nbr_correct_answers": int}

Task: single-question multiple-answer, French, five options. ~48% of questions have more than
one correct option, so the reference is a *set* of letters scored with exact set match
(``multiple_answer`` evaluator). Option letters are upper-cased from the source ``answers`` keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_answer_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letters
from healthcorebench.schemas.sample import EvaluationSample


class FrenchMedMCQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "FrenchMedMCQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_answer"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"FrenchMedMCQA provides only 'test'; requested '{self.split}'.")
        return [directory / "frenchmedmcqa_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            answers = rec.get("answers") or {}
            correct = [str(c).upper() for c in (rec.get("correct_answers") or [])]
            valid = {str(k).upper() for k in answers}
            if len(answers) < 2 or not correct or not set(correct).issubset(valid):
                continue
            yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(rec["question"]).strip()
        # order options by their (lower-case) source key: a, b, c, d, e.
        ordered_keys = sorted(rec["answers"], key=lambda k: str(k).lower())
        letters = [str(k).upper() for k in ordered_keys]
        choices = [str(rec["answers"][k]) for k in ordered_keys]
        block, _ = format_lettered_choices(choices, letters=letters)
        answers = sorted({str(c).upper() for c in rec["correct_answers"]})
        reference = ",".join(answers)

        source_id = str(rec.get("id") or f"{rel}:{rec_index}")
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
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=rec.get("subject_name"),
            language="fr",
            modality="Text",
            answer_format="multi_choice",
            evaluation_metric="set_match",
            source_content={"question": question, "choices": choices, "letters": letters},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"letters": letters, "answers": answers,
                      "nbr_correct": rec.get("nbr_correct_answers")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]], letters=c["letters"])
        return [{"role": "user", "content": multiple_answer_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letters(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
