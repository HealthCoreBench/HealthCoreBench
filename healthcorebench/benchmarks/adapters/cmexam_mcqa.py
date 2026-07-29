"""CMExam adapter (Chinese medical licensing exam).

Fixed data: ``56_CMExam/cmexam_test.json`` — a JSON list of records::

    {"Question": str, "Options": str (newline-joined "A xxx\\nB yyy\\n..."),
     "Answer": str (one or more letters, e.g. "C" or "ABD"), "Explanation": str,
     "Disease Group": str, "Area of Competency": str, "Clinical Department": str,
     "Medical Discipline": str, "Difficulty level": int}

CMExam ships no ``question_type``, so the answer cardinality is read off ``Answer`` itself:
6,607/6,811 records name a single letter. The two kinds are split into two tasks instead of
scoring every record with set match:

  - ``CMExamAdapter`` (``CMExam/mcqa``): the single-letter records, prompted for exactly one
    letter and scored with accuracy.
  - ``CMExamMultipleAnswerAdapter`` (``CMExam/multiple_answer``): the multi-letter records,
    prompted for one-or-more letters and scored with exact set match.

Options are parsed from the ``Options`` block, preserving their original letters.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_answer_prompt, multiple_choice_prompt
from healthcorebench.benchmarks.answer_parsing import (
    parse_multiple_choice_letter,
    parse_multiple_choice_letters,
)
from healthcorebench.schemas.sample import EvaluationSample

# Matches a leading option letter on each line: "A xxx", "A. xxx", "A、xxx", "A) xxx".
_OPTION_RE = re.compile(r"^\s*([A-Z])\s*[\.、\)）:：]?\s*(.*\S)\s*$")


def _parse_options(block: str) -> list[tuple[str, str]]:
    """Parse an 'A xxx\\nB yyy' block into ordered (letter, text) pairs."""
    pairs: list[tuple[str, str]] = []
    for line in str(block).splitlines():
        m = _OPTION_RE.match(line)
        if m:
            pairs.append((m.group(1).upper(), m.group(2).strip()))
    return pairs


class CMExamAdapter(BaseBenchmarkAdapter):
    benchmark_name = "CMExam"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    # Cardinality of this task: the single-answer subset unless a subclass flips it.
    multiple_answer = False

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"CMExam provides only 'test'; requested '{self.split}'.")
        return [directory / "cmexam_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            pairs = _parse_options(rec.get("Options", ""))
            answer_letters = [c for c in str(rec.get("Answer", "")).upper() if c.isalpha()]
            valid = {p[0] for p in pairs}
            # drop malformed rows: need options and every answer letter must be a real option.
            if len(pairs) < 2 or not answer_letters or not set(answer_letters).issubset(valid):
                continue
            answers = sorted(set(answer_letters))
            # Gold cardinality decides the subset, so each task scores only what it prompts for.
            if (len(answers) > 1) != self.multiple_answer:
                continue
            yield {"record": rec, "pairs": pairs, "answers": answers,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        pairs: list[tuple[str, str]] = raw_sample["pairs"]
        answers: list[str] = raw_sample["answers"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(rec["Question"]).strip()
        letters = [p[0] for p in pairs]
        choices = [p[1] for p in pairs]
        block = "\n".join(f"{l}. {t}" for l, t in pairs)
        # a single-answer task reports a bare letter; a multi-answer task a letter set.
        reference = ",".join(answers) if self.multiple_answer else answers[0]

        source_id = f"{rel}:{rec_index}"
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
            specialty=rec.get("Medical Discipline"),
            # CMExam ships an ordinal 1..5 difficulty; expose it as L1..L5 for grouping.
            difficulty=(f"L{rec.get('Difficulty level')}" if rec.get("Difficulty level") not in (None, "") else None),
            language="zh",
            modality="Text",
            answer_format="multi_choice" if self.multiple_answer else "single_choice",
            evaluation_metric="set_match" if self.multiple_answer else "accuracy",
            source_content={"question": question, "choices": choices, "block": block},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"letters": letters, "answers": answers,
                      "area": rec.get("Area of Competency"),
                      "difficulty": rec.get("Difficulty level")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], c["block"], lang="zh")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])


class CMExamMultipleAnswerAdapter(CMExamAdapter):
    """The multi-letter-gold subset: one-or-more correct options, scored with exact set match."""

    adapter_version = "1.0"
    prompt_template_name = "multiple_answer"
    multiple_answer = True

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        return [{"role": "user", "content": multiple_answer_prompt(c["question"], c["block"], lang="zh")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letters(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
