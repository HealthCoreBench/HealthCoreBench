"""EHRNoteQA adapter — **registered but disabled**, the shipped data cannot be answered.

Fixed data: ``46_EHRNoteQA/ehrnoteqa_test.jsonl`` — 962 lines, one JSON object each::

    {"category": str, "num_notes": int, "patient_id": int, "clinician": str,
     "question": str, "choice_A": str, ... "choice_E": str, "answer": "A".."E"}

Task: single-choice (A-E), English, grounded in EHR discharge notes. Only non-empty
``choice_*`` fields are used as options (some questions have fewer than five choices).

Why ``EHRNoteQA/mcqa`` has ``enabled=False`` in the registry
------------------------------------------------------------
Every question is *about one specific patient's discharge notes* — e.g. "What was the patient's
condition like at the time of discharge, particularly focused on his vital signs, pain
management and mobility?", with ``num_notes`` saying how many notes it spans — but the note text
is not in this file and not anywhere under ``benchmarks/``: a record holds only the question and
the choices (the longest string in the whole file is 464 characters), and MIMIC-IV notes need
credentialed PhysioNet access, which the fixed data tree does not have. A model answering from
question + options alone is guessing among 5 options with no evidence, so the resulting accuracy
measures option-prior bias, not clinical note comprehension — a number that looks like a score
but is not one. It was scored in three shipped configs; it has been removed from them.

The adapter is kept working so the task can be revived in one step: drop the per-patient note
text into the source file, render it ahead of the question in ``build_messages`` and budget it
with ``context_window.fit_context_to_window`` (these notes routinely exceed the model window),
then delete the ``EHRNoteQA/mcqa`` entry from ``registry._DISABLED``.
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

_ALL_LETTERS = ["A", "B", "C", "D", "E"]


class EHRNoteQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "EHRNoteQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"EHRNoteQA provides only 'test'; requested '{self.split}'.")
        return [directory / "ehrnoteqa_test.jsonl"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
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

        question = rec["question"]
        options = {l: rec[f"choice_{l}"] for l in _ALL_LETTERS
                   if rec.get(f"choice_{l}") not in (None, "")}
        letters = sorted(options.keys())
        block = "\n".join(f"{l}. {options[l]}" for l in letters)
        answer_letter = str(rec["answer"]).strip().upper()

        source_id = f"{rec.get('patient_id')}:{rec_index}"
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
            specialty=rec.get("category"),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "options": options, "letters": letters},
            reference_answer=answer_letter,
            reference_answer_normalized=answer_letter,
            metadata={"letters": letters, "num_notes": rec.get("num_notes")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        letters = c["letters"]
        block = "\n".join(f"{l}. {c['options'][l]}" for l in letters)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or _ALL_LETTERS)
