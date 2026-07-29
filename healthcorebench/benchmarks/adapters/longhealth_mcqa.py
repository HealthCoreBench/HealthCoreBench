"""LongHealth adapter (long-context clinical document MCQA, English).

Fixed data: ``22_LongHealth/data/benchmark_v5.json`` — a dict of 20 patients, each::

    {"texts": {"text_0": str, "text_1": str, ...} (long clinical documents),
     "name": str, "diagnosis": str,
     "questions": [{"No": int, "question": str, "answer_a".."answer_e": str,
                    "correct": str (the correct option's text), "answer_location": ...}, ...]}

Task: single-choice question answering over long clinical documents. The patient's documents are
concatenated as context; each question's five options are re-lettered A..E and the ``correct``
answer text is matched to its option letter. Five of the 400 questions repeat an option text and
in two of those the *correct* text appears twice (``patient_16:11``, ``patient_20:11``), so every
position holding that text is accepted: the first is the reference letter and the rest are
``reference_aliases``. This is a long-context task (tens of thousands of characters per patient).
Patients are exposed as splits (``patient_01``..``patient_20``); the default ``test`` uses all
patients.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.context_window import fit_context_to_window
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

_OPTION_KEYS = ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"]


class LongHealthMCQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "LongHealth"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _patients(self, data: dict) -> list[str]:
        if self.split == "test":
            return sorted(data.keys())
        if self.split in data:
            return [self.split]
        raise BenchmarkSplitNotFoundError(
            f"LongHealth split must be 'test' or a patient id like 'patient_01'; got '{self.split}'."
        )

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / "data" / "benchmark_v5.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for pid in self._patients(data):
            patient = data[pid]
            texts = patient.get("texts") or {}
            section_offsets = {}
            if isinstance(texts, dict):
                ordered_texts = [(str(key), str(texts[key])) for key in sorted(texts)]
            elif isinstance(texts, list):
                ordered_texts = [(f"text_{index}", str(text)) for index, text in enumerate(texts)]
            else:
                ordered_texts = [("text_0", str(texts))]
            context_parts = []
            cursor = 0
            for key, text in ordered_texts:
                if context_parts:
                    context_parts.append("\n\n")
                    cursor += 2
                section_offsets[key] = (cursor, len(text))
                context_parts.append(text)
                cursor += len(text)
            context = "".join(context_parts)
            for q in patient.get("questions", []):
                choices = [str(q.get(k) or "").strip() for k in _OPTION_KEYS]
                choices = [c for c in choices if c]
                correct = str(q.get("correct") or "").strip()
                if len(choices) < 2 or correct not in choices:
                    continue
                evidence_spans = []
                for key, location in (q.get("answer_location") or {}).items():
                    if key not in section_offsets or not isinstance(location, dict):
                        continue
                    section_start, section_length = section_offsets[key]
                    starts = location.get("start") or []
                    ends = location.get("end") or []
                    for start, end in zip(starts, ends):
                        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                            evidence_spans.append((
                                section_start + round(max(0.0, min(1.0, start)) * section_length),
                                section_start + round(max(0.0, min(1.0, end)) * section_length),
                            ))
                yield {"patient_id": pid, "context": context, "question": q,
                       "choices": choices, "correct": correct,
                       "evidence_spans": evidence_spans,
                       "source_file_rel": rel, "source_record_index": q.get("No")}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        pid = raw_sample["patient_id"]
        context = raw_sample["context"]
        q = raw_sample["question"]
        choices = raw_sample["choices"]
        correct = raw_sample["correct"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(q["question"]).strip()
        block, letters = format_lettered_choices(choices)
        # ``correct`` is the option *text*, and a handful of questions list it twice. Taking the
        # first match alone made the gold letter arbitrary and marked the identical option wrong.
        correct_letters = [letters[i] for i, choice in enumerate(choices) if choice == correct]
        correct_letter = correct_letters[0]
        alias_letters = correct_letters[1:]
        generation = getattr(self.config, "generation", None)
        output_budget = self.output_token_budget_for_format("single_choice")
        fixed_prompt = (
            f"Clinical documents:\n\nQuestion: {question}\nOptions:\n{block}\n"
            "Answer with the option's letter from the given choices directly."
        )
        context, context_meta = fit_context_to_window(
            context,
            fixed_prompt=fixed_prompt,
            max_model_len=getattr(getattr(self.config, "hardware", None), "max_model_len", None),
            max_output_tokens=output_budget,
            reserve_tokens=getattr(generation, "context_token_reserve", 512),
            policy=getattr(generation, "context_overflow_policy", "error"),
            protected_spans=raw_sample.get("evidence_spans") or [],
            protected_span_source="benchmark_answer_location",
        )

        source_id = f"{pid}:{q.get('No')}"
        content_hash = self.input_hash({"q": question, "c": choices, "p": pid})
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
            source_record_index=rec_index if isinstance(rec_index, int) else None,
            source_record_hash=self.input_hash(q),
            input_hash=self.input_hash({
                "question": question, "choices_block": block, "patient": pid, "context": context,
            }),
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
            source_content={"question": question, "choices": choices, "context": context},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            reference_aliases=alias_letters or None,
            metadata={
                "letters": letters,
                "patient_id": pid,
                "equivalent_option_letters": correct_letters,
                "request_max_tokens": output_budget,
                "output_token_budget_policy": "configured_generation_max_tokens",
                "benchmark_answer_location_available": bool(raw_sample.get("evidence_spans")),
                **context_meta,
            },
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        prompt = (
            f"Clinical documents:\n{c['context']}\n\n"
            f"Question: {c['question']}\nOptions:\n{block}\n"
            "Answer with the option's letter from the given choices directly."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        letter = parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
        # Options holding the identical gold text are the same answer. The accuracy evaluator
        # compares against a single reference letter, so collapse an equivalent pick onto the
        # reference here as well as declaring it in ``reference_aliases``.
        equivalent = sample.metadata.get("equivalent_option_letters") or []
        if letter is not None and letter in equivalent:
            return equivalent[0]
        return letter
