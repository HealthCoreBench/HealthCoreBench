"""MediQ adapter (clinical MCQA with patient context, English).

Fixed data: ``13_MediQ/data/all_dev_good.jsonl`` — one JSON object per line::

    {"id": ..., "question": str, "context": [str, ...] (patient facts/history),
     "options": {"A": str, ...}, "answer": str (option text), "answer_idx": str (letter),
     "explanation": str, "facts": ..., "patient": ...}

Task: single-choice question answering conditioned on the patient context. MediQ's full benchmark
is an *interactive* information-seeking task; this adapter uses the provided static context to pose
the question as a standard multiple-choice item (context sentences are prepended). Options are
re-lettered locally in sorted key order and ``answer_idx`` maps to the correct position.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample


class MediQMCQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MediQ"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MediQ provides only 'test'; requested '{self.split}'.")
        return [directory / "data" / "all_dev_good.jsonl", directory / "data" / "all_craft_md.jsonl"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
          rel = self.rel_path(f)
          with open(f, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                opts = rec.get("options") or {}
                idx = str(rec.get("answer_idx") or "").strip().upper()
                if len(opts) < 2 or idx not in {str(k).upper() for k in opts}:
                    continue
                yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(rec["question"]).strip()
        context = rec.get("context")
        context_text = "\n".join(str(x) for x in context) if isinstance(context, list) else str(context or "")
        src_keys = sorted(rec["options"], key=lambda k: str(k).upper())
        choices = [str(rec["options"][k]) for k in src_keys]
        block, letters = format_lettered_choices(choices)
        correct_pos = [str(k).upper() for k in src_keys].index(str(rec["answer_idx"]).strip().upper())
        correct_letter = letters[correct_pos]

        source_id = str(rec.get("id") if rec.get("id") is not None else f"{rel}:{rec_index}")
        content_hash = self.input_hash({"q": question, "ctx": context_text, "c": choices})
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
            input_hash=self.input_hash({"question": question, "context": context_text, "choices_block": block}),
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
            source_content={"question": question, "context": context_text, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        ctx = f"Patient information:\n{c['context']}\n\n" if c.get("context") else ""
        prompt = (
            f"{ctx}Question: {c['question']}\nOptions:\n{block}\n"
            "Answer with the option's letter from the given choices directly."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
