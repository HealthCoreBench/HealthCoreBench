"""HLE-Med adapter — text-only multiple-choice subset (Humanity's Last Exam, medicine).

Fixed data: ``11_HLE_med/hle_med_test_text.json`` — the image-free HLE medicine subset. Records::

    {"id": str, "question": str, "image": "" (empty for this subset), "answer": str,
     "answer_type": "multipleChoice" | "exactMatch", "raw_subject": str, "category": str, ...}

This adapter covers only ``answer_type == "multipleChoice"`` records. Their options are embedded in
the ``question`` after an ``Answer Choices:`` header, one ``A. text`` per line (up to many
options). The stem and options are split apart; options keep their original letters and ``answer``
(a letter) is the reference. The ``exactMatch`` records are handled by the separate HLE-Med exact
adapter.
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

_CHOICES_HEADER = re.compile(r"\n\s*Answer Choices?\s*:\s*\n", re.IGNORECASE)
_OPT_LINE = re.compile(r"^\s*([A-Z])[\.\)]\s+(.*\S)\s*$")


def _split_stem_options(question: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a HLE MCQ question into (stem, [(letter, text), ...]).

    Options follow an 'Answer Choices:' header; fall back to scanning trailing 'A. ...' lines
    if the header is absent.
    """
    parts = _CHOICES_HEADER.split(question, maxsplit=1)
    if len(parts) == 2:
        stem, block = parts[0].strip(), parts[1]
    else:
        # no explicit header: find the first line that looks like an option and split there.
        lines = question.split("\n")
        idx = next((i for i, l in enumerate(lines) if _OPT_LINE.match(l)), None)
        if idx is None:
            return question.strip(), []
        stem, block = "\n".join(lines[:idx]).strip(), "\n".join(lines[idx:])
    options: list[tuple[str, str]] = []
    for line in block.split("\n"):
        m = _OPT_LINE.match(line)
        if m:
            options.append((m.group(1).upper(), m.group(2).strip()))
    return stem, options


class HLEMedMCQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "HLE_med"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"HLE_med provides only 'test'; requested '{self.split}'.")
        return [directory / "hle_med_test_text.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            if rec.get("answer_type") != "multipleChoice" or rec.get("image"):
                continue
            stem, options = _split_stem_options(str(rec.get("question") or ""))
            answer = str(rec.get("answer") or "").strip().upper()
            letters = {l for l, _ in options}
            if len(options) < 2 or answer not in letters:
                continue
            yield {"record": rec, "stem": stem, "options": options, "answer": answer,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        stem = raw_sample["stem"]
        options: list[tuple[str, str]] = raw_sample["options"]
        answer = raw_sample["answer"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        letters = [l for l, _ in options]
        choices = [t for _, t in options]
        block = "\n".join(f"{l}. {t}" for l, t in options)

        source_id = str(rec.get("id") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"q": stem, "c": choices})
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
            input_hash=self.input_hash({"question": stem, "choices_block": block}),
            reference_hash=self.reference_hash(answer),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("raw_subject"),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": stem, "choices": choices, "block": block},
            reference_answer=answer,
            reference_answer_normalized=answer,
            metadata={"letters": letters, "raw_subject": rec.get("raw_subject")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], c["block"], lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
