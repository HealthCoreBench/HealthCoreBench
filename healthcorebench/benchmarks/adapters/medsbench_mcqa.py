"""MedS-Bench MCQA tasks that ship as ``MCQA/*.jsonl`` (six multilingual exam sets).

``medsbench_open.py`` covers the 20 profile-driven tasks whose answers are free text, labels or
entity sets. The ``MCQA/`` directory additionally holds six files that were never registered even
though they are ordinary single-choice exams with a gold letter::

    {"Definition": [...], "Instances": [{"input": "Question: ...\\nOptions: A: x\\tB: y\\t",
                                         "output": "The right answer is B: y"}, ...]}

Despite the ``.jsonl`` extension these are *not* line-delimited: each file is one pretty-printed
JSON document, so it is read with ``json.load`` and ``count_source_records`` is overridden (the
base implementation would count physical lines for a ``.jsonl`` suffix and report tens of
thousands of "records").

The input is ``"<question label>: <stem>\\n<options label>: A: t1\\tB: t2\\t"`` and the output is
``"<localized prefix> <LETTER>: <option text>"``; both labels and the prefix are localized per
task. All 8,518 instances across the six files parse, and in every one of them the gold letter's
option text equals the answer text, so the letter is a safe reference.

Only ``task61`` (RuMedBench) is enabled in the registry. The other five are registered but
disabled because their content is already scored elsewhere at the same or better fidelity — see
``overlap_note`` on each entry:

===============  =====  ====  ================================================================
task             n      lang  overlap
===============  =====  ====  ================================================================
task57_medqa_en  1,273  en    the MedQA-USMLE test split (``MedQA_USMLE/mcqa``)
task58_medqa_zh  3,426  zh    the MedQA-MCMLE test split (``MedQA_MCMLE/mcqa``)
task59_igakuqa     199  ja    a subset of ``IgakuQA/mcqa``
task60_frenchmed   622  fr    the FrenchMedMCQA test split (``FrenchMedMCQA/mcqa``)
task129_headqa   2,742  es    the HEAD-QA es test split (``HEAD-QA_v2/mcqa_es``); *worse*, since
                              it keeps all 2,742 records including the 67 that reference an
                              image the text-only file cannot supply
task61_rumedbench  256  ru    none — RuMedBench is not otherwise in the suite
===============  =====  ====  ================================================================
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkDataNotFoundError
from healthcorebench.benchmarks.prompts import format_lettered_choices, multiple_choice_prompt
from healthcorebench.schemas.sample import EvaluationSample

# task key -> (relative file, language, source dataset shown in metadata)
MEDS_MCQA_TASKS: dict[str, tuple[str, str, str]] = {
    "task57": ("MCQA/task57_medqa_question_answering_en.jsonl", "en", "MedQA (USMLE)"),
    "task58": ("MCQA/task58_medqa_question_answering_zh.jsonl", "zh", "MedQA (MCMLE)"),
    "task59": ("MCQA/task59_igakuqa_question_answering.jsonl", "ja", "IgakuQA"),
    "task60": ("MCQA/task60_frenchmedmcqa_question_answering.jsonl", "fr", "FrenchMedMCQA"),
    "task61": ("MCQA/task61_rumedbench_question_answering.jsonl", "ru", "RuMedBench"),
    "task129": ("MCQA/task129_headqa_question_answering.jsonl", "es", "HEAD-QA (es)"),
}

# The options label is localized ("Options", "选项", "オプション", "Варианты", ...), so the line is
# found by its "A:" payload rather than by the label. Matching the *last* such line keeps an
# "A:" inside the vignette from being mistaken for the option block.
_OPTIONS_LINE = re.compile(r"^(?P<label>[^\n:]{1,24}):[ \t]*(?P<body>A[ \t]*:.*)$", re.M)
_OPTION = re.compile(r"^\s*([A-Z])\s*:\s*(.*)$", re.S)
_GOLD_LETTER = re.compile(r"([A-Z])\s*:")


def _parse_input(text: str) -> tuple[str, list[str], list[str]] | None:
    """Split one ``input`` into (stem, option texts, source letters)."""
    matches = list(_OPTIONS_LINE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    stem = text[: match.start()].strip()
    options = []
    for piece in match.group("body").split("\t"):
        if not piece.strip():
            continue
        parsed = _OPTION.match(piece)
        if not parsed:
            return None
        options.append((parsed.group(1).upper(), parsed.group(2).strip()))
    letters = [letter for letter, _ in options]
    if len(letters) < 2 or letters != [chr(ord("A") + i) for i in range(len(letters))]:
        return None
    return stem, [option for _, option in options], letters


class MedSBenchMCQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedS-Bench"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    @property
    def task_key(self) -> str:
        task = self.entry.task
        if task not in MEDS_MCQA_TASKS:
            raise BenchmarkDataNotFoundError(
                f"Unknown MedS-Bench MCQA task '{task}'; expected one of {sorted(MEDS_MCQA_TASKS)}."
            )
        return task

    def discover_source_files(self) -> list[Path]:
        return [self.get_benchmark_directory() / MEDS_MCQA_TASKS[self.task_key][0]]

    def count_source_records(self, path: Path) -> int | None:
        # The ``.jsonl`` suffix lies: these files are single pretty-printed JSON documents, so the
        # base line-counting path would report the file's physical line count as a record count.
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        instances = payload.get("Instances") if isinstance(payload, dict) else None
        return len(instances) if isinstance(instances, list) else None

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        source = files[0]
        rel = self.rel_path(source)
        with open(source, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for index, instance in enumerate(payload.get("Instances") or []):
            parsed = _parse_input(str(instance.get("input") or ""))
            if parsed is None:
                self.drop_source_record("unparseable_option_block")
                continue
            stem, choices, letters = parsed
            output = instance.get("output")
            output = output[0] if isinstance(output, list) and output else output
            match = _GOLD_LETTER.search(str(output or ""))
            if not match or match.group(1) not in letters:
                self.drop_source_record("gold_letter_missing_or_out_of_range")
                continue
            if not stem:
                self.drop_source_record("empty_question")
                continue
            yield {
                "instance": instance,
                "question": stem,
                "choices": choices,
                "correct_pos": letters.index(match.group(1)),
                "source_file_rel": rel,
                "source_record_index": index,
            }

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        instance = raw_sample["instance"]
        rel = raw_sample["source_file_rel"]
        record_index = raw_sample["source_record_index"]
        question = raw_sample["question"]
        choices = raw_sample["choices"]
        _, language, source_dataset = MEDS_MCQA_TASKS[self.task_key]

        block, letters = format_lettered_choices(choices)
        correct_letter = letters[raw_sample["correct_pos"]]
        source_id = f"{self.task_key}:{record_index}"

        return EvaluationSample(
            sample_id=self.make_sample_id(
                source_file_rel=rel,
                source_sample_id=source_id,
                content_hash=self.input_hash({"q": question, "c": choices}),
            ),
            source_sample_id=source_id,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=record_index,
            source_record_hash=self.input_hash(instance),
            input_hash=self.input_hash({"question": question, "choices_block": block}),
            reference_hash=self.reference_hash(correct_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=self.task_key,
            language=language,
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "task": self.task_key,
                      "source_dataset": source_dataset,
                      "source_task_file": MEDS_MCQA_TASKS[self.task_key][0]},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        content = sample.source_content
        block, _ = format_lettered_choices([str(choice) for choice in content["choices"]])
        return [{
            "role": "user",
            "content": multiple_choice_prompt(content["question"], block, lang=sample.language),
        }]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(
            raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"]
        )
