"""Med-HALT reasoning-hallucination-test adapter (MCQA, English).

Fixed data under ``60_Med-HALT/``:
  - ``reasoning_FCT.json`` — False Confidence Test: standard medical MCQs.
  - ``reasoning_nota.json`` — None-of-the-Above test: one option is "None of the above".

Each is a JSON list of records::

    {"id": str, "dataset": str, "question": str, "options": str (JSON dict {"0": txt, ...}),
     "correct_answer": str, "correct_index": int, "split_type": "test"|"val"|"dev", ...}

Task: single-choice. ``options`` is a JSON *string* mapping stringified index -> text;
``correct_index`` gives the correct 0-based position. Only ``split_type == "test"`` rows are used.
Each subset is its own registry task (``Med-HALT/reasoning`` = FCT, ``Med-HALT/reasoning_nota`` =
NoTA) so both are scored in an ALL run; ``benchmark.split`` is global and could only ever select
one of them. ``--split fct|nota`` still works for a single-benchmark run.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

_FILES = {"fct": "reasoning_FCT.json", "nota": "reasoning_nota.json"}
_DEFAULT = "fct"


def _parse_options(raw: Any) -> list[str] | None:
    """Parse the options field into an ordered text list.

    ``options`` ships as a *Python-literal* dict string (single quotes, e.g. ``{'0': 'txt'}``),
    which is not valid JSON — try ``json.loads`` first, then fall back to ``ast.literal_eval``.
    """
    d = raw
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                d = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return None
    if not isinstance(d, dict):
        return None
    # some rows carry an extra non-numeric key (e.g. 'correct answer'); keep only integer keys.
    numeric_keys = [k for k in d if str(k).strip().lstrip("-").isdigit()]
    if not numeric_keys:
        return None
    keys = sorted(numeric_keys, key=lambda k: int(k))
    return [str(d[k]) for k in keys]


class MedHALTReasoningAdapter(BaseBenchmarkAdapter):
    benchmark_name = "Med-HALT"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _subset(self) -> str:
        # "reasoning_nota" -> "nota"; plain "reasoning" falls back to the split.
        task = (self.entry.task or "") if getattr(self, "entry", None) else ""
        if task.startswith("reasoning_"):
            s = task.removeprefix("reasoning_")
        else:
            s = _DEFAULT if self.split == "test" else self.split
        if s not in _FILES:
            raise BenchmarkSplitNotFoundError(f"Med-HALT reasoning subset must be 'fct'/'nota'; got '{s}'.")
        return s

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / _FILES[self._subset()]]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            if str(rec.get("split_type")) != "test":
                # the file bundles train/val/dev/test; evaluate only the test split, and say so
                # in the manifest rather than letting 7,790 of 18,866 records vanish silently.
                self.drop_source_record("not_test_split")
                continue
            opts = _parse_options(rec.get("options"))
            ci = rec.get("correct_index")
            if not opts or len(opts) < 2 or not isinstance(ci, int) or not (0 <= ci < len(opts)):
                self.drop_source_record("unparseable_options_or_correct_index")
                continue
            yield {"record": rec, "options": opts, "correct_index": ci,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        opts: list[str] = raw_sample["options"]
        ci: int = raw_sample["correct_index"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(rec["question"]).strip()
        block, letters = format_lettered_choices(opts)
        correct_letter = letters[ci]

        source_id = str(rec.get("id") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"q": question, "c": opts})
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
            reference_hash=self.reference_hash(correct_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("subject_name"),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": opts},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "dataset": rec.get("dataset"),
                      "subset": self._subset()},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
