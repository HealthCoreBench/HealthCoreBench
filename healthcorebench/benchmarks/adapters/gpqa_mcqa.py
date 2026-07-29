"""GPQA (biology subset) adapter.

Fixed data: ``17_GPQA/gpqa_<subset>_bio.json``. Only ``gpqa_extended_bio.json`` (105
questions) is present in this fixed dataset, so ``test`` maps to **extended**, not to the
diamond subset that published GPQA numbers are usually quoted over — scores from here are
not comparable with a diamond figure. ``diamond`` and ``main`` remain accepted split names
so the adapter keeps working if those files are added later; requesting one now fails on the
missing file. Each record::

    {"Question": str, "Correct Answer": str,
     "Incorrect Answer 1": str, "Incorrect Answer 2": str, "Incorrect Answer 3": str, ...}

Task: single-choice (A-D). The four options are assembled from the correct answer plus the
three distractors and shuffled *deterministically* (seeded by a hash of the question text),
so the correct-answer position is not fixed but the ordering is stable across runs and the
resulting ``sample_id`` is reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

_SUBSETS = {"test": "extended", "diamond": "diamond", "extended": "extended", "main": "main"}
_LETTERS = ["A", "B", "C", "D"]


def _difficulty_bucket(estimate: str | None) -> str | None:
    """Map GPQA's free-text "Writer's Difficulty Estimate" to a short grouping label.

    Check ``undergraduate`` before ``graduate`` because the former contains the latter as a
    substring.
    """
    if not estimate:
        return None
    e = estimate.lower()
    if "post-graduate" in e or "post graduate" in e:
        return "postgraduate"
    if "undergraduate" in e:
        return "hard_undergraduate" if "hard" in e else "easy_undergraduate"
    if "graduate" in e:
        return "graduate"
    return None


def _deterministic_order(question: str, n: int) -> list[int]:
    """Return a stable permutation of range(n) seeded by the question text.

    Uses a SHA-256 digest of the question as the sort key material so the shuffle is
    reproducible across processes (unlike ``random`` without a fixed seed) and independent
    of dict/insertion order.
    """
    keys = []
    for i in range(n):
        h = hashlib.sha256(f"{question}\x00{i}".encode("utf-8")).hexdigest()
        keys.append((h, i))
    keys.sort()
    return [i for _, i in keys]


class GPQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "GPQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _subset(self) -> str:
        if self.split not in _SUBSETS:
            raise BenchmarkSplitNotFoundError(f"GPQA split must be one of {sorted(_SUBSETS)}; got '{self.split}'.")
        return _SUBSETS[self.split]

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / f"gpqa_{self._subset()}_bio.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = rec["Question"]
        correct = str(rec["Correct Answer"]).strip()
        distractors = [str(rec[f"Incorrect Answer {k}"]).strip() for k in (1, 2, 3)]
        # index 0 is the correct answer before shuffling
        pool = [correct] + distractors
        order = _deterministic_order(question, len(pool))
        choices = [pool[i] for i in order]
        correct_pos = order.index(0)  # where the correct answer landed
        block, letters = format_lettered_choices(choices, _LETTERS)
        correct_letter = _LETTERS[correct_pos]

        source_id = f"{rel}:{rec_index}"
        content_hash = self.input_hash({"q": question, "correct": correct, "distractors": distractors})
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
            specialty="biology",
            difficulty=_difficulty_bucket(rec.get("Writer's Difficulty Estimate")),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "correct_position": correct_pos},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]], _LETTERS)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or _LETTERS)
