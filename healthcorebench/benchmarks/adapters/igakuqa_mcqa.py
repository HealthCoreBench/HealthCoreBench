"""IgakuQA adapter (Japanese National Medical License Exam).

Fixed data: ``39_IgakuQA/data/<year>/<exam>-<block>.jsonl`` — one JSON object per line::

    {"problem_id": str, "problem_text": str, "choices": [str, ...],
     "text_only": bool, "answer": [str, ...] (lower-case option letters), "points": str}

(``*_metadata.jsonl`` and ``*_translate.jsonl`` siblings are ignored.)

Choices map to letters a.. in order; ``answer`` is a list of the correct lower-case letters,
upper-cased here. The exam states the answer cardinality in the question itself: a ``Nつ選べ``
("choose N") instruction appears in ``problem_text`` for exactly the 298 multi-answer items and
for none of the 1,689 single-answer ones, and the stated N always equals the gold size. That
declared cardinality splits the data into two tasks instead of scoring every item with set match:

  - ``IgakuQAAdapter`` (``IgakuQA/mcqa``): items without the hint, prompted for exactly one
    letter and scored with accuracy.
  - ``IgakuQAMultipleAnswerAdapter`` (``IgakuQA/multiple_answer``): items with the hint,
    prompted for one-or-more letters and scored with exact set match.

Years are exposed as splits (``2018``..``2022``); the default ``test`` uses all years. Rows whose
answer is not a clean set of option letters (a handful of malformed entries) are skipped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import (
    format_lettered_choices,
    multiple_answer_prompt,
    multiple_choice_prompt,
)
from healthcorebench.benchmarks.answer_parsing import (
    parse_multiple_choice_letter,
    parse_multiple_choice_letters,
)
from healthcorebench.schemas.sample import EvaluationSample

_YEARS = ("2018", "2019", "2020", "2021", "2022")
# The exam's own "choose N" instruction, e.g. "2つ選べ" — present iff the item is multi-answer.
_CHOOSE_N_RE = re.compile(r"([0-9])つ選べ")


class IgakuQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "IgakuQA"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    # Cardinality of this task: the single-answer subset unless a subclass flips it.
    multiple_answer = False

    def _years(self) -> list[str]:
        if self.split == "test":
            return list(_YEARS)
        if self.split in _YEARS:
            return [self.split]
        raise BenchmarkSplitNotFoundError(f"IgakuQA split must be 'test' or one of {list(_YEARS)}; got '{self.split}'.")

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        files: list[Path] = []
        for yr in self._years():
            ydir = directory / "data" / yr
            for p in sorted(ydir.glob("*.jsonl")):
                if p.name.endswith("_metadata.jsonl") or p.name.endswith("_translate.jsonl"):
                    continue
                files.append(p)
        return files

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
            rel = self.rel_path(f)
            with open(f, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    choices = rec.get("choices") or []
                    n = len(choices)
                    if n < 2:
                        continue
                    letters = [chr(ord("a") + k) for k in range(n)]
                    ans = [str(a).strip().lower() for a in (rec.get("answer") or [])]
                    if not ans or not all(a in letters for a in ans):
                        continue  # skip malformed answers (numbers, "a or d", etc.)
                    # Route on the exam's own "choose N" instruction, which is part of the
                    # question text the model reads anyway — not a hint derived from the gold.
                    hint = _CHOOSE_N_RE.search(str(rec.get("problem_text") or ""))
                    if bool(hint) != self.multiple_answer:
                        continue
                    yield {"record": rec, "answers": sorted(set(ans)),
                           "choose_n": int(hint.group(1)) if hint else 1,
                           "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        answers: list[str] = raw_sample["answers"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(rec["problem_text"]).strip()
        choices = [str(c) for c in rec["choices"]]
        block, letters = format_lettered_choices(choices)  # A, B, C, ...
        # map lower-case source answer letters (a,b,..) to canonical upper letters by position.
        reference_letters = sorted({letters[ord(a) - ord("a")] for a in answers})
        # a single-answer task reports a bare letter; a multi-answer task a letter set.
        reference = ",".join(reference_letters) if self.multiple_answer else reference_letters[0]

        source_id = str(rec.get("problem_id") or f"{rel}:{rec_index}")
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
            specialty=None,
            language="ja",
            modality="Text",
            answer_format="multi_choice" if self.multiple_answer else "single_choice",
            evaluation_metric="set_match" if self.multiple_answer else "accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"letters": letters, "points": rec.get("points"),
                      "text_only": rec.get("text_only"),
                      "choose_n": raw_sample.get("choose_n")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])


class IgakuQAMultipleAnswerAdapter(IgakuQAAdapter):
    """The ``Nつ選べ`` subset: one-or-more correct options, scored with exact set match."""

    adapter_version = "1.0"
    prompt_template_name = "multiple_answer"
    multiple_answer = True

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_answer_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letters(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
