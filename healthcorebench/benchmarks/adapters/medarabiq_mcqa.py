"""MedArabiQ adapter (Arabic medical MCQA).

Fixed data: ``48_MedArabiQ/multiple-choice-questions.csv`` (UTF-8 BOM) with columns
``Question``, ``Answer``, ``Category``. The ``Question`` cell holds the stem on the first line
followed by one option per line, each prefixed with an Arabic ordinal letter and a dot::

    كل ما هو آت صحيح ...:
    أ. option one
    ب. option two
    ج. option three
    د. option four

``Answer`` repeats the full text of the correct option (including its Arabic-letter prefix).

Task: single-choice, Arabic. Options are parsed by their Arabic ordinal prefix (أ,ب,ج,د,ه),
re-lettered to Latin A.. for a canonical prompt, and the answer is matched to an option by its
Arabic prefix (falling back to full-text match).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

# Arabic ordinal option letters, in order: alif, ba, jim, dal, ha.
_AR_ORDINALS = ["أ", "ب", "ج", "د", "ه"]
_AR_SET = set(_AR_ORDINALS) | {"ا", "هـ"}
# a line beginning with an Arabic ordinal followed by . / ) / - / : separator. Tatweel (U+0640)
# is deliberately *not* a separator: it is a letter-joining glyph that occurs inside words
# ("بـ", "دم" written "دـ"), so accepting it split stems and option bodies mid-word.
_OPT_RE = re.compile(r"^\s*(هـ|[أ-ي])\s*[\.\)\-:]\s*(.*\S)\s*$")


def _ar_index(letter: str) -> int | None:
    letter = letter.replace("هـ", "ه").replace("ا", "أ")
    return _AR_ORDINALS.index(letter) if letter in _AR_ORDINALS else None


class MedArabiQAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedArabiQ"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedArabiQ provides only 'test'; requested '{self.split}'.")
        return [directory / "multiple-choice-questions.csv"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for i, rec in enumerate(rows):
            stem, opts = self._parse_question(rec.get("Question", ""))
            if len(opts) < 2:
                continue
            correct_pos = self._match_answer(rec.get("Answer", ""), opts)
            if correct_pos is None:
                continue
            yield {"record": rec, "stem": stem, "opts": opts, "correct_pos": correct_pos,
                   "source_file_rel": rel, "source_record_index": i}

    @staticmethod
    def _parse_question(q: str) -> tuple[str, list[tuple[str, str]]]:
        """Split the Question cell into (stem, [(arabic_letter, text), ...])."""
        stem_lines: list[str] = []
        opts: list[tuple[str, str]] = []
        # Some rows put option A on the same line as the stem. Insert a line break before
        # every Arabic option marker so those records are parsed the same as multiline rows.
        # Same tatweel exclusion as ``_OPT_RE``; it also lets "هـ." be recognised whole
        # instead of "ه" + tatweel-as-separator, which left a stray "." on the option text.
        normalized = re.sub(r"(?<!^)\s+([اأبجده]|هـ)\s*[\.\)\-:]", r"\n\1. ", str(q))
        for ln in normalized.splitlines():
            if not ln.strip():
                continue
            m = _OPT_RE.match(ln)
            if m and (m.group(1) in _AR_SET):
                opts.append((m.group(1).replace("هـ", "ه").replace("ا", "أ"), m.group(2).strip()))
            elif not opts:  # lines before the first option belong to the stem
                stem_lines.append(ln.strip())
        return " ".join(stem_lines).strip(), opts

    @staticmethod
    def _match_answer(answer: str, opts: list[tuple[str, str]]) -> int | None:
        a = str(answer).strip()
        if not a:
            return None
        m = _OPT_RE.match(a)
        # 1) match by the Arabic ordinal prefix of the answer.
        if m and m.group(1) in _AR_SET:
            letter = m.group(1).replace("هـ", "ه").replace("ا", "أ")
            for pos, (ol, _) in enumerate(opts):
                if ol == letter:
                    return pos
        # 2) fall back to full-text match of the option body.
        body = (m.group(2).strip() if m else a)
        for pos, (_, otext) in enumerate(opts):
            if otext == body or otext == a:
                return pos
        return None

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        stem = raw_sample["stem"]
        opts: list[tuple[str, str]] = raw_sample["opts"]
        correct_pos: int = raw_sample["correct_pos"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        choices = [t for _, t in opts]
        block, letters = format_lettered_choices(choices)
        correct_letter = letters[correct_pos]

        source_id = f"{rel}:{rec_index}"
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
            reference_hash=self.reference_hash(correct_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=rec.get("Category"),
            language="ar",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": stem, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "category": rec.get("Category")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
