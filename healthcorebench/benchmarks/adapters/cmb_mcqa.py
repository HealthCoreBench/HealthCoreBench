"""CMB adapter (Chinese Medical Benchmark — CMB-Exam MCQA).

Fixed data: two files under ``8_CMB/CMB-Exam/CMB-test/`` joined by ``id``:
  - ``CMB-test-choice-question-merge.json``: {"id", "exam_type", "exam_class", "exam_subject",
    "question", "question_type", "option": {"A": str, ...}}
  - ``CMB-test-choice-answer.json``: {"id", ..., "answer": str (one or more letters, e.g. "D" or "ABD")}

CMB ships the answer cardinality per record in ``question_type``: ``单项选择题`` (9,999) and
``C型选择题`` (11) are single-answer, ``多项选择题`` (1,190) is one-or-more. The two kinds are
therefore split into two tasks instead of scoring every record with set match:

  - ``CMBAdapter`` (``CMB/mcqa``): the 10,010 single-answer records, prompted for exactly one
    letter and scored with accuracy.
  - ``CMBMultipleAnswerAdapter`` (``CMB/multiple_answer``): the 1,190 ``多项选择题`` records,
    prompted for one-or-more letters and scored with exact set match.

Routing on ``question_type`` uses the benchmark's own declared question kind, so it is a subset
definition rather than a per-record hint about the answer: the two subsets are reconstructible
from the question file alone, without consulting the answer key. 25 of the 1,190 ``多项选择题``
have a single-letter gold; they stay in the multi-answer task, because the prompt that task sends
already says 该题可能有一个或多个正确选项, so a one-letter answer is a fair thing to ask for and
exact set match scores it correctly. Moving them to ``mcqa`` would make the split depend on the
answer key, and dropping them would discard 25 answerable questions. The two subsets are
complementary and total the source: 10,010 + 1,190 + 0 dropped = 11,200 records.

Options are re-lettered locally in sorted key order, after ``_strip_answer_key_leak`` removes the
scraped answer indices described there.
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

# The one ``question_type`` value CMB uses for one-or-more-correct questions.
_MULTI_QUESTION_TYPE = "多项选择题"

# A whole option text that is just a digit — never a real CMB option.
_BARE_DIGIT = re.compile(r"^\s*[1-9]\s*$")


def _strip_answer_key_leak(option: dict) -> tuple[dict, list[str]]:
    """Remove options that are the scraped answer index rather than an answer.

    19 records (all ``初级中药士`` 多项选择题) carry a sixth option ``F`` whose entire text is a
    single digit, and in 18 of them that digit is the 1-based position of the gold option — the
    source page's answer marker was captured as if it were another choice. Left in, the prompt
    hands the model the answer.

    The rule is deliberately narrow: the record must already offer A-E, the extra key must be one
    character past ``E``, and its text must be nothing but a digit. Measured against the fixed
    data it matches exactly those 19 records, never a key that is part of a gold answer, and
    leaves the 23 records with a genuine sixth option untouched.
    """
    upper = {str(k).strip().upper(): k for k in option}
    if not set("ABCDE").issubset(upper):
        return dict(option), []
    leaked = [key for up, key in upper.items()
              if len(up) == 1 and up > "E" and _BARE_DIGIT.match(str(option[key]))]
    if not leaked:
        return dict(option), []
    return {k: v for k, v in option.items() if k not in leaked}, sorted(str(k) for k in leaked)


class CMBAdapter(BaseBenchmarkAdapter):
    benchmark_name = "CMB"
    benchmark_version = "1.0"
    adapter_version = "1.2"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    # Cardinality of this task: the single-answer subset unless a subclass flips it.
    multiple_answer = False

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"CMB provides only 'test'; requested '{self.split}'.")
        base = directory / "CMB-Exam" / "CMB-test"
        return [base / "CMB-test-choice-question-merge.json", base / "CMB-test-choice-answer.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        qfile, afile = files[0], files[1]
        qrel = self.rel_path(qfile)
        with open(qfile, "r", encoding="utf-8") as fh:
            questions = json.load(fh)
        with open(afile, "r", encoding="utf-8") as fh:
            answers = {rec["id"]: rec for rec in json.load(fh)}
        for i, q in enumerate(questions):
            # Route on the benchmark's declared question kind so each task scores only the
            # cardinality it prompts for. This ``continue`` hands the record to the sibling task
            # rather than excluding it, so it is not counted as a drop.
            if (str(q.get("question_type") or "") == _MULTI_QUESTION_TYPE) != self.multiple_answer:
                continue
            arec = answers.get(q["id"])
            if arec is None:
                # Guard only: the answer file covers all 11,200 ids in the question file.
                self.drop_source_record("no_answer_record")
                continue
            options, leaked = _strip_answer_key_leak(q.get("option") or {})
            answer_letters = sorted({c for c in str(arec.get("answer", "")).upper() if c.isalpha()})
            valid = {str(k).upper() for k in options}
            # Guards only: no record in the fixed data trips any of the three. Reported so that a
            # future data refresh cannot shrink either subset without the manifest showing it —
            # the two subsets are expected to add up to the 11,200 source records.
            if len(options) < 2:
                self.drop_source_record("fewer_than_two_options")
                continue
            if not answer_letters:
                self.drop_source_record("no_gold_letters")
                continue
            if not set(answer_letters).issubset(valid):
                self.drop_source_record("gold_letter_not_an_option")
                continue
            yield {"question": q, "options": options, "leaked_option_keys": leaked,
                   "answers": answer_letters, "source_file_rel": qrel,
                   "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        q = raw_sample["question"]
        options: dict = raw_sample["options"]
        answers: list[str] = raw_sample["answers"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(q["question"]).strip()
        src_keys = sorted(options, key=lambda k: str(k).upper())
        choices = [str(options[k]) for k in src_keys]
        block, letters = format_lettered_choices(choices)
        # map source answer letters -> local canonical letters by position.
        src_upper = [str(k).upper() for k in src_keys]
        reference_letters = sorted({letters[src_upper.index(a)] for a in answers})
        # a single-answer task reports a bare letter; a multi-answer task a letter set.
        reference = ",".join(reference_letters) if self.multiple_answer else reference_letters[0]

        source_id = str(q.get("id") or f"{rel}:{rec_index}")
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
            source_record_hash=self.input_hash(q),
            input_hash=self.input_hash({"question": question, "choices_block": block}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=q.get("exam_subject"),
            language="zh",
            modality="Text",
            answer_format="multi_choice" if self.multiple_answer else "single_choice",
            evaluation_metric="set_match" if self.multiple_answer else "accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"letters": letters, "exam_type": q.get("exam_type"),
                      "exam_class": q.get("exam_class"),
                      "question_type": q.get("question_type"),
                      # Says on the sample that the shown options are not verbatim the source's.
                      **({"removed_option_keys": raw_sample["leaked_option_keys"]}
                         if raw_sample.get("leaked_option_keys") else {})},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="zh")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])


class CMBMultipleAnswerAdapter(CMBAdapter):
    """The ``多项选择题`` subset: one-or-more correct options, scored with exact set match."""

    adapter_version = "1.2"
    prompt_template_name = "multiple_answer"
    multiple_answer = True

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_answer_prompt(c["question"], block, lang="zh")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letters(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
