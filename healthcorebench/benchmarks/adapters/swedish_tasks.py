"""Swedish Medical LLM Benchmark — the three MCQA collections outside ``medqa-swe``.

``49_Swedish_Medical_LLM_Benchmark/`` ships far more than the licensing-exam file already covered
by ``swedish_medqa_mcqa.py``. Three more collections are multiple-choice with a usable gold answer;
each gets its own task because the item style differs (single fact question vs. case vignette vs.
national theory exam), and pooling them would hide which one a model fails:

``Swedish_Medical_LLM_Benchmark/specialist_mcqa`` (596 items)
    ``specialist_questions/<specialty>.json`` (7 files) + ``general_practitioner/
    general_practioner.json``, all sharing::

        {"question": str, "options": [str, ...], "correct_answer": str (an option's *text*),
         "justification": str}

    The gold is the option text, so it is matched back to its position and re-lettered.

``Swedish_Medical_LLM_Benchmark/clinical_case_mcqa`` (1,130 items)
    ``specialist_questions/emergency_medicine/emergency_medicine_corrected.json`` (116 cases) and
    ``specialist_questions/gp/fall_descriptions.json`` (137 cases), shaped as::

        {"case_description": str, "questions": [{"question": str,
          "options": ["A) text", ...], "correct_answer": "C) text", "explanation": str}], ...}

    Every question is scored against its own case vignette, which is prepended to the prompt.

``Swedish_Medical_LLM_Benchmark/theory_mcqa`` (~498 items)
    ``swetheoreticaldoctorsexam/clinical_case.json`` — ``{"1": {"QUESTION", "ANSWER", "EXAM"}}``.
    The options are embedded at the end of ``QUESTION`` as ``a)``..``e)`` lines and ``ANSWER`` is
    ``"a) <text>"``; the option block is parsed out so the prompt is canonical and the gold letter
    can be validated. Items with no embedded option block (free-text questions) and the three
    items whose gold is not a single option letter are dropped with a reason rather than guessed.

Not registered, on purpose: ``*_tiny.json`` (3–8 cases sampled from the full case files),
``*_clinical_format.json`` (the same case questions flattened one-per-entry),
``emergency_medicine.json`` (superseded by ``_corrected``), ``*.json.gpg`` (encrypted held-out
splits) and ``pubmedqa/data/ori_pqal_swe.json`` (a Swedish translation of the same 1,000 PubMedQA
records already scored by ``PubMedQA/classification``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.context_window import fit_context_to_window
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

_SPECIALIST_FILES = [
    ("specialist_questions/anestesi.json", "anesthesiology"),
    ("specialist_questions/cardiology.json", "cardiology"),
    ("specialist_questions/dermatology.json", "dermatology"),
    ("specialist_questions/endocrinology.json", "endocrinology"),
    ("specialist_questions/hematologi.json", "hematology"),
    ("specialist_questions/neurologi.json", "neurology"),
    ("specialist_questions/psychiatri.json", "psychiatry"),
    ("general_practitioner/general_practioner.json", "general_practice"),
]
_CASE_FILES = [
    ("specialist_questions/emergency_medicine/emergency_medicine_corrected.json", "emergency_medicine"),
    ("specialist_questions/gp/fall_descriptions.json", "general_practice"),
]
_THEORY_FILE = "swetheoreticaldoctorsexam/clinical_case.json"

# "A) text" / "a. text" / "B: text" -> (letter, text).
_LETTERED = re.compile(r"^\s*([A-Za-z])\s*[\)\.:]\s*(.*)$", re.DOTALL)
# An option line inside the theory-exam question body.
_OPTION_LINE = re.compile(r"^\s*([a-e])\)\s*(\S.*)$")


def _strip_letter(text: str) -> tuple[str | None, str]:
    """Split a source option/answer into its own letter marker and its text."""
    match = _LETTERED.match(str(text))
    if not match:
        return None, str(text).strip()
    return match.group(1).upper(), match.group(2).strip()


class _SwedishBase(BaseBenchmarkAdapter):
    benchmark_name = "Swedish_Medical_LLM_Benchmark"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"
    capability = "Knowledge"

    def _require_test_split(self) -> None:
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(
                f"Swedish_Medical_LLM_Benchmark provides only 'test'; requested '{self.split}'."
            )

    def _sample(
        self,
        *,
        raw_sample: dict,
        sample_index: int,
        question: str,
        choices: list[str],
        correct_pos: int,
        source_id: str,
        specialty: str | None,
        context: str = "",
        extra_metadata: dict | None = None,
    ) -> EvaluationSample:
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        record = raw_sample["record"]
        block, letters = format_lettered_choices(choices)
        correct_letter = letters[correct_pos]

        context_meta: dict[str, Any] = {}
        if context:
            generation = getattr(self.config, "generation", None)
            context, context_meta = fit_context_to_window(
                context,
                fixed_prompt=multiple_choice_prompt(question, block, lang="sv"),
                max_model_len=getattr(getattr(self.config, "hardware", None), "max_model_len", None),
                max_output_tokens=self.output_token_budget_for_format("single_choice"),
                reserve_tokens=getattr(generation, "context_token_reserve", 512),
                policy=getattr(generation, "context_overflow_policy", "error"),
            )

        content_hash = self.input_hash({"q": question, "c": choices, "ctx": context})
        return EvaluationSample(
            sample_id=self.make_sample_id(
                source_file_rel=rel, source_sample_id=source_id, content_hash=content_hash
            ),
            source_sample_id=source_id,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=rec_index,
            source_record_hash=self.input_hash(record),
            input_hash=self.input_hash(
                {"question": question, "choices_block": block, "context": context}
            ),
            reference_hash=self.reference_hash(correct_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability=self.capability,
            specialty=specialty,
            language="sv",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices, "context": context},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, **(extra_metadata or {}), **context_meta},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        content = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in content["choices"]])
        prompt = multiple_choice_prompt(content["question"], block, lang="sv")
        context = str(content.get("context") or "")
        if context:
            prompt = f"Patientfall:\n{context}\n\n{prompt}"
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(
            raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"]
        )


class SwedishSpecialistMCQAAdapter(_SwedishBase):
    """Single-fact specialty questions whose gold answer is one option's text."""

    def discover_source_files(self) -> list[Path]:
        self._require_test_split()
        directory = self.get_benchmark_directory()
        return [directory / rel for rel, _ in _SPECIALIST_FILES]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        specialties = {name: specialty for name, specialty in _SPECIALIST_FILES}
        for f in files:
            rel = self.rel_path(f)
            specialty = next(
                (s for name, s in specialties.items() if f.as_posix().endswith(name)), None
            )
            with open(f, "r", encoding="utf-8") as handle:
                records = json.load(handle)
            for i, rec in enumerate(records):
                choices = [str(o).strip() for o in (rec.get("options") or []) if str(o).strip()]
                gold = str(rec.get("correct_answer") or "").strip()
                question = str(rec.get("question") or "").strip()
                if not question or len(choices) < 2:
                    self.drop_source_record("missing_question_or_options")
                    continue
                if gold not in choices:
                    self.drop_source_record("gold_answer_not_among_options")
                    continue
                yield {"record": rec, "choices": choices, "correct_pos": choices.index(gold),
                       "question": question, "specialty": specialty,
                       "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        return self._sample(
            raw_sample=raw_sample,
            sample_index=sample_index,
            question=raw_sample["question"],
            choices=raw_sample["choices"],
            correct_pos=raw_sample["correct_pos"],
            source_id=f"{raw_sample['specialty']}:{raw_sample['source_record_index']}",
            specialty=raw_sample["specialty"],
        )


class SwedishClinicalCaseMCQAAdapter(_SwedishBase):
    """Case-vignette questions: several questions share one ``case_description``."""

    capability = "Reasoning"

    def discover_source_files(self) -> list[Path]:
        self._require_test_split()
        directory = self.get_benchmark_directory()
        return [directory / rel for rel, _ in _CASE_FILES]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        specialties = {name: specialty for name, specialty in _CASE_FILES}
        for f in files:
            rel = self.rel_path(f)
            specialty = next(
                (s for name, s in specialties.items() if f.as_posix().endswith(name)), None
            )
            with open(f, "r", encoding="utf-8") as handle:
                cases = json.load(handle)
            for case_index, case in enumerate(cases):
                context = str(case.get("case_description") or "").strip()
                for q_index, item in enumerate(case.get("questions") or []):
                    question = str(item.get("question") or "").strip()
                    raw_options = [str(o) for o in (item.get("options") or [])]
                    pairs = [_strip_letter(o) for o in raw_options]
                    choices = [text for _, text in pairs if text]
                    if not question or len(choices) < 2:
                        self.drop_source_record("missing_question_or_options")
                        continue
                    gold_letter, gold_text = _strip_letter(item.get("correct_answer") or "")
                    if gold_text in choices:
                        correct_pos = choices.index(gold_text)
                    elif gold_letter and gold_letter in [letter for letter, _ in pairs if letter]:
                        # one item's gold text was re-worded; its letter marker still resolves.
                        correct_pos = [letter for letter, _ in pairs].index(gold_letter)
                    else:
                        self.drop_source_record("gold_answer_not_among_options")
                        continue
                    yield {"record": item, "question": question, "choices": choices,
                           "correct_pos": correct_pos, "context": context, "specialty": specialty,
                           "source_file_rel": rel, "source_record_index": case_index,
                           "question_index": q_index}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        return self._sample(
            raw_sample=raw_sample,
            sample_index=sample_index,
            question=raw_sample["question"],
            choices=raw_sample["choices"],
            correct_pos=raw_sample["correct_pos"],
            source_id=f"{raw_sample['specialty']}:{raw_sample['source_record_index']}:{raw_sample['question_index']}",
            specialty=raw_sample["specialty"],
            context=raw_sample["context"],
            extra_metadata={"question_index": raw_sample["question_index"]},
        )


class SwedishTheoryExamAdapter(_SwedishBase):
    """Swedish national theory exam: options are embedded in the question body."""

    capability = "Reasoning"

    def discover_source_files(self) -> list[Path]:
        self._require_test_split()
        return [self.get_benchmark_directory() / _THEORY_FILE]

    def count_source_records(self, path: Path) -> int | None:
        # The file is ``{"1": {...}, "2": {...}}``; the base counter only understands a JSON list
        # or a dict wrapping exactly one list, and would report None, leaving the manifest unable
        # to reconcile the 43 dropped items against the file.
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return len(payload) if isinstance(payload, dict) else None

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        for i, (key, rec) in enumerate(records.items()):
            body = str(rec.get("QUESTION") or "")
            stem, choices, letters = _split_embedded_options(body)
            if len(choices) < 2:
                self.drop_source_record("no_embedded_option_block")
                continue
            gold_letter, _ = _strip_letter(rec.get("ANSWER") or "")
            if not gold_letter or gold_letter not in letters:
                # e.g. a free-text gold, or "bc) EKG" naming two options at once.
                self.drop_source_record("gold_answer_is_not_a_single_option_letter")
                continue
            yield {"record": rec, "question": stem, "choices": choices,
                   "correct_pos": letters.index(gold_letter), "exam": rec.get("EXAM"),
                   "source_id": str(key), "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        return self._sample(
            raw_sample=raw_sample,
            sample_index=sample_index,
            question=raw_sample["question"],
            choices=raw_sample["choices"],
            correct_pos=raw_sample["correct_pos"],
            source_id=raw_sample["source_id"],
            specialty=None,
            extra_metadata={"exam": raw_sample["exam"]},
        )


def _split_embedded_options(body: str) -> tuple[str, list[str], list[str]]:
    """Split ``"<stem>\\na) x\\nb) y"`` into stem, option texts and their source letters.

    Only the trailing run of ``a)``..``e)`` lines counts: an ``a)`` appearing mid-vignette would
    otherwise swallow the stem. Letters must start at ``a`` and be consecutive, which is what
    distinguishes a real option block from an enumeration inside the case text.
    """
    lines = body.splitlines()
    options: list[tuple[str, str]] = []
    cut = len(lines)
    for index in range(len(lines) - 1, -1, -1):
        match = _OPTION_LINE.match(lines[index])
        if match:
            options.append((match.group(1), match.group(2).strip()))
            cut = index
            continue
        if options and lines[index].strip():
            break
    options.reverse()
    letters = [letter.upper() for letter, _ in options]
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    if not letters or letters != expected:
        return body.strip(), [], []
    stem = "\n".join(lines[:cut]).strip()
    return stem, [text for _, text in options], letters
