"""AfriMedQA adapter (pan-African medical MCQA, English).

Fixed data: ``7_AfriMedQA_v2/afrimedqa_v2_test.json`` — a JSON list of 5,903 mixed records. Only
``question_type == "mcq"`` records are used here (the ``consumer_queries``/``saq`` open-ended
records are a different task). MCQ records look like::

    {"sample_id": str, "question": str, "question_clean": str,
     "answer_options": str (JSON: {"option1": str, ..., "option5": "n/a"}),
     "correct_answer": str (e.g. "option4", or "option2,option4" when several are correct),
     "specialty": str, "country": str, ...}

Task: single-choice. ``answer_options`` is a JSON *string* mapping optionN -> text; options whose
text is "n/a" are dropped. ``correct_answer`` names the correct optionN key; it is mapped to the
letter of that option's position among the retained options.

306 of the 3,910 MCQ records name **several** correct options. They used to fail the
``correct not in kept`` guard and disappear without a trace; they are now scored by
``AfriMedQA_v2/multiple_answer`` (``afrimedqa_multiple_answer.py``) with ``set_match``, following
the GlobalDentBench split. Both adapters partition the file with the single shared classifier
``classify_mcq_record`` below and record everything they do not keep via ``drop_source_record``,
so each task's ``kept + dropped`` adds back up to the file's 5,903 records::

    3,598 single-answer MCQ + 306 multi-answer MCQ + 6 unusable MCQ + 1,993 open-ended = 5,903

The 6 unusable MCQ records are 1 with fewer than two non-"n/a" options and 5 whose
``correct_answer`` points at an option whose text is "n/a" (2 single, 3 multi).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample


def _parse_options(raw: Any) -> dict[str, str]:
    """Coerce answer_options (a JSON string or already-dict) into {optionN: text}."""
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str) and raw.strip():
        try:
            d = json.loads(raw)
            if isinstance(d, dict):
                return {str(k): str(v) for k, v in d.items()}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


# Drop reasons the two MCQ adapters use for each other's records, so a reader of either task's
# manifest can see where the rest of the file went.
SINGLE_ANSWER_TASK_REASON = "single_answer_mcq_scored_by_afrimedqa_mcqa"
MULTI_ANSWER_TASK_REASON = "multi_answer_mcq_scored_by_afrimedqa_multiple_answer"
OPEN_ENDED_REASON = "open_ended_record_scored_by_afrimedqa_open"


def classify_mcq_record(rec: dict) -> tuple[str, dict[str, str], list[str]]:
    """Assign one source record to exactly one bucket.

    Returns ``(kind, kept_options, gold_keys)`` where ``kind`` is ``"single"``, ``"multiple"``, or
    a drop reason. Both AfriMedQA MCQ adapters route every record through this one function, which
    is what makes their filters provably complementary: each keeps its own ``kind`` and reports
    every other record through ``drop_source_record``.
    """
    if rec.get("question_type") != "mcq":
        return OPEN_ENDED_REASON, {}, []
    options = _parse_options(rec.get("answer_options"))
    kept = {k: v for k, v in options.items() if v.strip().lower() != "n/a"}
    gold_keys = [k.strip() for k in str(rec.get("correct_answer") or "").split(",") if k.strip()]
    if len(kept) < 2:
        return "fewer_than_two_usable_options", kept, gold_keys
    if not gold_keys:
        return "no_correct_answer_key", kept, gold_keys
    if any(key not in kept for key in gold_keys):
        # the named option exists but its text is "n/a", so there is no answer to score against.
        return "gold_option_text_is_not_available", kept, gold_keys
    return ("single" if len(gold_keys) == 1 else "multiple"), kept, gold_keys


def ordered_option_keys(kept: dict[str, str]) -> list[str]:
    """Option keys ordered by their option number, for a deterministic letter assignment."""
    return sorted(kept, key=lambda k: int("".join(ch for ch in k if ch.isdigit()) or 0))


class AfriMedQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "AfriMedQA_v2"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"AfriMedQA_v2 provides only 'test'; requested '{self.split}'.")
        return [directory / "afrimedqa_v2_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            kind, kept, gold_keys = classify_mcq_record(rec)
            if kind != "single":
                self.drop_source_record(MULTI_ANSWER_TASK_REASON if kind == "multiple" else kind)
                continue
            yield {"record": rec, "kept": kept, "gold_key": gold_keys[0],
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        kept: dict[str, str] = raw_sample["kept"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(rec.get("question_clean") or rec["question"]).strip()
        # order options by their option-number key for determinism.
        ordered_keys = ordered_option_keys(kept)
        choices = [kept[k] for k in ordered_keys]
        block, letters = format_lettered_choices(choices)
        correct_pos = ordered_keys.index(raw_sample["gold_key"])
        correct_letter = letters[correct_pos]

        source_id = str(rec.get("sample_id") or f"{rel}:{rec_index}")
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
            reference_hash=self.reference_hash(correct_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=rec.get("specialty"),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "country": rec.get("country"), "tier": rec.get("tier")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
