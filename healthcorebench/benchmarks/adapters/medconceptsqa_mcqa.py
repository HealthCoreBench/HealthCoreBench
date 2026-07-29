"""MedConceptsQA adapter (medical-coding MCQA, English).

Fixed data: ``21_MedConceptsQA/<vocab>_<difficulty>/<vocab>_<difficulty>_test.json`` — one
directory per (vocabulary, difficulty) combination, each a JSON list of records::

    {"question_id": int, "answer": str (option text), "answer_id": str (letter "A"-"D"),
     "option1": str, "option2": str, "option3": str, "option4": str,
     "question": str (already contains the lettered options), "vocab": str, "level": str}

Task: single-choice, four options, over ICD-9/10 and ATC codes. Options are re-lettered locally
from ``option1..4`` so the prompt is canonical and independent of the pre-formatted ``question``
text.

One task per difficulty (``mcqa_easy`` / ``mcqa_medium`` / ``mcqa_hard``), each pooling the five
vocabularies at that difficulty. The three levels ask about *the same concepts* and differ only in
how the distractors were drawn — measured on the fixed data, ``easy`` and ``medium`` hold an
identical record count per vocabulary and their correct-answer sets overlap 99.9% (``icd9proc``
4,560/4,564; ``atc`` 5,729/5,733). Pooling all fifteen subsets into one accuracy therefore scored
every concept about three times, at three different distractor difficulties, under a single number
— and difficulty is the axis this benchmark exists to vary, so collapsing it discards its point.

Per-vocabulary numbers are not lost: ``specialty`` carries the vocabulary and the run summary
breaks the score down by it.

Individual subsets stay reachable as splits named ``<vocab>_<difficulty>`` (e.g. ``atc_hard``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample
from healthcorebench.utils.stream_json import iter_json_array

_VOCABS = ("atc", "icd10cm", "icd10proc", "icd9cm", "icd9proc")
_LEVELS = ("easy", "medium", "hard")
_SUBSETS = {f"{v}_{l}" for v in _VOCABS for l in _LEVELS}


class MedConceptsQAAdapter(BaseBenchmarkAdapter):
    """Base class; each registered task fixes ``level`` to one difficulty."""

    benchmark_name = "MedConceptsQA"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    # Set by the subclasses below. ``None`` would mean "every difficulty", which is the pooling
    # this split exists to avoid, so the base class is not registered.
    level: str | None = None

    def _subset(self) -> str:
        if self.split not in _SUBSETS:
            raise BenchmarkSplitNotFoundError(
                f"MedConceptsQA split must be 'test' or one of {sorted(_SUBSETS)}; "
                f"got '{self.split}'."
            )
        return self.split

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        # ``test`` means "this task's difficulty across all five vocabularies"; an explicit
        # ``<vocab>_<difficulty>`` split narrows to that one subset.
        if self.split == "test":
            if self.level is None:
                raise BenchmarkSplitNotFoundError(
                    "MedConceptsQAAdapter is abstract: run MedConceptsQA/mcqa_easy, "
                    "/mcqa_medium or /mcqa_hard, or name an explicit <vocab>_<difficulty> split."
                )
            return [directory / f"{vocab}_{self.level}" / f"{vocab}_{self.level}_test.json"
                    for vocab in _VOCABS]
        sub = self._subset()
        return [directory / sub / f"{sub}_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
            rel = self.rel_path(f)
            for i, rec in enumerate(iter_json_array(f)):
                yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        # The stem is the question text with its pre-formatted options stripped: keep only the
        # first line (the actual question) and re-letter option1..4 ourselves.
        stem = str(rec["question"]).split("\n", 1)[0].strip()
        choices = [str(rec[f"option{n}"]) for n in (1, 2, 3, 4)]
        block, letters = format_lettered_choices(choices)
        correct_letter = str(rec["answer_id"]).strip().upper()

        source_id = str(rec.get("question_id") or f"{rel}:{rec_index}")
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
            specialty=rec.get("vocab"),
            difficulty=rec.get("level"),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": stem, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "vocab": rec.get("vocab"), "level": rec.get("level")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])


class MedConceptsQAEasyAdapter(MedConceptsQAAdapter):
    level = "easy"


class MedConceptsQAMediumAdapter(MedConceptsQAAdapter):
    level = "medium"


class MedConceptsQAHardAdapter(MedConceptsQAAdapter):
    level = "hard"
