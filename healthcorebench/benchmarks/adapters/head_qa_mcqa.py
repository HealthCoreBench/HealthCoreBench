"""HEAD-QA adapter (multilingual healthcare exam QA).

Fixed data: ``19_HEAD-QA_v2/head_qa_<lang>_test.json`` — five languages (en, es, gl, it, ru),
each a JSON list of records::

    {"qid": int, "qtext": str, "ra": int (correct answer id, 1-based),
     "answers": [{"aid": int, "atext": str}, ...], "year": int, "category": str,
     "name": str, "image": str|None}

Task: single-choice. Options come from ``answers`` (ordered by ``aid``); ``ra`` is the
correct 1-based answer id. Records carrying an image are skipped (this is the text-only
adapter).

Each language is its own registry task (``HEAD-QA_v2/mcqa`` = en, ``.../mcqa_es``, ``mcqa_gl``,
``mcqa_it``, ``mcqa_ru``), because ``benchmark.split`` is a single global setting: in an ALL run
every task shares it, so a split-driven language would collapse the five languages into one.
The split is still honoured when a task does not name a language, so ``--split ru`` keeps
working for a single-benchmark run.
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

_LANGS = {"test": "en", "en": "en", "es": "es", "gl": "gl", "it": "it", "ru": "ru"}


class HEADQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "HEAD-QA_v2"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _lang(self) -> str:
        # A task suffix ("mcqa_es") pins the language; plain "mcqa" falls back to the split so
        # that a single-benchmark run can still switch languages with --split.
        task = (self.entry.task or "") if getattr(self, "entry", None) else ""
        if task.startswith("mcqa_"):
            lang = task.removeprefix("mcqa_")
            if lang not in _LANGS:
                raise BenchmarkSplitNotFoundError(f"HEAD-QA task must name one of {sorted(_LANGS)}; got '{task}'.")
            return _LANGS[lang]
        if self.split not in _LANGS:
            raise BenchmarkSplitNotFoundError(f"HEAD-QA split must be one of {sorted(_LANGS)}; got '{self.split}'.")
        return _LANGS[self.split]

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / f"head_qa_{self._lang()}_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            if rec.get("image"):  # text-only adapter skips image-bearing questions
                self.drop_source_record("image_bearing_question")
                continue
            yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        lang = self._lang()

        question = rec["qtext"]
        answers = sorted(rec["answers"], key=lambda a: int(a["aid"]))
        choices = [str(a["atext"]) for a in answers]
        aids = [int(a["aid"]) for a in answers]
        block, letters = format_lettered_choices(choices)
        # ra is the correct aid (1-based); map it to the option position.
        correct_pos = aids.index(int(rec["ra"]))
        correct_letter = letters[correct_pos]

        source_id = str(rec.get("qid", f"{rel}:{rec_index}"))
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
            specialty=rec.get("category"),
            language=lang,
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "year": rec.get("year")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
