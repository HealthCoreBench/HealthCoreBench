"""Medbullets adapter.

Fixed data: ``4_Medbullets/medbullets_op4.json`` and ``medbullets_op5.json`` — two variants
with *different* record shapes:

- op4: ``{"question": str, "options": {"A".."D": str}, "answer": str, "answer_idx": "A".."D"}``
- op5: ``{"link": str, "question": str, "opa".."ope": str, "answer_idx": "A".."E",
         "answer": str, "explanation": str}``

Task: single-choice. Splits: ``op4`` (4 options) | ``op5`` (5 options). Default ``test`` → op5.

The two files are the *same* 298 questions asked under two option counts (308 records each, 298
unique question texts, 100% overlap). Reading both for ``test`` therefore double-counted every
item, blended the 4- and 5-option conditions into one score and computed the confidence interval
on n=616 instead of n=308. ``test`` reads op5 only — the harder, complete condition — and each
condition remains available on its own via the ``op4``/``op5`` splits.

Each file also repeats items internally, and the two files repeat them differently, so the
duplicate criterion is the tuple the model actually sees and is graded on — question, option
texts, and gold letter:

- op5: 10 of the 308 records are exact repeats on that tuple (they differ only in ``link``, a
  canonical ``step2.medbullets.com`` URL versus a ``bit.ly`` shortlink), leaving 298.
- op4: only 4 of its 10 repeated question texts are exact repeats, leaving 304. The other 6 pairs
  share a vignette and a gold *answer text* but were built with different distractor sets — e.g.
  "Cerebral salt wasting" appears once as option A against three distractors and once as option B
  against a different three. Those are two genuinely different questions, so both are kept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

_OP5_LETTERS = ["A", "B", "C", "D", "E"]


class MedbulletsAdapter(BaseBenchmarkAdapter):
    benchmark_name = "Medbullets"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _variants(self) -> list[str]:
        # "test" (default) is op5 alone: op4 holds the same questions with one option removed,
        # so reading both would score every item twice under two different conditions.
        if self.split == "test":
            return ["op5"]
        if self.split in ("op4", "op5"):
            return [self.split]
        raise BenchmarkSplitNotFoundError(f"Medbullets split must be 'op4'/'op5' (or 'test'=op5); got '{self.split}'.")

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / f"medbullets_{v}.json" for v in self._variants()]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        seen: set[tuple] = set()
        for f in files:
            rel = self.rel_path(f)
            with open(f, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            for i, rec in enumerate(records):
                question, options, answer_letter = self._extract(rec)
                # Keyed on what is presented and graded, not on the raw record: the repeats carry
                # a different ``link``, so whole-record identity would have found none of them.
                key = (question, tuple(sorted(options.items())), answer_letter)
                if key in seen:
                    self.drop_source_record("intra_file_duplicate")
                    continue
                seen.add(key)
                yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def _extract(self, rec: dict) -> tuple[str, dict, str]:
        """Return (question, options-dict, answer_letter) handling both op4/op5 shapes."""
        question = rec["question"]
        answer_letter = str(rec["answer_idx"]).strip().upper()
        if isinstance(rec.get("options"), dict):          # op4 shape
            options = {k: str(v) for k, v in rec["options"].items()}
        else:                                              # op5 shape: opa..ope
            keymap = {"A": "opa", "B": "opb", "C": "opc", "D": "opd", "E": "ope"}
            options = {L: str(rec[k]) for L, k in keymap.items() if rec.get(k) not in (None, "")}
        return question, options, answer_letter

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question, options, answer_letter = self._extract(rec)
        letters = sorted(options.keys())
        block = "\n".join(f"{l}. {options[l]}" for l in letters)

        source_id = f"{rel}:{rec_index}"
        content_hash = self.input_hash({"q": question, "o": options})
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
            reference_hash=self.reference_hash(answer_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "options": options, "letters": letters},
            reference_answer=answer_letter,
            reference_answer_normalized=answer_letter,
            metadata={"letters": letters},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        letters = c["letters"]
        block = "\n".join(f"{l}. {c['options'][l]}" for l in letters)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or _OP5_LETTERS)
