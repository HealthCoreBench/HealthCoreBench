"""MedXpertQA (Text) adapter.

Fixed data: ``5_MedXpertQA_Text/MedXpertQA_Text_test.jsonl`` — one JSON object per line::

    {"id": str, "question": str, "options": {"A".."J": str}, "label": "A"..,
     "medical_task": str, "body_system": str, "question_type": str}

Task: single-choice with up to 10 options (A-J). ``label`` is the correct letter.

``question`` embeds its own inline ``Answer Choices: (A) … (J) …`` block in all 2,450 records.
That block is stripped so the prompt lists the options once, in the canonical ``Options:`` form
built from ``options`` — the dict ``label`` and ``letters`` refer to, and the authority wherever
the two disagree (2 records).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

# The inline option block appended to every ``question``; everything from here on is a copy of
# ``options`` and is dropped in favour of the adapter's own choices block.
_EMBEDDED_CHOICES_RE = re.compile(r"\s*Answer\s+Choices\s*:\s*", re.IGNORECASE)


def _strip_embedded_choices(question: str) -> str:
    """Return the question stem without its trailing inline ``Answer Choices:`` block."""
    return _EMBEDDED_CHOICES_RE.split(str(question), maxsplit=1)[0].strip()


class MedXpertQATextAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedXpertQA_Text"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedXpertQA_Text provides only 'test'; requested '{self.split}'.")
        return [directory / "MedXpertQA_Text_test.jsonl"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                yield {"record": json.loads(line), "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = _strip_embedded_choices(rec["question"])
        options = {k: str(v) for k, v in rec["options"].items()}
        letters = sorted(options.keys())
        block = "\n".join(f"{l}. {options[l]}" for l in letters)
        answer_letter = str(rec["label"]).strip().upper()

        source_id = str(rec.get("id", f"{rel}:{rec_index}"))
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
            capability="Reasoning",
            specialty=rec.get("body_system"),
            language="en",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "options": options, "letters": letters},
            reference_answer=answer_letter,
            reference_answer_normalized=answer_letter,
            metadata={"letters": letters, "medical_task": rec.get("medical_task"),
                      "question_type": rec.get("question_type")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        letters = c["letters"]
        block = "\n".join(f"{l}. {c['options'][l]}" for l in letters)
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang="en")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D"])
