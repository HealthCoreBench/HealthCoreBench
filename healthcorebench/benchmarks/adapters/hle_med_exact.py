"""HLE-Med adapter — text-only exact-answer subset (Humanity's Last Exam, medicine).

Fixed data: ``11_HLE_med/hle_med_test_text.json`` — the image-free HLE medicine subset. This
adapter covers ``answer_type == "exactMatch"`` records::

    {"id": str, "question": str, "answer": str (short exact answer: a value, word, tuple, ...),
     "answer_type": "exactMatch", "raw_subject": str, ...}

Task: short exact-answer QA scored with EM + token-F1. The 75 references are 1-42 characters
(median 12), so the scored span has to be the model's final answer and not its working: asked the
bare question, the model returns 495-2,718 characters of reasoning, EM is unreachable, and token-F1
collapses to the length ratio (measured 0.0085 over 10 rows). The prompt therefore asks for the
answer alone and ``parse_response`` extracts the final short answer, the same pattern
``bioasq_factoid_open`` already uses for its own short references.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.answer_parsing import final_answer_region
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

BREVITY_INSTRUCTION = (
    "Answer with the exact short answer only — a value, word, name, or tuple — on a single line "
    "prefixed with 'Answer:'. Do not show your reasoning."
)

# "Answer: X" / "Final answer: X" and the localized-free English forms the models actually emit.
_ANSWER_LINE = re.compile(
    r"^\s*(?:\*\*|__)?\s*(?:final\s+answer|answer|result|value)\s*(?:\*\*|__)?\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_ANSWER_PHRASE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|was)\s*[:=]?\s*(.+?)(?:[.\n]|$)", re.IGNORECASE
)
# Some prompts prescribe their own label ("INGREDIENT: name", "Start date: YYYY-MM"); the value
# after the label is the answer, not the label itself.
_LABELLED_VALUE = re.compile(r"^\s*(?:\*\*|__)?[A-Za-z][A-Za-z /_-]{0,30}\s*[:=]\s*(\S.*)$")


def extract_short_answer(raw_response: str) -> str | None:
    """Return the model's final short answer, or ``None`` when there is nothing to score.

    Layered like the multiple-choice extractor: an explicit marker wins, then a labelled value on
    the last line, then the last non-empty line. The full response is never returned — with a
    1-42-character reference an essay makes token-F1 a length ratio rather than a correctness
    measure.
    """
    region = final_answer_region(raw_response or "").strip()
    if not region:
        return None

    boxed = re.findall(r"\\boxed\{([^{}]*)\}", region)
    if boxed:
        return _clean_short_answer(boxed[-1])

    lines = [line.strip() for line in region.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _ANSWER_LINE.match(line)
        if match:
            return _clean_short_answer(match.group(1))

    phrases = _ANSWER_PHRASE.findall(region)
    if phrases:
        return _clean_short_answer(phrases[-1])

    # No marker: a short standalone opening line followed by long prose is a lead answer with an
    # explanation after it, so the answer is the first line and not the closing sentence.
    if len(lines) >= 2 and len(lines[0]) <= 40 and len(lines[-1]) > 80:
        return _clean_short_answer(lines[0])
    if lines:
        return _clean_short_answer(lines[-1])
    return None


def _clean_short_answer(value: str) -> str | None:
    value = value.strip()
    value = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", value)
    value = value.replace("**", "").replace("__", "").strip().strip("$").strip()
    # Prompts that prescribe their own label get the label echoed back; keep only the value.
    labelled = _LABELLED_VALUE.match(value)
    if labelled:
        value = labelled.group(1).strip()
    value = value.strip().strip('"').strip("'").strip()
    value = value.rstrip(".").strip()
    return value or None


class HLEMedExactAdapter(BaseBenchmarkAdapter):
    benchmark_name = "HLE_med"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "short_answer"
    prompt_template_version = "1.1"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"HLE_med provides only 'test'; requested '{self.split}'.")
        return [directory / "hle_med_test_text.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            if rec.get("answer_type") != "exactMatch" or rec.get("image"):
                continue
            question = str(rec.get("question") or "").strip()
            answer = str(rec.get("answer") or "").strip()
            if not question or not answer:
                continue
            yield {"record": rec, "question": question, "reference": answer,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        question = raw_sample["question"]
        reference = raw_sample["reference"]

        source_id = str(rec.get("id") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"q": question})
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
            input_hash=self.input_hash({"question": question}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("raw_subject"),
            language="en",
            modality="Text",
            answer_format="short_answer",
            # short canonical answer (e.g. "False", a number, an entity) — rule-based EM +
            # token-F1 is exact and free; no LLM judge needed.
            evaluation_metric="text_f1",
            source_content={"question": question},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"raw_subject": rec.get("raw_subject"), "answer_type": "exactMatch"},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        question = sample.source_content["question"]
        return [{"role": "user", "content": f"{question}\n\n{BREVITY_INSTRUCTION}"}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return extract_short_answer(raw_response)
