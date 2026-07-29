"""MedThink-Bench adapter (medical reasoning, free-text, English).

Fixed data: ``53_MedThink-Bench/data/QA_data.json`` — a JSON list of records::

    {"Index": int, "QA_Type": str, "question": str (may embed lettered options),
     "answer": str, "Scoring_Points": [str, ...]}

Task: open-ended answering with reasoning. Questions span diagnosis, treatment, pharmacology,
ethics, etc. The reference ``answer`` (plus optional ``Scoring_Points`` rubric, carried in
metadata) is scored by an LLM judge. The question text is passed through verbatim, including any
embedded options, so choice-style items are answered in-context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.adapters.hle_med_exact import extract_short_answer
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

# Despite the "open_ended" framing, every shipped reference is an option pick: 495/500 answers
# match "``<letter>. <text>``" and the remaining 5 are the same thing without the space (or a bare
# letter). They are short — median 4 words / 29 characters. The question text embeds the option
# list, but nothing told the model where to put its choice, so the secondary token-F1 compared a
# whole essay against "B. Serum lipase" (measured 0.031, exact-match 0.000 on the recorded run).
#
# Reasoning is deliberately kept: the LLM judge is the primary metric, it reads the full raw
# response, and its rubric is fed this benchmark's ``Scoring_Points``, which grade the reasoning
# path. Only a locatable final line is added, using the ASCII "Answer:" marker that
# ``extract_short_answer`` already recognizes.
_FINAL_ANSWER_INSTRUCTION = (
    "Reason as much as you need, then end your reply with a final line in exactly this form:\n"
    "Answer: <your final answer — if the question lists options, give the option letter followed "
    "by its full text, e.g. \"B. Serum lipase\">"
)


class MedThinkBenchOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedThink-Bench"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.1"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedThink-Bench provides only 'test'; requested '{self.split}'.")
        return [directory / "data" / "QA_data.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            q = str(rec.get("question") or "").strip()
            a = str(rec.get("answer") or "").strip()
            if not q or not a:
                continue
            yield {"record": rec, "question": q, "reference": a,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        question = raw_sample["question"]
        reference = raw_sample["reference"]

        source_id = str(rec.get("Index") if rec.get("Index") is not None else f"{rel}:{rec_index}")
        content_hash = self.input_hash({"q": question})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=source_id, content_hash=content_hash)

        scoring = rec.get("Scoring_Points")
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
            specialty=rec.get("QA_Type"),
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"question": question},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"qa_type": rec.get("QA_Type"),
                      "scoring_points": scoring if isinstance(scoring, list) else None},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        prompt = f"{sample.source_content['question']}\n\n{_FINAL_ANSWER_INSTRUCTION}"
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return extract_short_answer(raw_response)
