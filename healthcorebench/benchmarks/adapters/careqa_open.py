"""CareQA open-ended adapter (Spanish MIR exam, free-text short answers, English/Spanish).

Fixed data: ``10_CareQA/CareQA_<lang>_open.json`` — a JSON list of records::

    {"exam_id": int, "question": str, "answer": str (short free-text reference),
     "year": int, "category": str, "unique_id": str}

Task: open-ended short-answer. Distinct from the multiple-choice ``CareQA`` adapter — here the
reference is a free-text answer scored by an LLM judge. Languages are exposed as splits
(``en`` default, ``es``); ``test`` maps to en.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

_LANGS = {"test": "en", "en": "en", "es": "es"}


class CareQAOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "CareQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.0"

    def _lang(self) -> str:
        if self.split not in _LANGS:
            raise BenchmarkSplitNotFoundError(f"CareQA open split must be one of {sorted(_LANGS)}; got '{self.split}'.")
        return _LANGS[self.split]

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / f"CareQA_{self._lang()}_open.json"]

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
        lang = self._lang()

        source_id = str(rec.get("unique_id") or rec.get("exam_id") or f"{rel}:{rec_index}")
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
            capability="Knowledge",
            specialty=rec.get("category"),
            language=lang,
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"question": question},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"category": rec.get("category"), "year": rec.get("year")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": sample.source_content["question"]}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
