"""MedQuAD adapter (consumer medical QA from NIH/NLM sources, English).

Fixed data: ``54_MedQuAD/medquad_all_qa.json`` — a JSON list of records::

    {"doc_id": str, "source": str, "url": str, "focus": str, "question": str, "answer": str}

Task: open-ended free-text answering. A large fraction of records have an empty ``answer``
(the source pages for several subsets could not redistribute answers); those are skipped, so only
question/answer pairs with a real reference remain. An LLM judge scores the model's answer against
the reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


class MedQuADOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedQuAD"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedQuAD provides only 'test'; requested '{self.split}'.")
        return [directory / "medquad_all_qa.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        grouped: dict[tuple[str, str], dict] = {}
        for i, rec in enumerate(records):
            question = str(rec.get("question") or "").strip()
            answer = str(rec.get("answer") or "").strip()
            if not question or not answer:
                continue  # many rows ship without a redistributable answer
            doc_id = str(rec.get("doc_id") or "").strip()
            # MedQuAD contains multiple valid answers for some document/question pairs. Treat
            # them as aliases of one logical question instead of emitting colliding sample IDs
            # and overweighting those questions during evaluation.
            key = (doc_id or f"{rel}:{i}", " ".join(question.split()).casefold())
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = {
                    "record": rec,
                    "question": question,
                    "reference": answer,
                    "reference_aliases": [],
                    "source_file_rel": rel,
                    "source_record_index": i,
                    "source_record_indices": [i],
                }
                continue
            existing["source_record_indices"].append(i)
            if answer != existing["reference"] and answer not in existing["reference_aliases"]:
                existing["reference_aliases"].append(answer)
        yield from grouped.values()

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        question = raw_sample["question"]
        reference = raw_sample["reference"]
        reference_aliases = raw_sample.get("reference_aliases") or []

        source_id = str(rec.get("doc_id") or f"{rel}:{rec_index}")
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
            specialty=rec.get("source"),
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"question": question},
            reference_answer=reference,
            reference_answer_normalized=reference,
            reference_aliases=reference_aliases or None,
            metadata={
                "source": rec.get("source"),
                "focus": rec.get("focus"),
                "source_record_indices": raw_sample.get("source_record_indices") or [rec_index],
            },
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": sample.source_content["question"]}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
