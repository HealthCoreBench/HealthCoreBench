"""BioHopR adapter (biomedical multi-hop QA — single-answer subset, English).

Fixed data: ``50_BioHopR/biohopr_test.json`` — a JSON list of multi-hop KG questions::

    {"hop1_question": str, "hop2_question": str, "prompt": str, "answer": [str, ...],
     "target_type": str, "relation_hop1": str, ...}

Task: open-ended 2-hop biomedical reasoning. This adapter uses only records with a **single**
accepted answer (``len(answer) == 1``) so scoring is unambiguous; the multi-answer records (which
need recall-style set scoring) are excluded. The ``hop2_question`` (the full 2-hop question) is the
prompt and the single ``answer`` is the reference, scored by an LLM judge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


class BioHopRSingleOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "BioHopR"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"BioHopR provides only 'test'; requested '{self.split}'.")
        return [directory / "biohopr_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            answer = rec.get("answer") or []
            question = str(rec.get("hop2_question") or rec.get("prompt") or "").strip()
            if not isinstance(answer, list) or len(answer) != 1 or not str(answer[0]).strip() or not question:
                continue
            yield {"record": rec, "question": question, "reference": str(answer[0]).strip(),
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        question = raw_sample["question"]
        reference = raw_sample["reference"]

        source_id = f"{rel}:{rec_index}"
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
            specialty=rec.get("target_type"),
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"question": question},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"target_type": rec.get("target_type"),
                      "relation_hop1": rec.get("relation_hop1"),
                      "relation_hop2": rec.get("relation_hop2")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": sample.source_content["question"]}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
