"""PubMedQA adapter.

Fixed data: ``2_PubMedQA/pubmedqa_test.json`` — a dict keyed by PMID; each value has
``QUESTION``, ``CONTEXTS`` (list of abstract paragraphs) and ``final_decision`` in
{yes, no, maybe}. ``test_ground_truth.json`` maps PMID -> decision (used to cross-check).

Task: three-way classification (yes/no/maybe) conditioned on the provided context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.answer_parsing import parse_yes_no_maybe
from healthcorebench.schemas.sample import EvaluationSample


class PubMedQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "PubMedQA"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "yes_no_maybe"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"PubMedQA provides only 'test'; requested '{self.split}'.")
        return [directory / "pubmedqa_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Deterministic order: sort by PMID key.
        for pmid in sorted(data.keys()):
            yield {"pmid": pmid, "record": data[pmid], "source_file_rel": rel}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        pmid = raw_sample["pmid"]

        question = rec["QUESTION"]
        contexts = rec.get("CONTEXTS") or []
        context_text = "\n".join(contexts)
        decision = (rec.get("final_decision") or "").strip().lower()

        content_hash = self.input_hash({"q": question, "ctx": contexts})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=str(pmid), content_hash=content_hash)
        input_payload = {"question": question, "context": context_text}

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=str(pmid),
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=None,
            source_record_hash=self.input_hash(rec),
            input_hash=self.input_hash(input_payload),
            reference_hash=self.reference_hash(decision),
            input_type="text",
            task_type="classification",
            component="Language",
            capability="Reasoning",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="yes_no_maybe",
            evaluation_metric="accuracy",
            source_content={"question": question, "context": context_text},
            reference_answer=decision,
            reference_answer_normalized=decision,
            metadata={"pmid": pmid},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        prompt = (
            f"Context:\n{c['context']}\n\n"
            f"Question: {c['question']}\n"
            "Based on the context, answer with exactly one of: yes, no, or maybe."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_yes_no_maybe(raw_response)
