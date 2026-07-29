"""LiveQA adapter (TREC-2017 LiveQA Medical, consumer health QA, English).

Fixed data: ``9_LiveQA/TREC-2017-LiveQA-Medical-Test.json`` — a JSON list of records::

    {"qid": str, "original_question": {"qfile": str, "subject": str, "message": str},
     "nist_paraphrase": str, "annotations": [...],
     "reference_answers": [{"aid": str, "answer": str, ...}, ...]}

Task: open-ended consumer health question answering. The prompt uses the NIST paraphrase when
available (a concise reformulation), otherwise the original message. TREC-2017 supplies one to
five independently authored reference answers per question (measured: 50 of the 104 questions
have more than one), and the official task treats each of them as an acceptable gold. All of
them are therefore kept — the first as the primary reference, the rest as aliases — and the LLM
judge scores the model against the best-matching one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


class LiveQAOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "LiveQA"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"LiveQA provides only 'test'; requested '{self.split}'.")
        return [directory / "TREC-2017-LiveQA-Medical-Test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            # Collect every reference answer, not just the first: they are alternative gold
            # answers from different sources, and dropping them made the judge compare against
            # one arbitrary phrasing of an answer the question accepts several forms of.
            references: list[str] = []
            for r in rec.get("reference_answers") or []:
                text = str(r.get("answer") or "").strip()
                if text and text not in references:
                    references.append(text)
            oq = rec.get("original_question") or {}
            question = str(rec.get("nist_paraphrase") or oq.get("message") or "").strip()
            if not question:
                self.drop_source_record("empty_question")
                continue
            if not references:
                self.drop_source_record("no_reference_answer")
                continue
            yield {"record": rec, "question": question, "references": references,
                   "subject": oq.get("subject"), "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        question = raw_sample["question"]
        references: list[str] = raw_sample["references"]
        reference = references[0]
        aliases = references[1:]

        source_id = str(rec.get("qid") or f"{rel}:{rec_index}")
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
            reference_hash=self.reference_hash(references),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Knowledge",
            specialty=raw_sample.get("subject"),
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"question": question},
            reference_answer=reference,
            reference_answer_normalized=reference,
            reference_aliases=aliases or None,
            metadata={"subject": raw_sample.get("subject"), "num_references": len(references)},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": sample.source_content["question"]}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
