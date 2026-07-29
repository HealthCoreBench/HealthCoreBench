"""MedCaseReasoning adapter (clinical case → diagnosis, free-text, English).

Fixed data: ``31_MedCaseReasoning/medcasereasoning_test.json`` — a JSON list of records::

    {"pmcid": str, "title": str, "journal": str, "article_link": str, "publication_date": str,
     "text": str (full case report), "case_prompt": str (the case presentation shown to the model),
     "diagnostic_reasoning": str, "final_diagnosis": str}

Task: open-ended diagnosis. The model is given ``case_prompt`` (case presentation) and must state
the final diagnosis; an LLM judge compares its answer to ``final_diagnosis`` (the reasoning trace
is kept in metadata for analysis, not shown to the model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.adapters.hle_med_exact import extract_short_answer
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

# ``final_diagnosis`` is a diagnosis phrase (measured over the shipped data: median 5 words / 42
# characters, max 24 words), but the prompt asked an open question, so the secondary token-F1
# scored a median 1,700-character case discussion against it and reported the length ratio
# (measured 0.041) instead of diagnostic accuracy.
#
# The reasoning trace is kept — the LLM judge is the primary metric and reads the full reply, and
# MedCaseReasoning is explicitly about the reasoning path — so only a locatable final line is
# requested, using the ASCII "Answer:" marker ``extract_short_answer`` already recognizes.
_FINAL_ANSWER_INSTRUCTION = (
    "Reason as much as you need, then end your reply with a final line in exactly this form:\n"
    "Answer: <the single most likely final diagnosis, name only>"
)


class MedCaseReasoningOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedCaseReasoning"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.1"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedCaseReasoning provides only 'test'; requested '{self.split}'.")
        return [directory / "medcasereasoning_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            prompt = str(rec.get("case_prompt") or "").strip()
            dx = str(rec.get("final_diagnosis") or "").strip()
            if not prompt or not dx:
                continue
            yield {"record": rec, "prompt": prompt, "reference": dx,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        prompt = raw_sample["prompt"]
        reference = raw_sample["reference"]

        source_id = str(rec.get("pmcid") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"case": prompt})
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
            input_hash=self.input_hash({"case_prompt": prompt}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"case_prompt": prompt},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"pmcid": rec.get("pmcid"),
                      "diagnostic_reasoning": rec.get("diagnostic_reasoning")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        prompt = (
            f"{sample.source_content['case_prompt']}\n\n"
            "Based on the case above, what is the most likely final diagnosis?\n\n"
            f"{_FINAL_ANSWER_INSTRUCTION}"
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return extract_short_answer(raw_response)
