"""MEDEC adapter (medical error detection, English).

Fixed data: ``28_MEDEC/medec_test.json`` — a JSON list of clinical texts::

    {"Text ID": str, "Text": str (clinical note, sentences numbered), "Sentences": ...,
     "Error Flag": 0 | 1 | None, "Error Type": str, "Error Sentence ID": int,
     "Error Sentence": str, "Corrected Sentence": str, "Corrected Text": str}

Task: binary error detection — does the clinical note contain a medical error? ``Error Flag`` is 1
(contains an error) or 0 (no error); it is mapped to a yes/no reference and scored with the
``classification`` evaluator. Rows with ``Error Flag`` = None (a handful of empty/placeholder texts)
are skipped. This adapter covers detection only; error localization and correction are separate
tasks not implemented here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.answer_parsing import parse_yes_no_maybe
from healthcorebench.schemas.sample import EvaluationSample


class MEDECDetectionAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MEDEC"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "error_detection"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MEDEC provides only 'test'; requested '{self.split}'.")
        return [directory / "medec_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            flag = rec.get("Error Flag")
            text = str(rec.get("Text") or "").strip()
            if flag not in (0, 1) or not text:
                continue
            label = "yes" if flag == 1 else "no"
            yield {"record": rec, "text": text, "label": label,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        text = raw_sample["text"]
        label = raw_sample["label"]

        source_id = str(rec.get("Text ID") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"t": text})
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
            input_hash=self.input_hash({"text": text}),
            reference_hash=self.reference_hash(label),
            input_type="text",
            task_type="classification",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("Error Type") if label == "yes" else None,
            language="en",
            modality="Text",
            answer_format="yes_no",
            evaluation_metric="accuracy",
            source_content={"text": text},
            reference_answer=label,
            reference_answer_normalized=label,
            metadata={"labels": ["yes", "no"], "error_type": rec.get("Error Type")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        text = sample.source_content["text"]
        prompt = (
            "The following clinical note may or may not contain a medical error "
            "(e.g. an incorrect diagnosis, management, treatment, or causal statement).\n\n"
            f"{text}\n\n"
            "Does this note contain a medical error? Answer with exactly one word: yes or no."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        v = parse_yes_no_maybe(raw_response)
        return v if v in ("yes", "no") else None
