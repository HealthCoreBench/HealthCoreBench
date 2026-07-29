"""MIMIC-CDM adapter (abdominal-pain differential diagnosis, English).

Fixed data: ``45_MIMIC-CDM/mimic_cdm_test.json`` — a JSON list of records::

    {"hadm_id": ..., "pathology": str (one of appendicitis / cholecystitis / pancreatitis /
     diverticulitis), "Patient History": str, "Physical Examination": str,
     "Laboratory Tests": str, ... , "Discharge Diagnosis": str, "ICD Diagnosis": ...}

Task: four-way differential diagnosis of acute abdominal pain. Given the patient's history,
physical examination and laboratory tests, choose the diagnosis; the reference is ``pathology``.
Scored with the ``classification`` evaluator (exact label match). The ``.pkl`` files shipped
alongside are not required. (Companion patient-info pickles are ignored.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.answer_parsing import parse_label
from healthcorebench.schemas.sample import EvaluationSample

_LABELS = ["appendicitis", "cholecystitis", "pancreatitis", "diverticulitis"]


class MIMICCDMDiagnosisAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MIMIC-CDM"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "diagnosis"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MIMIC-CDM provides only 'test'; requested '{self.split}'.")
        return [directory / "mimic_cdm_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            label = str(rec.get("pathology") or "").strip().lower()
            if label not in _LABELS:
                continue
            hist = str(rec.get("Patient History") or "").strip()
            exam = str(rec.get("Physical Examination") or "").strip()
            labs = str(rec.get("Laboratory Tests") or "").strip()
            if not (hist or exam or labs):
                continue
            yield {"record": rec, "label": label, "history": hist, "exam": exam, "labs": labs,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        label = raw_sample["label"]

        source_id = str(rec.get("hadm_id") or f"{rel}:{rec_index}")
        content = {"history": raw_sample["history"], "exam": raw_sample["exam"], "labs": raw_sample["labs"]}
        content_hash = self.input_hash(content)
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
            input_hash=self.input_hash(content),
            reference_hash=self.reference_hash(label),
            input_type="text",
            task_type="classification",
            component="Language",
            capability="Reasoning",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="label",
            evaluation_metric="accuracy",
            source_content=content,
            reference_answer=label,
            reference_answer_normalized=label,
            metadata={"labels": _LABELS},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        parts = []
        if c.get("history"):
            parts.append(f"Patient History:\n{c['history']}")
        if c.get("exam"):
            parts.append(f"Physical Examination:\n{c['exam']}")
        if c.get("labs"):
            parts.append(f"Laboratory Tests:\n{c['labs']}")
        body = "\n\n".join(parts)
        prompt = (
            f"{body}\n\n"
            "Based on the information above, what is the most likely diagnosis? "
            f"Answer with exactly one of: {', '.join(_LABELS)}."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_label(raw_response, sample.metadata.get("labels") or _LABELS)
