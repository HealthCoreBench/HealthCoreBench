"""ClinicalBench adapter (MIMIC clinical-prediction classification, English).

Fixed data under ``65_ClinicalBench/``: six ``<task>_<db>_test.json`` files, each a JSON list::

    {"ID": ..., "VISIT_ID": ..., "SUBJECT_ID": ..., "QUESTION": str (full patient info + question),
     "ANSWER": int/str (class label)}

Prediction tasks / label sets:
  - ``mortality_pred_mimic3`` / ``mortality_pred_mimic4``  : 0 / 1 (survives / dies)
  - ``readmission_pred_mimic3`` / ``readmission_pred_mimic4``: 0 / 1 (no readmit / readmit)
  - ``length_pred_mimic3`` / ``length_pred_mimic4``          : 1 / 2 / 3 (length-of-stay bucket)

Task: label classification. The ``QUESTION`` field already contains the full prompt; the model's
label is parsed and exact-matched against ``ANSWER`` (``classification`` evaluator).

The three MIMIC-IV prediction tasks are registered separately (``ClinicalBench/mortality``,
``/readmission``, ``/length_of_stay``) instead of being merged into one ``classification`` task:
they predict different targets over different label sets (binary vs ternary), so a single pooled
accuracy over all 1,498 records mixed two majority-class base rates (96.6% / 86.1%) with a
three-way problem and was not interpretable. The ``_mimic3`` counterparts remain reachable as
splits for a single-benchmark run. The ``*_ICL`` files are few-shot-prompt copies of the very
same records and are deliberately not registered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.answer_parsing import parse_label
from healthcorebench.schemas.sample import EvaluationSample

_BINARY = ["0", "1"]
_TERNARY = ["1", "2", "3"]
_SPLITS = {
    "mortality_pred_mimic3": _BINARY, "mortality_pred_mimic4": _BINARY,
    "readmission_pred_mimic3": _BINARY, "readmission_pred_mimic4": _BINARY,
    "length_pred_mimic3": _TERNARY, "length_pred_mimic4": _TERNARY,
}
# registry task -> MIMIC-IV file stem. The task wins over the split so that an ALL run (one
# global split) still scores all three prediction targets.
_TASKS = {
    "mortality": "mortality_pred_mimic4",
    "readmission": "readmission_pred_mimic4",
    "length_of_stay": "length_pred_mimic4",
}
_DEFAULT = "mortality_pred_mimic4"


class ClinicalBenchAdapter(BaseBenchmarkAdapter):
    benchmark_name = "ClinicalBench"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "classification"
    prompt_template_version = "1.1"

    def _split(self) -> str:
        task = (self.entry.task or "") if getattr(self, "entry", None) else ""
        if task in _TASKS:
            return _TASKS[task]
        s = _DEFAULT if self.split == "test" else self.split
        if s not in _SPLITS:
            raise BenchmarkSplitNotFoundError(f"ClinicalBench split must be 'test' or one of {sorted(_SPLITS)}; got '{self.split}'.")
        return s

    def _labels(self) -> list[str]:
        return _SPLITS[self._split()]

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / f"{self._split()}_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
            rel = self.rel_path(f)
            task = f.name.removesuffix("_test.json")
            labels = set(_SPLITS[task])
            with open(f, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            for i, rec in enumerate(records):
                q = str(rec.get("QUESTION") or "").strip()
                label = str(rec.get("ANSWER")).strip()
                if not q:
                    self.drop_source_record("empty_question")
                    continue
                if label not in labels:
                    self.drop_source_record("label_outside_class_set")
                    continue
                yield {"record": rec, "question": q, "label": label, "task": task,
                       "labels": sorted(labels), "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        question = raw_sample["question"]
        label = raw_sample["label"]
        split = raw_sample.get("task") or self._split()
        labels = raw_sample.get("labels") or self._labels()

        source_id = str(rec.get("ID") if rec.get("ID") is not None else f"{rel}:{rec_index}")
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
            reference_hash=self.reference_hash(label),
            input_type="text",
            task_type="classification",
            component="Language",
            capability="Reasoning",
            specialty=split,
            language="en",
            modality="Text",
            answer_format="label",
            evaluation_metric="accuracy",
            source_content={"question": question, "labels": labels},
            reference_answer=label,
            reference_answer_normalized=label,
            metadata={"labels": labels, "split": split,
                      "visit_id": rec.get("VISIT_ID"), "subject_id": rec.get("SUBJECT_ID")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        labels = c["labels"]
        prompt = (
            f"{c['question']}\n\nAllowed labels: {', '.join(labels)}. "
            "Return exactly one allowed label and no other text."
        )
        return [
            {"role": "system", "content": "You are a strict classifier. Output one label only, with no reasoning."},
            {"role": "user", "content": prompt},
        ]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_label(raw_response, sample.metadata.get("labels") or _BINARY)
