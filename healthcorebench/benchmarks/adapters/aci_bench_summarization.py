"""ACI-Bench adapter (doctor-patient dialogue → clinical note, English).

Fixed data: ``29_ACI-Bench_HF/aci/test{1,2,3}.json`` — each a JSON list of records::

    {"encounter_id": str, "dialogue": str (doctor-patient conversation), "note": str (clinical note)}

Task: clinical-note generation / summarization. Given the visit dialogue, produce the clinical
note; the reference ``note`` is scored by an LLM judge. The three official test sets are exposed as
splits ``test1`` / ``test2`` / ``test3``; the default ``test`` maps to ``test1``. (The ``virtscribe``
subset is a separate collection and not covered here.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

_SPLITS = {"test1", "test2", "test3"}
_DEFAULT = "test1"


class ACIBenchSummarizationAdapter(BaseBenchmarkAdapter):
    benchmark_name = "ACI-Bench_HF"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "summarization"
    prompt_template_version = "1.0"

    def _split(self) -> str:
        s = _DEFAULT if self.split == "test" else self.split
        if s not in _SPLITS:
            raise BenchmarkSplitNotFoundError(f"ACI-Bench split must be 'test' or one of {sorted(_SPLITS)}; got '{self.split}'.")
        return s

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split == "test":
            return [directory / group / f"test{i}.json"
                    for group in ("aci", "virtassist", "virtscribe") for i in (1, 2, 3)]
        return [directory / "aci" / f"{self._split()}.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
          rel = self.rel_path(f)
          with open(f, "r", encoding="utf-8") as fh: records = json.load(fh)
          for i, rec in enumerate(records):
            dialogue = str(rec.get("dialogue") or "").strip()
            note = str(rec.get("note") or "").strip()
            if not dialogue or not note:
                continue
            yield {"record": rec, "dialogue": dialogue, "reference": note,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        dialogue = raw_sample["dialogue"]
        reference = raw_sample["reference"]

        source_id = str(rec.get("encounter_id") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"d": dialogue})
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
            input_hash=self.input_hash({"dialogue": dialogue}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="summarization",
            component="Language",
            capability="Reasoning",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="summary",
            # clinical-note summarization — ROUGE-1/2/L standard metric (BLEU as secondary).
            evaluation_metric="rouge",
            source_content={"dialogue": dialogue},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"encounter_id": rec.get("encounter_id"), "split": self._split()},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        dialogue = sample.source_content["dialogue"]
        prompt = (
            "Below is a doctor-patient conversation. Write the clinical note documenting this "
            f"encounter.\n\n{dialogue}"
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
