"""MedSafetyBench adapter (medical safety / harmful-request refusal, English).

Fixed data: ``41_MedSafetyBench/medsafetybench_{gpt4,llama2}_test.json`` — each a JSON list::

    {"harmful_medical_request": str, "safe_response": str, "category": str, "source": str}

Task: safety. The model is given a harmful medical request; a safe model should refuse and/or
redirect. The reference ``safe_response`` demonstrates the desired safe behavior and is used by an
LLM judge to score whether the model's response is appropriately safe. The two source files
(harmful requests elicited from GPT-4 vs Llama-2) are exposed as splits ``gpt4`` / ``llama2``;
the default ``test`` maps to ``gpt4``.

NOTE: this is a *defensive* evaluation — it measures whether a model refuses harmful medical
requests. The prompts are the benchmark's fixed adversarial inputs; the reference is a safe refusal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

_SPLITS = {"gpt4": "medsafetybench_gpt4_test.json", "llama2": "medsafetybench_llama2_test.json"}
_DEFAULT = "gpt4"


class MedSafetyBenchAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedSafetyBench"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "safety"
    prompt_template_version = "1.0"

    def _split(self) -> str:
        s = _DEFAULT if self.split == "test" else self.split
        if s not in _SPLITS:
            raise BenchmarkSplitNotFoundError(f"MedSafetyBench split must be 'test'/'gpt4'/'llama2'; got '{self.split}'.")
        return s

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split == "test":
            return [directory / _SPLITS[name] for name in sorted(_SPLITS)]
        return [directory / _SPLITS[self._split()]]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
            rel = self.rel_path(f)
            with open(f, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            for i, rec in enumerate(records):
                request = str(rec.get("harmful_medical_request") or "").strip()
                safe = str(rec.get("safe_response") or "").strip()
                if request and safe:
                    yield {"record": rec, "request": request, "reference": safe,
                           "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        request = raw_sample["request"]
        reference = raw_sample["reference"]

        source_id = f"{rel}:{rec_index}"
        content_hash = self.input_hash({"req": request})
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
            input_hash=self.input_hash({"request": request}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Safety",
            specialty=rec.get("category"),
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"request": request},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"category": rec.get("category"), "source": rec.get("source"),
                      "split": self._split(), "judge_kind": "safety"},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": sample.source_content["request"]}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
