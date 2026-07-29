"""EHRBench risk-prediction adapter (EHR-conditioned yes/no prediction, English).

Fixed data: ``47_EHRBench/ehr_bench_risk_prediction.jsonl`` — one JSON object per line::

    {"idx": int, "instruction": str, "input": str (patient EHR context), "output": "yes"|"no",
     "candidates": ["yes", "no"], "task_info": {"task": str, "metric": str, ...}}

Task: binary (yes/no) risk prediction conditioned on a patient's EHR. Every record here is a
two-way choice (``candidates == ["yes","no"]``); scored with the ``classification`` evaluator.
The companion ``ehr_bench_decision_making.jsonl`` has large per-record candidate sets and is not
a clean MCQA task, so it is served as open generation by ``ehrbench_decision_open`` instead of
here. (An earlier note put "roughly half" of its answers outside their candidate list; the actual
figure is 889 of 13,500, all traced to eight incomplete candidate vocabularies.)

71% of records (5,492/7,721) do not fit a 16k window, so which part of the chart is retained is a
scoring decision. EHRBench declares no answer location, so it is made by explicit policy —
``fit_ehr_context`` keeps the recent tail and records that choice on every sample — rather than by
falling through to the answer-blind 50/50 head_tail slice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.answer_parsing import parse_yes_no_maybe
from healthcorebench.benchmarks.adapters.ehrbench_decision_open import fit_ehr_context
from healthcorebench.schemas.sample import EvaluationSample

_ANSWER_INSTRUCTION = "Answer with exactly one word: yes or no."


class EHRBenchRiskYesNoAdapter(BaseBenchmarkAdapter):
    benchmark_name = "EHRBench"
    benchmark_version = "1.0"
    adapter_version = "1.2"
    prompt_template_name = "yes_no"
    prompt_template_version = "1.1"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"EHRBench risk-prediction provides only 'test'; requested '{self.split}'.")
        return [directory / "ehr_bench_risk_prediction.jsonl"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                label = str(rec.get("output") or "").strip().lower()
                cands = [str(c).strip().lower() for c in (rec.get("candidates") or [])]
                if label not in ("yes", "no"):
                    self.drop_source_record("label_not_yes_no")
                    continue
                if set(cands) != {"yes", "no"}:
                    # This file is meant to be the two-way slice of EHRBench; a record with any
                    # other option set belongs to the decision-making task, not here.
                    self.drop_source_record("candidates_not_yes_no")
                    continue
                yield {"record": rec, "label": label, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        label = raw_sample["label"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        instruction = str(rec.get("instruction") or "").strip()
        context = str(rec.get("input") or "").strip()
        context, context_meta = fit_ehr_context(
            context,
            fixed_prompt=f"{instruction}\n\n\n\n{_ANSWER_INSTRUCTION}",
            config=self.config,
            max_output_tokens=getattr(getattr(self.config, "generation", None), "max_tokens", None),
        )
        task_info = rec.get("task_info") or {}

        source_id = str(rec.get("idx") if rec.get("idx") is not None else f"{rel}:{rec_index}")
        content_hash = self.input_hash({"instr": instruction, "ctx": context})
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
            input_hash=self.input_hash({"instruction": instruction, "context": context}),
            reference_hash=self.reference_hash(label),
            input_type="text",
            task_type="classification",
            component="Language",
            capability="Reasoning",
            specialty=task_info.get("task"),
            language="en",
            modality="Text",
            answer_format="yes_no",
            evaluation_metric="accuracy",
            source_content={"instruction": instruction, "context": context},
            reference_answer=label,
            reference_answer_normalized=label,
            metadata={"labels": ["yes", "no"], "task": task_info.get("task"),
                      "task_type": task_info.get("task_type"), **context_meta},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        prompt = f"{c['instruction']}\n\n{c['context']}\n\n{_ANSWER_INSTRUCTION}"
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        v = parse_yes_no_maybe(raw_response)
        return v if v in ("yes", "no") else None
