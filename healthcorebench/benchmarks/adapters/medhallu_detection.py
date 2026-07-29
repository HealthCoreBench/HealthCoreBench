"""MedHallu adapter (medical hallucination detection, English).

Fixed data: ``23_MedHallu/medhallu_pqa_labeled_test.json`` — a JSON list of records::

    {"Question": str, "Knowledge": [str, ...] (supporting passages), "Ground Truth": str,
     "Difficulty Level": str, "Hallucinated Answer": str, "Category of Hallucination": str}

Task: binary hallucination detection. Each source record yields **two** evaluation samples — the
faithful ``Ground Truth`` answer (label "no" = not a hallucination) and the ``Hallucinated Answer``
(label "yes" = hallucination). Given the question, supporting knowledge, and a candidate answer,
the model judges whether the candidate is a hallucination. Scored with the ``classification``
evaluator. (The larger ``medhallu_pqa_artificial_test.json`` variant is not used by default.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.answer_parsing import parse_yes_no_maybe
from healthcorebench.schemas.sample import EvaluationSample


class MedHalluDetectionAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedHallu"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "hallucination_detection"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedHallu provides only 'test'; requested '{self.split}'.")
        return [directory / "medhallu_pqa_labeled_test.json",
                directory / "medhallu_pqa_artificial_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
          rel = self.rel_path(f)
          with open(f, "r", encoding="utf-8") as fh: records = json.load(fh)
          for i, rec in enumerate(records):
            question = str(rec.get("Question") or "").strip()
            gt = str(rec.get("Ground Truth") or "").strip()
            hal = str(rec.get("Hallucinated Answer") or "").strip()
            if not question or not gt or not hal:
                continue
            # two samples per record: the faithful answer and the hallucinated one.
            yield {"record": rec, "candidate": gt, "label": "no", "variant": "truth",
                   "source_file_rel": rel, "source_record_index": i}
            yield {"record": rec, "candidate": hal, "label": "yes", "variant": "hallucination",
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        candidate = raw_sample["candidate"]
        label = raw_sample["label"]
        variant = raw_sample["variant"]

        question = str(rec["Question"]).strip()
        knowledge = rec.get("Knowledge") or []
        knowledge_text = "\n".join(str(k) for k in knowledge) if isinstance(knowledge, list) else str(knowledge)

        source_id = f"{rel}:{rec_index}:{variant}"
        content_hash = self.input_hash({"q": question, "cand": candidate})
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
            input_hash=self.input_hash({"question": question, "knowledge": knowledge_text, "candidate": candidate}),
            reference_hash=self.reference_hash(label),
            input_type="text",
            task_type="classification",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("Category of Hallucination") if variant == "hallucination" else None,
            difficulty=rec.get("Difficulty Level"),
            language="en",
            modality="Text",
            answer_format="yes_no",
            evaluation_metric="accuracy",
            source_content={"question": question, "knowledge": knowledge_text, "candidate": candidate},
            reference_answer=label,
            reference_answer_normalized=label,
            metadata={"labels": ["yes", "no"], "variant": variant,
                      "difficulty": rec.get("Difficulty Level"),
                      "category": rec.get("Category of Hallucination")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        prompt = (
            f"Question: {c['question']}\n\n"
            f"Supporting knowledge:\n{c['knowledge']}\n\n"
            f"Candidate answer: {c['candidate']}\n\n"
            "Is the candidate answer a hallucination (i.e. contains information not supported by, "
            "or contradicting, the knowledge)? Answer with exactly one word: yes or no."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        v = parse_yes_no_maybe(raw_response)
        return v if v in ("yes", "no") else None
