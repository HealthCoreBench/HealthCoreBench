"""MedNLI adapter (clinical natural-language inference, English).

Fixed data: ``12_MedNLI/mli_test_v1.jsonl`` — one JSON object per line::

    {"pairID": str, "sentence1": str (premise), "sentence2": str (hypothesis),
     "gold_label": "entailment" | "contradiction" | "neutral", ...(parse trees ignored)}

Task: three-way classification. Given a premise and a hypothesis, decide whether the hypothesis
is entailed by, contradicts, or is neutral with respect to the premise. Scored with the
``classification`` evaluator (exact label match).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.answer_parsing import parse_label
from healthcorebench.schemas.sample import EvaluationSample

_LABELS = ["entailment", "contradiction", "neutral"]
_ALIASES = {
    "entails": "entailment", "entail": "entailment", "entailed": "entailment",
    "contradicts": "contradiction", "contradict": "contradiction", "contradictory": "contradiction",
    "neutral.": "neutral", "unrelated": "neutral",
}


class MedNLIAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedNLI"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "nli"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedNLI provides only 'test'; requested '{self.split}'.")
        return [directory / "mli_test_v1.jsonl"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if (rec.get("gold_label") or "").strip().lower() not in _LABELS:
                    continue  # skip the rare "-" (no gold consensus) rows
                yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        premise = str(rec["sentence1"]).strip()
        hypothesis = str(rec["sentence2"]).strip()
        label = str(rec["gold_label"]).strip().lower()

        source_id = str(rec.get("pairID") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"p": premise, "h": hypothesis})
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
            input_hash=self.input_hash({"premise": premise, "hypothesis": hypothesis}),
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
            source_content={"premise": premise, "hypothesis": hypothesis},
            reference_answer=label,
            reference_answer_normalized=label,
            metadata={"labels": _LABELS},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        prompt = (
            f"Premise: {c['premise']}\n"
            f"Hypothesis: {c['hypothesis']}\n\n"
            "Does the premise entail the hypothesis, contradict it, or is it neutral? "
            "Answer with exactly one word: entailment, contradiction, or neutral."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_label(raw_response, sample.metadata.get("labels") or _LABELS, _ALIASES)
