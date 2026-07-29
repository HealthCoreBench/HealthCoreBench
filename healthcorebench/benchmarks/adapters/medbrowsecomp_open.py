"""MedBrowseComp adapter (medical fact-finding QA, English).

Fixed data: ``52_MedBrowseComp/MedBrowseComp_{605,50,CUA}.json`` — each a JSON list::

    {"gold": str (the exact answer, e.g. a drug ingredient / trial fact), "prompt": str,
     "task_name": str}

Task: fact-finding QA. MedBrowseComp is designed as a *web-browsing* benchmark; this adapter runs
it **closed-book** — the model answers ``prompt`` from its own knowledge and the response is scored
against ``gold`` with EM + token-F1. This measures parametric knowledge / reasoning rather than live
retrieval, which is the appropriate mode for an offline evaluation framework. The default ``test``
split is only ``605``; ``50`` and ``cua`` are explicit alternatives, and ``all`` is the only mode
that concatenates all three. Records with an empty gold/prompt are skipped.

Golds are 5-56 characters (median 12), so the scored span must be the final answer rather than the
whole response: asked the bare prompt the model returns 478-1,368 characters and token-F1 collapses
to the length ratio (measured 0.0024 over 10 rows). A brevity instruction is appended and
``parse_response`` extracts the final short answer, including the value behind a prescribed label
such as ``INGREDIENT:`` / ``COMPANY:`` / ``Start date:``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.adapters.hle_med_exact import extract_short_answer
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

_SPLITS = {"605": "MedBrowseComp_605.json", "50": "MedBrowseComp_50.json", "cua": "MedBrowseComp_CUA.json"}
_FILE_SPLITS = {file_name: split for split, file_name in _SPLITS.items()}
_DEFAULT = "605"
_BREVITY_INSTRUCTION = (
    "Answer with the requested value only, in the format the question asks for, on a single line. "
    "Do not show your reasoning."
)


class MedBrowseCompOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedBrowseComp"
    benchmark_version = "1.0"
    adapter_version = "1.2"
    prompt_template_name = "short_answer"
    prompt_template_version = "1.1"

    def _split(self) -> str:
        s = _DEFAULT if self.split == "test" else self.split
        if s not in {*_SPLITS, "all"}:
            raise BenchmarkSplitNotFoundError(
                "MedBrowseComp split must be 'test'/'605'/'50'/'cua'/'all'; "
                f"got '{self.split}'."
            )
        return s

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self._split() == "all":
            return [directory / name for name in _SPLITS.values()]
        return [directory / _SPLITS[self._split()]]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
          rel = self.rel_path(f)
          with open(f, "r", encoding="utf-8") as fh: records = json.load(fh)
          for i, rec in enumerate(records):
            prompt = str(rec.get("prompt") or "").strip()
            gold = str(rec.get("gold") or "").strip()
            if not prompt or not gold:
                continue
            yield {"record": rec, "prompt": prompt, "reference": gold,
                   "source_file_rel": rel, "source_record_index": i,
                   "resolved_split": _FILE_SPLITS[f.name]}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        prompt = raw_sample["prompt"]
        reference = raw_sample["reference"]

        source_id = f"{rel}:{rec_index}"
        content_hash = self.input_hash({"p": prompt})
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
            input_hash=self.input_hash({"prompt": prompt}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Knowledge",
            specialty=rec.get("task_name"),
            language="en",
            modality="Text",
            answer_format="short_answer",
            # short canonical answer (drug/entity) — rule-based EM + token-F1, no LLM judge.
            evaluation_metric="text_f1",
            source_content={"prompt": prompt},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"task_name": rec.get("task_name"),
                      "split": raw_sample["resolved_split"],
                      "requested_split": self.split},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        prompt = sample.source_content["prompt"]
        return [{"role": "user", "content": f"{prompt}\n\n{_BREVITY_INSTRUCTION}"}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return extract_short_answer(raw_response)
