"""BioASQ factoid adapter (biomedical factoid QA, free-text, English).

Fixed data: ``43_BioASQ/*/*.json`` — each ``{"questions": [...]}``; only ``type == "factoid"``
questions are used here. A factoid question has::

    {"body": str, "type": "factoid", "exact_answer": [[str, ...], ...], "ideal_answer": [str, ...]}

``exact_answer`` is a list of accepted-answer groups, each group a list of synonymous surface
forms; ``ideal_answer`` is a longer natural-language answer. This adapter uses the first exact
answer as the primary reference (with all accepted synonyms carried in metadata) and scores with
an LLM judge. Questions are de-duplicated by ``id`` across the overlapping golden files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


def _flatten_exact(exact: Any) -> list[str]:
    """Flatten BioASQ exact_answer ([[syn, ...], ...] or [str, ...]) into a flat synonym list."""
    out: list[str] = []
    if isinstance(exact, list):
        for grp in exact:
            if isinstance(grp, list):
                out.extend(str(x).strip() for x in grp if str(x).strip())
            elif str(grp).strip():
                out.append(str(grp).strip())
    elif str(exact or "").strip():
        out.append(str(exact).strip())
    return out


class BioASQFactoidOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "BioASQ"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"BioASQ (factoid) provides only 'test'; requested '{self.split}'.")
        return sorted(p for p in directory.rglob("*.json") if p.is_file())

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        seen: set[str] = set()
        for f in files:
            rel = self.rel_path(f)
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict) or "questions" not in data:
                continue
            for i, q in enumerate(data.get("questions", [])):
                if q.get("type") != "factoid":
                    continue
                synonyms = _flatten_exact(q.get("exact_answer"))
                body = str(q.get("body") or "").strip()
                if not synonyms or not body:
                    continue
                qid = str(q.get("id") or f"{rel}:{i}")
                if qid in seen:
                    continue
                seen.add(qid)
                yield {"question": q, "synonyms": synonyms, "qid": qid,
                       "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        q = raw_sample["question"]
        synonyms: list[str] = raw_sample["synonyms"]
        qid: str = raw_sample["qid"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        body = str(q["body"]).strip()
        reference = synonyms[0]
        ideal = q.get("ideal_answer")
        ideal_text = " ".join(ideal) if isinstance(ideal, list) else (str(ideal) if ideal else None)

        content_hash = self.input_hash({"q": body})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=qid, content_hash=content_hash)

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=qid,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=rec_index,
            source_record_hash=self.input_hash(q),
            input_hash=self.input_hash({"body": body}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Knowledge",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="short_answer",
            # factoid: short entity answer — rule-based EM + token-F1 (BioASQ-style), no judge.
            evaluation_metric="text_f1",
            source_content={"body": body},
            reference_answer=reference,
            reference_answer_normalized=reference,
            reference_aliases=synonyms,
            metadata={"accepted_answers": synonyms, "ideal_answer": ideal_text},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        body = sample.source_content["body"]
        return [{"role": "user", "content": f"{body}\nAnswer with the specific entity or fact requested."}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
