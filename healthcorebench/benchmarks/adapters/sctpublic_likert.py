"""SCTPublic adapter (Script Concordance Test, Likert partial-credit, English).

Fixed data: ``32_SCTPublic/sct_test.json`` — a JSON list of SCT items::

    {"question_id": int, "sct_stem": str (clinical scenario),
     "question": str ("If you were thinking of: <hypothesis>"),
     "additional_info": str ("And then you find: <new finding>"),
     "-2": float, "-1": float, "0": float, "1": float, "2": float  (expert credit per Likert option),
     "source": str, ...}

Task: Script Concordance Testing. Given the scenario, a diagnostic/management hypothesis, and a new
finding, the model rates how the finding changes the hypothesis on a -2..+2 Likert scale
(-2 = much less likely ... +2 = much more likely). There is no single correct answer: each option
carries an expert-panel credit, and the score is the (max-normalized) credit of the chosen option.
Scored with the ``likert_credit`` evaluator; the credit vector is passed in metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

_LEVELS = ("-2", "-1", "0", "1", "2")
# parse a Likert rating from model output: prefer an explicit "Rating: +1" marker.
_RATING_RE = re.compile(r"rating\s*[:=]?\s*(\+?-?[012])", re.IGNORECASE)
_BARE_RE = re.compile(r"(?<![\d.])([+-]?[012])(?![\d.])")


def parse_likert(text: str) -> int | None:
    if not text:
        return None
    m = _RATING_RE.search(text)
    if m:
        try:
            v = int(m.group(1).replace("+", ""))
            if v in (-2, -1, 0, 1, 2):
                return v
        except ValueError:
            pass
    # fall back to a single isolated -2..+2 token
    found = {int(x.replace("+", "")) for x in _BARE_RE.findall(text.strip())}
    found = {v for v in found if v in (-2, -1, 0, 1, 2)}
    return next(iter(found)) if len(found) == 1 else None


class SCTPublicLikertAdapter(BaseBenchmarkAdapter):
    benchmark_name = "SCTPublic"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "script_concordance"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"SCTPublic provides only 'test'; requested '{self.split}'.")
        return [directory / "sct_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            stem = str(rec.get("sct_stem") or "").strip()
            question = str(rec.get("question") or "").strip()
            credit = {k: rec.get(k) for k in _LEVELS}
            # need a stem, a hypothesis, and a numeric credit vector with at least one positive.
            if not stem or not question:
                continue
            try:
                vals = [float(credit[k] or 0.0) for k in _LEVELS]
            except (TypeError, ValueError):
                continue
            if max(vals) <= 0:
                continue
            yield {"record": rec, "credit": {k: float(credit[k] or 0.0) for k in _LEVELS},
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        credit = raw_sample["credit"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        stem = str(rec.get("sct_stem") or "").strip()
        question = str(rec.get("question") or "").strip()
        additional = str(rec.get("additional_info") or "").strip()
        # reference = the modal expert option (max credit), for reporting; scoring uses full credit.
        modal = max(_LEVELS, key=lambda k: credit.get(k, 0.0))

        source_id = str(rec.get("question_id") if rec.get("question_id") is not None else f"{rel}:{rec_index}")
        content_hash = self.input_hash({"s": stem, "q": question, "a": additional})
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
            input_hash=self.input_hash({"stem": stem, "question": question, "additional": additional}),
            reference_hash=self.reference_hash(modal),
            input_type="text",
            task_type="script_concordance",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("source"),
            language="en",
            modality="Text",
            answer_format="likert",
            evaluation_metric="likert_credit",
            source_content={"stem": stem, "question": question, "additional": additional},
            reference_answer=modal,
            reference_answer_normalized=modal,
            metadata={"credit": credit, "source": rec.get("source")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        prompt = (
            f"Scenario: {c['stem']}\n\n"
            f"{c['question']}\n"
            f"{c['additional']}\n\n"
            "On the following scale, how does this new finding affect the hypothesis?\n"
            "  -2 = much less likely / much less indicated\n"
            "  -1 = less likely / less indicated\n"
            "   0 = no effect\n"
            "  +1 = more likely / more indicated\n"
            "  +2 = much more likely / much more indicated\n\n"
            "Respond with exactly: 'Rating: <value>' where <value> is one of -2, -1, 0, +1, +2."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_likert(raw_response or "")
