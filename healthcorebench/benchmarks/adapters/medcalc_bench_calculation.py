"""MedCalc-Bench adapter (medical value calculation, English).

Fixed data: ``59_MedCalc-Bench/datasets/medcalc_test.json`` — a JSON list of records::

    {"Row Number": int, "Calculator ID": ..., "Calculator Name": str, "Category": str,
     "Output Type": "decimal"|"integer"|"date", "Patient Note": str, "Question": str,
     "Ground Truth Answer": str, "Lower Limit": str, "Upper Limit": str,
     "Ground Truth Explanation": str}

Task: compute a clinical value from a patient note. The reference is a number with an accepted
``[Lower Limit, Upper Limit]`` range (date-typed answers compare by exact string). Scored with the
``numeric_tolerance`` evaluator, which reads the limits and output type from sample metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


class MedCalcBenchAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedCalc-Bench"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "calculation"
    prompt_template_version = "1.1"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedCalc-Bench provides only 'test'; requested '{self.split}'.")
        return [directory / "datasets" / "medcalc_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            note = str(rec.get("Patient Note") or "").strip()
            question = str(rec.get("Question") or "").strip()
            gt = str(rec.get("Ground Truth Answer") or "").strip()
            if not note or not question or not gt:
                continue
            yield {"record": rec, "note": note, "question": question, "reference": gt,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        note = raw_sample["note"]
        question = raw_sample["question"]
        reference = raw_sample["reference"]
        output_type = rec.get("Output Type")

        source_id = str(rec.get("Row Number") if rec.get("Row Number") is not None else f"{rel}:{rec_index}")
        content_hash = self.input_hash({"note": note, "q": question})
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
            input_hash=self.input_hash({"note": note, "question": question}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="calculation",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("Category"),
            language="en",
            modality="Text",
            answer_format="numeric",
            evaluation_metric="numeric_tolerance",
            source_content={"note": note, "question": question,
                            "output_type": output_type},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"lower_limit": rec.get("Lower Limit"), "upper_limit": rec.get("Upper Limit"),
                      "output_type": output_type, "calculator": rec.get("Calculator Name"),
                      "category": rec.get("Category")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        ot = c.get("output_type")
        if ot == "date":
            if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", str(sample.reference_answer or "")):
                fmt = "Give the final answer as a date (MM/DD/YYYY)."
            else:
                fmt = "Give the final gestational age as a '(N weeks, N days)' tuple."
        elif ot == "integer":
            fmt = "Give the final answer as a single integer."
        else:
            fmt = "Give the final answer as a single number."
        prompt = (
            f"Patient note:\n{c['note']}\n\n"
            f"Question: {c['question']}\n\n"
            f"{fmt} End your response with 'Answer: <value>'."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        text = (raw_response or "").strip()
        if not text:
            return None
        output_type = (sample.metadata or {}).get("output_type")
        token_pattern = _DATE_TOKEN if output_type == "date" else _NUMBER_TOKEN

        # The last explicit final-answer marker wins. Restrict extraction to the marker's own
        # line so numbers from a trailing explanation cannot leak into scoring.
        marker_matches = list(re.finditer(
            r"(?:final\s+answer|answer)\s*[:=]\s*([^\r\n]+)", text, re.IGNORECASE
        ))
        if marker_matches:
            return _extract_single_token(marker_matches[-1].group(1), token_pattern)

        # Accept a clearly signposted concluding clause or a final line containing only the
        # requested value. Without either signal, multiple numbers are intentionally ambiguous.
        conclusion = re.search(
            r"(?:therefore|thus|hence|so|the\s+(?:final\s+)?(?:value|result))"
            r"[^\r\n]*?\b(?:is|equals?)\s*(" + token_pattern + r")\s*[.%]?\s*$",
            text,
            re.IGNORECASE,
        )
        if conclusion:
            return conclusion.group(1)
        last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
        if re.fullmatch(r"(?:[A-Za-z ]+\s*[:=]\s*)?(?:" + token_pattern + r")\s*%?\.?", last_line):
            return _extract_single_token(last_line, token_pattern)

        tokens = re.findall(token_pattern, text)
        return tokens[0] if len(tokens) == 1 else None


_NUMBER_TOKEN = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?"
_DATE_TOKEN = r"\d{1,2}/\d{1,2}/\d{4}|\(\s*['\"]?\d+\s+weeks?['\"]?\s*,\s*['\"]?\d+\s+days?['\"]?\s*\)"


def _extract_single_token(text: str, pattern: str) -> str | None:
    matches = list(re.finditer(pattern, text))
    # Even an explicit marker is unsafe when its own answer span contains several values
    # (for example, ``Answer: 20 or 30``). Never silently select the first number.
    return matches[0].group(0).strip() if len(matches) == 1 else None
