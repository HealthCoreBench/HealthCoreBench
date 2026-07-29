"""MedR-Bench adapter — Oracle Diagnosis task (clinical reasoning, free-text, English).

Fixed data: ``27_MedR-Bench/data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json`` — a
dict of PMC cases; each case has a ``generate_case`` field (a JSON string) containing::

    {"case_summary": str (full patient case incl. ancillary tests),
     "differential_diagnosis": str, "final_diagnosis": str, "diagnosis_results": str}

Task: **Oracle Diagnosis** (per the official MedR-Bench task suite): the model is given the complete
``case_summary`` and must produce a diagnosis directly. This adapter reproduces the official
Oracle-Diagnosis prompt (``src/Inference/instructions/oracle_diagnose.txt``). The reference is
``final_diagnosis``; the official metric compares the predicted vs ground-truth diagnosis with a
GPT-4o judge (``outcome_accuracy_eval.eval_accuracy``, offline / no web search), which maps to this
framework's ``llm_judge``. The One-Turn / Free-Turn diagnosis and treatment tasks require an
interactive GPT-4o patient agent and/or web-search factuality checks and are not covered here.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

# Official Oracle-Diagnosis instruction (src/Inference/instructions/oracle_diagnose.txt).
_ORACLE_PROMPT = (
    "Please thoroughly examine the patient case summary presented below. Your objective is to "
    "perform a detailed diagnostic analysis utilizing all available information. Note that due to "
    "the potentially limited details, the preliminary diagnosis may encompass several possible "
    "conditions. Should you ascertain that the provided data is inadequate for a definitive "
    "conclusion, please enumerate any additional diagnostic tests or information that would be "
    "necessary. However, if you can deduce a conclusive diagnosis, please proceed to provide it. "
    "Too many requests for information are also inappropriate.\n\n"
    "Patient Case Summary:\n{case}\n\n"
    "Guidelines:\n"
    "Evaluate the patient's symptoms, medical history, and all pertinent details from the case summary.\n"
    "Formulate differential diagnoses based on your analysis.\n"
    "If the information is not sufficient for a conclusive diagnosis, specify the further tests or details required."
)


def _parse_generate_case(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(value)
                if isinstance(d, dict):
                    return d
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
    return None


class MedRBenchDiagnosisAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedR-Bench"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "oracle_diagnosis"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedR-Bench provides only 'test'; requested '{self.split}'.")
        return [directory / "data" / "MedRBench" / "diagnosis_957_cases_with_rare_disease_491.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for key, case in data.items():
            gc = _parse_generate_case(case.get("generate_case"))
            if not gc:
                continue
            case_summary = str(gc.get("case_summary") or "").strip()
            final_dx = str(gc.get("final_diagnosis") or gc.get("diagnosis_results") or "").strip()
            if not case_summary or not final_dx:
                continue
            yield {"case_key": key, "case": case, "generate_case": gc,
                   "case_summary": case_summary, "reference": final_dx, "source_file_rel": rel}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        key = raw_sample["case_key"]
        case = raw_sample["case"]
        gc = raw_sample["generate_case"]
        case_summary = raw_sample["case_summary"]
        reference = raw_sample["reference"]
        rel = raw_sample["source_file_rel"]

        body = case.get("body_category")
        disorder = case.get("disorder_category")

        content_hash = self.input_hash({"cs": case_summary})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=str(key), content_hash=content_hash)

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=str(key),
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=None,
            source_record_hash=self.input_hash(case_summary),
            input_hash=self.input_hash({"case_summary": case_summary}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=str(disorder) if disorder else None,
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"case_summary": case_summary},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"body_category": str(body) if body else None,
                      "disorder_category": str(disorder) if disorder else None,
                      "differential_diagnosis": gc.get("differential_diagnosis"),
                      "checked_rare_disease": case.get("checked_rare_disease"),
                      "task": "oracle_diagnosis"},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": _ORACLE_PROMPT.format(case=sample.source_content["case_summary"])}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
