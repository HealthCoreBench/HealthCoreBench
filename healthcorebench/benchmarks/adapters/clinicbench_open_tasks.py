"""ClinicBench patient education, treatment, hospitalization, and interaction tasks.

``Treatment-Recommendation.json`` stores ``TestsAndProcedures`` / ``commonMedications`` as
Python-repr *strings* of lists. They are recovered with ``ast.literal_eval`` into real lists and
scored as a set-overlap task (``multilabel``): the gold is an enumerated controlled vocabulary of
up to a few dozen tests/procedures and medications, and binary judge equivalence against such a
list has a ceiling near zero (measured 0.000 on 10/10 rows before this change).
"""
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.answer_parsing import final_answer_region, parse_yes_no_maybe
from healthcorebench.benchmarks.prompts import judgement_prompt
from healthcorebench.schemas.sample import EvaluationSample


def _as_list(value: Any) -> list[str]:
    """Recover a real list from ClinicBench's Python-repr list strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def _list_items(text: str) -> list[str]:
    """Split a list-shaped answer into items: one per line, or comma separated within a line.

    Models annotate list items ("**Visual Acuity Test**: To determine ...") even when asked not
    to, so the trailing rationale after a colon and any Markdown emphasis/heading decoration are
    dropped; what remains is the item name that can be compared with the reference vocabulary.
    """
    items: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*#+\s*", "", line)
        line = re.sub(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+", "", line)
        line = line.replace("**", "").replace("__", "").strip()
        if not line or line.endswith(":"):
            # A line that is only a section heading ("Medications:") is structure, not an item.
            continue
        # "Name: explanation" — keep the name.
        head = line.split(":", 1)[0].strip(" .;")
        if not head:
            continue
        parts = [part for part in head.split(",")] if "," in head else [head]
        for part in parts:
            part = part.strip(" .;")
            if part and part not in items:
                items.append(part)
    return items


class _ClinicOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "ClinicBench"
    benchmark_version = "1.0"
    filename = ""
    capability = "Reasoning"
    metric = "llm_judge"
    prompt_template_version = "1.1"

    def discover_source_files(self) -> list[Path]:
        return [self.get_benchmark_directory() / self.filename]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        for i, rec in enumerate(json.loads(f.read_text(encoding="utf-8"))):
            question, reference = self.fields(rec)
            if question and reference:
                yield {"record": rec, "question": question, "reference": reference,
                       "source_file_rel": rel, "source_record_index": i}

    def fields(self, rec: dict) -> tuple[str, Any]:
        raise NotImplementedError

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        question = raw_sample["question"]
        reference = raw_sample["reference"]
        source_id = str(rec.get("key") or rec.get("idx") or rec.get("hadm_id") or
                        f"{rel}:{raw_sample['source_record_index']}")
        return EvaluationSample(
            sample_id=self.make_sample_id(source_file_rel=rel, source_sample_id=source_id,
                                          content_hash=self.input_hash(question)),
            source_sample_id=source_id, sample_index=sample_index,
            benchmark_name=self.benchmark_name, benchmark_version=self.benchmark_version,
            benchmark_split=self.split, source_file=rel,
            source_record_index=raw_sample["source_record_index"], source_record_hash=self.input_hash(rec),
            input_hash=self.input_hash(question), reference_hash=self.reference_hash(reference),
            task_type="open_ended", component="Language", capability=self.capability,
            language="en", modality="Text", answer_format="free_text", evaluation_metric=self.metric,
            source_content={"question": question}, reference_answer=reference,
            reference_answer_normalized=reference,
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": sample.source_content["question"]}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()


class ClinicBenchPatientEducationAdapter(_ClinicOpenAdapter):
    filename = "Patient-Education.json"

    def fields(self, rec: dict) -> tuple[str, str]:
        return (f"Write patient-friendly discharge instructions from this discharge summary:\n{rec.get('discharge_summary', '')}",
                str(rec.get("discharge_instruction") or ""))


class ClinicBenchTreatmentAdapter(_ClinicOpenAdapter):
    filename = "Treatment-Recommendation.json"
    # The gold is an enumerated controlled vocabulary of tests/procedures and medications, so the
    # answer is scored by set overlap. Binary judge equivalence against such a list scored 0.000
    # on every sampled row with rationales that only ever cited omissions.
    metric = "multilabel"

    def fields(self, rec: dict) -> tuple[str, list[str]]:
        tests = _as_list(rec.get("TestsAndProcedures"))
        medications = _as_list(rec.get("commonMedications"))
        question = (
            f"Disease: {rec.get('disease')}\n"
            f"Symptoms: {', '.join(_as_list(rec.get('Symptom'))) or rec.get('Symptom')}\n"
            f"Clinical background: {rec.get('reason')}\n\n"
            "Recommend the tests, procedures, and medications for this patient. Your answer is "
            f"scored against an enumerated reference list of {len(tests)} accepted "
            f"tests/procedures and {len(medications)} accepted medications, by overlap with that "
            "list. Return only the recommended items, one per line, under the headings "
            "'Tests and procedures:' and 'Medications:'. Name each item on its own line, use the "
            "usual clinical or generic name, and add no explanation."
        )
        return question, tests + medications

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        sample = super().normalize_sample(raw_sample, sample_index)
        rec = raw_sample["record"]
        tests = _as_list(rec.get("TestsAndProcedures"))
        medications = _as_list(rec.get("commonMedications"))
        sample.task_type = "multilabel_classification"
        sample.answer_format = "multi_label"
        sample.reference_aliases = raw_sample["reference"]
        sample.metadata = {
            "tests_and_procedures": tests,
            "medications": medications,
            "reference_item_count": len(raw_sample["reference"]),
        }
        return sample

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        text = final_answer_region(raw_response or "")
        if not text.strip():
            return None
        return _list_items(text)


class ClinicBenchHospitalizationAdapter(_ClinicOpenAdapter):
    filename = "Hospitalization-Summarization.json"
    metric = "rouge"

    def fields(self, rec: dict) -> tuple[str, str]:
        source = str(rec.get("instruct") or "")
        prompt = (
            "Write a concise hospitalization summary based only on the clinical record below. "
            "Do not reconstruct the source document, fill placeholders, or repeat the full chart. "
            "Include the admission reason, key diagnoses and interventions, hospital course, and "
            f"discharge plan. Return only the summary, preferably under 300 words.\n\n{source}"
        )
        return prompt, str(rec.get("answer") or "")


class ClinicBenchDrugInteractionAdapter(_ClinicOpenAdapter):
    filename = "Drug-Interaction-for-Emerging-Drugs.json"
    metric = "accuracy"

    def fields(self, rec: dict) -> tuple[str, str]:
        return str(rec.get("question") or ""), str(rec.get("answer") or "").lower()

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        sample = super().normalize_sample(raw_sample, sample_index)
        sample.task_type = "classification"
        sample.answer_format = "yes_no"
        sample.metadata = {"labels": ["yes", "no"]}
        return sample

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": judgement_prompt(
            sample.source_content["question"], lang=sample.language or "en"
        )}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        value = parse_yes_no_maybe(raw_response)
        return value if value in {"yes", "no"} else None
