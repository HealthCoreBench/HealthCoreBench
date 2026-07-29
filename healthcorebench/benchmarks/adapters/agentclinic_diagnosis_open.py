"""AgentClinic adapter (OSCE case → diagnosis, free-text, English).

Fixed data: ``30_AgentClinic/agentclinic_medqa{,_extended}.jsonl`` — one JSON object per line,
each ``{"OSCE_Examination": {...}}`` with::

    {"Objective_for_Doctor": str, "Patient_Actor": {Demographics, History, Symptoms, ...},
     "Physical_Examination_Findings": {...}, "Test_Results": {...}, "Correct_Diagnosis": str}

Task: open-ended diagnosis. AgentClinic is designed as an interactive doctor-agent simulation; this
adapter uses the static OSCE case — presenting the patient history, physical exam and test results
up front — and asks for the diagnosis. The reference is ``Correct_Diagnosis`` (LLM judge). The
``agentclinic_nejm*`` files carry ``image_url`` (multimodal) and are not covered here. Splits:
``medqa_extended`` (default complete text set) and ``medqa`` (107-case subset).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.adapters.hle_med_exact import extract_short_answer
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

# The reference is a bare diagnosis name (measured over the shipped data: median 2 words / 23
# characters, max 12 words), but the prompt asked an open question and the secondary token-F1
# scored the model's entire reply against it. On the recorded run the replies are a median 1,534
# characters, so that metric reported the length ratio (0.024) rather than diagnostic accuracy.
#
# Reasoning is deliberately *not* suppressed — the LLM judge is the primary metric here and reads
# the full raw response — but the reply must end with the diagnosis on its own line so the
# rule-based cross-check has a comparable span to score. The marker is the ASCII "Answer:" form
# that ``extract_short_answer`` already recognizes; it is not re-implemented here.
_FINAL_ANSWER_INSTRUCTION = (
    "Reason as much as you need, then end your reply with a final line in exactly this form:\n"
    "Answer: <the single most likely diagnosis, name only>"
)

_SPLITS = {"medqa": "agentclinic_medqa.jsonl", "medqa_extended": "agentclinic_medqa_extended.jsonl"}
_DEFAULT = "medqa_extended"


def _render(value: Any) -> str:
    """Render a (possibly nested dict) OSCE field into readable 'Key: value' lines."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            label = str(k).replace("_", " ")
            if isinstance(v, (dict, list)):
                parts.append(f"{label}: {json.dumps(v, ensure_ascii=False)}")
            else:
                parts.append(f"{label}: {v}")
        return "\n".join(parts)
    return str(value or "").strip()


class AgentClinicDiagnosisAdapter(BaseBenchmarkAdapter):
    benchmark_name = "AgentClinic"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.1"

    def _split(self) -> str:
        s = _DEFAULT if self.split == "test" else self.split
        if s not in _SPLITS:
            raise BenchmarkSplitNotFoundError(f"AgentClinic split must be 'test'/'medqa'/'medqa_extended'; got '{self.split}'.")
        return s

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / _SPLITS[self._split()]]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                osce = rec.get("OSCE_Examination") or {}
                dx = str(osce.get("Correct_Diagnosis") or "").strip()
                if not dx:
                    continue
                yield {"osce": osce, "reference": dx, "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        osce = raw_sample["osce"]
        reference = raw_sample["reference"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        patient = _render(osce.get("Patient_Actor"))
        exam = _render(osce.get("Physical_Examination_Findings"))
        tests = _render(osce.get("Test_Results"))

        source_id = f"{rel}:{rec_index}"
        content_hash = self.input_hash({"p": patient, "e": exam, "t": tests})
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
            source_record_hash=self.input_hash(osce),
            input_hash=self.input_hash({"patient": patient, "exam": exam, "tests": tests}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=None,
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"patient": patient, "exam": exam, "tests": tests},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"split": self._split()},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        parts = []
        if c.get("patient"):
            parts.append(f"Patient:\n{c['patient']}")
        if c.get("exam"):
            parts.append(f"Physical Examination Findings:\n{c['exam']}")
        if c.get("tests"):
            parts.append(f"Test Results:\n{c['tests']}")
        body = "\n\n".join(parts)
        prompt = (f"{body}\n\nBased on the clinical information above, what is the most likely "
                  f"diagnosis?\n\n{_FINAL_ANSWER_INSTRUCTION}")
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return extract_short_answer(raw_response)
