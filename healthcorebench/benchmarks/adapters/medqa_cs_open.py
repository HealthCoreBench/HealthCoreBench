"""MedQA-CS adapter (USMLE Step-2 CS clinical skills, free-text, English).

Fixed data: ``34_MedQA-CS/data/med-student.json`` — a JSON list of 1,667 records covering 44
cases, each record shaped::

    {"unique_id": str (unique), "section": str, "case_id": str, "conversation_turn_id": int,
     "input": {...}, "ground_truth_output": str, "prompt": {"template": str, ...}, "output": ...}

``section`` selects the exam stage, and each stage carries its own ``input`` schema — measured on
the fixed file, with the field list taken from the record's own ``prompt.input_variables``:

* ``qa`` (1,535 records, turns 1-39) — ``{opening, chat_history}``. History taking: ask the next
  question. ``chat_history`` is the literal string ``"N/A"`` on the 44 first turns.
* ``physical_exam`` (44, one per case) — ``{opening, chat_history}``. Name the physical exams and
  maneuvers the history justifies; ``chat_history`` is the full completed interview.
* ``closure`` (44) — ``{opening, chat_history, pre_closure, challenge_question}``. Summarise for
  the patient and answer their question. ``pre_closure`` is the physical-exam findings and
  ``challenge_question`` is what the patient asks ("Is it a myocardial infarction?").
* ``diagnosis`` (44) — ``{opening}`` only. Rank three differential diagnoses with supporting
  findings. Its ``opening`` is not the short scenario the other stages get but the whole encounter
  (7.1k-12.6k chars: vitals, full history, physical examination), which is why it needs no
  ``chat_history``.

All four stages are loaded — the ``section`` is recorded on every sample, so they stay separable
downstream. The prompt is assembled per stage from the fields that stage actually has;
``pre_closure`` and ``challenge_question`` used to be dropped, which asked the model for a closure
without the exam findings and without the question it was supposed to answer.

``ground_truth_output`` (free text) is scored by an LLM judge rather than against the JSON output
format ``prompt.template`` requests, because the reference itself is prose, not JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

# What the examinee is asked to produce at each stage, paraphrasing that stage's own
# ``prompt.template``. Without this the four stages shared one "provide the appropriate next
# response" instruction, so a physical-exam record and a diagnosis record were posed identically.
_SECTION_TASKS = {
    "qa": ("Take a focused history: ask the single most useful next question, and say which "
           "symptom or aspect it is meant to establish."),
    "physical_exam": ("Decide which physical examinations to perform and the corresponding "
                      "maneuvers, restricted to what the history above justifies, with a reason "
                      "for each."),
    "closure": ("Write a brief closure for the patient — summarise the history and physical "
                "findings, discuss the diagnostic possibilities without committing to one, and "
                "outline the planned workup in plain language — then answer the patient's "
                "question."),
    "diagnosis": ("Write a differential diagnosis: list the three most likely diagnoses in "
                  "descending order of probability, each with the historical and physical-exam "
                  "findings that support it."),
}
_DEFAULT_TASK = "Based on the information above, provide the appropriate next response."

# ``chat_history`` on a first turn. Rendering it as a "Conversation so far: N/A" block would tell
# the model a transcript exists and is empty, so the block is left out instead.
_NO_HISTORY = "N/A"

# Section field -> prompt heading, in the order the upstream templates present them.
_CONTEXT_BLOCKS = (
    ("chat_history", "Previous dialogue"),
    ("pre_closure", "Physical examination findings"),
    ("challenge_question", "Question from the patient"),
)


class MedQACSOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedQA-CS"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.1"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedQA-CS provides only 'test'; requested '{self.split}'.")
        return [directory / "data" / "med-student.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            inp = rec.get("input")
            if not isinstance(inp, dict):
                # Guard only: all 1,667 records ship ``input`` as a JSON object. (The previous
                # code also tried ``ast.literal_eval`` on a string form; no record uses one, so
                # that branch could never fire and has been removed.)
                self.drop_source_record("input_not_an_object")
                continue
            opening = str(inp.get("opening") or "").strip()
            reference = str(rec.get("ground_truth_output") or "").strip()
            if not opening:
                self.drop_source_record("empty_opening")
                continue
            if not reference:
                # 39/1,667 ``qa`` turns ship ``ground_truth_output: ""`` — no expected doctor
                # question, so there is nothing for the judge to score. Reported so the 1,628
                # scored items are not read as the full file.
                self.drop_source_record("empty_reference_answer")
                continue
            # Only the fields this section defines; a missing one stays absent rather than
            # becoming an empty heading in the prompt.
            context = {}
            for field, _ in _CONTEXT_BLOCKS:
                value = str(inp.get(field) or "").strip()
                if value and value != _NO_HISTORY:
                    context[field] = value
            yield {"record": rec, "opening": opening, "context": context, "reference": reference,
                   "section": rec.get("section"),
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        opening = raw_sample["opening"]
        context: dict = raw_sample["context"]
        reference = raw_sample["reference"]
        section = raw_sample.get("section")

        source_id = str(rec.get("unique_id") or f"{rel}:{rec_index}")
        # ``section`` participates in the hash because it selects the task instruction: two stages
        # of the same case can share an opening yet ask for entirely different output.
        content = {"section": section, "opening": opening, **context}
        content_hash = self.input_hash(content)
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
            input_hash=content_hash,
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=section or "history_taking",
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"section": section, "opening": opening, **context},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={"case_id": rec.get("case_id"), "turn": rec.get("conversation_turn_id"),
                      "section": section, "context_fields": sorted(context)},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        section = c.get("section") or (sample.metadata or {}).get("section")
        parts = [
            f"You are completing the USMLE clinical-skills section '{section}'.",
            _SECTION_TASKS.get(section, _DEFAULT_TASK),
            "",
            c["opening"],
        ]
        parts.extend(f"\n{heading}:\n{c[field]}" for field, heading in _CONTEXT_BLOCKS
                     if c.get(field))
        return [{"role": "user", "content": "\n".join(parts)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()
