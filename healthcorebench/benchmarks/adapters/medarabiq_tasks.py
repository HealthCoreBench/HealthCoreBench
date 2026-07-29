"""Additional MedArabiQ task adapters for the six non-baseline source sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.adapters.medarabiq_mcqa import MedArabiQAdapter
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import format_lettered_choices, multiple_choice_prompt
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample


class MedArabiQBiasMCQAAdapter(MedArabiQAdapter):
    """MCQA with an intentionally bias-inducing clinical preamble."""

    def discover_source_files(self) -> list[Path]:
        if self.split != "test":
            raise BenchmarkSplitNotFoundError("MedArabiQ bias MCQA provides only 'test'.")
        return [self.get_benchmark_directory() / "multiple-choice-withbias.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        path = files[0]
        rel = self.rel_path(path)
        records = json.loads(path.read_text(encoding="utf-8"))
        for index, record in enumerate(records):
            stem, options = self._parse_question(record.get("Question with Bias", ""))
            answer_position = self._match_answer(record.get("Answer", ""), options)
            if len(options) >= 2 and answer_position is not None:
                yield {
                    "record": record,
                    "stem": stem,
                    "opts": options,
                    "correct_pos": answer_position,
                    "source_file_rel": rel,
                    "source_record_index": index,
                }


class MedArabiQFillChoiceAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedArabiQ"
    benchmark_version = "1.0"
    prompt_template_name = "multiple_choice"

    def discover_source_files(self) -> list[Path]:
        if self.split != "test":
            raise BenchmarkSplitNotFoundError("MedArabiQ fill-choice provides only 'test'.")
        return [self.get_benchmark_directory() / "fill-in-the-blank-choices.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        path = files[0]
        rel = self.rel_path(path)
        for index, record in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            stem, options = MedArabiQAdapter._parse_question(record.get("Question - Arabic", ""))
            position = MedArabiQAdapter._match_answer(record.get("Answer - Arabic", ""), options)
            if len(options) >= 2 and position is not None:
                yield {"record": record, "stem": stem, "options": options, "position": position,
                       "source_file_rel": rel, "source_record_index": index}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        record = raw_sample["record"]
        choices = [text for _, text in raw_sample["options"]]
        _, letters = format_lettered_choices(choices)
        reference = letters[raw_sample["position"]]
        rel = raw_sample["source_file_rel"]
        index = raw_sample["source_record_index"]
        source_id = f"{rel}:{index}"
        return EvaluationSample(
            sample_id=self.make_sample_id(source_file_rel=rel, source_sample_id=source_id,
                                          content_hash=self.input_hash({"q": raw_sample["stem"], "c": choices})),
            source_sample_id=source_id, sample_index=sample_index, benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version, benchmark_split=self.split, source_file=rel,
            source_record_index=index, source_record_hash=self.input_hash(record),
            input_hash=self.input_hash({"question": raw_sample["stem"], "choices": choices}),
            reference_hash=self.reference_hash(reference), task_type="multiple_choice", component="Language",
            capability="Knowledge", specialty=record.get("Category"), language="ar", modality="Text",
            answer_format="single_choice", evaluation_metric="accuracy",
            source_content={"question": raw_sample["stem"], "choices": choices},
            reference_answer=reference, reference_answer_normalized=reference, metadata={"letters": letters},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        block, _ = format_lettered_choices(sample.source_content["choices"])
        return [{"role": "user", "content": multiple_choice_prompt(sample.source_content["question"], block, lang="ar")}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata["letters"])


class _MedArabiQOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MedArabiQ"
    benchmark_version = "1.0"
    prompt_template_name = "open_ended"
    file_name: str
    question_field: str
    answer_field: str
    instruction = "أجب عن السؤال الطبي التالي."

    def discover_source_files(self) -> list[Path]:
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"MedArabiQ {self.file_name} provides only 'test'.")
        return [self.get_benchmark_directory() / self.file_name]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        path = files[0]
        rel = self.rel_path(path)
        for index, record in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            question = str(record.get(self.question_field) or "").strip()
            answer = str(record.get(self.answer_field) or "").strip()
            if question and answer:
                yield {"record": record, "question": question, "answer": answer,
                       "source_file_rel": rel, "source_record_index": index}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rel = raw_sample["source_file_rel"]
        index = raw_sample["source_record_index"]
        source_id = f"{rel}:{index}"
        return EvaluationSample(
            sample_id=self.make_sample_id(source_file_rel=rel, source_sample_id=source_id,
                                          content_hash=self.input_hash(raw_sample["question"])),
            source_sample_id=source_id, sample_index=sample_index, benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version, benchmark_split=self.split, source_file=rel,
            source_record_index=index, source_record_hash=self.input_hash(raw_sample["record"]),
            input_hash=self.input_hash(raw_sample["question"]), reference_hash=self.reference_hash(raw_sample["answer"]),
            task_type="open_ended", component="Language", capability="Reasoning", language="ar", modality="Text",
            answer_format="free_text", evaluation_metric="llm_judge",
            source_content={"question": raw_sample["question"]}, reference_answer=raw_sample["answer"],
            reference_answer_normalized=raw_sample["answer"], metadata={"source_set": self.file_name},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": f"{self.instruction}\n\n{sample.source_content['question']}"}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()


class MedArabiQFillOpenAdapter(_MedArabiQOpenAdapter):
    file_name = "fill-in-the-blank-nochoices.json"
    question_field = "Question - Arabic"
    answer_field = "Answer - Arabic"
    instruction = "املأ الفراغ في الجملة الطبية التالية بإجابة قصيرة."


class MedArabiQPatientQAAdapter(_MedArabiQOpenAdapter):
    file_name = "patient-doctor-qa.json"
    question_field = "Question_description"
    answer_field = "Answer_details"


class MedArabiQPatientLLMQAAdapter(_MedArabiQOpenAdapter):
    file_name = "patient-doctor-qa-llm.json"
    question_field = "Modified question description"
    answer_field = "Unmodified answer"


class MedArabiQPatientGECQAAdapter(_MedArabiQOpenAdapter):
    file_name = "patient-doctor-qa-gec.json"
    question_field = "GEC Question description"
    answer_field = "GEC Answer details"
