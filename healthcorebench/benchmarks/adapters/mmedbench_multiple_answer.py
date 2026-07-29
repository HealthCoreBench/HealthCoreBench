"""MMedBench French/Japanese multiple-answer questions.

Only items whose ``answer_idx`` names at least two options belong here (340 of 821). The
single-element lists are single-choice questions: prompting them for "one or more" letters and
scoring exact set match measures letter-count compliance, not knowledge.
"""
import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.prompts import multiple_answer_prompt
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letters
from healthcorebench.schemas.sample import EvaluationSample

_FILES = {"French.jsonl": "fr", "Japanese.jsonl": "ja"}


class MMedBenchMultipleAnswerAdapter(BaseBenchmarkAdapter):
    benchmark_name = "MMedBench"
    benchmark_version = "1.0"
    prompt_template_name = "multiple_answer"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / name for name in _FILES]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        for f in files:
            rel = self.rel_path(f)
            with open(f, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    answers = rec.get("answer_idx")
                    options = rec.get("options") or {}
                    if (isinstance(answers, list) and len(set(answers)) >= 2
                            and set(answers) <= set(options)):
                        yield {"record": rec, "source_file_rel": rel,
                               "source_record_index": i, "language": _FILES[f.name]}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        letters = sorted(rec["options"])
        reference = ",".join(sorted(set(rec["answer_idx"])))
        block = "\n".join(f"{letter}. {rec['options'][letter]}" for letter in letters)
        source_id = f"{rel}:{raw_sample['source_record_index']}"
        return EvaluationSample(
            sample_id=self.make_sample_id(source_file_rel=rel, source_sample_id=source_id,
                                          content_hash=self.input_hash({"q": rec["question"], "o": rec["options"]})),
            source_sample_id=source_id, sample_index=sample_index,
            benchmark_name=self.benchmark_name, benchmark_version=self.benchmark_version,
            benchmark_split=self.split, source_file=rel,
            source_record_index=raw_sample["source_record_index"], source_record_hash=self.input_hash(rec),
            input_hash=self.input_hash({"question": rec["question"], "choices_block": block}),
            reference_hash=self.reference_hash(reference), task_type="multiple_choice",
            component="Language", capability="Knowledge", language=raw_sample["language"],
            modality="Text", answer_format="multi_choice", evaluation_metric="set_match",
            source_content={"question": rec["question"], "options": rec["options"], "letters": letters},
            reference_answer=reference, reference_answer_normalized=reference,
            metadata={"letters": letters},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block = "\n".join(f"{letter}. {c['options'][letter]}" for letter in c["letters"])
        return [{"role": "user", "content": multiple_answer_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letters(raw_response, sample.metadata.get("letters") or [])
