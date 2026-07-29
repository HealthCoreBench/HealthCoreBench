"""BioHopR multi-answer task scored against any accepted answer."""
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.schemas.sample import EvaluationSample


class BioHopRMultiOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "BioHopR"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "short_answer"
    prompt_template_version = "1.1"

    def discover_source_files(self) -> list[Path]:
        return [self.get_benchmark_directory() / "biohopr_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        import json
        for i, rec in enumerate(json.loads(f.read_text(encoding="utf-8"))):
            answers = rec.get("answer") or []
            question = str(rec.get("hop2_question") or rec.get("prompt") or "").strip()
            if isinstance(answers, list) and len(answers) > 1 and question:
                yield {"record": rec, "question": question, "answers": answers,
                       "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        question = raw_sample["question"]
        references = [str(answer).strip() for answer in raw_sample["answers"] if str(answer).strip()]
        source_id = f"{rel}:{raw_sample['source_record_index']}"
        return EvaluationSample(
            sample_id=self.make_sample_id(source_file_rel=rel, source_sample_id=source_id,
                                          content_hash=self.input_hash(question)),
            source_sample_id=source_id, sample_index=sample_index,
            benchmark_name=self.benchmark_name, benchmark_version=self.benchmark_version,
            benchmark_split=self.split, source_file=rel,
            source_record_index=raw_sample["source_record_index"], source_record_hash=self.input_hash(rec),
            input_hash=self.input_hash(question), reference_hash=self.reference_hash(references),
            task_type="open_ended", component="Language", capability="Reasoning",
            specialty=rec.get("target_type"), language="en", modality="Text",
            answer_format="short_answer", evaluation_metric="any_of_match",
            source_content={"question": question}, reference_answer=references[0],
            reference_answer_normalized=references,
            metadata={"answer_count": len(references), "match_policy": "any_accepted_phrase"},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{
            "role": "user",
            "content": (
                f"{sample.source_content['question']}\n\n"
                "Return one final disease, drug, effect, or other requested name. "
                "Do not include reasoning or a list of alternatives."
            ),
        }]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip() or None
