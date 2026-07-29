"""BioASQ summary and list open-ended tasks.

``ideal_answer`` is the human-written answer text. BioASQ stores it either as a plain string or
as a list of several independently written gold answers for the same question (measured over the
summary and list tasks: 2,131 of 2,349 records are lists, up to 112 entries; after dropping
repeats within a list, 818 records still hold more than one distinct gold). Both are natural
language and both are what the LLM judge must compare the model's prose against.
"""
import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.schemas.sample import EvaluationSample


def _reference_texts(value: Any) -> list[str]:
    """Flatten ``ideal_answer`` / ``exact_answer`` into de-duplicated answer strings.

    The reference used to be ``json.dumps(value)``, which handed the judge a JSON literal —
    brackets, quotes and ``\\u`` escapes around the answer — so a model replying in plain prose
    could never match it verbatim. Take the text itself instead, and keep every gold variant as
    a separate reference rather than concatenating them into one impossible-to-match blob.
    """
    out: list[str] = []

    def _add(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for sub in item:
                _add(sub)
            return
        text = str(item).strip()
        if text and text not in out:
            out.append(text)

    _add(value)
    return out


class _BioASQLongAdapter(BaseBenchmarkAdapter):
    benchmark_name = "BioASQ"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    question_type = ""

    def discover_source_files(self) -> list[Path]:
        return sorted(self.get_benchmark_directory().rglob("*.json"))

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        seen = set()
        for f in files:
            rel = self.rel_path(f)
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (ValueError, json.JSONDecodeError):
                self.drop_source_record("unparseable_source_file")
                continue
            for i, question in enumerate(data.get("questions", [])):
                if question.get("type") != self.question_type:
                    continue  # belongs to a sibling BioASQ task, not an exclusion
                qid = str(question.get("id") or f"{rel}:{i}")
                if qid in seen:
                    # Guard only: the fixed release ships 5,486 distinct ids and no repeats.
                    self.drop_source_record("duplicate_question_id")
                    continue
                # ``ideal_answer`` is preferred; ``exact_answer`` is the entity-level gold and is
                # only reached when a record ships no ideal answer at all.
                references = _reference_texts(question.get("ideal_answer")
                                              or question.get("exact_answer"))
                if not question.get("body"):
                    self.drop_source_record("empty_question_body")
                    continue
                if not references:
                    self.drop_source_record("no_reference_answer")
                    continue
                seen.add(qid)
                yield {"question": question, "qid": qid, "references": references,
                       "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        q = raw_sample["question"]
        rel = raw_sample["source_file_rel"]
        body = str(q["body"])
        references: list[str] = raw_sample["references"]
        # The first gold answer is the primary reference; the remaining ones are equally valid
        # phrasings by other annotators, so they go in ``reference_aliases`` where both the judge
        # and the text metrics score against the best match.
        reference = references[0]
        aliases = references[1:]
        return EvaluationSample(
            sample_id=self.make_sample_id(source_file_rel=rel, source_sample_id=raw_sample["qid"],
                                          content_hash=self.input_hash(body)),
            source_sample_id=raw_sample["qid"], sample_index=sample_index,
            benchmark_name=self.benchmark_name, benchmark_version=self.benchmark_version,
            benchmark_split=self.split, source_file=rel,
            source_record_index=raw_sample["source_record_index"], source_record_hash=self.input_hash(q),
            input_hash=self.input_hash(body), reference_hash=self.reference_hash(references),
            task_type="open_ended", component="Language", capability="Knowledge",
            language="en", modality="Text", answer_format="free_text", evaluation_metric="llm_judge",
            source_content={"body": body}, reference_answer=reference,
            reference_answer_normalized=reference,
            reference_aliases=aliases or None,
            metadata={"question_type": self.question_type, "num_references": len(references)},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        return [{"role": "user", "content": sample.source_content["body"]}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip()


class BioASQSummaryAdapter(_BioASQLongAdapter):
    question_type = "summary"


class BioASQListAdapter(_BioASQLongAdapter):
    question_type = "list"
