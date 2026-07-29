"""Swedish Medical LLM Benchmark — MedQA-SWE adapter (Swedish medical exam MCQA).

Fixed data: ``49_Swedish_Medical_LLM_Benchmark/medqa-swe/medqa_swe.json`` — a dict with a
``questions`` list of records::

    {"question": str, "options": ["A: text", "B: text", ...], "answer": str (letter),
     "date": str, "part": str}

Task: single-choice. Option strings carry their own "X: " letter prefix; the prefix is stripped
and options are re-lettered locally so the prompt is canonical, then ``answer`` (a letter) is
mapped to the corresponding position. The source ``options`` list is line-wrapped: a long option
is split across consecutive entries and the continuation entries carry no letter prefix, so they
are appended to the option they continue (264/3,180 records) rather than dropped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

# "A: text" / "A. text" / "A) text" -> (letter, text). The body may be empty: one row carries a
# bare "C:" whose text the source lost, and treating it as a continuation line would silently
# re-letter D/E into C/D.
_OPT_RE = re.compile(r"^\s*([A-Za-z])\s*[:\.\)]\s*(.*)$", re.DOTALL)


class SwedishMedQAAdapter(BaseBenchmarkAdapter):
    benchmark_name = "Swedish_Medical_LLM_Benchmark"
    benchmark_version = "1.0"
    adapter_version = "1.0"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(
                f"Swedish_Medical_LLM_Benchmark provides only 'test'; requested '{self.split}'."
            )
        return [directory / "medqa-swe" / "medqa_swe.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for i, rec in enumerate(data.get("questions", [])):
            parsed: list[list[str]] = []
            for opt in rec.get("options", []):
                m = _OPT_RE.match(str(opt))
                if m:
                    parsed.append([m.group(1).upper(), m.group(2).strip()])
                elif parsed:
                    # wrapped continuation of the option opened above; dropping it truncated
                    # the option text mid-sentence.
                    parsed[-1][1] = f"{parsed[-1][1]} {str(opt).strip()}".strip()
            pairs = [(letter, text) for letter, text in parsed]
            ans = str(rec.get("answer", "")).strip().upper()
            valid = {p[0] for p in pairs}
            if len(pairs) < 2 or ans not in valid:
                continue
            yield {"record": rec, "pairs": pairs, "answer": ans,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        pairs: list[tuple[str, str]] = raw_sample["pairs"]
        src_answer: str = raw_sample["answer"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]

        question = str(rec["question"]).strip()
        # re-letter locally (A.. in source order) so the prompt is canonical.
        src_letters = [p[0] for p in pairs]
        choices = [p[1] for p in pairs]
        block, letters = format_lettered_choices(choices)
        correct_pos = src_letters.index(src_answer)
        correct_letter = letters[correct_pos]

        source_id = f"{rel}:{rec_index}"
        content_hash = self.input_hash({"q": question, "c": choices})
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
            input_hash=self.input_hash({"question": question, "choices_block": block}),
            reference_hash=self.reference_hash(correct_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Knowledge",
            specialty=None,
            language="sv",
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "part": rec.get("part")},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        return [{"role": "user", "content": multiple_choice_prompt(c["question"], block, lang=sample.language)}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or ["A", "B", "C", "D", "E"])
