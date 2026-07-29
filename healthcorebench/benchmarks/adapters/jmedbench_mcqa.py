"""JMedBench adapter (Japanese medical benchmark suite — MCQA subsets).

Fixed data: ``40_JMedBench/<subset>/test.jsonl`` — one JSON object per line. Only the subsets
whose records share the multiple-choice schema are handled here::

    {"sample_id": str, "question": str, "options": [str, ...], "answer_idx": int (0-based),
     "n_options": int, "metadata": {..., "context": str (only some subsets)}}

The suite's NER / span / summarization subsets (bc2gm_jp, jcsts, mrner_*, etc.) use different
schemas and are not exposed by this adapter.

Subset selection is driven by the registry task (``JMedBench/crade`` -> ``crade``), falling back
to the split for a single-benchmark run; ``JMedBench/mcqa`` means ``jmmlu_medical``. The default
used to be ``medmcqa_jp``, which is a machine translation of the *same 4,183 records* as the
registered ``MedMCQA/mcqa`` task (identical ``sample_id`` set), so the suite's single scored task
was pure duplicate coverage. ``jmmlu_medical`` is the largest subset that is not a record-level
copy of another registered task (JMMLU is an independently built Japanese exam set, capped at 150
questions per subject, versus the 1,871 English MMLU medical items).

``crade`` / ``rrtnm`` / ``smdis`` are the only Japanese-native clinical subsets. Their question
text ("in the case report above, ...") refers to a case report, radiology report or social-media
post that lives in ``metadata.context`` — the prompt is unanswerable without it, so the context is
rendered into the prompt (and budgeted against the model window) rather than dropped.
Language is inferred from the subset name (``*_jp``/igakuqa*/native subsets → Japanese).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.context_window import fit_context_to_window
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.benchmarks.prompts import multiple_choice_prompt, format_lettered_choices
from healthcorebench.benchmarks.answer_parsing import parse_multiple_choice_letter
from healthcorebench.schemas.sample import EvaluationSample

# MCQA-format subsets (question/options/answer_idx). Kept explicit so an unexpected new
# directory with a different schema is not silently treated as MCQA.
_SUBSETS = {
    "crade", "igakuqa", "igakuqa_en", "igakuqa_sa", "igakuqa_sa_to", "jmmlu_medical",
    "medmcqa", "medmcqa_jp", "medqa", "medqa_jp", "mmlu_medical", "mmlu_medical_jp",
    "mmlu_pro_medical", "mmlu_pro_medical_jp", "pubmedqa", "pubmedqa_jp", "rrtnm", "smdis",
    "usmleqa", "usmleqa_jp",
}
# Japanese-native subsets: no English source dataset, so they are not translation duplicates.
_NATIVE_JA = {"crade", "rrtnm", "smdis"}
_DEFAULT = "jmmlu_medical"
_MAX_LETTERS = list("ABCDEFGHIJKLMNOP")


def _lang_of(subset: str) -> str:
    if subset.endswith("_en"):
        return "en"
    if subset.endswith("_jp") or subset.startswith("igakuqa") or subset in _NATIVE_JA:
        return "ja"
    if subset == "jmmlu_medical":
        return "ja"
    return "en"


class JMedBenchAdapter(BaseBenchmarkAdapter):
    benchmark_name = "JMedBench"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "multiple_choice"
    prompt_template_version = "1.0"

    def _subset(self) -> str:
        # The registry task pins the subset; "mcqa" keeps the historical default key working and
        # the split remains available for ad-hoc single-benchmark runs.
        task = (self.entry.task or "") if getattr(self, "entry", None) else ""
        if task in _SUBSETS:
            return task
        s = _DEFAULT if self.split == "test" else self.split
        if s not in _SUBSETS:
            raise BenchmarkSplitNotFoundError(
                f"JMedBench MCQA split must be 'test' or one of {sorted(_SUBSETS)}; got '{self.split}'."
            )
        return s

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / self._subset() / "test.jsonl"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                opts = rec.get("options")
                ai = rec.get("answer_idx")
                if not isinstance(opts, list) or len(opts) < 2 or not isinstance(ai, int):
                    self.drop_source_record("unparseable_options")
                    continue
                if not (0 <= ai < len(opts)):
                    self.drop_source_record("answer_index_out_of_range")
                    continue
                yield {"record": rec, "source_file_rel": rel, "source_record_index": i}

    def _context_of(self, rec: dict) -> str:
        meta = rec.get("metadata")
        return str(meta.get("context") or "").strip() if isinstance(meta, dict) else ""

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        subset = self._subset()
        lang = _lang_of(subset)

        question = str(rec["question"]).strip()
        choices = [str(o) for o in rec["options"]]
        block, letters = format_lettered_choices(choices)
        correct_letter = letters[int(rec["answer_idx"])]

        # crade/rrtnm/smdis/pubmedqa* ask about a document that only exists in metadata.context.
        context = self._context_of(rec)
        context_meta: dict[str, Any] = {}
        if context:
            generation = getattr(self.config, "generation", None)
            context, context_meta = fit_context_to_window(
                context,
                fixed_prompt=multiple_choice_prompt(question, block, lang=lang),
                max_model_len=getattr(getattr(self.config, "hardware", None), "max_model_len", None),
                max_output_tokens=self.output_token_budget_for_format("single_choice"),
                reserve_tokens=getattr(generation, "context_token_reserve", 512),
                policy=getattr(generation, "context_overflow_policy", "error"),
            )

        source_id = str(rec.get("sample_id") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"q": question, "c": choices})
        sample_id = self.make_sample_id(source_file_rel=rel, source_sample_id=source_id, content_hash=content_hash)

        return EvaluationSample(
            sample_id=sample_id,
            source_sample_id=source_id,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=f"{rel}",
            source_file=rel,
            source_record_index=rec_index,
            source_record_hash=self.input_hash(rec),
            input_hash=self.input_hash({"question": question, "choices_block": block, "context": context}),
            reference_hash=self.reference_hash(correct_letter),
            input_type="text",
            task_type="multiple_choice",
            component="Language",
            capability="Reasoning" if subset in _NATIVE_JA else "Knowledge",
            specialty=subset,
            language=lang,
            modality="Text",
            answer_format="single_choice",
            evaluation_metric="accuracy",
            source_content={"question": question, "choices": choices, "context": context},
            reference_answer=correct_letter,
            reference_answer_normalized=correct_letter,
            metadata={"letters": letters, "subset": subset, **context_meta},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        c = sample.source_content
        block, _ = format_lettered_choices([str(x) for x in c["choices"]])
        prompt = multiple_choice_prompt(c["question"], block, lang=sample.language)
        context = str(c.get("context") or "")
        if context:
            prompt = f"{context}\n\n{prompt}"
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return parse_multiple_choice_letter(raw_response, sample.metadata.get("letters") or _MAX_LETTERS)
