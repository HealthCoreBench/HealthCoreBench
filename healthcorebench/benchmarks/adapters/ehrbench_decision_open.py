"""EHRBench decision-making task adapter with request-time context budgeting.

Every record ships a controlled-vocabulary ``candidates`` list (1-165 options, median 98) and an
official metric of ``em`` / ``recall`` / ``acc`` against ``task_info.label``. The candidates are
part of the task: record 0's gold is ``OBSERVATION ADMIT`` while ``EU OBSERVATION``,
``DIRECT OBSERVATION`` and ``AMBULATORY OBSERVATION`` are also admission types, and nothing in
the instruction distinguishes them — so the candidate list is rendered into the prompt. 6,080 of
13,500 records carry a multi-entry ``task_info.label``; those entries are the concurrently
recorded next events, so each is an accepted answer (``reference_aliases``) rather than being
newline-joined into one unreachable reference string.

Eight of the 27 tasks repeat one invariant candidate vocabulary on all 500 of their records and
each leaves out a single value its own golds use, which put 889 records' gold outside the options
the prompt showed. ``_vocabulary_repairs`` completes those vocabularies rather than dropping the
records; see its docstring for why the omission reads as an enumeration gap.

The answer directive is deliberately gold-independent: it used to announce how many events were
recorded at the next step, which handed the model ``len(task_info.label)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.context_window import fit_context_to_window
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

# Fraction of the fittable context reserved for the *end* of the EHR when truncation is needed.
# EHRBench ships no answer-location field, so no genuine evidence span can be protected; the
# retention policy is therefore declared explicitly instead of falling through to the
# answer-blind 50/50 head_tail slice. Recent EHR content is the more predictive half for both
# "next event" decisions and risk prediction, so the tail is protected and the remaining budget
# goes to the record header.
_RECENCY_TAIL_SHARE = 0.75
_RETENTION_POLICY = "prefer_recent_tail"
_RETENTION_SPAN_SOURCE = "adapter_recency_policy_tail"

# Sent after the EHR context. Deliberately free of any gold-derived quantity: the previous
# wording announced how many events were recorded at the next step, which is exactly
# ``len(task_info.label)`` and told the model the size of the answer it had to produce.
_ANSWER_DIRECTIVE = (
    "List every event recorded at the next step, one per line, copied verbatim from the "
    "options above."
)

# Marks a task whose candidate list still has to be inspected.
_UNSET = object()


def fit_ehr_context(
    context: str,
    *,
    fixed_prompt: str,
    config,
    max_output_tokens: int | None,
) -> tuple[str, dict]:
    """Fit an EHR context to the model window, preferring the most recent content.

    The first pass measures how much source text fits at all. If nothing had to be dropped the
    result is used as-is; otherwise a second pass protects the trailing
    ``_RECENCY_TAIL_SHARE`` of that amount so recency survives, and the leftover budget keeps the
    record header. The chosen policy is recorded on the sample so a reader knows which half of
    the chart the score was earned on.
    """
    generation = getattr(config, "generation", None)
    window = dict(
        fixed_prompt=fixed_prompt,
        max_model_len=getattr(getattr(config, "hardware", None), "max_model_len", None),
        max_output_tokens=max_output_tokens,
        reserve_tokens=getattr(generation, "context_token_reserve", 512),
        policy=getattr(generation, "context_overflow_policy", "error"),
        # EHRs contain dense IDs, values, abbreviations, and tables; use the conservative
        # estimator so the first provider request stays inside the configured window.
        ascii_chars_per_token=2,
    )
    fitted, metadata = fit_context_to_window(context, **window)
    if not metadata.get("context_truncated"):
        return fitted, metadata
    retained_source = max(0, len(context) - metadata.get("omitted_context_chars", 0))
    tail_chars = int(retained_source * _RECENCY_TAIL_SHARE)
    if tail_chars > 0:
        fitted, metadata = fit_context_to_window(
            context,
            protected_spans=[(len(context) - tail_chars, len(context))],
            protected_span_source=_RETENTION_SPAN_SOURCE,
            **window,
        )
    metadata.update({
        "context_retention_policy": _RETENTION_POLICY,
        "context_retention_tail_share": _RECENCY_TAIL_SHARE,
        "context_retention_rationale": (
            "EHRBench declares no answer location, so the retained window is chosen by policy: "
            "recent chart content is generally the more predictive part, so the tail is kept and "
            "the remaining budget keeps the record header."
        ),
    })
    return fitted, metadata


def candidate_block(candidates: Any) -> str:
    """Render the controlled-vocabulary options a record's answer must be drawn from."""
    options = [str(option).strip() for option in (candidates or []) if str(option).strip()]
    if not options:
        return ""
    listed = "\n".join(f"- {option}" for option in options)
    return f"Choose from the following {len(options)} options:\n{listed}"


class EHRBenchDecisionAdapter(BaseBenchmarkAdapter):
    benchmark_name = "EHRBench"
    benchmark_version = "1.0"
    adapter_version = "1.3"

    def discover_source_files(self) -> list[Path]:
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(
                f"EHRBench decision-making provides only 'test'; requested '{self.split}'."
            )
        return [self.get_benchmark_directory() / "ehr_bench_decision_making.jsonl"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        source = files[0]
        rel = self.rel_path(source)
        repairs = self._vocabulary_repairs(source)
        for index, record in self._iter_records(source):
            instruction = str(record.get("instruction") or "").strip()
            context = str(record.get("input") or "").strip()
            reference = str(record.get("output") or "").strip()
            if not ((instruction or context) and reference):
                self.drop_source_record("empty_instruction_or_reference")
                continue
            accepted = self._accepted_answers(record, reference)
            candidates = self._candidates(record)
            known = set(candidates)
            added = [value for value in repairs.get(self._task(record), ())
                     if value not in known]
            candidates = candidates + added
            if not known.union(added).issuperset(accepted):
                # Nothing links this gold to the options the prompt shows, so the record is
                # unanswerable as posed rather than merely hard. Excluding it here keeps the
                # count in the manifest instead of quietly deflating the score.
                self.drop_source_record("gold_not_in_candidates")
                continue
            yield {
                "record": record,
                "instruction": instruction,
                "context": context,
                "reference": reference,
                "accepted_answers": accepted,
                "candidates": candidates,
                "candidates_added": added,
                "source_file_rel": rel,
                "source_record_index": index,
            }

    @staticmethod
    def _iter_records(source: Path) -> Iterable[tuple[int, dict]]:
        with source.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    yield index, json.loads(line)

    @staticmethod
    def _task(record: dict) -> Any:
        return (record.get("task_info") or {}).get("task")

    @staticmethod
    def _candidates(record: dict) -> list[str]:
        return [str(option).strip() for option in (record.get("candidates") or [])
                if str(option).strip()]

    def _vocabulary_repairs(self, source: Path) -> dict[Any, list[str]]:
        """Gold values that a task's fixed candidate vocabulary leaves out, keyed by task.

        13,500 records span 27 tasks in two shapes. Nineteen tasks draw a fresh candidate subset
        per record (hundreds of distinct lists, 1-165 options each) and every one of their golds
        appears in its own record's list. The other eight repeat *one* identical vocabulary on all
        500 of their records — and each of those eight omits exactly one value that its own golds
        use: ``URGENT`` for admissions, ``MED`` for services, ``Medications`` for poe,
        ``Height (Inches)`` for omr, and so on, 889 records in all.

        A value that is invariably absent from an invariant vocabulary, yet is the recorded answer
        elsewhere in the same task, is a gap in how that vocabulary was enumerated — MIMIC-IV's
        ``admission_type`` does contain ``URGENT`` — not a per-record decision to rule the answer
        out. So the vocabulary is completed and the records are kept. Tasks with per-record
        candidate subsets carry no task-level vocabulary and are therefore never repaired; a gold
        outside such a record's own list would be dropped by ``load_raw_samples`` instead.
        """
        vocabularies: dict[Any, Any] = {}
        golds: dict[Any, set[str]] = {}
        for _, record in self._iter_records(source):
            task = self._task(record)
            candidates = frozenset(self._candidates(record))
            known = vocabularies.get(task, _UNSET)
            if known is _UNSET:
                vocabularies[task] = candidates
            elif known is not None and known != candidates:
                vocabularies[task] = None
                golds.pop(task, None)  # varies per record: nothing task-level to complete
            if vocabularies[task] is None:
                continue
            reference = str(record.get("output") or "").strip()
            golds.setdefault(task, set()).update(self._accepted_answers(record, reference))
        return {
            task: sorted(values - vocabularies[task])
            for task, values in golds.items()
            if vocabularies[task] and values - vocabularies[task]
        }

    @staticmethod
    def _accepted_answers(record: dict, reference: str) -> list[str]:
        """The individual gold entries, which the source joins with newlines into ``output``."""
        label = (record.get("task_info") or {}).get("label")
        entries = label if isinstance(label, list) else [label]
        accepted = [str(entry).strip() for entry in entries if str(entry or "").strip()]
        if not accepted:
            accepted = [part.strip() for part in reference.split("\n") if part.strip()]
        return accepted

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        record = raw_sample["record"]
        instruction = raw_sample["instruction"]
        context = raw_sample["context"]
        original_input = {"instruction": instruction, "context": context}
        reference = raw_sample["reference"]
        accepted = raw_sample["accepted_answers"]
        rel = raw_sample["source_file_rel"]
        record_index = raw_sample["source_record_index"]
        task_info = record.get("task_info") or {}
        options = candidate_block(raw_sample["candidates"])
        context, context_meta = fit_ehr_context(
            context,
            fixed_prompt=self._fixed_prompt(instruction, options),
            config=self.config,
            max_output_tokens=getattr(getattr(self.config, "generation", None), "max_tokens", None),
        )
        source_id = str(
            record.get("idx") if record.get("idx") is not None else f"{rel}:{record_index}"
        )
        normalized_input = {"instruction": instruction, "context": context}
        metadata = {
            "official_metric": task_info.get("metric"),
            "candidates": raw_sample["candidates"],
            "accepted_answers": accepted,
            "reference_label_count": len(accepted),
            **context_meta,
        }
        if raw_sample["candidates_added"]:
            # Says on the sample itself that the shown options are not verbatim the source's.
            metadata["candidates_added"] = raw_sample["candidates_added"]
        # Applied to every record, not just multi-gold ones: the prompt no longer tells the model
        # how many events to name, so a single-gold answer may legitimately arrive as a list.
        metadata["judge_rubric"] = (
            "The gold answer is the set of events recorded together at the next step, drawn from "
            "the listed candidate options. Score 1 when the answer names one of the accepted "
            "references (or all of them); score 0 for an option outside the accepted set."
        )

        return EvaluationSample(
            sample_id=self.make_sample_id(
                source_file_rel=rel,
                source_sample_id=source_id,
                # Sample identity follows the untruncated source record, while input_hash below
                # records the exact context sent for this model-window configuration.
                content_hash=self.input_hash(original_input),
            ),
            source_sample_id=source_id,
            sample_index=sample_index,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            benchmark_split=self.split,
            source_benchmark_entry=rel,
            source_file=rel,
            source_record_index=record_index,
            source_record_hash=self.input_hash(record),
            input_hash=self.input_hash(normalized_input),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=task_info.get("task"),
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"instruction": instruction, "context": context,
                            "options": options},
            reference_answer=reference,
            reference_answer_normalized=reference,
            reference_aliases=accepted if len(accepted) > 1 else None,
            metadata=metadata,
        )

    @staticmethod
    def _answer_block(options: str) -> str:
        """The candidate list plus the answer-shape directive, sent after the EHR context."""
        return "\n\n".join(part for part in (options, _ANSWER_DIRECTIVE) if part)

    def _fixed_prompt(self, instruction: str, options: str) -> str:
        """Everything sent besides the EHR context, with the same separators, so the context
        budget accounts for the candidate list exactly as ``build_messages`` emits it."""
        return f"{instruction}\n\n\n\n{self._answer_block(options)}"

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        content = sample.source_content
        answers = self._answer_block(content.get("options") or "")
        prompt = f"{content['instruction']}\n\n{content['context']}\n\n{answers}"
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return (raw_response or "").strip() or None
