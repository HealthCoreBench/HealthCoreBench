"""GeneTuring adapter (genomics QA, free-text, English).

Fixed data: ``33_GeneTuring/geneturing_test.json`` — a JSON list of records::

    {"Model": str, "Module": str, "Question": str, "Goldstandard": str}

Task: open-ended genomics QA across 16 modules (gene alias, gene location, SNP association,
DNA-sequence tasks, etc.). ``Goldstandard`` is the reference answer (a gene symbol, chromosome
coordinate, sequence, ontology term, ...). Because answers span very different formats, scoring
uses an LLM judge against the gold answer. Rows without a gold answer are skipped (reported via
``drop_source_record``). ``Module`` is exposed both as the specialty and in metadata so
per-module accuracy can be aggregated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample


class GeneTuringOpenAdapter(BaseBenchmarkAdapter):
    benchmark_name = "GeneTuring"
    benchmark_version = "1.0"
    adapter_version = "1.2"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.2"

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split != "test":
            raise BenchmarkSplitNotFoundError(f"GeneTuring provides only 'test'; requested '{self.split}'.")
        return [directory / "geneturing_test.json"]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        for i, rec in enumerate(records):
            q = str(rec.get("Question") or "").strip()
            gold = str(rec.get("Goldstandard") or "").strip()
            if not q:
                self.drop_source_record("empty_question")
                continue
            if not gold:
                # 65/1600 rows ship ``Goldstandard: null`` (mostly the "Protein-coding genes"
                # module). Nothing to judge against, so they are excluded — but reported, so the
                # 1,535 scored items are not mistaken for full coverage of the 1,600-row file.
                self.drop_source_record("missing_gold_answer")
                continue
            yield {"record": rec, "question": q, "reference": gold,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        question = raw_sample["question"]
        reference = raw_sample["reference"]
        module = rec.get("Module")

        source_id = f"{rel}:{rec_index}"
        content_hash = self.input_hash({"q": question, "m": module})
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
            input_hash=self.input_hash({"question": question}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Knowledge",
            specialty=module,
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"question": question},
            reference_answer=reference,
            reference_answer_normalized=reference,
            metadata={
                "module": module,
                "expected_reference_length": len(reference),
                "gene_output_kind": (
                    "sequence" if module in {"Amino acid translation", "DNA sequence extraction"}
                    else "short_answer"
                ),
            },
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        prompt = (
            f"{sample.source_content['question']}\n\n"
            "Return only the final answer in the requested format, without explanation, "
            "reasoning, or Markdown. For sequence tasks, return exactly one sequence and stop; "
            "do not add spaces, restart the translation, or repeat sequence motifs."
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        text = (raw_response or "").strip()
        if not text:
            return None
        if (sample.metadata or {}).get("gene_output_kind") != "sequence":
            return text

        sequence = _clean_sequence(text)
        if sequence is None:
            return None
        module = (sample.metadata or {}).get("module")
        allowed = set("ACGTN") if module == "DNA sequence extraction" else set("ACDEFGHIKLMNPQRSTVWY*X")
        if any(letter not in allowed for letter in sequence):
            return None
        reference_length = int((sample.metadata or {}).get("expected_reference_length") or 0)
        if _is_runaway_periodic(sequence, reference_length):
            return None
        return sequence


def _clean_sequence(text: str) -> str | None:
    marker = re.fullmatch(r"(?:final\s+answer|answer)\s*[:=]\s*([A-Za-z*\s]+)", text, re.I | re.S)
    if marker:
        text = marker.group(1)
    fenced = re.fullmatch(r"```(?:text|dna|protein|sequence)?\s*([A-Za-z*\s]+)```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1)
    sequence = re.sub(r"\s+", "", text).upper()
    return sequence or None


def _is_runaway_periodic(sequence: str, reference_length: int) -> bool:
    """Reject only severe over-generation whose tail is dominated by a short period."""
    if reference_length <= 0 or len(sequence) <= max(reference_length * 2, reference_length + 64):
        return False
    tail = sequence[reference_length:]
    if len(tail) < 64:
        return False
    for period in range(1, min(32, len(tail) // 4) + 1):
        compared = len(tail) - period
        agreement = sum(tail[index] == tail[index - period] for index in range(period, len(tail)))
        if compared and agreement / compared >= 0.95:
            return True
    return False
