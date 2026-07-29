"""VivaBench adapter (clinical vignette → diagnosis, free-text, English).

Fixed data: ``68_VivaBench/dataset/{pubmed_reviewed,dataset_generated}.csv`` — columns include
``uid``, ``source``, ``vignette`` (case presentation), ``specialty_group``, ``diagnosis``
(Python-literal list of accepted diagnoses, primary first), ``differentials``, ``clinicalcase``.

Task: open-ended diagnosis. Given the vignette, name the most likely diagnosis; the primary
(first) accepted diagnosis is the reference, with the full accepted list carried in metadata for
the LLM judge. VivaBench's full interactive viva format (progressive case, differentials) is not
modeled here — this is the single-shot diagnosis task. The two CSVs are exposed as splits
``pubmed_reviewed`` (default) and ``dataset_generated``.
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.adapters.hle_med_exact import extract_short_answer
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

# The reference is a diagnosis phrase (measured: median 4 words / 40 characters, max 13 words),
# while the recorded replies are a median 1,382 characters — so the secondary token-F1 was a
# length ratio (measured 0.075) rather than a diagnostic signal. Reasoning is kept (the LLM judge
# is the primary metric and reads the full reply); only a locatable final line is added, using the
# ASCII "Answer:" marker ``extract_short_answer`` already recognizes.
_FINAL_ANSWER_INSTRUCTION = (
    "Reason as much as you need, then end your reply with a final line in exactly this form:\n"
    "Answer: <the single most likely diagnosis, name only>"
)

_SPLITS = {"pubmed_reviewed": "pubmed_reviewed.csv", "dataset_generated": "dataset_generated.csv"}
# ``test`` resolves to the reviewed split, which is what this module's docstring has always
# claimed and what upstream marks as its default config. ``dataset_generated.csv`` is the raw
# pre-review superset: 1,952 rows whose uids contain all 990 reviewed ones, plus 162 unreviewed
# PubMed cases and 800 MedQA-derived ones, and 2 uids appear twice in it. Defaulting to that file
# silently scored the unreviewed generation output instead of the benchmark.
_DEFAULT = "pubmed_reviewed"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        try:
            d = ast.literal_eval(value)
            if isinstance(d, list):
                return [str(x).strip() for x in d if str(x).strip()]
        except (ValueError, SyntaxError):
            return [value.strip()]
    return []


class VivaBenchDiagnosisAdapter(BaseBenchmarkAdapter):
    benchmark_name = "VivaBench"
    benchmark_version = "1.0"
    adapter_version = "1.2"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.1"

    def _split(self) -> str:
        s = _DEFAULT if self.split == "test" else self.split
        if s not in _SPLITS:
            raise BenchmarkSplitNotFoundError(f"VivaBench split must be 'test'/'pubmed_reviewed'/'dataset_generated'; got '{self.split}'.")
        return s

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        return [directory / "dataset" / _SPLITS[self._split()]]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        f = files[0]
        rel = self.rel_path(f)
        with open(f, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for i, rec in enumerate(rows):
            vignette = str(rec.get("vignette") or "").strip()
            diagnoses = _as_list(rec.get("diagnosis"))
            # Guards only: both CSVs are complete on these two columns (0 of 990 and 0 of 1,952
            # rows affected). Reported anyway so a future data refresh cannot shrink the split
            # without the count showing up in the manifest.
            if not vignette:
                self.drop_source_record("empty_vignette")
                continue
            if not diagnoses:
                self.drop_source_record("no_reference_diagnosis")
                continue
            yield {"record": rec, "vignette": vignette, "diagnoses": diagnoses,
                   "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        vignette = raw_sample["vignette"]
        diagnoses = raw_sample["diagnoses"]
        reference = diagnoses[0]

        source_id = str(rec.get("uid") or f"{rel}:{rec_index}")
        content_hash = self.input_hash({"v": vignette})
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
            source_record_hash=self.input_hash(dict(rec)),
            input_hash=self.input_hash({"vignette": vignette}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=rec.get("specialty_group"),
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"vignette": vignette},
            reference_answer=reference,
            reference_answer_normalized=reference,
            reference_aliases=diagnoses,
            metadata={"accepted_diagnoses": diagnoses, "specialty_group": rec.get("specialty_group"),
                      "split": self._split()},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        vignette = sample.source_content["vignette"]
        prompt = (f"{vignette}\n\nBased on the case above, what is the most likely diagnosis?"
                  f"\n\n{_FINAL_ANSWER_INSTRUCTION}")
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return extract_short_answer(raw_response)
