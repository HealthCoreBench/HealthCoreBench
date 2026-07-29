"""RareBench adapter (rare-disease diagnosis from phenotypes, English).

Fixed data: ``15_RareBench/data/<dataset>.jsonl`` — one JSON object per line::

    {"Phenotype": ["HP:0000509", ...], "RareDisease": ["OMIM:191900", "ORPHA:575", ...],
     "Department": str | null}

plus ``15_RareBench/mapping/phenotype_mapping.json`` (HPO code -> name) and
``mapping/disease_mapping.json`` (OMIM/ORPHA/... code -> disease name).

Task: open-ended rare-disease diagnosis. The phenotype (HPO) codes are mapped to human-readable
symptom names and presented; the model names the most likely rare disease. The reference is the
first RareDisease code mapped to its disease name (all accepted disease names carried in metadata
for the LLM judge). Datasets are exposed as splits (``lirical`` / ``hms`` / ``mme`` / ``ramedis``);
the default ``test`` maps to ``ramedis``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.adapters.hle_med_exact import extract_short_answer
from healthcorebench.benchmarks.base import BaseBenchmarkAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
from healthcorebench.schemas.sample import EvaluationSample

# The reference is a disease name, optionally with slash-joined aliases (measured: median 4 words
# / 44 characters). The prompt already said "Give the specific disease name", but nothing pinned
# the answer to a locatable span, so the secondary token-F1 scored a median 673-character essay
# against it (measured 0.076 on the recorded run) and, worse, ``extract_short_answer``'s
# last-line fallback landed on the closing explanation instead of the diagnosis.
#
# The reasoning is kept — the LLM judge is the primary metric and reads the full reply — and the
# ASCII "Answer:" marker that ``extract_short_answer`` already recognizes is requested explicitly
# so the cross-check scores the diagnosis rather than the prose around it.
_FINAL_ANSWER_INSTRUCTION = (
    "Reason as much as you need, then end your reply with a final line in exactly this form:\n"
    "Answer: <the specific disease name only>"
)

_FILES = {"lirical": "LIRICAL.jsonl", "hms": "HMS.jsonl", "mme": "MME.jsonl", "ramedis": "RAMEDIS.jsonl"}
_DEFAULT = "ramedis"


class RareBenchDiagnosisAdapter(BaseBenchmarkAdapter):
    benchmark_name = "RareBench"
    benchmark_version = "1.0"
    adapter_version = "1.1"
    prompt_template_name = "open_ended"
    prompt_template_version = "1.1"

    def _subset(self) -> str:
        s = _DEFAULT if self.split == "test" else self.split
        if s not in _FILES:
            raise BenchmarkSplitNotFoundError(f"RareBench split must be 'test' or one of {sorted(_FILES)}; got '{self.split}'.")
        return s

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        data_files = ([directory / "data" / name for name in _FILES.values()]
                      if self.split == "test" else
                      [directory / "data" / _FILES[self._subset()]])
        return [
            *data_files,
            directory / "mapping" / "phenotype_mapping.json",
            directory / "mapping" / "disease_mapping.json",
        ]

    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        data_files, phe_map_file, dis_map_file = files[:-2], files[-2], files[-1]
        with open(phe_map_file, "r", encoding="utf-8") as fh:
            phe_map = json.load(fh)
        with open(dis_map_file, "r", encoding="utf-8") as fh:
            dis_map = json.load(fh)
        for data_file in data_files:
          rel = self.rel_path(data_file)
          with open(data_file, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                phenotypes = [phe_map.get(code, code) for code in (rec.get("Phenotype") or [])]
                disease_names = [dis_map.get(code) for code in (rec.get("RareDisease") or [])]
                disease_names = [d for d in disease_names if d]
                if not phenotypes or not disease_names:
                    continue
                yield {"record": rec, "phenotypes": phenotypes, "diseases": disease_names,
                       "source_file_rel": rel, "source_record_index": i}

    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        rec = raw_sample["record"]
        phenotypes: list[str] = raw_sample["phenotypes"]
        diseases: list[str] = raw_sample["diseases"]
        rel = raw_sample["source_file_rel"]
        rec_index = raw_sample["source_record_index"]
        reference = diseases[0]

        # RareBench encodes alternate disease names with "/" (and a case may list several
        # diseases). Expose every accepted name as an explicit alias, since reference_candidates
        # no longer auto-splits "/". This keeps the token-F1 secondary metric fair.
        aliases: list[str] = []
        for d in diseases:
            for part in [d, *str(d).split("/")]:
                p = part.strip()
                if p and p not in aliases:
                    aliases.append(p)

        source_id = f"{rel}:{rec_index}"
        content_hash = self.input_hash({"phe": phenotypes})
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
            input_hash=self.input_hash({"phenotypes": phenotypes}),
            reference_hash=self.reference_hash(reference),
            input_type="text",
            task_type="open_ended",
            component="Language",
            capability="Reasoning",
            specialty=self._subset(),
            language="en",
            modality="Text",
            answer_format="free_text",
            evaluation_metric="llm_judge",
            source_content={"phenotypes": phenotypes},
            reference_answer=reference,
            reference_answer_normalized=reference,
            reference_aliases=aliases,
            metadata={"accepted_diseases": diseases, "subset": self._subset()},
        )

    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        phenotypes = sample.source_content["phenotypes"]
        bullet = "\n".join(f"- {p}" for p in phenotypes)
        prompt = (
            "A patient presents with the following clinical features (phenotypes):\n"
            f"{bullet}\n\n"
            "What is the most likely rare disease? Give the specific disease name.\n\n"
            f"{_FINAL_ANSWER_INSTRUCTION}"
        )
        return [{"role": "user", "content": prompt}]

    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        return extract_short_answer(raw_response)
