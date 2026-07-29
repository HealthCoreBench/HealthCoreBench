"""MMedBench adapter (multilingual medical QA).

Fixed data: ``16_MMedBench/<Language>.jsonl`` — six languages (English, Chinese, French,
Japanese, Russian, Spanish), one JSON object per line::

    {"question": str, "answer": str, "options": {"A".."D"|...: str}, "meta_info": str,
     "answer_idx": "A".., "metamap_phrases": [...], "rationale": str, ...}

Task: single-choice. Same ``options``-dict + ``answer_idx`` shape as MedQA-USMLE, so the
loading/normalization logic is inherited. Languages are exposed as splits; default ``test``
maps to English.
"""

from __future__ import annotations

from pathlib import Path
import json

from healthcorebench.benchmarks.adapters.medqa_usmle_mcqa import MedQAUSMLEAdapter
from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError

# All six languages are read. French and Japanese store ``answer_idx`` as a list and are mostly
# multi-answer (301/622 and 39/199), which is why they were originally excluded — but their
# single-answer items (321 French, 160 Japanese) then belonged to no task at all once
# ``MMedBench/multiple_answer`` began requiring two or more gold letters. A one-element list is
# unwrapped here so those 481 items are scored as the single-choice questions they are.
_LANGS = {
    "test": ("English.jsonl", "en"),
    "en": ("English.jsonl", "en"),
    "zh": ("Chinese.jsonl", "zh"),
    "ru": ("Russian.jsonl", "ru"),
    "es": ("Spanish.jsonl", "es"),
    "fr": ("French.jsonl", "fr"),
    "ja": ("Japanese.jsonl", "ja"),
}
_SINGLE_FILES = {"English.jsonl": "en", "Chinese.jsonl": "zh",
                 "Russian.jsonl": "ru", "Spanish.jsonl": "es",
                 "French.jsonl": "fr", "Japanese.jsonl": "ja"}


class MMedBenchAdapter(MedQAUSMLEAdapter):
    benchmark_name = "MMedBench"
    benchmark_version = "1.0"
    adapter_version = "1.0"

    def _resolve(self) -> tuple[str, str]:
        if self.split not in _LANGS:
            raise BenchmarkSplitNotFoundError(
                f"MMedBench split must be one of {sorted(_LANGS)}; got '{self.split}'."
            )
        return _LANGS[self.split]

    @property
    def _lang(self) -> str:  # used by inherited normalize_sample / build_messages
        return self._resolve()[1]

    def discover_source_files(self) -> list[Path]:
        directory = self.get_benchmark_directory()
        if self.split == "test":
            return [directory / name for name in _SINGLE_FILES]
        return [directory / self._resolve()[0]]

    def load_raw_samples(self, files: list[Path]):
        for f in files:
            rel = self.rel_path(f)
            language = _SINGLE_FILES.get(f.name, self._resolve()[1])
            with open(f, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    answer_idx = record.get("answer_idx")
                    if isinstance(answer_idx, list):
                        # Multi-answer items belong to MMedBench/multiple_answer; a one-element
                        # list is a single-choice question stored in the list-valued schema.
                        if len(set(answer_idx)) != 1:
                            continue
                        record = dict(record, answer_idx=answer_idx[0])
                    yield {"record": record, "source_file_rel": rel,
                           "source_record_index": i, "language": language}
