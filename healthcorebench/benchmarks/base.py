"""Base benchmark adapter.

Splits data loading into explicit stages (directory → discover → validate → load raw →
normalize) and defines the message/parse/evaluate/aggregate hooks. Concrete adapters
implement the abstract methods; everything about model calls, concurrency, retries and
persistence lives outside the adapter.

The base also provides shared helpers: source-file discovery + hashing (for the manifest
and the effective benchmark revision), and stable sample-id construction.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from healthcorebench.benchmarks.errors import BenchmarkDataNotFoundError
from healthcorebench.benchmarks.registry import BenchmarkRegistryEntry
from healthcorebench.config import get_project_root
from healthcorebench.schemas.sample import EvaluationSample
from healthcorebench.utils.hashing import (
    hash_file,
    combined_files_hash,
    hash_json,
    stable_sample_id,
)
from healthcorebench.utils.timestamps import _to_iso
from healthcorebench.version import DEFAULT_ADAPTER_VERSION, DEFAULT_PROMPT_TEMPLATE_VERSION
from datetime import datetime, timezone


class BaseBenchmarkAdapter(ABC):
    benchmark_name: str = "base"
    benchmark_version: str | None = None
    adapter_version: str = DEFAULT_ADAPTER_VERSION
    prompt_template_name: str = "default"
    prompt_template_version: str = DEFAULT_PROMPT_TEMPLATE_VERSION

    # ------------------------------------------------------------------ #
    # data-license declarations (see spec §27)
    #
    # Read from ``benchmarks/data_licenses.py``, keyed on this benchmark's directory, so a
    # licence is stated once per corpus instead of once per task adapter. Corpora with no
    # entry get the permissive defaults below. These were plain ``True`` class attributes
    # that nothing ever overrode, which left ``_check_data_protection`` and the artifact
    # redaction in ``_record_samples`` as dead gates -- PhysioNet credentialed corpora were
    # shipped to a remote judge and written into ``samples.jsonl`` like any public corpus.
    #
    # Still overridable: a subclass assigning ``redistribution_allowed = False`` shadows the
    # property, since attribute lookup takes the first match along the MRO.
    # ------------------------------------------------------------------ #
    @property
    def redistribution_allowed(self) -> bool:
        """Whether the corpus may be sent to a third party (e.g. a remote judge endpoint)."""
        return self._data_license("redistribution_allowed")

    @property
    def store_full_input_allowed(self) -> bool:
        """Whether the source records may be persisted into the run directory."""
        return self._data_license("store_full_input_allowed")

    @property
    def store_reference_allowed(self) -> bool:
        """Whether the gold answers may be persisted into the run directory."""
        return self._data_license("store_reference_allowed")

    def _data_license(self, field: str) -> bool:
        from healthcorebench.benchmarks.data_licenses import license_for

        entry = getattr(self, "entry", None)
        declaration = license_for(getattr(entry, "benchmark_dir", "") or "")
        return True if declaration is None else getattr(declaration, field)

    def __init__(self, entry: BenchmarkRegistryEntry, config=None) -> None:
        self.entry = entry
        self.config = config
        self.split = getattr(getattr(config, "benchmark", None), "split", "test") if config else "test"
        self._combined_hash: str | None = None
        self._source_record_drops: Counter[str] = Counter()

    # ------------------------------------------------------------------ #
    # excluded source records
    # ------------------------------------------------------------------ #
    def drop_source_record(self, reason: str, count: int = 1) -> None:
        """Record that ``count`` source records were excluded from the task, and why.

        Adapters legitimately skip records (unparseable option blocks, missing images,
        the wrong question type for this task). Routing every such ``continue`` through
        this method is what keeps the exclusion visible: the totals reach the manifest
        and then the batch report, instead of silently shrinking the denominator so that
        a task looks fully covered when it is not.
        """
        if count > 0:
            self._drop_counter()[reason] += count

    def _drop_counter(self) -> Counter[str]:
        # Tolerate adapters that build state before ``BaseBenchmarkAdapter.__init__``.
        counter = getattr(self, "_source_record_drops", None)
        if counter is None:
            counter = Counter()
            self._source_record_drops = counter
        return counter

    @property
    def num_source_records_dropped(self) -> int:
        return sum(self._drop_counter().values())

    @property
    def source_record_drop_reasons(self) -> dict[str, int]:
        return dict(sorted(self._drop_counter().items()))

    # ------------------------------------------------------------------ #
    # directory / file discovery
    # ------------------------------------------------------------------ #
    def get_benchmark_directory(self) -> Path:
        # Debug override (non-standard runs only).
        override = getattr(getattr(self.config, "benchmark", None), "debug_data_path_override", None) if self.config else None
        if override:
            d = Path(override)
        else:
            d = self.entry.directory()
        if not d.exists():
            raise BenchmarkDataNotFoundError(
                f"Expected benchmark data directory:\n{d}\n"
                f"Check the project's fixed benchmark files; automatic download is disabled."
            )
        return d

    @abstractmethod
    def discover_source_files(self) -> list[Path]:
        """Return the fixed files this adapter reads for the configured split."""

    def validate_source_files(self, files: list[Path]) -> None:
        """Default: ensure at least one file exists and is readable."""
        if not files:
            raise BenchmarkDataNotFoundError(
                f"No source files discovered for benchmark '{self.benchmark_name}' "
                f"(split={self.split}) under {self.get_benchmark_directory()}."
            )
        for f in files:
            if not f.exists():
                raise BenchmarkDataNotFoundError(f"Source file missing: {f}")
            if not os.access(f, os.R_OK):
                raise BenchmarkDataNotFoundError(f"Source file not readable: {f}")

    # ------------------------------------------------------------------ #
    # source-file manifest + effective revision
    # ------------------------------------------------------------------ #
    def source_file_manifest(self, files: list[Path]) -> tuple[list[dict], str]:
        """Return per-file metadata entries and a deterministic combined hash."""
        root = get_project_root()
        entries = []
        pairs = []
        for f in files:
            digest = hash_file(f)
            try:
                rel = str(f.resolve().relative_to(root))
            except ValueError:
                rel = str(f)
            stat = f.stat()
            entries.append({
                "relative_path": rel,
                "file_name": f.name,
                "file_size_bytes": stat.st_size,
                "sha256": digest,
                "modified_time": _to_iso(datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)),
                "record_count": self.count_source_records(f),
            })
            pairs.append((rel, digest))
        combined = combined_files_hash(pairs)
        self._combined_hash = combined
        return entries, combined

    # Above this size a second full parse of a ``.json`` file costs more than the
    # record count is worth; those entries report ``None`` instead.
    MAX_RECORD_COUNT_PARSE_BYTES = 64 * 1024 * 1024

    def count_source_records(self, path: Path) -> int | None:
        """Records physically present in a source file, or ``None`` if not countable.

        This is the file's own record count, deliberately independent of how many the
        adapter keeps — the difference between the two is what
        ``num_source_records_dropped`` reports. Line-delimited formats are counted by
        streaming; a JSON document is parsed only when it is small enough that the
        second pass is free.
        """
        suffix = path.suffix.lower()
        try:
            if suffix in {".jsonl", ".ndjson"}:
                with open(path, "r", encoding="utf-8") as handle:
                    return sum(1 for line in handle if line.strip())
            if suffix in {".csv", ".tsv"}:
                with open(path, "r", encoding="utf-8", newline="") as handle:
                    rows = sum(1 for line in handle if line.strip())
                return max(rows - 1, 0)  # drop the header row
            if suffix == ".json":
                if path.stat().st_size > self.MAX_RECORD_COUNT_PARSE_BYTES:
                    return None
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    return len(payload)
                if isinstance(payload, dict):
                    # A single split wrapped in a dict, e.g. {"questions": [...]}.
                    lists = [value for value in payload.values() if isinstance(value, list)]
                    if len(lists) == 1:
                        return len(lists[0])
                    return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return None

    def effective_revision(self) -> str | None:
        if self._combined_hash is None:
            return None
        return self._combined_hash

    # ------------------------------------------------------------------ #
    # raw loading + normalization
    # ------------------------------------------------------------------ #
    @abstractmethod
    def load_raw_samples(self, files: list[Path]) -> Iterable[dict]:
        """Yield raw records (with enough context to build a stable id) in deterministic order."""

    @abstractmethod
    def normalize_sample(self, raw_sample: dict, sample_index: int) -> EvaluationSample:
        """Convert a raw record into a framework ``EvaluationSample``."""

    # ------------------------------------------------------------------ #
    # message / parse / evaluate hooks (implemented by concrete adapters)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_messages(self, sample: EvaluationSample) -> list[dict]:
        """Build logical messages (see clients.messages) for a sample."""

    @abstractmethod
    def parse_response(self, sample: EvaluationSample, raw_response: str) -> Any:
        """Extract the answer span from the raw model output (no scoring here)."""

    def aggregate(self, results: Iterable[dict], judgments: Iterable[dict]) -> dict:
        """Optional adapter-specific aggregation. Default: none (generic aggregation used)."""
        return {}

    def output_token_budget_for_format(self, answer_format: str | None) -> int | None:
        """Use the configured generation budget regardless of answer format."""
        return getattr(getattr(self.config, "generation", None), "max_tokens", None)

    def max_output_tokens(self, sample: EvaluationSample) -> int | None:
        """Allow adapters to apply a deterministic per-sample output budget."""
        return self.output_token_budget_for_format(sample.answer_format)

    # ------------------------------------------------------------------ #
    # shared helpers
    # ------------------------------------------------------------------ #
    def make_sample_id(self, *, source_file_rel: str, source_sample_id: str | None, content_hash: str | None) -> str:
        return stable_sample_id(
            benchmark_name=self.benchmark_name,
            benchmark_revision=self.effective_revision(),
            split=self.split,
            source_file=source_file_rel,
            source_sample_id=source_sample_id,
            content_hash=content_hash,
        )

    def rel_path(self, path: Path) -> str:
        root = get_project_root()
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return str(path)

    def input_hash(self, payload: Any) -> str:
        return hash_json(payload)

    def reference_hash(self, reference: Any) -> str:
        return hash_json(reference)
