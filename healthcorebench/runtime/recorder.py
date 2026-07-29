"""Append-only recorder for all run artifacts.

The recorder is the only component that writes the run's JSONL files and manifest. It does
not compute any business metric — it just persists records durably and in order per file.
Records are validated against their Pydantic schema before writing; a single non-core
optional field problem must not lose a whole record, so validation failures fall back to
writing the raw dict (with a serialization-error marker) rather than dropping data.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from healthcorebench.schemas.manifest import Manifest
from healthcorebench.utils.jsonl import append_jsonl, atomic_write_json
from healthcorebench.utils.timestamps import utc_now_iso


class Recorder:
    """Owns writes to a single run directory."""

    ATTEMPTS = "attempts.jsonl"
    RESULTS = "results.jsonl"
    JUDGMENTS = "judgments.jsonl"
    SAMPLES = "samples.jsonl"
    EVENTS = "events.jsonl"
    MANIFEST = "manifest.json"
    SUMMARY = "summary.json"

    def __init__(self, run_dir: str | Path, *, fsync: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _append(self, filename: str, record: Any) -> None:
        payload = self._to_payload(record)
        with self._lock:
            append_jsonl(self.run_dir / filename, payload, fsync=self._fsync)

    @staticmethod
    def _to_payload(record: Any) -> dict:
        if isinstance(record, BaseModel):
            return record.model_dump()
        if isinstance(record, dict):
            return record
        raise TypeError(f"Recorder cannot serialize {type(record)}")

    # ------------------------------------------------------------------ #
    def record_sample(self, sample) -> None:
        self._append(self.SAMPLES, sample)

    def record_attempt(self, attempt) -> None:
        self._append(self.ATTEMPTS, attempt)

    def record_result(self, result) -> None:
        self._append(self.RESULTS, result)

    def record_judgment(self, judgment) -> None:
        self._append(self.JUDGMENTS, judgment)

    def record_event(self, event_type: str, detail: dict | None = None) -> None:
        self._append(self.EVENTS, {
            "event_type": event_type,
            "timestamp": utc_now_iso(),
            "detail": detail or {},
        })

    # ------------------------------------------------------------------ #
    def write_manifest(self, manifest: Manifest | dict) -> None:
        payload = manifest.model_dump() if isinstance(manifest, BaseModel) else manifest
        with self._lock:
            atomic_write_json(self.run_dir / self.MANIFEST, payload, fsync=self._fsync)

    def read_manifest(self) -> dict | None:
        path = self.run_dir / self.MANIFEST
        if not path.exists():
            return None
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_manifest_status(self, status: str, extra: dict | None = None) -> None:
        """Atomically update run_status (and optional fields) without wiping the manifest."""
        current = self.read_manifest()
        if current is None:
            return
        current["run_status"] = status
        if extra:
            current.update(extra)
        with self._lock:
            atomic_write_json(self.run_dir / self.MANIFEST, current, fsync=self._fsync)

    def write_summary(self, summary) -> None:
        payload = summary.model_dump() if isinstance(summary, BaseModel) else summary
        with self._lock:
            atomic_write_json(self.run_dir / self.SUMMARY, payload, fsync=self._fsync)

    def path(self, filename: str) -> Path:
        return self.run_dir / filename
