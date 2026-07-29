"""Append-only JSONL IO and atomic JSON writes.

Durability model:

* Records are appended one line at a time and flushed immediately (optionally ``fsync``),
  so a crash never loses an already-written record and never leaves a torn line at the end
  that a reader can't skip.
* Whole-file JSON (manifest, summary) is written to a ``.tmp`` sibling then atomically
  renamed, so readers never observe a partially written file.
* JSON is emitted with ``allow_nan=False`` — NaN / Infinity are not valid JSON and must
  never enter the logs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from healthcorebench.utils.hashing import replace_non_finite


def _dump(obj: Any) -> str:
    return json.dumps(replace_non_finite(obj), ensure_ascii=False, allow_nan=False)


def append_jsonl(path: str | Path, record: Any, *, fsync: bool = False) -> None:
    """Append a single record as one JSON line, flushing immediately.

    When ``fsync`` is true the file descriptor is also fsync'd for stronger durability at
    the cost of throughput.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _dump(record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def iter_jsonl(path: str | Path, *, skip_errors: bool = True) -> Iterator[dict]:
    """Yield records from a JSONL file.

    A trailing torn line (e.g. from a crash mid-write) is skipped when ``skip_errors`` is
    true rather than raising, so partially written logs remain readable.
    """
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if skip_errors:
                    continue
                raise


def read_jsonl(path: str | Path, *, skip_errors: bool = True) -> list[dict]:
    """Materialize all records from a JSONL file into a list."""
    return list(iter_jsonl(path, skip_errors=skip_errors))


def count_jsonl(path: str | Path, *, skip_errors: bool = True) -> int:
    """Count valid records in a JSONL file without materializing them."""
    return sum(1 for _ in iter_jsonl(path, skip_errors=skip_errors))


def atomic_write_json(path: str | Path, obj: Any, *, indent: int = 2, fsync: bool = False) -> None:
    """Write JSON to ``path`` atomically via a temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(replace_non_finite(obj), f, ensure_ascii=False, allow_nan=False, indent=indent)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp, path)
