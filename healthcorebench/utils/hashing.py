"""Stable, cross-process hashing and canonical JSON.

Design constraints (from the refactor spec):

* Never use Python's built-in ``hash()`` — it is salted per-process and unstable.
* JSON hashing must use sorted keys and a fixed Unicode encoding so the same logical
  content always hashes identically, regardless of dict insertion order or platform.
* ``sample_id`` must be globally unique and stable across models and runs, and must not
  depend on transient file ordering. We use UUIDv5 over canonical JSON of the sample's
  identity fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Iterable

# Fixed namespace for all HealthCoreBench sample IDs. Never change this value: doing so would
# alter every generated sample_id and break cross-run alignment. Derived deterministically
# from a fixed URL so it is reproducible and documented.
SAMPLE_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "urn:healthcorebench:sample-id-namespace")


def replace_non_finite(obj: Any) -> Any:
    """Recursively replace NaN / ±Infinity floats with ``None``.

    Source datasets (often exported via pandas) can contain NaN for missing values. NaN /
    Infinity are not valid JSON, so they must never enter a hash or a log line. Rather than
    crash on real-world data, we canonicalize them to ``null`` at the serialization boundary —
    they carry no meaning in an identity hash and represent "no value" in a record.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: replace_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [replace_non_finite(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to a canonical JSON string.

    Uses sorted keys, no insignificant whitespace, ``ensure_ascii=False`` (so Unicode is
    represented consistently as UTF-8 text rather than ``\\uXXXX`` escapes). Non-finite floats
    (NaN / Infinity) are canonicalized to ``null`` first so hashing is total over real data
    while such values never enter a hash; ``allow_nan=False`` then guards against any missed path.
    """
    return json.dumps(
        replace_non_finite(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    """SHA256 of a UTF-8 string, returned as a ``sha256:``-prefixed hex digest."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def blake2b_hex(text: str) -> str:
    """BLAKE2b of a UTF-8 string, returned as a ``blake2b:``-prefixed hex digest."""
    return "blake2b:" + hashlib.blake2b(text.encode("utf-8")).hexdigest()


def hash_json(obj: Any) -> str:
    """SHA256 over the canonical JSON representation of ``obj``."""
    return sha256_hex(canonical_json(obj))


def hash_bytes(data: bytes) -> str:
    """SHA256 of raw bytes (e.g. media content), ``sha256:``-prefixed."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA256 without loading it fully into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def combined_files_hash(entries: Iterable[tuple[str, str]]) -> str:
    """Deterministic combined hash over multiple files.

    ``entries`` is an iterable of ``(relative_path, file_sha256)`` pairs. The combination
    is order-independent of the filesystem: entries are sorted by normalized relative path
    before hashing, so the same set of files always yields the same combined hash
    regardless of directory traversal order.
    """
    normalized = sorted(
        (str(rel).replace("\\", "/"), digest) for rel, digest in entries
    )
    return hash_json(normalized)


def stable_sample_id(
    *,
    benchmark_name: str,
    benchmark_revision: str | None,
    split: str,
    source_file: str | None,
    source_sample_id: str | None,
    content_hash: str | None,
) -> str:
    """Compute a stable ``sample_id`` as a ``urn:healthcorebench:sample:<uuid>``.

    The identity payload deliberately excludes ``run_id`` and any prompt-template detail so
    the id is stable across runs and across prompt changes. It includes the benchmark
    revision, split, source file and either the source's own stable id or a content hash,
    so inserting a row elsewhere in a file does not renumber unrelated samples.
    """
    payload = {
        "benchmark_name": benchmark_name,
        "benchmark_revision": benchmark_revision,
        "split": split,
        "source_file": source_file,
        "source_sample_id": source_sample_id,
        "content_hash": content_hash,
    }
    digest = uuid.uuid5(SAMPLE_ID_NAMESPACE, canonical_json(payload))
    return f"urn:healthcorebench:sample:{digest}"
