"""Low-level, dependency-free utilities used across the framework.

Nothing in this subpackage imports benchmark, client, or runtime code: these are the
foundation everything else builds on (hashing, canonical JSON, timestamps, JSONL IO,
git/environment capture).
"""

from healthcorebench.utils.hashing import (
    canonical_json,
    sha256_hex,
    blake2b_hex,
    hash_json,
    hash_bytes,
    hash_file,
    combined_files_hash,
    stable_sample_id,
    SAMPLE_ID_NAMESPACE,
)
from healthcorebench.utils.timestamps import utc_now_iso, parse_iso, duration_seconds
from healthcorebench.utils.jsonl import (
    append_jsonl,
    read_jsonl,
    iter_jsonl,
    atomic_write_json,
    count_jsonl,
)
from healthcorebench.utils.git import collect_git_info
from healthcorebench.utils.environment import collect_environment_info

__all__ = [
    "canonical_json",
    "sha256_hex",
    "blake2b_hex",
    "hash_json",
    "hash_bytes",
    "hash_file",
    "combined_files_hash",
    "stable_sample_id",
    "SAMPLE_ID_NAMESPACE",
    "utc_now_iso",
    "parse_iso",
    "duration_seconds",
    "append_jsonl",
    "read_jsonl",
    "iter_jsonl",
    "atomic_write_json",
    "count_jsonl",
    "collect_git_info",
    "collect_environment_info",
]
