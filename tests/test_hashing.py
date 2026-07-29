"""Unit tests: canonical JSON, hashing, and cross-process-stable sample IDs."""

import subprocess
import sys

from healthcorebench.utils.hashing import (
    canonical_json,
    hash_json,
    stable_sample_id,
    combined_files_hash,
)


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_sanitizes_non_finite():
    # NaN / ±Infinity are canonicalized to null (never crash on real pandas-exported data,
    # never emit invalid JSON). Finite floats are untouched.
    import json
    out = json.loads(canonical_json({"x": float("nan"), "y": float("inf"),
                                     "z": float("-inf"), "ok": 1.5, "s": "t"}))
    assert out == {"x": None, "y": None, "z": None, "ok": 1.5, "s": "t"}
    # a record with NaN and the same record with the NaN replaced by null hash identically
    assert canonical_json({"a": float("nan")}) == canonical_json({"a": None})


def test_hash_json_stable():
    h1 = hash_json({"a": [1, 2, 3], "b": "x"})
    h2 = hash_json({"b": "x", "a": [1, 2, 3]})
    assert h1 == h2 and h1.startswith("sha256:")


def test_sample_id_stable_and_prefixed():
    kwargs = dict(
        benchmark_name="MMLU", benchmark_revision="sha256:rev", split="test",
        source_file="f.json", source_sample_id="42", content_hash=None,
    )
    a = stable_sample_id(**kwargs)
    b = stable_sample_id(**kwargs)
    assert a == b
    assert a.startswith("urn:healthcorebench:sample:")


def test_sample_id_changes_with_identity():
    base = dict(
        benchmark_name="MMLU", benchmark_revision="sha256:rev", split="test",
        source_file="f.json", source_sample_id="42", content_hash=None,
    )
    other = {**base, "source_sample_id": "43"}
    assert stable_sample_id(**base) != stable_sample_id(**other)


def test_sample_id_cross_process_stable():
    """The whole point of not using built-in hash(): stable across interpreters."""
    code = (
        "from healthcorebench.utils.hashing import stable_sample_id;"
        "print(stable_sample_id(benchmark_name='MMLU',benchmark_revision='r',"
        "split='test',source_file='f',source_sample_id='42',content_hash=None))"
    )
    outs = set()
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        outs.add(out.stdout.strip())
    assert len(outs) == 1


def test_combined_files_hash_order_independent():
    a = combined_files_hash([("b.json", "sha256:2"), ("a.json", "sha256:1")])
    b = combined_files_hash([("a.json", "sha256:1"), ("b.json", "sha256:2")])
    assert a == b
