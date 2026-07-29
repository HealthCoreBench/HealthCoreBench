"""Migrate legacy HealthCoreBench results into the new run layout (best-effort).

Old outputs (``results.json`` / ``metrics.json`` / ``total_results.json``) held per-sample
records with a ``response`` and often a ``correct`` flag, but no token/latency/model
identity. This converts what exists into new-style ``results.jsonl`` + ``judgments.jsonl``,
marking each record ``provenance="legacy_import"`` with ``legacy_schema=true`` and listing
``missing_fields``. Never fabricates tokens or model version.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from healthcorebench.utils.jsonl import append_jsonl, atomic_write_json
from healthcorebench.utils.timestamps import utc_now_iso

_MISSING = ["prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds",
            "actual_model_version", "system_fingerprint"]


def migrate_legacy(legacy_dir: str | Path, output_dir: str | Path) -> dict:
    legacy_dir = Path(legacy_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _find_legacy_records(legacy_dir)
    run_id = f"legacy_{uuid.uuid4().hex[:8]}"
    n_results = n_judg = 0
    for i, rec in enumerate(records):
        sid = str(rec.get("id") or rec.get("sample_id") or i)
        rid = f"legacy_res_{i}"
        response = rec.get("response") or rec.get("prediction")
        result = {
            "schema_version": "1.0", "result_id": rid, "run_id": run_id, "sample_id": sid,
            "sample_repeat_index": 0, "benchmark_name": rec.get("benchmark_name") or legacy_dir.name,
            "raw_response": response, "parsed_answer": rec.get("parsed") or rec.get("extracted"),
            "reference_answer": rec.get("answer") or rec.get("reference"),
            "status": "success" if response is not None else "error",
            "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
            "latency_seconds": None,
            "provenance": "legacy_import", "legacy_schema": True, "missing_fields": _MISSING,
            "timestamp": utc_now_iso(),
        }
        append_jsonl(output_dir / "results.jsonl", result)
        n_results += 1
        if "correct" in rec:
            judgment = {
                "schema_version": "1.0", "judgment_id": f"legacy_jdg_{i}", "run_id": run_id,
                "result_id": rid, "sample_id": sid, "evaluator_type": "rule_based",
                "evaluator_name": "legacy_correct", "evaluator_version": "legacy",
                "raw_score": 1.0 if rec["correct"] else 0.0,
                "normalized_score": 1.0 if rec["correct"] else 0.0,
                "is_correct": bool(rec["correct"]), "evaluation_status": "success",
                "provenance": "legacy_import", "timestamp": utc_now_iso(),
            }
            append_jsonl(output_dir / "judgments.jsonl", judgment)
            n_judg += 1

    manifest = {
        "schema_version": "1.0", "run_id": run_id, "run_name": f"legacy_import_{legacy_dir.name}",
        "run_status": "completed", "provenance": "legacy_import", "legacy_schema": True,
        "benchmark": {"name": legacy_dir.name}, "model": {}, "missing_fields": _MISSING,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return {"results": n_results, "judgments": n_judg, "output_dir": str(output_dir)}


def _find_legacy_records(legacy_dir: Path) -> list[dict]:
    for name in ("results.json", "total_results.json"):
        p = legacy_dir / name
        if p.exists():
            data = json.loads(p.read_text())
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # total_results.json may be {benchmark: {...}} or {id: record}
                for v in data.values():
                    if isinstance(v, list):
                        return v
                return [ {"id": k, **(v if isinstance(v, dict) else {"response": v})} for k, v in data.items() ]
    return []
