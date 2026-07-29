"""Validate a run directory's integrity.

Checks: JSONL parseability, duplicate logical keys among latest results, usage consistency
(total ~= prompt + completion when all present), every judgment references a known result,
every result references a known sample, no NaN/Inf, and summary consistency vs a fresh
recomputation. Returns a report of issues (empty = valid).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.evaluators import get_evaluator
from healthcorebench.tools.evaluation_plan import resolved_extra_evaluators


def validate_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    issues: list[str] = []
    warnings: list[str] = []

    samples = _read_strict_jsonl(run_dir / "samples.jsonl", issues, warnings)
    results = _read_strict_jsonl(run_dir / "results.jsonl", issues, warnings)
    judgments = _read_strict_jsonl(run_dir / "judgments.jsonl", issues, warnings)
    attempts = _read_strict_jsonl(run_dir / "attempts.jsonl", issues, warnings)

    sample_ids = {s.get("sample_id") for s in samples}
    result_ids = {r.get("result_id") for r in results}
    attempts_by_id = {a.get("attempt_id"): a for a in attempts if a.get("attempt_id")}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            issues.append(f"manifest.json is not valid JSON: {exc}")
            manifest = {}
    else:
        issues.append("manifest.json is missing")
        manifest = {}

    authoritative_run_id = manifest.get("run_id")
    if authoritative_run_id:
        for name, rows in (("results", results), ("judgments", judgments),
                           ("attempts", attempts)):
            mismatched = [row for row in rows
                          if row.get("run_id") not in (None, authoritative_run_id)]
            if mismatched:
                issues.append(
                    f"{name} contain {len(mismatched)} record(s) with run_id different "
                    f"from manifest {authoritative_run_id}"
                )

    # Append-only logs may legitimately re-emit a record with the same result_id
    # (reparse/rescore/resume). Duplicates are expected: latest-wins on read. We flag only a
    # genuinely inconsistent case: two records sharing a result_id but disagreeing on their
    # (sample_id, sample_repeat_index) identity.
    identity_by_rid: dict = {}
    for r in results:
        rid = r.get("result_id")
        ident = (r.get("sample_id"), r.get("sample_repeat_index", 0))
        if rid in identity_by_rid and identity_by_rid[rid] != ident:
            issues.append(f"result_id {rid} reused for different samples: {identity_by_rid[rid]} vs {ident}")
        identity_by_rid[rid] = ident

    # result -> sample linkage
    for r in results:
        if r.get("sample_id") not in sample_ids and samples:
            issues.append(f"result {r.get('result_id')} references unknown sample {r.get('sample_id')}")
        if r.get("status") == "success":
            attempt_id = r.get("successful_attempt_id")
            attempt = attempts_by_id.get(attempt_id)
            if attempt_id is None:
                issues.append(f"successful result {r.get('result_id')} has no successful_attempt_id")
            elif attempt is None:
                issues.append(f"successful result {r.get('result_id')} references unknown attempt {attempt_id}")
            elif (attempt.get("status") != "success" or attempt.get("request_purpose") != "model_inference"
                  or attempt.get("sample_id") != r.get("sample_id")
                  or attempt.get("sample_repeat_index", 0) != r.get("sample_repeat_index", 0)):
                issues.append(f"successful result {r.get('result_id')} has incompatible attempt {attempt_id}")

    # judgment -> result linkage
    for j in judgments:
        if j.get("result_id") not in result_ids and results:
            issues.append(f"judgment {j.get('judgment_id')} references unknown result {j.get('result_id')}")

    latest_results: dict[tuple, dict] = {}
    for result in results:
        latest_results[(result.get("sample_id"), result.get("sample_repeat_index", 0))] = result
    repeats = int(((manifest.get("generation") or {}).get("n") or 1))
    if manifest.get("run_status") in {"completed", "completed_with_errors"}:
        expected = len(samples) * repeats
        if len(latest_results) != expected:
            issues.append(f"logical result coverage {len(latest_results)} != expected {expected}")

    dedup_judgments: dict[tuple, dict] = {}
    for judgment in judgments:
        dedup_judgments[(judgment.get("result_id"), judgment.get("evaluator_name"))] = judgment
    evaluation_cfg = (manifest.get("full_config") or {}).get("evaluation") or {}
    expected_evaluators: list[str] = []
    if evaluation_cfg.get("use_llm_judge"):
        expected_evaluators.append("llm_judge")
    elif evaluation_cfg.get("evaluator"):
        try:
            expected_evaluators.append(get_evaluator(evaluation_cfg["evaluator"]).evaluator_name)
        except ValueError as exc:
            issues.append(str(exc))
    # The manifest records only the configured secondaries. Resolve the set the run actually
    # applied, otherwise a missing secondary judgment cannot be detected at all.
    for name in resolved_extra_evaluators(manifest):
        try:
            expected_evaluators.append(get_evaluator(name).evaluator_name)
        except ValueError as exc:
            issues.append(str(exc))
    for result in latest_results.values():
        if result.get("status") != "success":
            continue
        length_policy = ((manifest.get("generation") or {}).get("length_finish_policy")
                         or "mark_incomplete")
        if result.get("finish_reason") == "length" and length_policy == "mark_incomplete":
            continue
        rid = result.get("result_id")
        for evaluator_name in dict.fromkeys(expected_evaluators):
            judgment = dedup_judgments.get((rid, evaluator_name))
            if judgment is None:
                issues.append(f"successful result {rid} missing judgment {evaluator_name}")

    # usage self-consistency (only when all three present)
    for a in attempts:
        u = a.get("usage") or {}
        pt, ct, tt = u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens")
        if pt is not None and ct is not None and tt is not None:
            if abs((pt + ct) - tt) > max(1, int(0.05 * tt)):
                issues.append(f"attempt {a.get('attempt_id')} usage inconsistent: {pt}+{ct}!={tt}")

    # NaN/Inf scan
    for name, rows in (("results", results), ("judgments", judgments), ("attempts", attempts)):
        for row in rows:
            if _has_nonfinite(row):
                issues.append(f"{name} record contains NaN/Inf: {row.get('result_id') or row.get('judgment_id') or row.get('attempt_id')}")

    # summary consistency
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            stored = json.loads(summary_path.read_text(encoding="utf-8"))
            fresh = summarize_run(run_dir).model_dump()
            if stored.get("summary_code_version") != fresh.get("summary_code_version"):
                warnings.append(
                    "summary.json was generated by aggregation code "
                    f"{stored.get('summary_code_version') or 'unknown'}; current code is "
                    f"{fresh.get('summary_code_version')}. Re-run summarize to refresh it."
                )
            else:
                for field in ("run_id", "benchmark_name", "counts", "metrics", "tokens",
                              "timing", "groups", "metrics_by_evaluator", "source_files"):
                    if stored.get(field) != fresh.get(field):
                        issues.append(f"summary {field} is stale or inconsistent")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            issues.append(f"summary.json is not valid JSON: {exc}")
    elif results:
        issues.append("summary.json is missing")

    return {"valid": not issues, "issues": issues, "warnings": warnings,
            "counts": {"samples": len(samples), "results": len(results),
                       "judgments": len(judgments), "attempts": len(attempts)}}


def _read_strict_jsonl(path: Path, issues: list[str], warnings: list[str]) -> list[dict]:
    """Read JSONL, tolerating only one unterminated malformed final line."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    for index, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_unterminated_final = index == len(lines) and not raw_line.endswith((b"\n", b"\r"))
            message = f"{path.name}:{index} is invalid JSON: {exc}"
            if is_unterminated_final:
                warnings.append(message + " (ignored torn final line)")
            else:
                issues.append(message)
    return rows


def _has_nonfinite(obj) -> bool:
    if isinstance(obj, float):
        return math.isnan(obj) or math.isinf(obj)
    if isinstance(obj, dict):
        return any(_has_nonfinite(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_nonfinite(v) for v in obj)
    return False
