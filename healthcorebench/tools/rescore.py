"""Re-score stored results without calling the evaluated model.

Reads the latest result per (sample, repeat) plus samples, applies a (possibly new)
evaluator, and appends fresh judgment records. Existing judgments are never mutated;
append-only, latest-wins on read.
"""

from __future__ import annotations

from pathlib import Path

from healthcorebench.evaluators import get_evaluator
from healthcorebench.utils.jsonl import read_jsonl, append_jsonl
from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.utils.jsonl import atomic_write_json


def rescore_run(run_dir: str | Path, *, evaluator_name: str | None = None,
                evaluator_version: str | None = None, regenerate_summary: bool = True,
                replace_primary: bool = True) -> dict:
    run_dir = Path(run_dir)
    manifest = _load_json(run_dir / "manifest.json")
    ev_name = evaluator_name or (manifest.get("full_config", {}).get("evaluation", {}) or {}).get("evaluator", "multiple_choice")
    if not ev_name or ev_name == "llm_judge":
        raise ValueError(
            "This run uses an LLM judge. Offline rescore requires an explicit rule-based "
            "--evaluator; judge rescoring needs a separately configured judge client."
        )
    evaluator = get_evaluator(ev_name)
    if evaluator_version:
        evaluator.evaluator_version = evaluator_version

    samples = {s["sample_id"]: s for s in read_jsonl(run_dir / "samples.jsonl")}
    results = read_jsonl(run_dir / "results.jsonl")
    existing_judgments = read_jsonl(run_dir / "judgments.jsonl")
    primary_names_by_result: dict[str, set[str]] = {}
    for existing in existing_judgments:
        if (existing.get("provider_metadata") or {}).get("primary_metric"):
            primary_names_by_result.setdefault(existing.get("result_id"), set()).add(
                existing.get("evaluator_name")
            )
    latest: dict[tuple, dict] = {}
    for r in results:
        latest[(r["sample_id"], r.get("sample_repeat_index", 0))] = r

    scored = 0
    skipped_incomplete = 0
    for r in latest.values():
        if r.get("status") != "success":
            continue
        if _scoring_ineligible(r, manifest):
            skipped_incomplete += 1
            continue
        sample = samples.get(r["sample_id"], {})
        judgment = evaluator.evaluate(r, sample)
        original_name = judgment.evaluator_name
        if not replace_primary and original_name in primary_names_by_result.get(r["result_id"], set()):
            # Judgments deduplicate by (result_id, evaluator_name). A same-name comparison must
            # have a separate identity or it silently replaces the tagged primary judgment.
            judgment.evaluator_name = f"{original_name}__secondary"
        judgment.provider_metadata.update({
            "primary_metric": replace_primary,
            "rescore_mode": "replace_primary" if replace_primary else "secondary_only",
            "base_evaluator_name": original_name,
        })
        append_jsonl(run_dir / "judgments.jsonl", judgment.model_dump())
        scored += 1
    if regenerate_summary:
        atomic_write_json(run_dir / "summary.json", summarize_run(run_dir).model_dump())
    return {"rescored": scored, "skipped_incomplete": skipped_incomplete,
            "evaluator": ev_name,
            "mode": "replace_primary" if replace_primary else "secondary_only"}


def _load_json(path: Path) -> dict:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scoring_ineligible(result: dict, manifest: dict) -> bool:
    generation = manifest.get("generation") or {}
    if not generation:
        generation = (manifest.get("full_config") or {}).get("generation") or {}
    return (
        result.get("finish_reason") == "length"
        and generation.get("length_finish_policy", "mark_incomplete") == "mark_incomplete"
    )
