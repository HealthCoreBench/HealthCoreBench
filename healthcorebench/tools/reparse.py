"""Re-parse stored raw responses without calling the model.

Reads results.jsonl, re-applies the adapter's parser to each stored ``raw_response``, and
appends updated result records (append-only — the original raw response is never mutated in
place; the latest record for a key supersedes earlier ones when read back). A new
``parser_version`` tag and parse timestamp are recorded.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from healthcorebench.benchmarks import get_adapter
from healthcorebench.utils.jsonl import read_jsonl, append_jsonl
from healthcorebench.utils.timestamps import utc_now_iso
from healthcorebench.aggregation.summarize import summarize_run
from healthcorebench.utils.jsonl import atomic_write_json
from healthcorebench.evaluators import get_evaluator
from healthcorebench.tools.evaluation_plan import resolved_extra_evaluators
from healthcorebench.version import DEFAULT_PARSER_VERSION


def reparse_run(run_dir: str | Path, *, parser_version: str | None = None,
                regenerate_summary: bool = False) -> dict:
    run_dir = Path(run_dir)
    manifest = _load_json(run_dir / "manifest.json")
    evaluation_cfg = (manifest.get("full_config") or {}).get("evaluation") or {}
    if regenerate_summary and evaluation_cfg.get("use_llm_judge"):
        raise ValueError(
            "Cannot safely regenerate a judge-based summary during offline reparse. "
            "Reparse first, then run judge-specific rescoring with an explicit judge client."
        )
    benchmark_name = manifest["benchmark"]["name"]
    adapter = get_adapter(benchmark_name, config=None)
    effective_parser_version = parser_version or DEFAULT_PARSER_VERSION

    samples = {s["sample_id"]: s for s in read_jsonl(run_dir / "samples.jsonl")}
    results = read_jsonl(run_dir / "results.jsonl")
    # Judgments deduplicate by (result_id, evaluator_name) and summarize._primary_judgment takes
    # the *last* tagged primary. Appending a second tagged primary under a different name would
    # therefore silently outrank the primary this run actually recorded.
    primary_names_by_result: dict[str, set[str]] = {}
    for existing in read_jsonl(run_dir / "judgments.jsonl"):
        if (existing.get("provider_metadata") or {}).get("primary_metric"):
            primary_names_by_result.setdefault(existing.get("result_id"), set()).add(
                existing.get("evaluator_name")
            )

    # latest success result per key
    latest: dict[tuple, dict] = {}
    for r in results:
        if r.get("status") == "success":
            latest[(r["sample_id"], r.get("sample_repeat_index", 0))] = r

    updated = 0
    skipped_incomplete = 0
    reparsed_records: list[tuple[dict, dict]] = []
    for r in latest.values():
        sample = samples.get(r["sample_id"])
        if sample is None:
            continue
        sample_obj = SimpleNamespace(**sample)
        parsed = adapter.parse_response(sample_obj, r.get("raw_response") or "")
        new_rec = dict(r)
        new_rec["parsed_answer"] = parsed
        new_rec["normalized_answer"] = parsed
        new_rec["parser_name"] = type(adapter).__name__
        new_rec["parser_version"] = effective_parser_version
        new_rec["parse_timestamp"] = utc_now_iso()
        new_rec["parsing_status"] = "success" if parsed is not None else "error"
        append_jsonl(run_dir / "results.jsonl", new_rec)
        reparsed_records.append((new_rec, sample))
        updated += 1

    if regenerate_summary:
        evaluator_name = evaluation_cfg.get("evaluator")
        if not evaluator_name:
            raise ValueError("No rule-based evaluator is recorded in the manifest.")
        evaluators = [get_evaluator(evaluator_name)]
        # The manifest may not list the secondaries the run actually applied, and a secondary
        # left un-regenerated keeps scoring the pre-reparse parsed_answer.
        evaluators.extend(
            get_evaluator(name) for name in resolved_extra_evaluators(manifest)
            if name != evaluator_name
        )
        for new_rec, sample in reparsed_records:
            if _scoring_ineligible(new_rec, manifest):
                skipped_incomplete += 1
                continue
            tagged = primary_names_by_result.get(new_rec.get("result_id"), set())
            for index, evaluator in enumerate(evaluators):
                judgment = evaluator.evaluate(new_rec, sample)
                original_name = judgment.evaluator_name
                # Only replace a primary that is already recorded under this evaluator's name.
                # A primary held by some other evaluator (an LLM judge, or an earlier
                # `rescore --evaluator`) stays authoritative.
                claims_primary = index == 0 and not (tagged and original_name not in tagged)
                if not claims_primary and original_name in tagged:
                    judgment.evaluator_name = f"{original_name}__secondary"
                judgment.provider_metadata["primary_metric"] = claims_primary
                append_jsonl(run_dir / "judgments.jsonl", judgment.model_dump())
        atomic_write_json(run_dir / "summary.json", summarize_run(run_dir).model_dump())

    return {
        "reparsed": updated,
        "skipped_incomplete": skipped_incomplete,
        "parser_version": effective_parser_version,
        "summary_regenerated": regenerate_summary,
    }


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
