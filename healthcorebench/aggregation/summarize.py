"""Recompute a run summary purely from results.jsonl + judgments.jsonl.

Denominator policy: by default only successful, scored results count toward the score
denominator; failed inferences and unscored results are counted separately and excluded, so
API failures never depress the metric as if they were wrong answers. The policy is recorded
in the summary.

The summary records the hashes of the source JSONL files it was computed from so a
verifier can confirm it corresponds to the current logs.
"""

from __future__ import annotations

import statistics
import json
from pathlib import Path

from healthcorebench.aggregation.confidence_interval import (
    bootstrap_clustered_weighted_mean_interval,
    bootstrap_mean_interval,
    bootstrap_weighted_mean_interval,
    wilson_interval,
)
from healthcorebench.aggregation.grouping import group_scores
from healthcorebench.schemas.summary import Summary, Counts, Metrics
from healthcorebench.utils.hashing import hash_file
from healthcorebench.utils.jsonl import read_jsonl
from healthcorebench.utils.timestamps import utc_now_iso

_GROUP_DIMS = {
    "score_by_difficulty": "difficulty",
    "score_by_capability": "capability",
    "score_by_task": "task_type",
    "score_by_specialty": "specialty",
    "score_by_language": "language",
    "score_by_modality": "modality",
    "score_by_component": "component",
}

# Dimensions whose combination identifies a benchmark subset for the macro average.
# MedConceptsQA is the case that forced this: one run of ``MedConceptsQA/mcqa`` loads all 15
# (vocabulary x difficulty) subsets at once and the benchmark's own protocol averages the 15
# subset accuracies. The subsets are not the same size, so the pooled micro mean is a different
# number, tilted towards the largest vocabulary — and ``macro_score`` used to be hardcoded to
# ``None``, so that second number was never available at all.
_MACRO_DIMS = ("specialty", "difficulty")


def _macro_score(scored_rows: list[dict]) -> float | None:
    """Unweighted mean of the per-subset means, or ``None`` when subsets are not well defined.

    A macro average is only honest when every scored sample belongs to a declared subset: with
    some samples untagged, averaging the tagged ones would quietly drop the rest and bucketing
    them together would invent a subset. So an untagged row makes this return ``None``, as does
    a single subset (where the macro is just the micro and reporting both would suggest the run
    said something it did not).
    """
    subsets: dict[tuple, list[tuple[float, float]]] = {}
    for row in scored_rows:
        result = row["result"]
        key = tuple(result.get(dim) for dim in _MACRO_DIMS)
        if all(value is None for value in key):
            return None
        subsets.setdefault(key, []).append((float(row["score"]), row["weight"]))
    if len(subsets) < 2:
        return None
    means = [
        sum(score * weight for score, weight in cluster) / sum(w for _, w in cluster)
        for cluster in subsets.values()
    ]
    return sum(means) / len(means)

# parsed_judgment keys that are bookkeeping, not metrics — excluded from sub-score averaging.
_SUBSCORE_SKIP = {"num_references", "parse_failed", "support"}
# The confidence-interval estimator follows the *metric's* type, never the scores it happened
# to produce: a judge that emits only 0/1 on one task must not get a binomial interval while
# the same judge gets a bootstrap on the next. Keys are judgment ``evaluator_name`` values.
# Only a genuinely Bernoulli metric is eligible for Wilson; every continuous metric — and any
# unrecognized or renamed evaluator — falls back to the assumption-free bootstrap.
_METRIC_KIND = {
    # Bernoulli: one right-or-wrong decision per response, ``is_correct`` always set.
    "multiple_choice_accuracy": "binary",
    "classification_accuracy": "binary",
    "multiple_answer_set_match": "binary",
    "exact_match": "binary",
    "any_of_match": "binary",
    # Continuous: the per-response score is a graded quantity, whatever values a given task
    # happened to produce.
    "llm_judge": "continuous",
    "rouge": "continuous",
    "bleu": "continuous",
    "text_f1_em": "continuous",
    "likert_credit": "continuous",
    "multilabel": "continuous",
    "vlm_multilabel": "continuous",
    "vlm_text_overlap": "continuous",
    "vlm_document_fields": "continuous",
    "vlm_grounding": "continuous",
    "vlm_multistage_choice": "continuous",
    # numeric_tolerance emits only 0/1 with is_correct set, so Wilson would fit it; it is
    # declared continuous by policy because a tolerance check is a graded comparison whose
    # thresholds can change. Move it to "binary" to restore a binomial interval.
    "numeric_tolerance": "continuous",
}
_BINARY_EVALUATORS = {name for name, kind in _METRIC_KIND.items() if kind == "binary"}
_BOOLEAN_METRICS = {
    "critical_hallucination", "critical_omission", "unsupported_claim",
}


def _flatten_numeric(d: dict, prefix: str = "") -> dict:
    """Flatten a parsed_judgment into dotted-key numeric leaves (skips bools + bookkeeping)."""
    out: dict = {}
    for k, v in (d or {}).items():
        if k in _SUBSCORE_SKIP:
            continue
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            if k in _BOOLEAN_METRICS:
                out[key] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_flatten_numeric(v, prefix=key + "."))
    return out


def _metrics_by_evaluator(
    dedup_judgments: dict,
    valid_result_ids: set,
    weights_by_result: dict[str, float],
) -> dict:
    """Mean normalized_score (+ accuracy, + every parsed sub-score) per evaluator name.

    Restricted to successful judgments on successfully-inferred results, so a secondary metric
    (ROUGE-L, BLEU, token-F1) is reported over the same population as the headline score. Every
    numeric detail in a judgment's parsed_judgment (ROUGE-1/2/L precision/recall/f, EM, F1, BLEU)
    is averaged into ``subscores`` so the summary is self-contained without re-reading judgments.
    """
    agg: dict = {}
    for j in dedup_judgments.values():
        if j.get("evaluation_status") != "success":
            continue
        if j.get("result_id") not in valid_result_ids:
            continue
        b = agg.setdefault(j.get("evaluator_name") or "?", {
            "scores": [], "correct": [], "subs": {}, "weight_sum": 0.0, "versions": set(),
        })
        # Recorded because a score is only comparable against another score from the same
        # evaluator version. Evaluator 1.1 stopped scoring an unparsed answer as a hard zero and
        # reports it unscorable instead, which moves both the mean and its denominator; IgakuQA
        # rose 0.4897 -> 0.5234 on that change alone, with the model untouched.
        b["versions"].add(j.get("evaluator_version") or "unknown")
        weight = weights_by_result.get(j.get("result_id"), 1.0)
        ns = j.get("normalized_score")
        if ns is not None:
            b["scores"].append((ns, weight))
            b["weight_sum"] += weight
        ic = j.get("is_correct")
        if ic is not None:
            b["correct"].append((1.0 if ic else 0.0, weight))
        for key, val in _flatten_numeric(j.get("parsed_judgment")).items():
            b["subs"].setdefault(key, []).append((val, weight))
    out: dict = {}
    for name, b in agg.items():
        def weighted_mean(values):
            return (sum(value * weight for value, weight in values)
                    / sum(weight for _, weight in values)) if values else None

        subscores = {k: round(weighted_mean(v), 6) for k, v in b["subs"].items() if v}
        out[name] = {
            "mean_score": round(weighted_mean(b["scores"]), 6) if b["scores"] else None,
            "accuracy": round(weighted_mean(b["correct"]), 6) if b["correct"] else None,
            "n": max(len(b["scores"]), len(b["correct"])),
            "weight_sum": round(b["weight_sum"], 6),
            "subscores": subscores or None,
            # Normally one version; more than one means the run mixes judgments written before
            # and after an evaluator change, and its mean spans both definitions.
            "evaluator_versions": sorted(b["versions"]),
        }
    return out


def summarize_run(run_dir: str | Path, *, denominator_policy: str = "successful_and_scored_only") -> Summary:
    run_dir = Path(run_dir)
    results = read_jsonl(run_dir / "results.jsonl")
    judgments = read_jsonl(run_dir / "judgments.jsonl")
    attempts = read_jsonl(run_dir / "attempts.jsonl")
    samples = read_jsonl(run_dir / "samples.jsonl")
    samples_by_id = {sample.get("sample_id"): sample for sample in samples}

    manifest_path = run_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})
    run_id = manifest.get("run_id") or (
        results[0]["run_id"] if results else (samples[0].get("run_id", "") if samples else "")
    )
    benchmark = ((manifest.get("benchmark") or {}).get("name") or
                 (results[0].get("benchmark_name") if results else
                  (samples[0].get("benchmark_name") if samples else "")))
    generation = manifest.get("generation") or {}
    if not generation:
        generation = (manifest.get("full_config") or {}).get("generation") or {}
    length_finish_policy = generation.get("length_finish_policy", "mark_incomplete")

    # Keep only the latest inference result per (sample_id, repeat): last-writer wins so a
    # resumed success supersedes an earlier failure.
    latest_result: dict[tuple, dict] = {}
    for r in results:
        key = (r.get("sample_id"), r.get("sample_repeat_index", 0))
        latest_result[key] = r

    # Primary judgment per result: prefer a rule-based accuracy/classification evaluator.
    # Latest-wins per (result_id, evaluator_name) so a rescore supersedes an earlier judgment
    # instead of both being counted.
    dedup_judgments: dict[tuple, dict] = {}
    judgment_order: dict[tuple, int] = {}
    for index, j in enumerate(judgments):
        key = (j.get("result_id"), j.get("evaluator_name"))
        dedup_judgments[key] = j
        judgment_order[key] = index
    judgments_by_result: dict[str, list[dict]] = {}
    # Keep the source-log order after latest-wins deduplication. Reassigning a dict key does
    # not move it, which previously made A -> B -> A rescoring select B as primary.
    for key in sorted(dedup_judgments, key=judgment_order.__getitem__):
        j = dedup_judgments[key]
        judgments_by_result.setdefault(j.get("result_id"), []).append(j)

    counts = Counts()
    counts.num_total = len(samples) if samples else len(latest_result)
    counts.num_unique_samples = len(samples) if samples else len({key[0] for key in latest_result})
    counts.num_logical_responses = len(latest_result)
    # Truncation is recorded per sample by the context-window fitter. Counting it here is the
    # only way a report can tell a clean run from one whose inputs lost their middles.
    counts.num_context_truncated = sum(
        1 for sample in samples if (sample.get("metadata") or {}).get("context_truncated")
    )
    dropped = ((manifest.get("benchmark") or {}).get("num_source_records_dropped"))
    counts.num_source_records_dropped = (
        int(dropped) if isinstance(dropped, (int, float)) and not isinstance(dropped, bool)
        else None
    )
    drop_reasons = (manifest.get("benchmark") or {}).get("source_record_drop_reasons")
    if isinstance(drop_reasons, dict):
        counts.source_record_drop_reasons = {
            str(reason): int(count) for reason, count in drop_reasons.items()
            if isinstance(count, (int, float)) and not isinstance(count, bool)
        }

    scored_rows = []  # for grouping
    successes = 0
    scored = 0
    prompt_tokens = completion_tokens = total_tokens = 0
    reasoning_tokens = cached_tokens = image_tokens = audio_tokens = 0
    latencies = []

    scorable_result_ids: set = set()

    for key, r in latest_result.items():
        counts.num_attempted += 1
        status = r.get("status")
        finish = r.get("finish_reason")
        if finish == "length":
            counts.num_max_length += 1
        if status == "success":
            counts.num_successful += 1
            successes += 1
            if r.get("parsing_status") == "error":
                counts.num_parsing_errors += 1
            if (r.get("provider_metadata") or {}).get("native_refusal"):
                counts.num_refusals += 1
        else:
            counts.num_failed += 1
            if r.get("error_type") == "content_filter":
                counts.num_content_filtered += 1
            if r.get("error_type") == "model_refusal":
                counts.num_refusals += 1
            continue

        # Token aggregation covers only successful logical results.
        prompt_tokens += r.get("prompt_tokens") or 0
        completion_tokens += r.get("completion_tokens") or 0
        total_tokens += r.get("total_tokens") or 0
        reasoning_tokens += r.get("reasoning_tokens") or 0
        cached_tokens += r.get("cached_input_tokens") or 0
        image_tokens += r.get("image_tokens") or 0
        audio_tokens += r.get("audio_tokens") or 0
        if r.get("latency_seconds") is not None:
            latencies.append(r["latency_seconds"])

        # The incomplete-output policy is an aggregation invariant, not merely a Runner
        # convention. This prevents stale/offline judgments from scoring truncated answers.
        if finish == "length" and length_finish_policy == "mark_incomplete":
            counts.num_missing_scoring += 1
            continue
        scorable_result_ids.add(r.get("result_id"))

        # scoring
        rjs = judgments_by_result.get(r.get("result_id"), [])
        primary = _primary_judgment(rjs)
        if primary is None:
            counts.num_missing_scoring += 1
            continue
        pstatus = primary.get("evaluation_status")
        if pstatus == "error":
            counts.num_evaluation_errors += 1
            continue
        if pstatus == "skipped":
            # A deliberately skipped evaluation is still a disposition of a successful
            # response; leaving it uncounted would drop the sample from every count.
            counts.num_evaluation_skipped += 1
            continue
        # Legacy records may only expose parse failures through parsed_judgment.
        pj = primary.get("parsed_judgment") or {}
        if pj.get("parse_failed") and r.get("parsing_status") != "error":
            counts.num_parsing_errors += 1
        norm = primary.get("normalized_score")
        correct = primary.get("is_correct")
        if norm is None:
            # ``evaluators.base.unscorable`` returns no score *and* tags a reason: the sample
            # gave the evaluator nothing to compare against. That is a property of the data,
            # not a failure of the evaluation, so it gets its own disposition. Counting it as
            # an evaluation error is what made MedS-Bench/task1 read as 67% broken evaluator
            # when in fact 67% of its references are empty.
            reason = pj.get("unscorable_reason")
            if reason:
                counts.num_unscorable += 1
                counts.unscorable_reasons[str(reason)] = (
                    counts.unscorable_reasons.get(str(reason), 0) + 1
                )
            else:
                counts.num_evaluation_errors += 1
            continue
        counts.num_scored += 1
        scored += 1
        weight = float((samples_by_id.get(r.get("sample_id")) or {}).get("sample_weight", 1.0))
        if weight <= 0:
            raise ValueError(f"sample {r.get('sample_id')} has non-positive sample_weight")
        scored_rows.append({
            "result": r, "sample_id": r.get("sample_id"), "score": norm, "correct": correct,
            "evaluator_name": primary.get("evaluator_name"), "weight": weight,
        })
    # metric
    num_scored = scored
    score = None
    ci = None
    if num_scored > 0:
        weight_sum = sum(row["weight"] for row in scored_rows)
        mean_score = sum(row["score"] * row["weight"] for row in scored_rows) / weight_sum
        score = mean_score
        unit_weights = all(row["weight"] == 1.0 for row in scored_rows)
        binary = all(row["evaluator_name"] in _BINARY_EVALUATORS
                     and row["correct"] is not None
                     for row in scored_rows)
        clusters_by_sample: dict[object, list[tuple[float, float]]] = {}
        for row in scored_rows:
            clusters_by_sample.setdefault(row["sample_id"], []).append(
                (float(row["score"]), row["weight"])
            )
        has_repeated_samples = any(len(cluster) > 1 for cluster in clusters_by_sample.values())
        if has_repeated_samples:
            ci = bootstrap_clustered_weighted_mean_interval(list(clusters_by_sample.values()))
            ci_method = "clustered_sample_bootstrap_95_seed_20260721"
        elif binary and unit_weights:
            num_correct = sum(1 for row in scored_rows if row["correct"] is True)
            ci = wilson_interval(num_correct, num_scored)
            ci_method = "wilson_95"
        else:
            values = [float(row["score"]) for row in scored_rows]
            weights = [row["weight"] for row in scored_rows]
            ci = (bootstrap_mean_interval(values) if unit_weights
                  else bootstrap_weighted_mean_interval(values, weights))
            ci_method = ("percentile_bootstrap_95_seed_20260721" if unit_weights
                         else "weighted_case_bootstrap_95_seed_20260721")
    else:
        ci_method = None

    # per-evaluator metrics: every evaluator (primary + any secondaries) averaged over the
    # successfully-inferred results, so ROUGE-L / BLEU / token-F1 appear beside the headline.
    weights_by_result = {
        result.get("result_id"): float(
            (samples_by_id.get(result.get("sample_id")) or {}).get("sample_weight", 1.0)
        )
        for result in latest_result.values()
    }
    metrics_by_evaluator = _metrics_by_evaluator(
        dedup_judgments, scorable_result_ids, weights_by_result,
    )
    from healthcorebench.aggregation.vlm import aggregate_vlm_profiles
    vlm_profile_metrics = aggregate_vlm_profiles(
        samples_by_id, list(dedup_judgments.values()), scorable_result_ids,
    )

    metrics = Metrics(
        score=score, micro_score=score, macro_score=_macro_score(scored_rows),
        confidence_interval=ci, confidence_interval_method=ci_method if ci else None,
        score_denominator_policy=denominator_policy,
        sample_weight_sum=(sum(row["weight"] for row in scored_rows) if scored_rows else 0.0),
    )

    # Current effective judge usage follows latest-wins judgment deduplication.
    eval_tokens = 0
    for j in dedup_judgments.values():
        eval_tokens += j.get("judge_total_tokens") or 0

    tokens = {
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens or None, "cached_input_tokens": cached_tokens or None,
        "image_tokens": image_tokens or None, "audio_tokens": audio_tokens or None,
        "evaluation_tokens": eval_tokens,
        "current_effective_evaluation_tokens": eval_tokens,
    }
    # These are named "attempt" tokens, but a request that failed carries no usage block: the
    # provider bills it and reports nothing, so it contributes 0 and the totals are in fact
    # "tokens over the attempts that returned usage". One run made 121,142 attempts for 7,721
    # samples, 95.7% of them rejected -- reading its cumulative figure as the cost of the run
    # understated it by that entire majority. The counts below say how much of the run the
    # totals actually cover, so the gap is visible instead of inferred.
    cumulative_eval_tokens = 0
    cumulative_inference_tokens = 0
    attempts_with_usage = 0
    attempts_without_usage = 0
    for attempt in attempts:
        purpose = attempt.get("request_purpose")
        usage_tokens = (attempt.get("usage") or {}).get("total_tokens")
        if usage_tokens is None:
            attempts_without_usage += 1
            usage_tokens = 0
        else:
            attempts_with_usage += 1
        if purpose == "evaluation_judge":
            cumulative_eval_tokens += usage_tokens
        elif purpose == "model_inference":
            cumulative_inference_tokens += usage_tokens
    tokens["cumulative_evaluation_attempt_tokens"] = cumulative_eval_tokens
    tokens["cumulative_inference_attempt_tokens"] = cumulative_inference_tokens
    tokens["attempts_with_usage"] = attempts_with_usage
    tokens["attempts_without_usage"] = attempts_without_usage

    effective_max_tokens: dict[str, int] = {}
    adaptive_recovered = 0
    for result in latest_result.values():
        value = result.get("effective_max_tokens")
        if value is not None:
            effective_max_tokens[str(value)] = effective_max_tokens.get(str(value), 0) + 1
        if result.get("adaptive_retry_count", 0) and result.get("status") == "success":
            adaptive_recovered += 1
    tokens["effective_max_tokens_distribution"] = effective_max_tokens
    tokens["adaptive_recovered_results"] = adaptive_recovered

    timing = _timing(latencies)

    # grouping
    groups = {}
    for group_name, dim in _GROUP_DIMS.items():
        rows = [{"value": row["result"].get(dim), "score": row["score"],
                 "correct": row["correct"], "weight": row["weight"]}
                for row in scored_rows]
        groups[group_name] = group_scores(rows, dim)

    source_files = {}
    rp = run_dir / "results.jsonl"
    jp = run_dir / "judgments.jsonl"
    source_files["results_hash"] = hash_file(rp) if rp.exists() else None
    source_files["judgments_hash"] = hash_file(jp) if jp.exists() else None

    return Summary(
        run_id=run_id, benchmark_name=benchmark or "", counts=counts, metrics=metrics,
        tokens=tokens, timing=timing, groups=groups,
        metrics_by_evaluator=metrics_by_evaluator,
        vlm_profile_metrics=vlm_profile_metrics,
        generated_at=utc_now_iso(), source_files=source_files,
    )


def _primary_judgment(judgments: list[dict]) -> dict | None:
    if not judgments:
        return None
    # An explicitly-tagged primary metric wins even if it errored — so a judge failure surfaces
    # as an evaluation error rather than being silently replaced by a secondary rule metric.
    tagged = [j for j in judgments
              if (j.get("provider_metadata") or {}).get("primary_metric")]
    if tagged:
        # Records are append-only. A later rescore explicitly replaces an earlier primary.
        return tagged[-1]
    # Legacy runs (no tag): prefer rule-based accuracy/classification, else first successful.
    for j in judgments:
        if j.get("evaluator_type") == "rule_based" and j.get("evaluation_status") == "success":
            return j
    for j in judgments:
        if j.get("evaluation_status") == "success":
            return j
    return judgments[0]


def _timing(latencies: list[float]) -> dict:
    if not latencies:
        return {"total_request_latency_seconds": 0.0, "average_latency_seconds": None,
                "median_latency_seconds": None, "p95_latency_seconds": None, "p99_latency_seconds": None}
    s = sorted(latencies)
    def pct(p):
        if not s:
            return None
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]
    return {
        "total_request_latency_seconds": sum(s),
        "average_latency_seconds": statistics.fmean(s),
        "median_latency_seconds": statistics.median(s),
        "p95_latency_seconds": pct(0.95),
        "p99_latency_seconds": pct(0.99),
    }
