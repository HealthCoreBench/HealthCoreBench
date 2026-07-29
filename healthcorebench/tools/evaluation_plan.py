"""Reconstruct a run's effective evaluator plan from its manifest.

``manifest.full_config.evaluation`` records the *configured* evaluators, which is not the set
that actually produced judgments. Two resolutions happen at run time and are not written back:

* ``extra_evaluators: null`` becomes the benchmark's per-task defaults;
* when the LLM judge holds the primary metric, the configured rule-based ``evaluator`` is
  demoted to a secondary metric and run alongside the judge (verified against 71 of 434 real
  run directories, whose ``judgments.jsonl`` carries a rule-based evaluator that the manifest's
  ``extra_evaluators: []`` does not mention).

Offline tools must reconstruct the same plan or they will regenerate only part of a run's
judgments (reparse) or fail to notice a missing one (validate-run).
"""

from __future__ import annotations

from healthcorebench.evaluators import default_extra_evaluators


def resolved_extra_evaluators(manifest: dict) -> list[str]:
    """Registry names of the secondary evaluators a run actually applied.

    ``extra_evaluators: null`` means the run never resolved them and the benchmark defaults
    applied; an empty list is a deliberate opt-out and is respected as such.
    """
    evaluation = (manifest.get("full_config") or {}).get("evaluation") or {}
    extra = evaluation.get("extra_evaluators")
    if extra is not None:
        names = list(extra)
    else:
        benchmark = manifest.get("benchmark") or {}
        names = default_extra_evaluators(
            benchmark.get("registry_key") or benchmark.get("name")
        )
    if evaluation.get("use_llm_judge") and evaluation.get("evaluator"):
        # With the judge scoring, the configured rule-based evaluator still runs — as a
        # secondary metric — so its judgment is expected even though nothing records it.
        names.insert(0, evaluation["evaluator"])
    return list(dict.fromkeys(names))


def rule_based_evaluator_names(manifest: dict) -> list[str]:
    """Every rule-based evaluator registry name expected to have produced judgments."""
    evaluation = (manifest.get("full_config") or {}).get("evaluation") or {}
    names = []
    if evaluation.get("evaluator"):
        names.append(evaluation["evaluator"])
    names.extend(resolved_extra_evaluators(manifest))
    return list(dict.fromkeys(names))
