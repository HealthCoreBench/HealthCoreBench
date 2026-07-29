"""Green, English-only human-facing progress for evaluation runs.

Everything here is *informational* output for the operator watching a run: what benchmark /
task is about to be evaluated, which scoring method is used, and where the final artifacts were
written. It is printed to stderr so stdout remains available for machine-readable subcommands.
"""

from __future__ import annotations

import sys
from pathlib import Path

# One-line description of each rule-based evaluator, so the operator can sanity-check the
# scoring method *before* a run burns model calls. Keyed by evaluator registry name.
_EVALUATOR_DESC = {
    "multiple_choice": "exact match between the parsed option and the reference option",
    "multiple_answer": "exact set match across all selected option letters",
    "exact_match": "exact match after answer normalization",
    "classification": "exact match between the normalized predicted and reference labels",
    "numeric_tolerance": "numeric range, tolerance, or normalized date matching",
    "likert_credit": "partial credit from the official SCT expert distribution",
    "text_f1_em": "normalized exact match and token F1 against all accepted references",
    "rouge": "ROUGE-1/2/L overlap with ROUGE-L F1 as the primary score",
    "bleu": "cumulative BLEU-1/2/3/4 n-gram overlap",
    "vlm_text_overlap": "EM, token P/R/F1, BLEU-1/2/3/4, and ROUGE-1/2/L",
    "multilabel": "exact set, Hamming loss, and multi-label precision/recall/F1",
    "multistage_choice": "ordered stage accuracy and all-stage correctness",
    "grounding": "phrase-aware bounding-box IoU and threshold precision/recall",
    "document_fields": "document field-pair precision, recall, and F1",
    "any_of": "exact phrase match against any accepted reference answer",
}

# Human label for the task-type suffix in a task key (``<bench>/<task>``).
_TASK_TYPE_DESC = {
    "mcqa": "multiple-choice QA",
    "open": "open-ended QA",
    "classification": "classification",
    "nli": "natural language inference",
    "yesno": "yes/no classification",
    "factoid": "factoid short answer",
    "summarization": "summarization",
    "detection": "error or hallucination detection",
    "risk": "risk prediction",
    "reasoning": "reasoning",
    "calculation": "numeric calculation",
    "triage": "triage",
    "diagnosis": "diagnosis",
    "safety": "safety evaluation",
    "single": "single-answer extraction",
    "exact": "short-answer matching",
    "likert": "SCT Likert scoring",
    "task1": "participant extraction",
    "task2": "intervention extraction",
    "task3": "outcome extraction",
    "task12": "answer verification",
    "task16": "claim classification",
    "task18": "evidence explanation",
    "task29": "drug-dose extraction",
    "task46": "entity explanation",
    "task50": "concept explanation",
    "task74": "patient information extraction",
    "task100": "evidence justification",
    "task106": "cancer hallmark classification",
    "task122": "multiple-choice QA",
    "task123": "yes/no/maybe QA",
    "task125": "chemical NER",
    "task126": "chemical NER",
    "task127": "disease NER",
    "task128": "species NER",
    "task130": "diagnosis classification",
    "task131": "treatment classification",
}

_WIDTH = 66
_GREEN = "\033[32m"
_RESET = "\033[0m"


def _p(msg: str = "") -> None:
    print(f"{_GREEN}{msg}{_RESET}", file=sys.stderr, flush=True)


def task_type_label(task: str | None) -> str:
    if not task:
        return "single task"
    return _TASK_TYPE_DESC.get(task, task)


def evaluator_method_line(evaluator: str | None, use_llm_judge: bool, judge_model: str | None) -> str:
    """Human description of how this task will be scored."""
    if use_llm_judge:
        tail = f" (judge model: {judge_model})" if judge_model else ""
        return f"llm_judge - independent LLM scoring{tail}"
    if evaluator and evaluator in _EVALUATOR_DESC:
        return f"{evaluator} - {_EVALUATOR_DESC[evaluator]}"
    if evaluator:
        return evaluator
    return "unresolved"


def print_bench_overview(requested_name: str, task_keys: list[str]) -> None:
    """Printed once, before the loop, so the operator sees the whole bench plan up front."""
    _p()
    _p("=" * _WIDTH)
    _p("Evaluation plan")
    _p(f"Requested benchmarks: {requested_name}")
    _p(f"Total tasks: {len(task_keys)}")
    for index, key in enumerate(task_keys, start=1):
        task = key.split("/", 1)[1] if "/" in key else None
        _p(f"  [{index}/{len(task_keys)}] {key:<34} Type: {task_type_label(task)}")
    _p("=" * _WIDTH)


def print_task_plan(
    *,
    task_key: str,
    bench_name: str,
    task: str | None,
    num_samples: int,
    evaluator: str | None,
    use_llm_judge: bool,
    judge_model: str | None,
    model_name: str,
    base_url_redacted: str,
    run_dir: str | Path,
    extra_evaluators: list[str] | None = None,
    task_number: int = 1,
    task_total: int = 1,
) -> None:
    """Printed right before a task's model calls begin — the 'is this run correct?' preview."""
    _p()
    _p("-" * _WIDTH)
    _p(f">>> Starting task [{task_number}/{task_total}]: {task_key}")
    _p(f"    Benchmark  : {bench_name}")
    _p(f"    Task type  : {task_type_label(task)}")
    _p(f"    Samples    : {num_samples}")
    _p(f"    Evaluator  : {evaluator_method_line(evaluator, use_llm_judge, judge_model)}")
    if extra_evaluators:
        for name in extra_evaluators:
            desc = _EVALUATOR_DESC.get(name, name)
            _p(f"    Extra metric: {name} - {desc} (secondary only)")
    _p(f"    Model      : {model_name} @ {base_url_redacted}")
    _p(f"    Task output: {run_dir}")
    _p("-" * _WIDTH)


def print_task_complete(*, task_key: str, status: str,
                        task_number: int = 1, task_total: int = 1) -> None:
    """Print one compact completion line without dumping every per-task artifact path."""
    _p(f"<<< Finished task [{task_number}/{task_total}]: {task_key} (status: {status})")


def print_final_paths(*, markdown_path: str | Path, run_dir: str | Path) -> None:
    """Print only the two final paths requested by the batch-run interface."""
    _p(f"Results Markdown: {Path(markdown_path).resolve()}")
    _p(f"Run directory: {Path(run_dir).resolve()}")
