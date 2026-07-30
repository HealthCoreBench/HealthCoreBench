"""Batch runs persist usable cross-task reports after every completed task."""

from __future__ import annotations

import json
from types import SimpleNamespace

from healthcorebench.cli import cmd_run
from healthcorebench.schemas.config import RunConfig


def _write_task_result(run_dir, task_key: str, score: float) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_status": "completed",
        "benchmark": {"registry_key": task_key},
        "full_config": {"evaluation": {
            "evaluator": "multiple_choice",
            "use_llm_judge": False,
            "extra_evaluators": [],
        }},
    }), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({
        "benchmark_name": task_key,
        "counts": {
            "num_total": 1,
            "num_logical_responses": 1,
            "num_successful": 1,
            "num_failed": 0,
            "num_scored": 1,
            "num_parsing_errors": 0,
            "num_evaluation_errors": 0,
            "num_max_length": 0,
            "num_missing_scoring": 0,
        },
        "metrics": {"score": score, "confidence_interval": None},
        "metrics_by_evaluator": {
            "multiple_choice_accuracy": {"mean_score": score, "accuracy": score},
        },
    }), encoding="utf-8")


def test_batch_report_refreshes_after_each_task_and_stops_after_interrupt(
    tmp_path, monkeypatch,
):
    task_keys = ["MMLU/mcqa", "PubMedQA/classification", "GPQA/mcqa"]
    config = RunConfig.model_validate({
        "experiment": {"experiment_id": "incremental", "run_name": "incremental"},
        "benchmark": {"name": ",".join(task_keys)},
        "model": {"base_url": "http://mock/v1", "requested_model_name": "mock"},
    })
    batch_dir = tmp_path / "batch"
    # Snapshot of {task_key: status} as each task starts, so the assertions can check *what*
    # the intermediate report claimed and not merely how many rows it had.
    observed_reports = []

    class FakeOrchestrator:
        def __init__(self, cfg, *, run_dir, **kwargs):
            self.task_key = cfg.benchmark.name
            self.run_dir = run_dir

        def run(self):
            report_path = batch_dir / "all_tasks_results.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                assert report["num_tasks"] == len(report["rows"])
                observed_reports.append(
                    {row["task_key"]: row["status"] for row in report["rows"]}
                )
            score = 1.0 if self.task_key == task_keys[0] else 0.5
            _write_task_result(batch_dir / self.task_key, self.task_key, score)
            status = "interrupted" if self.task_key == task_keys[1] else "completed"
            return {
                "run_dir": str(batch_dir / self.task_key),
                "status": status,
                "summary_metrics": {"score": score},
            }

    monkeypatch.setattr("healthcorebench.config.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        "healthcorebench.benchmarks.resolve_benchmark_keys", lambda key: [key],
    )
    monkeypatch.setattr(
        "healthcorebench.runtime.run_setup.RunOrchestrator", FakeOrchestrator,
    )

    exit_code = cmd_run(SimpleNamespace(
        config="unused.yaml",
        run_dir=str(batch_dir),
        benchmark=None,
        split=None,
        model=None,
        base_url=None,
        concurrency=None,
        max_samples=None,
    ))

    exported = json.loads((batch_dir / "all_tasks_results.json").read_text())
    rows = {row["task_key"]: row for row in exported["rows"]}
    assert exit_code == 1
    # Task 3 is skipped by the interrupt but stays in the report as not_run: a batch that
    # silently shrank to its finished tasks would read as a complete 2-task evaluation.
    assert observed_reports == [{
        task_keys[0]: "completed", task_keys[1]: "not_run", task_keys[2]: "not_run",
    }]
    assert exported["num_tasks"] == 3
    assert set(rows) == set(task_keys)
    assert rows[task_keys[2]]["status"] == "not_run"
    assert rows[task_keys[2]]["run_status"] == "not_run"
    assert rows[task_keys[2]]["has_summary"] is False
    assert rows[task_keys[2]]["run_dir"] is None
    assert rows[task_keys[2]]["num_total"] == 0
    # The two tasks that did run keep their real, non-placeholder rows.
    assert [rows[key]["has_summary"] for key in task_keys[:2]] == [True, True]
    assert not (batch_dir / task_keys[2] / "summary.json").exists()


def test_incremental_report_uses_experiment_root_without_explicit_run_dir(
    tmp_path, monkeypatch,
):
    task_keys = ["MMLU/mcqa", "GPQA/mcqa"]
    output_root = tmp_path / "runs"
    experiment_dir = output_root / "incremental-default"
    config = RunConfig.model_validate({
        "experiment": {
            "experiment_id": "incremental-default",
            "run_name": "incremental-default",
        },
        "benchmark": {"name": ",".join(task_keys)},
        "model": {"base_url": "http://mock/v1", "requested_model_name": "mock"},
        "output": {"root_dir": str(output_root)},
    })
    observed_reports = []

    class FakeOrchestrator:
        def __init__(self, cfg, *, run_dir, **kwargs):
            assert run_dir is None
            self.task_key = cfg.benchmark.name

        def run(self):
            report_path = experiment_dir / "all_tasks_results.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                observed_reports.append(
                    {row["task_key"]: row["status"] for row in report["rows"]}
                )
            task_dir = experiment_dir / self.task_key
            _write_task_result(task_dir, self.task_key, 1.0)
            return {
                "run_dir": str(task_dir),
                "status": "completed",
                "summary_metrics": {"score": 1.0},
            }

    monkeypatch.setattr("healthcorebench.config.load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        "healthcorebench.benchmarks.resolve_benchmark_keys", lambda key: [key],
    )
    monkeypatch.setattr(
        "healthcorebench.runtime.run_setup.RunOrchestrator", FakeOrchestrator,
    )

    exit_code = cmd_run(SimpleNamespace(
        config="unused.yaml",
        run_dir=None,
        benchmark=None,
        split=None,
        model=None,
        base_url=None,
        concurrency=None,
        max_samples=None,
    ))

    exported = json.loads(
        (experiment_dir / "all_tasks_results.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    # Mid-run the report is already full width: task 2 is present as not_run before it starts.
    assert observed_reports == [{task_keys[0]: "completed", task_keys[1]: "not_run"}]
    assert exported["num_tasks"] == 2
    # By the end both placeholders have been replaced by real rows.
    assert all(row["status"] == "completed" for row in exported["rows"])
    assert all(row["has_summary"] is True for row in exported["rows"])
    assert not (experiment_dir / task_keys[0] / "all_tasks_results.json").exists()


def test_retry_failed_flag_overrides_only_when_passed():
    """An absent flag must leave the config's own value alone, not force it to False.

    ``runtime.retry_failed`` is what turns a re-run into a gap-fill: without it a sample that
    failed once stays unscored forever and its task keeps rendering N/A. The flag exists so the
    behaviour is reachable without editing YAML.
    """
    from healthcorebench.cli import _overrides_from_args, build_parser

    parser = build_parser()
    with_flag = parser.parse_args(["run", "--config", "x.yaml", "--retry-failed"])
    assert _overrides_from_args(with_flag)["runtime.retry_failed"] is True

    without = parser.parse_args(["run", "--config", "x.yaml"])
    assert "runtime.retry_failed" not in _overrides_from_args(without)


def test_concurrency_override_preserves_zero_for_schema_validation():
    """Invalid zero must reach RunConfig instead of silently falling back to the YAML value."""
    from healthcorebench.cli import _overrides_from_args, build_parser

    args = build_parser().parse_args([
        "run", "--config", "x.yaml", "--concurrency", "0",
    ])

    assert _overrides_from_args(args)["runtime.concurrency"] == 0
