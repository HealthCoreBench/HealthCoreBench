"""HealthCoreBench command-line interface.

Subcommands:
  run               run one benchmark against one model (config-driven)
  summarize         rebuild summary.json from JSONL
  batch-report      rebuild all_tasks_results.{json,csv,md} from existing task summaries
  reparse           re-extract answers from stored raw responses (no model calls)
  rescore           re-run evaluators over stored results (no evaluated-model calls)
  validate-run      check a run directory's integrity
  export-parquet    convert JSONL logs to Parquet
  list-benchmarks   list registered benchmarks and whether a parser is implemented
  inspect-benchmark show a benchmark's fixed directory, source files, hashes, sample count
  migrate-legacy    convert old-format results into the new layout

API keys are never accepted through CLI flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_common_overrides(p):
    p.add_argument("--benchmark", help="override benchmark.name")
    p.add_argument("--split", help="override benchmark.split")
    p.add_argument("--model", help="override model.requested_model_name")
    p.add_argument("--base-url", help="override model.base_url")
    p.add_argument(
        "--concurrency", type=int,
        help="override runtime.concurrency (1 = serial, greater than 1 = concurrent requests)",
    )
    p.add_argument("--max-samples", type=int, help="override benchmark.max_samples")
    p.add_argument(
        "--retry-failed", action="store_true", default=None,
        help="override runtime.retry_failed: re-request samples whose earlier attempt failed "
             "instead of leaving them permanently unscored. Needs runtime.resume; successful "
             "samples are still skipped, and missing judgments are backfilled either way.",
    )


def _overrides_from_args(args) -> dict:
    ov = {}
    if getattr(args, "benchmark", None):
        ov["benchmark.name"] = args.benchmark
    if getattr(args, "split", None):
        ov["benchmark.split"] = args.split
    if getattr(args, "model", None):
        ov["model.requested_model_name"] = args.model
    if getattr(args, "base_url", None):
        ov["model.base_url"] = args.base_url
    if getattr(args, "concurrency", None) is not None:
        ov["runtime.concurrency"] = args.concurrency
    if getattr(args, "max_samples", None) is not None:
        ov["benchmark.max_samples"] = args.max_samples
    # ``store_true`` with default None so an absent flag leaves the config's own value alone
    # rather than overriding it to False.
    if getattr(args, "retry_failed", None):
        ov["runtime.retry_failed"] = True
    return ov


def cmd_run(args) -> int:
    from healthcorebench.config import get_project_root, load_config
    from healthcorebench.runtime.run_setup import RunOrchestrator, RunSetupError
    from healthcorebench.benchmarks import resolve_benchmark_keys
    from healthcorebench.aggregation.batch_results import (
        batch_output_dir, build_batch_result_rows, write_batch_result_files,
    )

    config = load_config(args.config, _overrides_from_args(args))
    # Support comma-separated list: "MMLU,CareQA,PubMedQA" or "MMLU/mcqa,CareQA/open".
    # Each name is resolved independently; multi-task benchmarks (CareQA -> mcqa+open) expand.
    raw_names = [n.strip() for n in config.benchmark.name.split(",")]
    task_keys = []
    for name in raw_names:
        if name in {"ALL", "ALL_VLM", "ALL_BENCHMARKS"}:
            from healthcorebench.benchmarks import list_benchmarks
            component = {"ALL": "Language", "ALL_VLM": "Multimodal"}.get(name)
            task_keys.extend(
                row["key"] for row in list_benchmarks()
                if row["implemented"] and row["enabled"]
                and (component is None or row["component"] == component)
            )
        else:
            task_keys.extend(resolve_benchmark_keys(name))
    task_keys = list(dict.fromkeys(task_keys))

    # Up-front plan: name the bench and list its task types before any run starts, so the
    # operator can confirm the scope. Each task then prints its own method/counts when it runs.
    from healthcorebench.runtime import reporting
    reporting.print_bench_overview(config.benchmark.name, task_keys)

    results = []
    skipped_task_reasons: dict[str, str] = {}
    output_paths = None
    output_dir = None
    incremental_output_dir = None
    if len(task_keys) > 1:
        incremental_output_dir = (
            Path(args.run_dir) if args.run_dir else
            get_project_root() / config.output.root_dir / config.experiment.experiment_id
        )

    def refresh_report():
        """Rewrite the cross-task report from whatever has finished so far.

        Configured tasks with no results are carried as ``not_run`` rows, so an interrupted
        batch reports its own incompleteness instead of shrinking to the tasks that finished.
        """
        run_dirs = [item["run_dir"] for item in results]
        directory = incremental_output_dir or (
            batch_output_dir(run_dirs) if run_dirs else None
        )
        if directory is None:
            return None, None
        rows = build_batch_result_rows(
            run_dirs,
            configured_task_keys=task_keys,
            skipped_task_reasons=skipped_task_reasons,
        )
        return write_batch_result_files(rows, directory), directory

    interrupted = False
    try:
        for i, key in enumerate(task_keys):
            cfg = config.model_copy(deep=True)
            cfg.benchmark.name = key
            # Default (no --run-dir): the orchestrator lays each task out at
            # <root>/<experiment_id>/<bench>/<task>. With an explicit --run-dir holding several
            # tasks, nest each as <run-dir>/<bench>/<task> so a benchmark's tasks don't collide.
            run_dir = args.run_dir
            if run_dir and len(task_keys) > 1:
                run_dir = f"{run_dir.rstrip('/')}/{key}"
            orch = RunOrchestrator(
                cfg,
                run_dir=run_dir,
                task_number=i + 1,
                task_total=len(task_keys),
            )
            try:
                result = orch.run()
            except RunSetupError as error:
                # A setup refusal is about this one task's configuration -- a corpus whose
                # licence forbids the configured judge endpoint, a split that is not there.
                # Aborting the batch would throw away the other tasks over a decision that
                # says nothing about them, so the task is recorded with its refusal and the
                # run continues. Failures raised once a task is actually running still
                # propagate: those can mean the whole batch is running against a broken
                # setup, and finding out at task 1 is the point.
                skipped_task_reasons[key] = str(error)
                print(f"  [{i + 1}/{len(task_keys)}] {key}: skipped — {error}")
                output_paths, output_dir = refresh_report()
                continue
            results.append({"benchmark": key, "run_dir": result["run_dir"],
                            "status": result["status"], "metrics": result["summary_metrics"]})

            # Keep the cross-task report usable if a later task is interrupted or force-stopped.
            output_paths, output_dir = refresh_report()
            if result["status"] == "interrupted":
                interrupted = True
                break
    except KeyboardInterrupt:
        # A report of the completed tasks is still worth more than no report at all.
        interrupted = True
        output_paths, output_dir = refresh_report()

    # Tables are not printed; the terminal receives only their final paths.
    if output_paths is None:
        return 0
    reporting.print_final_paths(markdown_path=output_paths["markdown"], run_dir=output_dir)

    if interrupted:
        return 1
    return 0 if all(result["status"] == "completed" for result in results) else 1


def cmd_batch_report(args) -> int:
    from healthcorebench.aggregation.batch_results import (
        build_batch_result_rows, discover_task_run_dirs, write_batch_result_files,
    )
    from healthcorebench.runtime import reporting

    run_root = Path(args.run_root)
    task_dirs = discover_task_run_dirs(run_root)
    if not task_dirs:
        print(f"No task run directories found under {run_root}", file=sys.stderr)
        return 1
    configured = [key.strip() for key in (args.task_keys or "").split(",") if key.strip()]
    rows = build_batch_result_rows(task_dirs, configured_task_keys=configured or None)
    output_paths = write_batch_result_files(rows, args.output_dir or run_root)
    reporting.print_final_paths(
        markdown_path=output_paths["markdown"], run_dir=args.output_dir or run_root,
    )
    print(json.dumps({"num_tasks": len(rows), **output_paths}, indent=2))
    return 0



def cmd_summarize(args) -> int:
    from healthcorebench.aggregation.summarize import summarize_run
    from healthcorebench.utils.jsonl import atomic_write_json
    summary = summarize_run(args.run_dir)
    atomic_write_json(Path(args.run_dir) / "summary.json", summary.model_dump())
    print(json.dumps(summary.metrics.model_dump(), ensure_ascii=False, indent=2))
    return 0


def cmd_reparse(args) -> int:
    from healthcorebench.tools import reparse_run
    print(json.dumps(reparse_run(
        args.run_dir, parser_version=args.parser_version,
        regenerate_summary=args.rescore_and_summarize,
    ), indent=2))
    return 0


def cmd_rescore(args) -> int:
    from healthcorebench.tools import rescore_run
    print(json.dumps(rescore_run(
        args.run_dir, evaluator_name=args.evaluator,
        evaluator_version=args.evaluator_version,
        replace_primary=not args.secondary_only,
    ), indent=2))
    return 0


def cmd_validate(args) -> int:
    from healthcorebench.tools import validate_run
    report = validate_run(args.run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


def cmd_export_parquet(args) -> int:
    from healthcorebench.tools import export_parquet
    print(json.dumps(export_parquet(args.run_dir, args.out_dir), indent=2))
    return 0


def cmd_list_benchmarks(args) -> int:
    from healthcorebench.benchmarks import list_benchmarks
    rows = list_benchmarks()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            flag = "impl" if r["implemented"] else "----"
            if r["implemented"] and not r["enabled"]:
                flag = "off "
            print(f"[{flag}] {r['key']:<32} {r['benchmark_dir']}")
            if r.get("disabled_reason"):
                print(f"         disabled: {r['disabled_reason']}")
        print(f"\n{sum(1 for r in rows if r['implemented'])}/{len(rows)} implemented adapters"
              f" ({sum(1 for r in rows if r['implemented'] and not r['enabled'])} disabled)")
    return 0


def cmd_inspect_benchmark(args) -> int:
    from healthcorebench.benchmarks import get_entry, get_adapter
    from healthcorebench.benchmarks.errors import BenchmarkFormatNotImplementedError
    entry = get_entry(args.benchmark)
    out = {"benchmark_name": entry.benchmark_name, "benchmark_dir": entry.benchmark_dir,
           "directory": str(entry.directory()), "implemented": entry.adapter_dotted is not None}
    try:
        adapter = get_adapter(args.benchmark, config=None)
        files = adapter.discover_source_files()
        adapter.validate_source_files(files)
        entries, combined = adapter.source_file_manifest(files)
        out["source_files"] = entries
        out["source_files_combined_hash"] = combined
        out["num_samples"] = sum(1 for _ in adapter.load_raw_samples(files))
        out["adapter_version"] = adapter.adapter_version
    except BenchmarkFormatNotImplementedError as e:
        out["parser_status"] = "not_implemented"
        out["note"] = str(e)
        # still list files if the directory exists
        try:
            d = entry.directory()
            out["files_present"] = sorted(p.name for p in d.iterdir()) if d.exists() else []
        except Exception:
            pass
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_migrate_legacy(args) -> int:
    from healthcorebench.tools import migrate_legacy
    print(json.dumps(migrate_legacy(args.legacy_dir, args.output_dir), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="healthcorebench", description="OpenAI-compatible medical model evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one benchmark against one model")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--run-dir", default=None, help="explicit run directory (else auto under runs/)")
    _add_common_overrides(p_run)
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summarize", help="rebuild summary.json from JSONL")
    p_sum.add_argument("--run-dir", required=True)
    p_sum.set_defaults(func=cmd_summarize)

    p_batch = sub.add_parser(
        "batch-report",
        help="rebuild all_tasks_results.{json,csv,md} from existing task summaries",
    )
    p_batch.add_argument("--run-root", required=True,
                         help="directory holding the per-task run directories")
    p_batch.add_argument("--output-dir", default=None,
                         help="where to write the report (default: --run-root)")
    p_batch.add_argument(
        "--task-keys", default=None,
        help="comma-separated task keys the batch was configured to run; any without results "
             "are reported as not_run",
    )
    p_batch.set_defaults(func=cmd_batch_report)

    p_rep = sub.add_parser("reparse", help="re-extract answers from stored raw responses")
    p_rep.add_argument("--run-dir", required=True)
    p_rep.add_argument("--parser-version", default=None)
    p_rep.add_argument(
        "--rescore-and-summarize", action="store_true",
        help="for rule-based runs, append fresh judgments and rebuild summary.json",
    )
    p_rep.set_defaults(func=cmd_reparse)

    p_res = sub.add_parser("rescore", help="re-run evaluators over stored results")
    p_res.add_argument("--run-dir", required=True)
    p_res.add_argument("--evaluator", default=None)
    p_res.add_argument("--evaluator-version", default=None)
    p_res.add_argument(
        "--secondary-only", action="store_true",
        help="append the evaluator as a comparison metric without replacing the primary score",
    )
    p_res.set_defaults(func=cmd_rescore)

    p_val = sub.add_parser("validate-run", help="check run directory integrity")
    p_val.add_argument("--run-dir", required=True)
    p_val.set_defaults(func=cmd_validate)

    p_exp = sub.add_parser("export-parquet", help="convert JSONL logs to Parquet")
    p_exp.add_argument("--run-dir", required=True)
    p_exp.add_argument("--out-dir", default=None)
    p_exp.set_defaults(func=cmd_export_parquet)

    p_list = sub.add_parser("list-benchmarks", help="list registered benchmarks")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list_benchmarks)

    p_ins = sub.add_parser("inspect-benchmark", help="inspect a benchmark's fixed files")
    p_ins.add_argument("--benchmark", required=True)
    p_ins.set_defaults(func=cmd_inspect_benchmark)

    p_mig = sub.add_parser("migrate-legacy", help="convert old-format results to the new layout")
    p_mig.add_argument("--legacy-dir", required=True)
    p_mig.add_argument("--output-dir", required=True)
    p_mig.set_defaults(func=cmd_migrate_legacy)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
