"""Human-facing run progress is English, green, compact, and progress-aware."""

from healthcorebench.runtime import reporting


GREEN = "\033[32m"
RESET = "\033[0m"


def _assert_green_lines(output: str) -> None:
    lines = output.splitlines()
    assert lines
    assert all(line.startswith(GREEN) and line.endswith(RESET) for line in lines)


def test_task_progress_is_green_english_and_numbered(capsys, tmp_path):
    reporting.print_task_plan(
        task_key="CareQA/open",
        bench_name="CareQA",
        task="open",
        num_samples=10,
        evaluator=None,
        use_llm_judge=True,
        judge_model="judge-model",
        model_name="model-under-test",
        base_url_redacted="https://example.invalid/v1",
        run_dir=tmp_path / "runs" / "exp" / "CareQA" / "open",
        extra_evaluators=["rouge"],
        task_number=2,
        task_total=30,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    _assert_green_lines(captured.err)
    assert ">>> Starting task [2/30]: CareQA/open" in captured.err
    assert "Task type  : open-ended QA" in captured.err
    assert "Extra metric: rouge" in captured.err
    assert not any("\u4e00" <= char <= "\u9fff" for char in captured.err)


def test_completion_and_final_paths_are_green(capsys, tmp_path):
    run_dir = tmp_path / "runs" / "experiment"
    markdown = run_dir / "all_tasks_results.md"
    reporting.print_task_complete(
        task_key="MMLU/mcqa",
        status="completed",
        task_number=30,
        task_total=30,
    )
    reporting.print_final_paths(markdown_path=markdown, run_dir=run_dir)

    captured = capsys.readouterr()
    _assert_green_lines(captured.err)
    assert "<<< Finished task [30/30]: MMLU/mcqa (status: completed)" in captured.err
    assert f"Results Markdown: {markdown.resolve()}" in captured.err
    assert f"Run directory: {run_dir.resolve()}" in captured.err
    assert captured.out == ""
