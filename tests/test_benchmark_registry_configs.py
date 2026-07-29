"""Keep the shipped all-task configs synchronized with the adapter registry."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from healthcorebench.benchmarks.registry import get_adapter, get_registry
from healthcorebench.benchmarks.vlm_adapters.catalog import VLM_TASK_SPECS
from healthcorebench.config import load_config
from healthcorebench.evaluators import get_evaluator, select_evaluator_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The output-budget ladder every shipped config shares. The first entry is also each config's
# generation.max_tokens, so the run starts at the top of its own ladder. The tiers above 8192
# (262144/64000/32000/12800) were removed because both served endpoints reject or time out on
# them — vLLM answers 262144 with "max_tokens cannot be greater than max_model_len" — so
# walking them only produced doomed requests.
TOKEN_BUDGET_LADDER = [8192, 4096, 2048, 1024, 512, 256, 128, 64]
TEXT_CONFIG = PROJECT_ROOT / "configs" / "run_all_benchmarks_text.yaml"
STRICT_TEXT_CONFIG = PROJECT_ROOT / "configs" / "run_all_benchmarks_text_strict.yaml"
GPTOSS_STRICT_TEXT_CONFIG = (
    PROJECT_ROOT / "configs" / "run_all_benchmarks_text_strict_gptoss.yaml"
)
MULTIMODAL_CONFIG = PROJECT_ROOT / "configs" / "run_all_benchmarks_multimodal.yaml"
STRICT_MULTIMODAL_CONFIG = (
    PROJECT_ROOT / "configs" / "run_all_benchmarks_multimodal_strict.yaml"
)

STRICT_METRIC_FORMATS = {
    ("accuracy", "single_choice"),
    ("accuracy", "label"),
    ("accuracy", "yes_no"),
    ("accuracy", "yes_no_maybe"),
    ("accuracy", "nli"),
    ("set_match", "multi_choice"),
    ("numeric_tolerance", "numeric"),
}
STRICT_MULTIMODAL_PROFILES = {
    "closed",
    "document_parse",
    "fixed_diagnosis",
    "fixed_text",
    "grounding",
    "multilabel",
    "multistage",
    "multistage_closed",
}
STRICT_MULTIMODAL_METRIC_FORMATS = {
    ("accuracy", "label"),
    ("accuracy", "single_choice"),
    ("accuracy", "yes_no"),
    ("document_fields", "free_text"),
    ("grounding", "grounding_json"),
    ("multilabel", "label_set"),
    ("multistage_choice", "ordered_choices"),
}


def _implemented_keys(component: str) -> set[str]:
    """Implemented *and enabled* tasks: what a shipped config is allowed to contain.

    A disabled task (``enabled=False``, e.g. ``EHRNoteQA/mcqa``) keeps its adapter and stays
    resolvable by its exact key, but its score would not be meaningful — see the entry's
    ``disabled_reason`` — so it must not appear in any config, and the configs are compared
    against this set rather than against every registered adapter.
    """
    return {
        key for key, entry in get_registry().items()
        if entry.adapter_dotted is not None and entry.component == component and entry.enabled
    }


def _configured_keys(config) -> list[str]:
    return [key.strip() for key in config.benchmark.name.split(",") if key.strip()]


def _experiment_ids_are_consistent(config, prefix: str) -> None:
    """Check the naming convention, never a specific version.

    ``experiment_id`` is bumped for *every* evaluation (runtime.resume=true reuses
    ``runs/<experiment_id>/``), so pinning its exact value here would make each new run a
    test failure. What must hold is the family the config belongs to and that the run name
    tracks the id.
    """
    assert config.experiment.experiment_id.startswith(prefix), config.experiment.experiment_id
    assert config.experiment.run_name == f"all_{config.experiment.experiment_id}"


def test_all_text_config_matches_registry() -> None:
    config = load_config(TEXT_CONFIG)
    configured = set(_configured_keys(config))

    assert configured == _implemented_keys("Language")
    # Guard against the registry itself shrinking: 136 is the number of implemented *and
    # enabled* Language task keys today (120 before HEAD-QA's four remaining languages,
    # Med-HALT/reasoning_nota, three JMedBench subsets, three Swedish collections,
    # AfriMedQA_v2/multiple_answer, MedS-Bench/task61, ClinicalBench's split into three
    # prediction tasks, MedS-Bench's six MCQA tasks and MedConceptsQA's split into three
    # difficulty tasks were added, and EHRNoteQA/mcqa plus six duplicate tasks were disabled).
    assert len(configured) == 136
    # Disabled tasks must be absent from the config, not merely absent from the comparison set.
    assert not configured & {
        key for key, entry in get_registry().items() if not entry.enabled
    }
    _experiment_ids_are_consistent(config, "text_benchmarks")
    assert config.model.api_key is not None
    assert config.evaluation.judge.api_key is not None
    assert config.generation.max_tokens == TOKEN_BUDGET_LADDER[0]
    assert config.generation.max_tokens_candidates == TOKEN_BUDGET_LADDER
    assert config.runtime.same_budget_error_retries == 2
    # resume + retry_failed is what makes a second run fill the gaps of the first instead of
    # inheriting them: without retry_failed a sample that failed once is never re-requested,
    # so one endpoint wobble leaves that task reading N/A in every later report.
    assert config.runtime.resume is True
    assert config.runtime.retry_failed is True
    # Prompt trimming must be budgeted: null silently disables it and biases the score
    # towards the records that happen to fit. 16384 is this endpoint's reported window.
    assert config.hardware.max_model_len == 16384


def test_every_text_adapter_can_be_imported() -> None:
    config = load_config(TEXT_CONFIG)
    for key in config.benchmark.name.split(","):
        assert get_adapter(key, config=config).entry.key == key


def test_gptoss_strict_text_config_uses_shared_retry_ladder() -> None:
    config = load_config(GPTOSS_STRICT_TEXT_CONFIG)

    assert config.model.api_key is not None
    assert "api_key_env:" not in GPTOSS_STRICT_TEXT_CONFIG.read_text(encoding="utf-8")
    _experiment_ids_are_consistent(config, "text_benchmarks_strict")
    assert config.generation.max_tokens == TOKEN_BUDGET_LADDER[0]
    assert config.generation.max_tokens_candidates == TOKEN_BUDGET_LADDER
    assert config.runtime.same_budget_error_retries == 2
    assert config.runtime.transient_error_ladder_steps == 2
    # This config only swaps the endpoint: it must score gpt-oss on exactly the same tasks,
    # with exactly the same trimming, as the phi-4 strict config, or the two scores are not
    # comparable. A null window here would disable trimming for one model only.
    assert config.benchmark.name == load_config(STRICT_TEXT_CONFIG).benchmark.name
    assert config.hardware.max_model_len == 16384


def test_strict_text_config_contains_only_deterministically_scored_tasks() -> None:
    config = load_config(STRICT_TEXT_CONFIG)
    configured = _configured_keys(config)

    # 73 = every enabled text task that a rule-based scorer can grade end to end, minus
    # MedS-Bench. It grew from 58 with HEAD-QA es/gl/it/ru, Med-HALT/reasoning_nota, the three
    # non-duplicate JMedBench subsets, the three Swedish collections,
    # AfriMedQA_v2/multiple_answer, ClinicalBench's three prediction tasks (which replaced the
    # single ClinicalBench/classification task) and MedConceptsQA's three difficulty tasks
    # (which replaced the single pooled MedConceptsQA/mcqa), and lost EHRNoteQA/mcqa, now
    # disabled.
    assert len(configured) == len(set(configured)) == 73
    # A strict task that is missing from the full text config would never be maintained.
    assert set(configured) <= set(_configured_keys(load_config(TEXT_CONFIG)))
    assert all(not key.startswith("MedS-Bench/") for key in configured)
    assert config.evaluation.use_llm_judge is False

    metric_formats = Counter()
    for key in configured:
        adapter = get_adapter(key, config=config)
        raw = next(iter(adapter.load_raw_samples(adapter.discover_source_files())))
        sample = adapter.normalize_sample(raw, 0)
        assert (sample.evaluation_metric, sample.answer_format) in STRICT_METRIC_FORMATS, key
        metric_formats[(sample.evaluation_metric, sample.answer_format)] += 1

    # Every task is classified by its first sample, so this is one bucket per configured task.
    assert sum(metric_formats.values()) == len(configured)
    # Measured shape of the strict set: 52 single-choice, 7 multiple-answer (set_match),
    # 13 fixed-label classifications (label / yes_no / yes_no_maybe), 1 numeric calculation.
    assert metric_formats == {
        ("accuracy", "single_choice"): 52,
        ("accuracy", "label"): 7,
        ("accuracy", "yes_no"): 5,
        ("accuracy", "yes_no_maybe"): 1,
        ("set_match", "multi_choice"): 7,
        ("numeric_tolerance", "numeric"): 1,
    }


def test_all_multimodal_config_matches_registry() -> None:
    config = load_config(MULTIMODAL_CONFIG)
    configured = set(_configured_keys(config))

    _experiment_ids_are_consistent(config, "multimodal_benchmarks")
    assert configured == _implemented_keys("Multimodal")
    assert len(configured) == 56
    assert not configured & _implemented_keys("Language")
    assert config.runtime.concurrency == 8
    assert config.runtime.request_timeout_seconds == 300
    assert config.runtime.same_budget_error_retries == 2
    assert config.generation.max_tokens == TOKEN_BUDGET_LADDER[0]
    assert config.generation.max_tokens_candidates == TOKEN_BUDGET_LADDER
    assert config.generation.reduce_max_tokens_on_timeout is True
    assert config.evaluation.judge.api_key is not None
    # fit_context_to_window counts text only, so the served 131072-token window has to be
    # split: half for text, half held back for image/frame prefill (the heaviest measured
    # request, SurgeryVideoQA at 32 frames, alone cost 65,399 prompt tokens).
    assert config.hardware.max_model_len == 65536


def test_every_multimodal_adapter_can_be_imported() -> None:
    config = load_config(MULTIMODAL_CONFIG)
    for key in config.benchmark.name.split(","):
        assert get_adapter(key, config=config).entry.key == key


def test_strict_multimodal_config_contains_only_deterministically_scored_tasks() -> None:
    config = load_config(STRICT_MULTIMODAL_CONFIG)
    configured = _configured_keys(config)

    _experiment_ids_are_consistent(config, "multimodal_benchmarks_strict")
    assert len(configured) == len(set(configured)) == 33
    assert set(configured) <= set(_configured_keys(load_config(MULTIMODAL_CONFIG)))
    assert config.evaluation.use_llm_judge is False
    assert config.evaluation.judge is None
    assert config.runtime.concurrency == 8
    assert config.runtime.request_timeout_seconds == 300
    assert config.model.api_key is not None
    assert config.generation.max_tokens == TOKEN_BUDGET_LADDER[0]
    assert config.generation.max_tokens_candidates == TOKEN_BUDGET_LADDER
    assert config.runtime.same_budget_error_retries == 2
    assert config.generation.reduce_max_tokens_on_timeout is True
    # Same text/vision split as the full multimodal config, so trimming is identical in both.
    assert config.hardware.max_model_len == 65536
    assert all(get_registry()[key].component == "Multimodal" for key in configured)
    assert set(configured) == {
        key for key, spec in VLM_TASK_SPECS.items()
        if spec.profile in STRICT_MULTIMODAL_PROFILES
    }

    profiles = Counter(VLM_TASK_SPECS[key].profile for key in configured)
    assert set(profiles) <= STRICT_MULTIMODAL_PROFILES
    assert profiles == {
        "closed": 23,
        "document_parse": 2,
        "fixed_diagnosis": 1,
        "fixed_text": 1,
        "grounding": 1,
        "multilabel": 3,
        "multistage": 1,
        "multistage_closed": 1,
    }

    for key in configured:
        adapter = get_adapter(key, config=config)
        files = adapter.discover_source_files()
        adapter.validate_source_files(files)
        raw = next(iter(adapter.load_raw_samples(files)))
        sample = adapter.normalize_sample(raw, 0)
        assert sample.component == "Multimodal", key
        assert (
            sample.evaluation_metric,
            sample.answer_format,
        ) in STRICT_MULTIMODAL_METRIC_FORMATS, key
        assert sample.evaluation_metric not in {"vlm_text_overlap", "llm_judge"}, key
        evaluator_name = select_evaluator_name(
            sample.evaluation_metric, sample.answer_format
        )
        assert evaluator_name is not None, key
        assert get_evaluator(evaluator_name).evaluator_type == "rule_based", key
