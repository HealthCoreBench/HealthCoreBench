"""The per-corpus data-licence declarations, and the gates they drive.

Both gates were dead before these declarations existed: ``redistribution_allowed`` and
friends were ``True`` class attributes that no adapter overrode, so PhysioNet credentialed
corpora were shipped to whatever judge endpoint the config named and written into
``samples.jsonl`` like any public corpus. These tests pin the wiring, not the policy —
a corpus moving in or out of the table is a deliberate edit to ``data_licenses.py``.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from healthcorebench.aggregation.batch_results import build_batch_result_rows
from healthcorebench.benchmarks import get_adapter, get_registry
from healthcorebench.benchmarks.data_licenses import DATA_LICENSES, license_for
from healthcorebench.config import get_project_root
from healthcorebench.runtime.run_setup import (
    RestrictedDataDisclosureWarning,
    RunOrchestrator,
    RunSetupError,
)
from healthcorebench.schemas.config import RunConfig

# Fixed here rather than recomputed from the table, so silently emptying ``DATA_LICENSES``
# turns the suite red instead of making every test below vacuously pass.
CREDENTIALED_DIRS = {
    "7_MIMIC-CXR", "12_MedNLI", "20_Medical-CXR-VQA", "25_MIMICEchoQA", "27_MX-CXR",
    "30_MIMIC-Ext-MIMIC-CXR-VQA", "45_MIMIC-CDM", "46_EHRNoteQA", "47_EHRBench",
}


def _config(benchmark: str, tmp_path, judge_base_url: str | None,
            acknowledged: bool = False, artifacts: bool = False) -> RunConfig:
    evaluation = {"use_llm_judge": bool(judge_base_url),
                  "allow_restricted_data_to_remote_judge": acknowledged}
    if judge_base_url:
        evaluation["judge"] = {
            "base_url": judge_base_url, "requested_model_name": "judge", "api_key": "x",
        }
    return RunConfig(
        experiment={"experiment_id": "licence_test", "run_name": "licence_test"},
        benchmark={"name": benchmark, "split": "test", "max_samples": 2},
        model={"base_url": "http://127.0.0.1:8077/v1", "requested_model_name": "m"},
        generation={"temperature": 0.0, "max_tokens": 32, "n": 1},
        runtime={"concurrency": 1, "max_retries": 1},
        output={"root_dir": str(tmp_path),
                "allow_restricted_data_in_artifacts": artifacts},
        evaluation=evaluation,
    )


def test_every_declaration_names_a_directory_that_exists():
    """A renumbered corpus must fail here, not silently drop its licence.

    The table is keyed on directory basenames (``47_EHRBench``), which is what makes one
    declaration cover every task derived from that corpus. The cost of that key is that a
    renumbering turns the entry into a no-op that still reads as protective.
    """
    roots = [
        get_project_root() / "benchmarks" / "medical_llm_benchmarks",
        get_project_root() / "benchmarks" / "medical_vlm_benchmarks",
    ]
    present = {path.name for root in roots if root.exists() for path in root.iterdir()}
    if not present:
        pytest.skip("benchmark data directories unavailable")
    assert set(DATA_LICENSES) <= present, sorted(set(DATA_LICENSES) - present)


def test_directory_basenames_are_unique_across_the_two_roots():
    """Keying on the basename is only sound while basenames do not collide."""
    llm = get_project_root() / "benchmarks" / "medical_llm_benchmarks"
    vlm = get_project_root() / "benchmarks" / "medical_vlm_benchmarks"
    if not (llm.exists() and vlm.exists()):
        pytest.skip("benchmark data directories unavailable")
    collisions = {p.name for p in llm.iterdir()} & {p.name for p in vlm.iterdir()}
    assert collisions == set()


def test_credentialed_corpora_are_declared_restricted():
    for directory in CREDENTIALED_DIRS:
        declaration = license_for(directory)
        assert declaration is not None, directory
        assert declaration.redistribution_allowed is False, directory
        assert declaration.store_full_input_allowed is False, directory
        # The references are labels and single clinical actions, not the restricted material;
        # blanking them would only make a run's artifacts impossible to re-score offline.
        assert declaration.store_reference_allowed is True, directory
        assert declaration.evidence, directory


def test_license_for_accepts_a_full_registry_path():
    assert license_for("benchmarks/medical_llm_benchmarks/47_EHRBench") is not None
    assert license_for("benchmarks/medical_llm_benchmarks/47_EHRBench/") is not None
    assert license_for("") is None
    assert license_for("benchmarks/medical_llm_benchmarks/1_MMLU") is None


def test_adapters_inherit_the_declaration_from_their_directory():
    """The flags have to arrive through the adapter — that is where the runtime reads them."""
    registry = get_registry()
    restricted = {
        key for key, entry in registry.items()
        if entry.adapter_dotted and Path(entry.benchmark_dir).name in CREDENTIALED_DIRS
    }
    assert restricted, "no registered task maps to a credentialed corpus"
    for key in restricted:
        adapter = get_adapter(key, None)
        assert adapter.redistribution_allowed is False, key
        assert adapter.store_full_input_allowed is False, key
        assert adapter.store_reference_allowed is True, key

    # An unrestricted corpus keeps the permissive defaults.
    assert get_adapter("MMLU/mcqa", None).redistribution_allowed is True


def test_a_subclass_can_still_override_the_declaration():
    """The flags became properties on the base; a plain class attribute must still win.

    Attribute lookup takes the first match along the MRO, so this holds — but it holds by
    a language rule that is easy to break with a ``__getattr__`` or a metaclass later.
    """
    adapter = get_adapter("MMLU/mcqa", None)

    class _Restricted(type(adapter)):
        redistribution_allowed = False

    assert _Restricted(adapter.entry, None).redistribution_allowed is False


def test_remote_judge_is_refused_for_a_restricted_corpus(tmp_path):
    config = _config("EHRBench/decision", tmp_path, "https://api.example.com/v1")
    orchestrator = RunOrchestrator(config, run_dir=str(tmp_path / "run"))
    with pytest.raises(RunSetupError, match="redistribution_allowed=False"):
        orchestrator._check_data_protection(config)


@pytest.mark.parametrize("judge_base_url", [
    "http://127.0.0.1:8077/v1", "http://localhost:8077/v1", "http://judge.internal/v1",
])
def test_a_local_judge_is_allowed_for_a_restricted_corpus(tmp_path, judge_base_url):
    """The gate is about third parties, not about judging restricted data at all."""
    config = _config("EHRBench/decision", tmp_path, judge_base_url)
    orchestrator = RunOrchestrator(config, run_dir=str(tmp_path / "run"))
    orchestrator._check_data_protection(config)


def test_rule_based_scoring_of_a_restricted_corpus_is_untouched(tmp_path):
    """Most restricted tasks score locally; the gate must not cost them anything."""
    config = _config("MedNLI/nli", tmp_path, None)
    orchestrator = RunOrchestrator(config, run_dir=str(tmp_path / "run"))
    orchestrator._check_data_protection(config)


def test_restricted_source_records_are_kept_out_of_samples_jsonl(tmp_path):
    """``store_full_input_allowed=False`` has to blank the persisted copy, not the live one."""
    config = _config("MedNLI/nli", tmp_path, None)
    orchestrator = RunOrchestrator(config, run_dir=str(tmp_path / "run"))
    sample = {
        "sample_id": "s1",
        "source_content": {"premise": "restricted clinical note text"},
        "reference_answer": "entailment",
        "reference_answer_normalized": "entailment",
        "reference_aliases": None,
        "logical_messages": [],
    }
    recorded: list[dict] = []
    orchestrator.recorder.record_sample = recorded.append
    # Built during run(); here the point is only that nothing is already on disk.
    orchestrator._resume_index = SimpleNamespace(sample_recorded=lambda _sample_id: False)

    orchestrator._record_samples([sample])

    assert recorded[0]["source_content"] == {}
    assert recorded[0]["reference_answer"] == "entailment"   # references stay scorable
    # The in-memory sample the run scores against is untouched.
    assert sample["source_content"] == {"premise": "restricted clinical note text"}


def test_a_declined_task_is_reported_with_its_reason_not_as_not_run():
    """A refusal has to survive into the batch report, or the task reads as merely missing."""
    rows = build_batch_result_rows(
        [],
        configured_task_keys=["EHRBench/decision", "MMLU/mcqa"],
        skipped_task_reasons={"EHRBench/decision": "declares redistribution_allowed=False"},
    )
    by_key = {row["task_key"]: row for row in rows}
    assert by_key["EHRBench/decision"]["status"] == "skipped"
    assert "redistribution_allowed=False" in by_key["EHRBench/decision"]["note"]
    assert by_key["MMLU/mcqa"]["status"] == "not_run"
    assert by_key["MMLU/mcqa"]["note"] is None


def test_operator_can_authorize_a_remote_judge_for_restricted_data(tmp_path):
    """The framework does not get the last word on what authorization the operator holds.

    It cannot know which data use agreements are signed, so the refusal is a safe default rather
    than a verdict. The data controller lifts it by asserting authorization in the config -- and
    the assertion has to leave a trace, or the audit value of the declarations is gone.
    """
    config = _config("EHRBench/decision", tmp_path, "https://api.example.com/v1",
                     acknowledged=True)
    orchestrator = RunOrchestrator(config, run_dir=str(tmp_path / "run"))
    events: list[tuple[str, dict]] = []
    orchestrator.recorder.record_event = lambda name, payload: events.append((name, payload))

    with pytest.warns(RestrictedDataDisclosureWarning) as caught:
        orchestrator._check_data_protection(config)

    # The warning still names the corpus, the licence and the endpoint: authorized is not silent.
    message = str(caught[0].message)
    assert "EHRBench/decision" in message
    assert "redistribution_allowed=False" in message
    assert "PhysioNet" in message
    assert "api.example.com" in message

    assert len(events) == 1
    name, payload = events[0]
    assert name == "restricted_data_sent_to_remote_judge"
    assert payload["benchmark"] == "EHRBench/decision"
    assert payload["license_name"].startswith("PhysioNet")
    assert payload["license_evidence"]
    assert payload["acknowledged_by_config"] == (
        "evaluation.allow_restricted_data_to_remote_judge"
    )


def test_the_authorization_is_recorded_in_the_run_config(tmp_path):
    """It reaches the manifest, and does not invalidate a resume.

    The flag changes neither the samples nor the prompts nor the model outputs, so making it part
    of the resume identity would force a full re-run for a bookkeeping change.
    """
    config = _config("EHRBench/decision", tmp_path, "https://api.example.com/v1",
                     acknowledged=True)
    assert config.evaluation.allow_restricted_data_to_remote_judge is True
    assert "allow_restricted_data_to_remote_judge" not in str(config.config_hash_payload())


def test_the_authorization_does_not_widen_to_anything_else(tmp_path):
    """Set on a corpus that was never restricted, it must be inert -- no warning, no event."""
    import warnings

    config = _config("MMLU/mcqa", tmp_path, "https://api.example.com/v1", acknowledged=True)
    orchestrator = RunOrchestrator(config, run_dir=str(tmp_path / "run"))
    events: list = []
    orchestrator.recorder.record_event = lambda name, payload: events.append(name)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RestrictedDataDisclosureWarning)
        orchestrator._check_data_protection(config)
    assert events == []


def test_a_restricted_corpus_is_still_refused_without_the_authorization(tmp_path):
    """The default has to stay a refusal, and its message has to name the way out."""
    config = _config("EHRBench/decision", tmp_path, "https://api.example.com/v1")
    orchestrator = RunOrchestrator(config, run_dir=str(tmp_path / "run"))
    with pytest.raises(RunSetupError) as error:
        orchestrator._check_data_protection(config)
    assert "allow_restricted_data_to_remote_judge" in str(error.value)


def _recording_orchestrator(config, tmp_path):
    orchestrator = RunOrchestrator(config, run_dir=str(tmp_path / "run"))
    recorded: list[dict] = []
    events: list[tuple[str, dict]] = []
    orchestrator.recorder.record_sample = recorded.append
    orchestrator.recorder.record_event = lambda name, payload: events.append((name, payload))
    orchestrator._resume_index = SimpleNamespace(sample_recorded=lambda _sample_id: False)
    return orchestrator, recorded, events


def _restricted_sample(index: int = 1) -> dict:
    return {
        "sample_id": f"s{index}",
        "source_content": {"premise": "restricted clinical note text"},
        "reference_answer": "entailment",
        "reference_answer_normalized": "entailment",
        "reference_aliases": None,
        "logical_messages": [],
    }


def test_the_operator_can_authorize_restricted_records_into_the_run_directory(tmp_path):
    """Scoring never depended on the redaction; only the persisted copy did.

    Blanking ``source_content`` protects a run directory that leaves the machine, but it also
    makes the operator unable to check what prompt their own run actually sent. That is their
    call, so it is a config assertion — disclosed, not silent.
    """
    config = _config("MedNLI/nli", tmp_path, None, artifacts=True)
    orchestrator, recorded, events = _recording_orchestrator(config, tmp_path)

    with pytest.warns(RestrictedDataDisclosureWarning, match="store_full_input_allowed=False"):
        orchestrator._record_samples([_restricted_sample()])

    assert recorded[0]["source_content"] == {"premise": "restricted clinical note text"}
    assert [name for name, _ in events] == ["restricted_data_written_to_artifacts"]
    payload = events[0][1]
    assert payload["field"] == "store_full_input_allowed"
    assert payload["license_name"].startswith("PhysioNet")
    assert payload["acknowledged_by_config"] == "output.allow_restricted_data_in_artifacts"


def test_the_artifact_disclosure_fires_once_per_run_not_once_per_sample(tmp_path):
    """A 7,000-sample task must not emit 7,000 identical warnings and events."""
    config = _config("MedNLI/nli", tmp_path, None, artifacts=True)
    orchestrator, recorded, events = _recording_orchestrator(config, tmp_path)

    with pytest.warns(RestrictedDataDisclosureWarning) as caught:
        orchestrator._record_samples([_restricted_sample(i) for i in range(5)])

    assert len(recorded) == 5
    assert len(caught) == 1
    assert len(events) == 1


def test_save_source_content_still_wins_over_the_authorization(tmp_path):
    """The authorization lifts the *licence* gate; it is not a blanket "store everything".

    ``save_source_content`` is an ordinary size/privacy switch, and an operator who turned it off
    did so for their own reasons.
    """
    config = _config("MedNLI/nli", tmp_path, None, artifacts=True)
    config.output.save_source_content = False
    orchestrator, recorded, _events = _recording_orchestrator(config, tmp_path)

    orchestrator._record_samples([_restricted_sample()])
    assert recorded[0]["source_content"] == {}


def test_artifacts_are_still_redacted_without_the_authorization(tmp_path):
    """The default stays redaction, with no warning and no event."""
    import warnings

    config = _config("MedNLI/nli", tmp_path, None)
    orchestrator, recorded, events = _recording_orchestrator(config, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RestrictedDataDisclosureWarning)
        orchestrator._record_samples([_restricted_sample()])
    assert recorded[0]["source_content"] == {}
    assert events == []


def test_an_unrestricted_corpus_never_triggers_the_artifact_disclosure(tmp_path):
    """Set globally, the flag must be inert for the ~95 corpora that were never restricted."""
    import warnings

    config = _config("MMLU/mcqa", tmp_path, None, artifacts=True)
    orchestrator, recorded, events = _recording_orchestrator(config, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RestrictedDataDisclosureWarning)
        orchestrator._record_samples([_restricted_sample()])
    assert recorded[0]["source_content"] == {"premise": "restricted clinical note text"}
    assert events == []
