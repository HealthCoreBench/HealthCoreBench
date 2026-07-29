"""Count and normalization regressions for newly split benchmark tasks."""

from __future__ import annotations

import pytest

from healthcorebench.benchmarks.registry import get_adapter
from healthcorebench.config import load_config


@pytest.fixture(scope="module")
def run_config():
    return load_config("configs/run_all_benchmarks_text.yaml")


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("AgentClinic/diagnosis", 214),
        ("MEDEC/correction", 311),
        ("MedArabiQ/mcqa", 100),
        ("MedArabiQ/bias_mcqa", 100),
        ("MedArabiQ/fill_choice", 100),
        ("MedArabiQ/fill_open", 100),
        ("MedArabiQ/patient_qa", 100),
        ("MedArabiQ/patient_qa_llm", 100),
        ("MedArabiQ/patient_qa_gec", 100),
    ],
)
def test_task_count_ids_and_messages(run_config, task: str, expected: int) -> None:
    adapter = get_adapter(task, config=run_config)
    files = adapter.discover_source_files()
    adapter.validate_source_files(files)
    raw_samples = list(adapter.load_raw_samples(files))

    assert len(raw_samples) == expected
    normalized = [adapter.normalize_sample(raw, i) for i, raw in enumerate(raw_samples)]
    assert len({sample.sample_id for sample in normalized}) == expected
    assert all(adapter.build_messages(sample) for sample in normalized[:5])


def test_medarabiq_task_family_totals_700(run_config) -> None:
    tasks = (
        "mcqa", "bias_mcqa", "fill_choice", "fill_open",
        "patient_qa", "patient_qa_llm", "patient_qa_gec",
    )
    total = 0
    for task in tasks:
        adapter = get_adapter(f"MedArabiQ/{task}", config=run_config)
        total += sum(1 for _ in adapter.load_raw_samples(adapter.discover_source_files()))
    assert total == 700


def test_medconceptsqa_is_split_by_difficulty_not_pooled():
    """The three difficulty levels ask about the same concepts, so pooling triple-counts them.

    ``easy``/``medium``/``hard`` differ only in how the distractors were drawn: per vocabulary the
    correct-answer sets overlap ~99.9%. One pooled ``accuracy`` over all fifteen subsets therefore
    scored every concept about three times and collapsed the one axis this benchmark exists to
    vary. Each task now holds exactly one difficulty, and no record is lost.
    """
    from healthcorebench.benchmarks import get_adapter, resolve_benchmark_keys
    from healthcorebench.benchmarks.registry import get_registry

    keys = resolve_benchmark_keys("MedConceptsQA")
    assert keys == ["MedConceptsQA/mcqa_easy", "MedConceptsQA/mcqa_hard",
                    "MedConceptsQA/mcqa_medium"]
    # The pooled key must be gone, not merely unused: a config naming it has to fail loudly.
    assert "MedConceptsQA/mcqa" not in get_registry()

    total = 0
    for level in ("easy", "medium", "hard"):
        adapter = get_adapter(f"MedConceptsQA/mcqa_{level}", None)
        files = adapter.discover_source_files()
        # One file per vocabulary, all at this task's difficulty.
        assert len(files) == 5, level
        assert all(path.parent.name.endswith(f"_{level}") for path in files), level
        count = sum(1 for _ in adapter.load_raw_samples(files))
        assert count > 0, level
        total += count

    # Splitting reassigns records, it does not drop them.
    assert total == 819_772


def test_medconceptsqa_base_adapter_refuses_to_pool_every_difficulty():
    """The abstract base must not silently fall back to "all fifteen subsets"."""
    import pytest

    from healthcorebench.benchmarks.adapters.medconceptsqa_mcqa import MedConceptsQAAdapter
    from healthcorebench.benchmarks.errors import BenchmarkSplitNotFoundError
    from healthcorebench.benchmarks.registry import get_registry

    entry = get_registry()["MedConceptsQA/mcqa_hard"]
    adapter = MedConceptsQAAdapter(entry, None)
    assert adapter.level is None
    with pytest.raises(BenchmarkSplitNotFoundError, match="abstract"):
        adapter.discover_source_files()
