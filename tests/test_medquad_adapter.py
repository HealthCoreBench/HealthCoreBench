"""Regression coverage for MedQuAD's repeated reference-answer rows."""

from healthcorebench.benchmarks import get_adapter


def test_medquad_merges_duplicate_questions_into_reference_aliases():
    adapter = get_adapter("MedQuAD/open", config=None)
    files = adapter.discover_source_files()
    adapter.validate_source_files(files)

    samples = [
        adapter.normalize_sample(raw, index)
        for index, raw in enumerate(adapter.load_raw_samples(files))
    ]

    assert len(samples) > 15000
    assert len({sample.sample_id for sample in samples}) == len(samples)
    assert any(sample.reference_aliases for sample in samples)
