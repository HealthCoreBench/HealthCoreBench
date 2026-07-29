"""Unit test for GPQA's free-text difficulty → short-label mapping (substring order matters:
'undergraduate' contains 'graduate')."""

from healthcorebench.benchmarks.adapters.gpqa_mcqa import _difficulty_bucket


def test_gpqa_difficulty_bucket():
    assert _difficulty_bucket("Hard undergraduate level (…)") == "hard_undergraduate"
    assert _difficulty_bucket("Easy undergraduate level (or easier)") == "easy_undergraduate"
    assert _difficulty_bucket("Hard graduate level (…)") == "graduate"
    assert _difficulty_bucket("Post-graduate level or harder (…)") == "postgraduate"
    assert _difficulty_bucket(None) is None
    assert _difficulty_bucket("") is None
