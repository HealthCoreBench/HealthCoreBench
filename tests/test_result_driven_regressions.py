"""Regression coverage for failures observed in quick_test_10case_v2."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from healthcorebench.benchmarks import get_adapter
from healthcorebench.benchmarks.adapters.ehrbench_decision_open import EHRBenchDecisionAdapter
from healthcorebench.benchmarks.adapters.geneturing_open import GeneTuringOpenAdapter
from healthcorebench.benchmarks.adapters.medcalc_bench_calculation import MedCalcBenchAdapter
from healthcorebench.benchmarks.answer_parsing import parse_label, parse_multiple_choice_letters
from healthcorebench.benchmarks.context_window import (
    ContextOverflowError,
    estimate_text_tokens,
)
from healthcorebench.evaluators import get_evaluator
from healthcorebench.schemas.config import RunConfig


def _config(benchmark: str, *, split: str = "test", policy: str = "head_tail") -> RunConfig:
    return RunConfig.model_validate({
        "experiment": {"experiment_id": "regression", "run_name": "regression"},
        "benchmark": {"name": benchmark, "split": split},
        "model": {"base_url": "http://localhost:8000/v1", "requested_model_name": "model"},
        "generation": {
            "max_tokens": 8192,
            "context_token_reserve": 512,
            "context_overflow_policy": policy,
        },
        "hardware": {"max_model_len": 16384},
    })


@pytest.mark.parametrize(("response", "expected"), [
    (
        "B. 兄が先天性免疫不全症である。\nC. 同居の祖父が肺結核で入院中である。",
        ["B", "C"],
    ),
    (
        "B. 自然増減数はマイナスに転じた。\nD. 交通事故の死亡者数は減少傾向にある。",
        ["B", "D"],
    ),
    ("B. bonne réponse\nD. bonne réponse", ["B", "D"]),
])
def test_multi_answer_parser_keeps_all_enumerated_choices(response, expected):
    assert parse_multiple_choice_letters(response, list("ABCD")) == expected


@pytest.mark.parametrize("response", [
    "B, D", "B D", "BD", "B، D", r"\boxed{B,D}", r"\boxed{B and D}",
])
def test_multi_answer_parser_supports_bare_structured_lists(response):
    assert parse_multiple_choice_letters(response, list("ABCD")) == ["B", "D"]


@pytest.mark.parametrize(("response", "expected"), [
    (
        "B, D\n\nExplanation:\nA is incorrect, while B and D are selected.",
        ["B", "D"],
    ),
    ("A、C\n\n解释：B选项错误。", ["A", "C"]),
    ("A et D\n\nExplication : B est une option incorrecte.", ["A", "D"]),
    ("B と C\n\n説明：Aは誤りです。", ["B", "C"]),
])
def test_multi_answer_parser_accepts_first_line_list_before_explanation(response, expected):
    assert parse_multiple_choice_letters(response, list("ABCD")) == expected


@pytest.mark.parametrize(("response", "expected"), [
    (
        "A. plausible\nB. plausible\nD. plausible\n\n"
        "Final answer: A, B, D.\n\nExplanation: option is incorrect; C must be excluded.",
        ["A", "B", "D"],
    ),
    (
        "逐项分析了 A、B和 C。\n因此，正确答案是 A、C。\n"
        "说明：B 选项错误。",
        ["A", "C"],
    ),
    (
        "A. proposition discutée\nB. proposition discutée\nD. proposition discutée\n\n"
        "Par conséquent, la réponse correcte est : A, B, D.\n\n"
        "Explication : l'option C est incorrecte.",
        ["A", "B", "D"],
    ),
    (
        "A. 尿沈渣\nB. 細菌培養\nD. 子宮卵管造影検査\n\n"
        "したがって、適切でない検査は：\n\n**D. 子宮卵管造影検査**",
        ["D"],
    ),
])
def test_multi_answer_parser_prioritizes_localized_final_answer(response, expected):
    assert parse_multiple_choice_letters(response, list("ABCD")) == expected


@pytest.mark.parametrize("response", [
    "**D**",
    "**D. 子宮卵管造影検査**",
    "The alternatives were reviewed.\n\n**D**",
])
def test_multi_answer_parser_accepts_single_markdown_final_option(response):
    assert parse_multiple_choice_letters(response, list("ABCD")) == ["D"]


@pytest.mark.parametrize("response", [
    "A. This option is incorrect.\nB. This option might be correct, but the answer is uncertain.",
    "A 选项错误，B 选项也需要讨论，无法确定。",
    "A. Cette option est incorrecte.\nB. Cette option pourrait être correcte.",
    "A. この選択肢は誤りです。\nB. この選択肢も検討中です。",
    "Analysis:\n**A. discussed option**\n**B. another discussed option**",
])
def test_multi_answer_parser_does_not_guess_from_multilingual_option_discussion(response):
    assert parse_multiple_choice_letters(response, list("ABCD")) is None


def test_multi_answer_parser_rejects_option_discussion_as_ambiguous():
    response = "A. incorrect because it is too broad\nB. correct\nC. wrong for this patient"
    assert parse_multiple_choice_letters(response, list("ABCD")) is None
    japanese = "A. この選択肢は誤りです。\nB. この選択肢が正しいです。"
    assert parse_multiple_choice_letters(japanese, list("ABCD")) is None
    assert parse_multiple_choice_letters("bad", list("ABCD")) is None


def test_label_parser_handles_medical_labels_ending_in_punctuation():
    labels = ["HIV (initial infection)", "Bronchitis"]
    assert parse_label("Diagnosis: HIV (initial infection).", labels) == labels[0]


def _ehr_raw() -> dict:
    """A raw sample in exactly the shape ``EHRBenchDecisionAdapter.load_raw_samples`` yields.

    The derived entries are produced by the adapter's own helpers rather than restated here,
    and ``test_ehr_raw_matches_the_loader_contract`` pins the key set. When this helper last
    drifted behind the loader, the two context-fitting tests below stopped exercising context
    fitting at all and failed on a bare ``KeyError`` inside ``normalize_sample`` instead.
    """
    record = {
        "idx": 7,
        "instruction": "Recommend the next clinical action.",
        "input": "patient event " * 10_000,
        "output": "Order the requested follow-up test.",
        # A real record lists its gold among the candidates. Without this the loader drops the
        # record as ``gold_not_in_candidates``, so it could never reach ``normalize_sample``
        # in production and anything asserted about it would be about a shape that never runs.
        "candidates": ["Order the requested follow-up test.", "Discharge the patient."],
        "task_info": {"task": "decision", "metric": "generation"},
    }
    reference = record["output"]
    return {
        "record": record,
        "instruction": record["instruction"],
        "context": record["input"],
        "reference": reference,
        "accepted_answers": EHRBenchDecisionAdapter._accepted_answers(record, reference),
        "candidates": EHRBenchDecisionAdapter._candidates(record),
        "candidates_added": [],
        "source_file_rel": "47_EHRBench/ehr_bench_decision_making.jsonl",
        "source_record_index": 0,
    }


def test_ehr_raw_matches_the_loader_contract():
    """The synthetic raw sample must carry the keys the real loader actually yields.

    Compared against a record read off disk, so adding a key to ``load_raw_samples`` fails
    here with the missing name rather than surfacing as a ``KeyError`` in an unrelated test.
    """
    adapter = get_adapter("EHRBench/decision", _config("EHRBench/decision"))
    files = adapter.discover_source_files()
    if not all(path.exists() for path in files):
        pytest.skip("EHRBench decision source data unavailable")
    produced = next(iter(adapter.load_raw_samples(files)))
    assert set(_ehr_raw()) == set(produced)


def test_ehrbench_decision_fits_context_before_provider_request():
    adapter = get_adapter("EHRBench/decision", _config("EHRBench/decision"))
    sample = adapter.normalize_sample(_ehr_raw(), 0)

    assert sample.metadata["context_truncated"] is True
    assert sample.metadata["context_token_budget"] < 8192
    assert sample.metadata["context_estimator"] == "conservative_multilingual_2ascii_v1"
    assert "middle of source context omitted" in sample.source_content["context"]
    assert "middle of source context omitted" in adapter.build_messages(sample)[0]["content"]


def test_ehrbench_decision_strict_policy_fails_before_request():
    adapter = get_adapter(
        "EHRBench/decision", _config("EHRBench/decision", policy="error")
    )
    with pytest.raises(ContextOverflowError):
        adapter.normalize_sample(_ehr_raw(), 0)


def test_ehrbench_risk_full_dataset_fits_before_first_provider_request():
    adapter = get_adapter("EHRBench/risk", _config("EHRBench/risk"))
    files = adapter.discover_source_files()
    adapter.validate_source_files(files)

    checked = 0
    overflow_records = 0
    worst_overflow_raw = None
    worst_overflow_tokens = -1
    answer_instruction = "Answer with exactly one word: yes or no."
    for raw in adapter.load_raw_samples(files):
        record = raw["record"]
        fixed_prompt = record["instruction"].strip() + "\n\n\n\n" + answer_instruction
        context_estimate = estimate_text_tokens(
            str(record.get("input") or "").strip(), ascii_chars_per_token=2
        )
        context_budget = (
            16384 - 8192 - 512
            - estimate_text_tokens(fixed_prompt, ascii_chars_per_token=2)
        )

        assert context_budget > 0, record.get("idx")
        overflow_tokens = context_estimate - context_budget
        overflow_records += overflow_tokens > 0
        if overflow_tokens > worst_overflow_tokens:
            worst_overflow_tokens = overflow_tokens
            worst_overflow_raw = raw
        checked += 1

    assert checked == 7721
    assert overflow_records > 0

    sample = adapter.normalize_sample(worst_overflow_raw, 0)
    estimated_request_tokens = (
        estimate_text_tokens(
            worst_overflow_raw["record"]["instruction"].strip()
            + "\n\n\n\n"
            + answer_instruction,
            ascii_chars_per_token=2,
        )
        + estimate_text_tokens(sample.source_content["context"], ascii_chars_per_token=2)
        + 8192
        + 512
    )
    assert sample.metadata["context_estimator"] == "conservative_multilingual_2ascii_v1"
    assert sample.metadata["context_truncated"] is True
    assert estimated_request_tokens <= 16384


@pytest.mark.parametrize(("split", "expected_files", "resolved"), [
    ("test", ["MedBrowseComp_605.json"], "605"),
    ("50", ["MedBrowseComp_50.json"], "50"),
    ("cua", ["MedBrowseComp_CUA.json"], "cua"),
    (
        "all",
        ["MedBrowseComp_605.json", "MedBrowseComp_50.json", "MedBrowseComp_CUA.json"],
        None,
    ),
])
def test_medbrowsecomp_split_discovery_and_metadata(split, expected_files, resolved):
    adapter = get_adapter("MedBrowseComp/open", _config("MedBrowseComp/open", split=split))
    files = adapter.discover_source_files()
    assert [path.name for path in files] == expected_files
    raw = next(iter(adapter.load_raw_samples([files[0]])))
    sample = adapter.normalize_sample(raw, 0)
    assert sample.metadata["split"] == (resolved or "605")
    assert sample.metadata["requested_split"] == split
    if split == "all":
        metadata_splits = {
            adapter.normalize_sample(next(iter(adapter.load_raw_samples([path]))), index)
            .metadata["split"]
            for index, path in enumerate(files)
        }
        assert metadata_splits == {"605", "50", "cua"}


def test_all_twenty_medsbench_tasks_load_with_homogeneous_metric_profiles():
    expected = {
        "task1": ("multi_label", "multilabel"),
        "task2": ("multi_label", "multilabel"),
        "task3": ("multi_label", "multilabel"),
        "task12": ("yes_no", "accuracy"),
        "task16": ("label", "accuracy"),
        "task18": ("free_text", "llm_judge"),
        "task29": ("short_answer", "any_of_match"),
        "task46": ("free_text", "llm_judge"),
        "task50": ("free_text", "llm_judge"),
        "task74": ("structured_text", "document_fields"),
        "task100": ("free_text", "llm_judge"),
        "task106": ("multi_label", "multilabel"),
        "task122": ("single_choice", "accuracy"),
        "task123": ("yes_no_maybe", "accuracy"),
        "task125": ("multi_label", "multilabel"),
        "task126": ("multi_label", "multilabel"),
        "task127": ("multi_label", "multilabel"),
        "task128": ("multi_label", "multilabel"),
        "task130": ("label", "accuracy"),
        "task131": ("label", "accuracy"),
    }
    for task, profile in expected.items():
        key = f"MedS-Bench/{task}"
        adapter = get_adapter(key, _config(key))
        files = adapter.discover_source_files()
        adapter.validate_source_files(files)
        adapter.source_file_manifest(files)
        raw = next(iter(adapter.load_raw_samples(files)))
        sample = adapter.normalize_sample(raw, 0)
        assert (sample.answer_format, sample.evaluation_metric) == profile
        assert sample.metadata["task"] == task
        assert sample.reference_answer_normalized is not None


def test_any_of_match_accepts_phrase_and_rejects_substring_false_positive():
    evaluator = get_evaluator("any_of")
    sample = {"reference_answer_normalized": ["Lipoatrophy", "Fat loss"]}
    result = evaluator.score(
        evaluator.normalize(
            "The finding is lipodystrophy with focal fat loss (lipoatrophy).", sample
        ),
        sample,
    )
    assert result[2] is True and result[3]["matched_reference"] in {"Lipoatrophy", "Fat loss"}

    _, score, correct, parsed = evaluator.score(
        evaluator.normalize("This is unrelated lipoatrophying text.", sample), sample
    )
    assert score == 0.0 and correct is False and parsed["matched_reference"] is None
    short_reference = {"reference_answer_normalized": ["AR"]}
    _, score, correct, _ = evaluator.score("cardiac disease", short_reference)
    assert score == 0.0 and correct is False


def test_any_of_match_distinguishes_parse_failure_and_missing_references():
    evaluator = get_evaluator("any_of")
    # Both cases are unscorable, but for different reasons, and the reason has to survive:
    # an unparsed response is the model's failure, missing references are the harness's.
    raw, score, correct, parsed = evaluator.score(None, {"reference_answer": ["term"]})
    assert (raw, score, correct) == (None, None, None)
    assert parsed["unscorable_reason"] == "unparsed_answer"
    assert parsed["parse_failed"] is True and parsed["reference_missing"] is False
    raw, score, correct, parsed = evaluator.score("term", {"reference_answer": []})
    assert (raw, score, correct) == (None, None, None)
    assert parsed["reference_missing"] is True


@pytest.mark.parametrize(("response", "expected"), [
    ("Age 70, intermediate score 41.2.\nAnswer: 53.1", "53.1"),
    ("The intermediate values are 20 and 40. Therefore, the value is 53.1.", "53.1"),
    ("The computed value is -1.2e-3.", "-1.2e-3"),
    ("Final answer: 80%\nThis is the final percentage.", "80"),
])
def test_medcalc_extracts_only_a_signposted_final_value(response, expected):
    adapter = object.__new__(MedCalcBenchAdapter)
    sample = SimpleNamespace(metadata={"output_type": "decimal"})
    assert adapter.parse_response(sample, response) == expected


def test_medcalc_rejects_ambiguous_intermediate_values_and_accepts_single_value():
    adapter = object.__new__(MedCalcBenchAdapter)
    sample = SimpleNamespace(metadata={"output_type": "decimal"})
    assert adapter.parse_response(sample, "First calculate 12, then multiply by 3.") is None
    assert adapter.parse_response(sample, "Answer: 20 or 30") is None
    assert adapter.parse_response(sample, "The calculated result is 53.1") == "53.1"


def test_medcalc_calendar_dates_are_normalized_and_invalid_dates_fail():
    evaluator = get_evaluator("numeric_tolerance")
    sample = {"reference_answer": "09/11/2014", "metadata": {"output_type": "date"}}
    _, _, correct, parsed = evaluator.score("9/11/2014", sample)
    assert correct is True and parsed["predicted"] == "09/11/2014"
    _, _, correct, parsed = evaluator.score("13/40/2014", sample)
    assert correct is False and parsed["parse_failed"] is True

    gestational = {
        "reference_answer": "('14 weeks', '1 days')",
        "metadata": {"output_type": "date"},
    }
    _, _, correct, parsed = evaluator.score("(14 weeks, 1 day)", gestational)
    assert correct is True and parsed["predicted"] == "(14 weeks, 1 days)"


def _gene_sample(reference_length=100, *, sequence=True, module="DNA sequence extraction"):
    return SimpleNamespace(
        metadata={
            "gene_output_kind": "sequence" if sequence else "short_answer",
            "expected_reference_length": reference_length,
            "module": module,
        }
    )


def test_geneturing_sequence_validation_accepts_normal_outputs_and_fences():
    adapter = object.__new__(GeneTuringOpenAdapter)
    assert adapter.parse_response(_gene_sample(12), "ACGTACGTACGT") == "ACGTACGTACGT"
    protein = _gene_sample(9, module="Amino acid translation")
    assert adapter.parse_response(protein, "```protein\nMTEQVLK*\n```") == "MTEQVLK*"


def test_geneturing_rejects_v2_style_runaway_periodic_sequences():
    adapter = object.__new__(GeneTuringOpenAdapter)
    assert adapter.parse_response(_gene_sample(30), "ATG" * 80) is None
    protein = _gene_sample(30, module="Amino acid translation")
    assert adapter.parse_response(protein, "MTE" + "QDQV" * 70) is None


def test_geneturing_does_not_reject_short_motifs_or_nonsequence_tasks():
    adapter = object.__new__(GeneTuringOpenAdapter)
    assert adapter.parse_response(_gene_sample(20), "ACGT" * 10) == "ACGT" * 10
    text = "The gene is BRCA1 because it is associated with this phenotype."
    assert adapter.parse_response(_gene_sample(sequence=False), text) == text
