from types import SimpleNamespace

import pytest

from healthcorebench.benchmarks.answer_parsing import (
    parse_label,
    parse_multiple_choice_letter,
    parse_multiple_choice_letters,
    parse_yes_no_maybe,
)
from healthcorebench.benchmarks.context_window import fit_context_to_window, estimate_text_tokens
from healthcorebench.benchmarks.prompts import judgement_prompt, multiple_answer_prompt
from healthcorebench.benchmarks.adapters.clinicbench_open_tasks import (
    ClinicBenchDrugInteractionAdapter,
    ClinicBenchHospitalizationAdapter,
)
from healthcorebench.benchmarks.adapters.clinicalbench_classification import ClinicalBenchAdapter
from healthcorebench.benchmarks.adapters.geneturing_open import GeneTuringOpenAdapter


@pytest.mark.parametrize("text", [
    "The correct answer is B.", "正确答案是 B。", "الإجابة الصحيحة هي B.",
    "正解は B です。", "정답은 B입니다.", "La bonne réponse est B.",
    "La respuesta correcta es B.", "Rätt svar är B.", "Правильный ответ: B.",
    "La risposta corretta è B.", "A resposta correcta é B.",
])
def test_mcqa_parser_supports_all_benchmark_languages(text):
    assert parse_multiple_choice_letter(text, list("ABCD")) == "B"


def test_mcqa_parser_handles_real_v3_arabic_outputs_without_guessing():
    assert parse_multiple_choice_letter(
        "الخيار الصحيح هو C. تخترق غلاف العصب البصري السحائي.", list("ABCD")
    ) == "C"
    assert parse_multiple_choice_letter("الإجابة الصحيحة هي B.", list("ABCD")) == "B"
    assert parse_multiple_choice_letter("A may fit, but B may also fit.", list("ABCD")) is None
    assert parse_multiple_choice_letter("The answer is A because B is wrong.", list("ABCD")) == "A"


@pytest.mark.parametrize("text", [
    "The answers are A and C", "正确答案是 A、C", "الإجابة الصحيحة هي A و C",
    "正解は A と C", "정답은 A 및 C", "Les réponses sont A et C",
    "Las respuestas son A y C", "Rätt svar är A och C", "Правильный ответ: A и C",
    "Le risposte corrette sono A e C", "As respostas correctas son A e C",
])
def test_multi_answer_parser_supports_localized_markers_and_joiners(text):
    assert parse_multiple_choice_letters(text, list("ABCD")) == ["A", "C"]


@pytest.mark.parametrize(("text", "expected"), [
    ("yes", "yes"), ("是", "yes"), ("نعم", "yes"), ("はい", "yes"), ("예", "yes"),
    ("oui", "yes"), ("sí", "yes"), ("ja", "yes"), ("да", "yes"), ("sì", "yes"),
    ("si", "yes"), ("لا", "no"), ("いいえ", "no"), ("아니요", "no"), ("nej", "no"),
    ("нет", "no"), ("non", "no"), ("ربما", "maybe"), ("forse", "maybe"),
])
def test_yes_no_maybe_parser_is_multilingual(text, expected):
    assert parse_yes_no_maybe(text) == expected


def test_numeric_label_requires_an_explicit_unambiguous_label():
    assert parse_label("Final answer: 2", ["1", "2", "3"]) == "2"
    assert parse_label("最终答案：2", ["1", "2", "3"]) == "2"
    assert parse_label("There are 2 concerns and 3 possible outcomes.", ["1", "2", "3"]) is None


@pytest.mark.parametrize("lang", ["en", "zh", "ar", "ja", "ko", "fr", "sv", "es", "ru", "it", "gl"])
def test_judgement_and_multiple_answer_prompts_cover_all_languages(lang):
    assert judgement_prompt("Q", lang=lang) != judgement_prompt("Q", lang="xx") or lang == "en"
    assert multiple_answer_prompt("Q", "A. x", lang=lang) != multiple_answer_prompt("Q", "A. x", lang="xx") or lang == "en"


def test_context_window_head_tail_is_bounded_deterministic_and_auditable():
    context = "HEADER\n" + ("patient event text " * 5000) + "\nLATEST EVENT"
    kwargs = dict(fixed_prompt="question", max_model_len=1024, max_output_tokens=128,
                  reserve_tokens=64, policy="head_tail")
    first, meta = fit_context_to_window(context, **kwargs)
    second, meta2 = fit_context_to_window(context, **kwargs)
    assert (first, meta) == (second, meta2)
    assert first.startswith("HEADER") and first.endswith("LATEST EVENT")
    assert "middle of source context omitted" in first
    assert meta["context_truncated"] is True
    assert estimate_text_tokens(first) <= meta["context_token_budget"]
    assert meta["original_context_chars"] == len(context)
    assert meta["omitted_context_chars"] > 0


def test_context_window_leaves_short_input_unchanged():
    context = "short context"
    fitted, meta = fit_context_to_window(
        context, fixed_prompt="q", max_model_len=1024, max_output_tokens=128,
        reserve_tokens=64, policy="head_tail",
    )
    assert fitted == context and meta["context_truncated"] is False


def test_context_window_supports_calibrated_ascii_ratio():
    context = "ordinary clinical prose " * 1000
    fitted, meta = fit_context_to_window(
        context, fixed_prompt="q", max_model_len=1024, max_output_tokens=128,
        reserve_tokens=64, policy="head_tail", ascii_chars_per_token=3,
    )
    assert estimate_text_tokens(fitted, ascii_chars_per_token=3) <= meta["context_token_budget"]
    assert "3ascii" in meta["context_estimator"]


def test_problem_adapters_enforce_concise_formats():
    sample = SimpleNamespace(source_content={"question": "Does X interact?"}, language="en")
    adapter = object.__new__(ClinicBenchDrugInteractionAdapter)
    assert "no extra output" in adapter.build_messages(sample)[0]["content"]

    hospital = object.__new__(ClinicBenchHospitalizationAdapter)
    prompt, _ = hospital.fields({"instruct": "chart", "answer": "summary"})
    assert "under 300 words" in prompt and "Do not reconstruct" in prompt

    gene = object.__new__(GeneTuringOpenAdapter)
    gene_sample = SimpleNamespace(
        source_content={"question": "Translate this sequence"},
        language="en",
        reference_answer="MST",
        answer_format="free_text",
        metadata={"gene_output_kind": "sequence"},
    )
    assert "without explanation" in gene.build_messages(gene_sample)[0]["content"]
    assert "do not add spaces" in gene.build_messages(gene_sample)[0]["content"]
    gene.config = SimpleNamespace(generation=SimpleNamespace(max_tokens=8192))
    for answer_format in (
        "single_choice", "multi_choice", "label", "yes_no", "yes_no_maybe",
        "likert", "numeric", "short_answer", "free_text",
    ):
        gene_sample.answer_format = answer_format
        assert gene.max_output_tokens(gene_sample) == 8192

    clinical = object.__new__(ClinicalBenchAdapter)
    classification_sample = SimpleNamespace(source_content={"question": "Q", "labels": ["1", "2", "3"]})
    messages = clinical.build_messages(classification_sample)
    assert messages[0]["role"] == "system" and "one label only" in messages[0]["content"]
