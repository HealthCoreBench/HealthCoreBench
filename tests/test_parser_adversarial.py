"""Adversarial regression tests for the multiple-choice / judgement / label parsers.

Every case here is a real failure mode observed in ``runs/`` on 97,805 stored model responses,
or a synthetic minimisation of one. The parsers are deliberately conservative: returning ``None``
is correct when a response is genuinely undecidable, because an unparsed answer is recorded as a
parse failure whereas a guessed letter is silently scored as if the model had chosen it.

Two enumeration heuristics ARE kept, each justified by measured precision on the stored responses
(chance is ~20% for a five-option question):

* a restated closing option line — 70.0% on 140 responses;
* the choice stated first, followed by the rejected options in alphabetical order while the prose
  calls them wrong — 65.2% on 187 responses.

The naive "last option line of an enumeration" rule they replaced scored 25.4% on 566 responses,
and the "single isolated letter anywhere" rule scored 0 legitimate hits in a 380-response audit.
"""

from __future__ import annotations

import pytest

from healthcorebench.benchmarks.answer_parsing import (
    parse_label,
    parse_multiple_choice_letter,
    parse_multiple_choice_letters,
    parse_yes_no_maybe,
)

ABCDE = ["A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------- #
# Mixed letter+digit medical terms must not contribute option letters.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "Après analyse, la réponse correcte est :\n\nE. C5a",
        r"\boxed{E. C5a}",
        "Answer: E. C3 deficiency",
        "The answer is E. Vitamin B12 deficiency",
        "答案：E. HbA1c 升高",
    ],
)
def test_alphanumeric_terms_do_not_leak_option_letters(text: str) -> None:
    """``C5a`` used to tokenize into C and a, turning one correct choice into the set A/C/E."""
    assert parse_multiple_choice_letters(text, ABCDE) == ["E"]


# --------------------------------------------------------------------------- #
# A letter embedded in ordinary prose is not an answer.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "This is a 45-year-old patient. Vitamin D deficiency is likely.",
        "The patient has hepatitis B infection with no cirrhosis.",
        "Type A behaviour pattern was described by Friedman.",
    ],
)
def test_isolated_letter_in_prose_is_not_an_answer(text: str) -> None:
    assert parse_multiple_choice_letter(text, ["A", "B", "C", "D"]) is None


def test_generic_option_noun_does_not_hijack_the_answer() -> None:
    """"...each option:" is reasoning prose, not a final-answer marker.

    This response used to parse as ``A`` — the first line of the enumeration that followed the
    colon — even though the model states ``B`` in its closing sentence.
    """
    text = (
        "To answer this we need to understand the characteristics of each option:\n\n"
        "A. Ventricular bigeminy is a rhythm with alternating beats.\n"
        "B. Electrical alternans is beat-to-beat amplitude variation.\n\n"
        "Therefore the answer would be **B**. Electrical alternans."
    )
    assert parse_multiple_choice_letter(text, ["A", "B", "C", "D"]) == "B"


# --------------------------------------------------------------------------- #
# Localized answer statements.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ("したがって、この状況では、最も適切な対応は**A. 経過観察**です。", "A"),
        ("したがって、最も可能性の高い疾患はE. ウイルス性心筋炎です。", "E"),
        ("最も考えられる診断はC. 急性膵炎です。", "C"),
        ("정답은 B입니다.", "B"),
        ("最有可能的诊断是D。", "D"),
        ("The most likely diagnosis is D. Neuroblastoma.\n\nKey findings include:\n- mass", "D"),
    ],
)
def test_localized_answer_markers(text: str, expected: str) -> None:
    assert parse_multiple_choice_letter(text, ABCDE) == expected


def test_bare_letter_on_the_first_line_is_an_answer() -> None:
    """The model obeyed "output one letter" and then explained anyway."""
    text = "A\n\n의료광고에 관한 「의료법」은 'B' 병원이 아니다."
    assert parse_multiple_choice_letter(text, ABCDE) == "A"


@pytest.mark.parametrize(
    "text, expected",
    [
        ('The answer is "B". Option A is wrong.', "B"),
        ("The answer is 'C'.", "C"),
    ],
)
def test_quoted_answer_after_a_marker(text: str, expected: str) -> None:
    assert parse_multiple_choice_letter(text, ABCDE) == expected


def test_french_answer_marker_tolerates_an_adjective() -> None:
    assert parse_multiple_choice_letters(
        "Les réponses exactes sont A, C, D.", ABCDE
    ) == ["A", "C", "D"]


# --------------------------------------------------------------------------- #
# Enumerations: only the two measured-reliable shapes are accepted.
# --------------------------------------------------------------------------- #
def test_restated_closing_option_line_is_accepted() -> None:
    text = (
        "Let us review the options.\n"
        "A. first\nB. second\nC. third\n\n"
        "正解は B です。\nB. second\n"
    )
    assert parse_multiple_choice_letter(text, ABCDE) == "B"


def test_answer_stated_first_then_rejections_in_order_is_accepted() -> None:
    text = (
        "La proposition correcte est :\n\nD. amoxicilline\n\n"
        "Les autres options sont incorrectes :\n\nA. x\nB. y\nC. z\nE. w"
    )
    assert parse_multiple_choice_letter(text, ABCDE) == "D"


def test_plain_enumeration_without_either_signal_is_undecidable() -> None:
    """Position in a walk-through is not evidence: the naive rule scored 25.4% (~chance)."""
    text = (
        "Dans les conditions physiologiques, le pH le plus élevé est mesuré dans :\n\n"
        "B. La bile vésiculaire\nC. Le suc pancréatique\nD. La salive\n"
        "E. Les sécrétions intestinales\n"
    )
    assert parse_multiple_choice_letter(text, ABCDE) is None


def test_single_option_line_wins_even_while_rejections_are_discussed() -> None:
    """"A は誤り" confirms the single presented option rather than making it ambiguous."""
    text = (
        "免疫性血小板減少性紫斑病について正しい選択肢は次の通りです：\n\n"
        "C. 皮下出血を起こしやすい。\n\n"
        "ITPは通常は後天性であり先天性ではありません（Aは誤り）。"
        "骨髄の巨核球は通常正常です（Bは誤り）。"
    )
    assert parse_multiple_choice_letter(text, ABCDE) == "C"


def test_several_emphasized_options_remain_ambiguous() -> None:
    text = "Analysis:\n**A. discussed option**\n**B. another discussed option**"
    assert parse_multiple_choice_letters(text, ABCDE) is None


# --------------------------------------------------------------------------- #
# yes / no / maybe.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        # Short non-ASCII tokens must not match inside longer words.
        ("الالتهاب الكبدي مرتبط بتليف الكبد.", None),          # لا ("no") inside الالتهاب
        ("예를 들어 폐렴이 있습니다.", None),                      # 예 ("yes") inside 예를
        ("The patient has si units of measurement recorded.", None),
        # Same ambiguity, but "si" *leads* the answer. The unanchored pass excludes "si"; the
        # leading-token path did not, so these were read as a Spanish "yes". What separates the
        # two readings is what follows: another word makes "si" modify it, punctuation or the end
        # of the text leaves "si" as the answer.
        ("si units", None),
        ("SI units are used throughout.", None),
        ("si", "yes"),
        ("si.", "yes"),
        ("si, es correcto.", "yes"),
        ("sí", "yes"),
        ("The lesion is not correct in size; it measures 3cm.", None),
        # Chinese has no word boundary, so the longer form must win.
        ("患者不是糖尿病。", "no"),
        ("患者是糖尿病。", "yes"),
        # Ordinary forms still parse.
        ("Yes, it is.", "yes"),
        ("no", "no"),
        ("maybe", "maybe"),
        ("はい", "yes"),
        ("いいえ", "no"),
        ("예", "yes"),
        ("아니요", "no"),
        ("نعم", "yes"),
        ("لا", "no"),
        ('The answer is "yes." There were no differences.', "yes"),
    ],
)
def test_yes_no_maybe(text: str, expected: str | None) -> None:
    assert parse_yes_no_maybe(text) == expected


# --------------------------------------------------------------------------- #
# Labels.
# --------------------------------------------------------------------------- #
def test_nested_labels_prefer_the_longer_surface_form() -> None:
    """IOR-Bench ships both ``风湿免疫科`` and ``风湿免疫科门诊``; they used to cancel out."""
    labels = ["风湿免疫科", "风湿免疫科门诊", "心内科"]
    assert parse_label("应该挂风湿免疫科门诊。", labels) == "风湿免疫科门诊"
    assert parse_label("应该挂风湿免疫科。", labels) == "风湿免疫科"


def test_two_competing_labels_stay_undecidable() -> None:
    labels = ["entailment", "contradiction", "neutral"]
    assert parse_label("Both entailment and contradiction are plausible.", labels) is None
    assert parse_label("entailment", labels) == "entailment"


# --------------------------------------------------------------------------- #
# Clean forms must keep working.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, single, multi",
    [
        ("B", "B", ["B"]),
        ("The answer is C.", "C", ["C"]),
        (r"\boxed{D}", "D", ["D"]),
        ("**A**", "A", ["A"]),
        ("A. Dog", "A", ["A"]),
        ("reasoning...\n\nE", "E", ["E"]),
    ],
)
def test_unambiguous_single_answers(text: str, single: str, multi: list[str]) -> None:
    assert parse_multiple_choice_letter(text, ABCDE) == single
    assert parse_multiple_choice_letters(text, ABCDE) == multi


@pytest.mark.parametrize(
    "text, expected",
    [
        ("B, D", ["B", "D"]),
        ("B and D", ["B", "D"]),
        (r"\boxed{BD}", ["B", "D"]),
        ("答案：A, B, E", ["A", "B", "E"]),
        ("A和C", ["A", "C"]),
    ],
)
def test_unambiguous_multi_answers(text: str, expected: list[str]) -> None:
    assert parse_multiple_choice_letters(text, ABCDE) == expected


# --------------------------------------------------------------------------- #
# The multi-answer parser's removed last-resort step.
#
# Step 7 used to return "every valid option letter that appears in isolation anywhere in the
# response". Over the 2,004,314 stored multiple-choice responses under runs/ it fired 84 times
# and 53 of those (63.1%) disagreed with the reference — it was not recovering answers, it was
# mining them out of prose. 62 of the 84 were the letter A, almost always the English article
# or the word "Analysis". Every input below must now be reported as undecidable rather than
# answered, because a fabricated concrete choice is scored as a real one by set_match while a
# parse failure is not.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        # Medical vocabulary whose letters are not option letters.
        "hepatitis B infection",
        "The patient has hepatitis B infection with no cirrhosis.",
        "vitamin D deficiency",
        "Vitamin D deficiency is the most likely underlying cause.",
        "C5a complement",
        "T2-weighted MRI",
        "Type 2 diabetes",
        "Elevated HbA1c and a positive B12 assay.",
        # Prose stating no choice at all.
        "The patient improved after treatment and was discharged home.",
        "I am not able to determine the correct option from the information given.",
        # The single most common real trigger: reasoning that stalled and named nothing.
        "The analysis got stuck in a loop. Let's step back and reconsider the question.",
        "The assistant's previous output is nonsense due to a glitch. We need to produce an answer.",
    ],
)
def test_prose_never_yields_a_fabricated_option_set(text: str) -> None:
    assert parse_multiple_choice_letters(text, ABCDE) is None
    assert parse_multiple_choice_letter(text, ABCDE) is None


# --------------------------------------------------------------------------- #
# A response that is nothing but a selection is still parsed.
# Dropping step 7 must not cost the shapes that genuinely state a choice.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ("A and C", ["A", "C"]),
        ("(A)(C)", ["A", "C"]),
        ("A, C", ["A", "C"]),
        ("选 A 和 C", ["A", "C"]),
        ("答案 A 和 C", ["A", "C"]),
        ("AとC", ["A", "C"]),
        ("Answer A and C", ["A", "C"]),
    ],
)
def test_bare_selection_is_still_parsed(text: str, expected: list[str]) -> None:
    assert parse_multiple_choice_letters(text, ABCDE) == expected


# --------------------------------------------------------------------------- #
# "E. B and D" is option E, whose option *text* happens to name other letters.
#
# Reading the option text as further selections turned a correct ['E'] into ['B','D','E'].
# 116 stored responses have this shape and the unguarded rule got every one of them wrong,
# which is why the bare-selection rule above refuses to fire behind an option label.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ("E. B and D.", "E"),
        ("D. A, B and C", "D"),
        ("A. a and b.", "A"),
        ("A. B.", "A"),
        ("D. B-100.", "D"),
        ("A. 250 °C.", "A"),
        ("C. A, B, E, D, C", "C"),
        (r"E. \((4a + b) / 5\)", "E"),
        ("B. a-III, b-IV, c-V", "B"),
    ],
)
def test_option_text_naming_other_letters_is_not_a_multi_answer(
    text: str, expected: str
) -> None:
    assert parse_multiple_choice_letters(text, ABCDE) == [expected]


# --------------------------------------------------------------------------- #
# The single-answer parser masks alphanumeric terms too.
#
# It previously read the answer region unmasked, so a drug or protein name in the final
# sentence produced a confident wrong letter instead of a parse failure.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        ("The answer is C5a.", None),
        ("The answer is B12 deficiency.", None),
        ("The answer is D5W infusion.", None),
        # \boxed{} is an explicit answer slot, so a letter followed by its option text is
        # still that letter — the box, not the prose, decides.
        (r"\boxed{E. C5a}", "E"),
        (r"\boxed{D}", "D"),
        ("The answer is C.", "C"),
    ],
)
def test_single_parser_masks_alphanumeric_terms(text: str, expected: str | None) -> None:
    assert parse_multiple_choice_letter(text, ABCDE) == expected


# --------------------------------------------------------------------------- #
# Known conservative misses, pinned deliberately.
#
# "A、" is both a Chinese list separator and a Chinese option label, so "选择A、C" could be
# options A and C or option A whose text is "C". The parser refuses rather than guessing;
# these assert that the refusal is stable, not that the refusal is desirable.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["选择A、C", "选 A、C", "选项A、C"])
def test_chinese_label_separator_stays_undecidable(text: str) -> None:
    assert parse_multiple_choice_letters(text, ABCDE) is None
