"""Prompt-localization tests: the instruction scaffold must follow the sample's language,
with an English fallback for languages without a template."""

from healthcorebench.benchmarks.prompts import (
    multiple_choice_prompt,
    multiple_answer_prompt,
)

_MC_MARK = {
    "en": "Return exactly one option letter",
    "zh": "请只输出一个正确选项",
    "fr": "Répondez uniquement par une lettre",
    "ja": "正しい選択肢の記号",
    "ko": "정답 선택지의 알파벳",
    "ar": "حرف الخيار الصحيح فقط",
    "sv": "exakt en bokstav",
    "es": "exactamente una letra",
    "ru": "одну букву правильного варианта",
    "it": "una sola lettera dell'opzione corretta",
    "gl": "única letra da opción correcta",
}


def test_multiple_choice_prompt_is_localized():
    for lang, marker in _MC_MARK.items():
        prompt = multiple_choice_prompt("Q?", "A. x\nB. y", lang=lang)
        assert marker in prompt, f"{lang} scaffold missing"
    # question label localized too (not always English "Question:")
    assert "質問" in multiple_choice_prompt("問い", "A. あ", lang="ja")


def test_unknown_language_falls_back_to_english():
    # a language with no template must not crash and must produce the English scaffold.
    prompt = multiple_choice_prompt("Q?", "A. x", lang="xx")
    assert "Return exactly one option letter" in prompt
    assert multiple_choice_prompt("Q?", "A. x", lang=None).startswith("Question: ")


def test_multiple_answer_prompt_localized_fr_ja():
    assert "Une ou plusieurs options" in multiple_answer_prompt("Q", "A. x", lang="fr")
    assert "正解は1つまたは複数" in multiple_answer_prompt("Q", "A. x", lang="ja")


def test_prompt_handles_braces_in_content():
    # case data can contain literal { } — must not raise (no str.format on content).
    p = multiple_choice_prompt("dict {'x': 1} in stem", "A. {y}\nB. z", lang="en")
    assert "{'x': 1}" in p and "{y}" in p


def test_autodetect_when_language_missing():
    # No declared language -> fall back to detecting the script of the question text.
    assert "正しい選択肢の記号" in multiple_choice_prompt("これは質問です", "A. あ", lang=None)   # kana -> ja
    assert "정답 선택지의 알파벳" in multiple_choice_prompt("이것은 질문입니다", "A. 가", lang=None)  # hangul -> ko
    assert "حرف الخيار الصحيح فقط" in multiple_choice_prompt("هذا سؤال", "A. x", lang=None)       # arabic
    assert "请只输出一个正确选项" in multiple_choice_prompt("这是一个问题", "A. 甲", lang=None)       # han -> zh
    assert "Return exactly one option letter" in multiple_choice_prompt("plain english", "A. x", lang=None)


def test_declared_language_overrides_detection():
    # Han-only text is ambiguous zh/ja; a declared language must win over script guessing.
    p = multiple_choice_prompt("漢字のみの質問", "A. 甲", lang="ja")
    assert "正しい選択肢の記号" in p            # ja template, not zh, despite Han-only script
