"""Shared prompt templates.

Kept faithful to the original HealthCoreBench prompt semantics so migrated benchmarks produce
comparable prompts. Each template is versioned via the adapter's ``prompt_template_version``.

Instruction scaffolding is localized: the wrapper around a question (the "Question:" /
"Options:" labels and the "answer with the letter" instruction) must be in the *same language*
as the question itself, or a non-English model is being asked in English to answer a
non-English question. Templates are keyed by language code with an English fallback; adapters
pass ``lang=sample.language`` so the wrapper always follows the data's language.

Templates are assembled by concatenation (not ``str.format``) on purpose: question/option text
can contain literal ``{`` / ``}`` (JSON-ish case data), which would break ``.format``.
"""

from __future__ import annotations

import re

# (question_label, options_label, instruction) per language for single-answer MCQA.
_MC = {
    "en": ("Question: ", "Options:", "Return exactly one option letter and no other text."),
    "zh": ("问题： ", "选项：", "请只输出一个正确选项的字母，不要输出任何其它内容。"),
    "fr": ("Question : ", "Options :", "Répondez uniquement par une lettre d'option correcte, sans aucun autre texte."),
    "ja": ("質問： ", "選択肢：", "正しい選択肢の記号（アルファベット）を1文字だけ出力し、他の内容は出力しないでください。"),
    "ko": ("질문: ", "선택지:", "정답 선택지의 알파벳 한 글자만 출력하고 다른 내용은 출력하지 마세요."),
    "ar": ("السؤال: ", "الخيارات:", "أخرج حرف الخيار الصحيح فقط دون أي نص آخر."),
    "sv": ("Fråga: ", "Alternativ:", "Svara med exakt en bokstav för rätt alternativ och ingen annan text."),
    "es": ("Pregunta: ", "Opciones:", "Responda con exactamente una letra de opción correcta y ningún otro texto."),
    "ru": ("Вопрос: ", "Варианты:", "Выведите ровно одну букву правильного варианта без другого текста."),
    "it": ("Domanda: ", "Opzioni:", "Rispondi con una sola lettera dell'opzione corretta e nient'altro."),
    "gl": ("Pregunta: ", "Opcións:", "Responde cunha única letra da opción correcta e sen ningún outro texto."),
}

# Same, for one-or-more-correct-answer MCQA.
_MA = {
    "en": ("Question: ", "Options:", "One or more options may be correct. Answer with all correct option letters from the given choices (comma-separated if more than one), directly."),
    "zh": ("问题： ", "选项：", "该题可能有一个或多个正确选项。请直接输出所有正确选项的字母（多个字母之间用逗号分隔），不要有任何其它输出。"),
    "fr": ("Question : ", "Options :", "Une ou plusieurs options peuvent être correctes. Répondez directement par toutes les lettres des options correctes (séparées par des virgules s'il y en a plusieurs)."),
    "ja": ("質問： ", "選択肢：", "正解は1つまたは複数の場合があります。正しい選択肢の記号をすべて（複数ある場合はカンマ区切りで）直接答えてください。"),
    "ko": ("질문: ", "선택지:", "정답은 하나 이상일 수 있습니다. 정답 선택지의 기호를 모두(여러 개인 경우 쉼표로 구분) 직접 답하세요."),
    "ar": ("السؤال: ", "الخيارات:", "قد يكون هناك خيار صحيح واحد أو أكثر. أجب مباشرةً بجميع حروف الخيارات الصحيحة (مفصولة بفواصل إن وُجد أكثر من واحد)."),
    "sv": ("Fråga: ", "Alternativ:", "Ett eller flera alternativ kan vara rätt. Svara direkt med alla bokstäver för de rätta alternativen (kommaseparerade om fler än ett)."),
    "es": ("Pregunta: ", "Opciones:", "Una o más opciones pueden ser correctas. Responda directamente con todas las letras de las opciones correctas (separadas por comas si hay más de una)."),
    "ru": ("Вопрос: ", "Варианты:", "Правильных вариантов может быть несколько. Выведите только их буквы через запятую, без другого текста."),
    "it": ("Domanda: ", "Opzioni:", "Una o più opzioni possono essere corrette. Scrivi solo le lettere corrette separate da virgole, senza altro testo."),
    "gl": ("Pregunta: ", "Opcións:", "Pode haber unha ou varias opcións correctas. Escribe só as letras correctas separadas por comas, sen outro texto."),
}

_YN = {
    "en": "Please output 'yes' or 'no' (no extra output).",
    "zh": "请输出'是'或'否'(不要有任何其它输出)。",
    "ar": "أخرج كلمة واحدة فقط: 'نعم' أو 'لا'.",
    "ja": "「はい」または「いいえ」のどちらか一語だけを出力してください。",
    "ko": "'예' 또는 '아니요' 중 한 단어만 출력하세요.",
    "fr": "Répondez uniquement par « oui » ou « non ».",
    "sv": "Svara endast med 'ja' eller 'nej'.",
    "es": "Responda únicamente 'sí' o 'no'.",
    "ru": "Ответьте только одним словом: «да» или «нет».",
    "it": "Rispondi soltanto con 'sì' o 'no'.",
    "gl": "Responde unicamente 'si' ou 'non'.",
}


def _detect_lang(text: str) -> str | None:
    """Best-effort language of a question from its script — a *fallback* only.

    The declared ``sample.language`` is always preferred (it distinguishes cases a script
    cannot, e.g. Han-only text that is Chinese vs Japanese). This is used solely when no usable
    language was passed. Japanese is detected by kana (Han-only text is treated as Chinese).
    """
    if not text:
        return None
    if re.search(r"[぀-ヿ]", text):   # hiragana / katakana -> Japanese
        return "ja"
    if re.search(r"[가-힣]", text):    # hangul -> Korean
        return "ko"
    if re.search(r"[؀-ۿ]", text):    # Arabic script
        return "ar"
    if re.search(r"[一-鿿]", text):    # Han (no kana) -> Chinese
        return "zh"
    return None


def _lang_key(table: dict, lang: str | None, text: str = "") -> str:
    """Resolve the template key: declared language first, then script fallback, then English."""
    if lang in table:
        return lang
    detected = _detect_lang(text)
    if detected in table:
        return detected
    return "en"


def multiple_choice_prompt(question: str, choices_block: str, lang: str = "en") -> str:
    q_label, o_label, instruction = _MC[_lang_key(_MC, lang, question)]
    return f"{q_label}{question}\n{o_label}\n{choices_block}\n{instruction}"


def multiple_answer_prompt(question: str, choices_block: str, lang: str = "en") -> str:
    """Prompt for single-question, one-or-more-correct-option tasks."""
    q_label, o_label, instruction = _MA[_lang_key(_MA, lang, question)]
    return f"{q_label}{question}\n{o_label}\n{choices_block}\n{instruction}"


def judgement_prompt(question: str, lang: str = "en") -> str:
    return f"{question}\n{_YN[_lang_key(_YN, lang, question)]}"


def format_lettered_choices(choices: list[str], letters: list[str] | None = None) -> tuple[str, list[str]]:
    """Return a newline-joined ``A. text`` block plus the letters used."""
    letters = letters or [chr(ord("A") + i) for i in range(len(choices))]
    block = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, choices))
    return block, letters[: len(choices)]
