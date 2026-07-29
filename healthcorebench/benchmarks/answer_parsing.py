"""Answer parsing/normalization helpers for multiple-choice and judgement tasks.

Parsing is layered and kept separate from scoring. The multiple-choice extractor avoids
naive substring matching (which is fooled by option text appearing inside reasoning): it
prefers explicit answer markers ("answer is B", "\\boxed{B}", a leading "B."), then falls
back to a lone letter, and returns ``None`` when it cannot decide — never a guess.
"""

from __future__ import annotations

import re


# Explicit answer markers used by the languages present in the benchmark collection.  A
# marker is deliberately required before accepting a letter embedded in prose; this keeps
# explanations that merely discuss option letters from being scored as a final answer.
#
# The bare nouns ``options``/``choices`` are deliberately absent: ordinary reasoning prose
# ("...the characteristics of each option:") ends in a marker+link and would hijack the
# extractor, returning the first enumerated option instead of the model's stated answer.
# ``answers?`` stays because "the answer is X" is the single most common explicit form.
# Localized diagnostic phrasings ("最も可能性の高い疾患は") are listed before the shorter
# alternatives so the regex prefers the specific form over a substring of it.
_ANSWER_MARKER = (
    r"(?:final\s+answer|correct\s+(?:answers?|options?|choices?)|answers?|"
    r"diagnosis|conclusion|impression|"
    r"最も可能性の高い(?:疾患|診断|もの)|最も考えられる(?:疾患|診断)|"
    r"最も適切な(?:対応|治療|検査|処置|薬剤|選択肢|もの)|"
    r"最有可能的(?:诊断|疾病|答案)|最恰当的(?:处理|治疗|选择)|最合适的(?:选项|治疗|处理)|"
    r"最终答案|正確答案|正确答案|正确选项|答案|选择|"
    r"الإجابات\s+الصحيحة|الإجابة\s+النهائية|الإجابة\s+الصحيحة|الجواب\s+الصحيح|الخيار\s+الصحيح|الإجابات|الإجابة|الجواب|الخيارات|الخيار|"
    r"最終回答|正解|答え|正しい選択肢|최종\s*답변|정답|올바른\s*선택지|"
    r"réponse\s+finale|bonnes\s+réponses|bonne\s+réponse|réponses?\s+correctes?|"
    r"réponses?\s+(?:exactes?|justes?|attendues?)|"
    r"options\s+correctes|option\s+correcte|réponses|réponse|"
    r"respuesta\s+final|respuestas\s+correctas|respuesta\s+correcta|respuestas|respuesta|opciones\s+correctas|opción\s+correcta|"
    r"slutligt\s+svar|rätt\s+svar|svar|korrekt\s+alternativ|"
    r"окончательный\s+ответ|правильный\s+ответ|правильный\s+вариант|ответ|"
    r"risposta\s+finale|risposte\s+corrette|risposta\s+corretta|risposte|risposta|opzioni\s+corrette|opzione\s+corretta|"
    r"resposta\s+final|respostas\s+correctas|resposta\s+correcta|respostas|resposta|opcións\s+correctas|opción\s+correcta|"
    r"適切でない(?:検査|選択肢|回答)|不適切な(?:検査|選択肢|回答))"
)
_MARKER_LINK = (
    r"(?:\s*(?:is|are|would\s+be|should\s+be|must\s+be|est|es|sont|son|ist|är|è|é|"
    r"是|为|為|は|が|은|는|이|가|هو|هي|هما|это)\s*|\s*[:：=\-]\s*)"
)

# Mixed letter+digit terms (C5a, B12, HbA1c, T2DM) are pervasive in medical option text.
# Splitting them on script runs yields bare letters indistinguishable from option labels —
# ``"E. C5a"`` used to parse as A/C/E — so they are blanked before any letter extraction.
_ALPHANUMERIC_TERM = re.compile(r"\b(?=[A-Za-z]*\d)[A-Za-z0-9]+\b")


def _mask_alphanumeric_terms(text: str) -> str:
    return _ALPHANUMERIC_TERM.sub(lambda m: "#" * len(m.group()), text)


# Scripts that separate words with spaces but have no ``\b`` support in ``re``. A short token in
# one of these needs an explicit same-script boundary; Han and kana are excluded on purpose
# because they are written without separators (see ``_word_present``).
_SPACED_SCRIPT_CHARS = r"؀-ۿ가-힯"

# Bare selection verbs. These are not answer markers — ``_ANSWER_MARKER`` requires a linking word
# or a colon after it, which "选 A 和 C" does not have — but a response consisting of nothing but
# such a verb and option letters states a selection unambiguously. Longest forms first so the
# alternation prefers ``选择`` over its own prefix ``选``.
_CHOICE_LEAD_IN = (
    r"answers?|choices?|choose|chose|selects?|selected|picks?|picked|"
    r"réponses?|respuestas?|risposte?|respostas?|svar|ответ|الإجابة|"
    r"选择|选项|选|答案|答|正解|正答|回答|정답|답"
)

_OPTION_DISCUSSION_TERMS = re.compile(
    r"\b(?:wrong|incorrect|false|not\s+(?:correct|selected)|"
    r"faux|fausses?|incorrectes?|pas\s+(?:correctes?|retenues?))\b|"
    r"(?:誤り|不正解|正しくない|選択しない|错误|錯誤|不正确|不正確|오답|틀린)",
    re.IGNORECASE,
)


def final_answer_region(text: str) -> str:
    """Prefer the answer emitted after a model's closed reasoning block."""
    parts = re.split(r"</think\s*>", text, flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def _explicit_answer_tail(text: str) -> str | None:
    """Return text following the last localized answer marker, if present."""
    return next(iter(_explicit_answer_tails(text)), None)


def _explicit_answer_tails(text: str):
    """Yield localized answer-marker tails from last to first."""
    matches = list(re.finditer(_ANSWER_MARKER + _MARKER_LINK, text, re.IGNORECASE))
    for match in reversed(matches):
        yield text[match.end():]


def _strip_leading_markdown(text: str) -> str:
    """Remove answer-token emphasis without altering Markdown in the explanation."""
    return re.sub(r"^\s*(?:\*\*|__)", "", text, count=1)


def parse_multiple_choice_letter(text: str, valid_letters: list[str]) -> str | None:
    """Extract the chosen option letter from free-form model output.

    Returns one of ``valid_letters`` (upper-cased) or ``None`` if undecidable.
    """
    if not text:
        return None
    letters = "".join(re.escape(letter) for letter in valid_letters)
    # Mixed letter+digit terms are blanked before any letter extraction, exactly as in the
    # multi-answer extractor: without this "the answer is C5a" returned C, "B12 deficiency"
    # returned B and "D5W infusion" returned D — a fabricated option letter read out of a
    # drug/protein name, which scores as a confident wrong answer rather than a parse failure.
    upper = _mask_alphanumeric_terms(final_answer_region(text))
    if not upper:
        return None

    # 1) \boxed{X}. Text may follow the letter ("\boxed{E. C5a}"): the box is an explicit answer
    # slot, so the leading letter is the choice. A second bare letter ("\boxed{BD}") is a
    # multi-answer form and stays undecidable here.
    m = re.search(
        r"\\boxed\{\s*\(?\s*([" + letters + r"])\s*\)?(?![A-Za-z])"
        r"(?:\s*[\.\):、，,:：;；]\s*[^}]*)?\s*\}",
        upper,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Markdown emphasis is commonly used to mark a final choice. Accept an emphasized
    # response directly, or a unique emphasized choice on the last non-empty line; multiple
    # emphasized option lines remain ambiguous discussion rather than a guessed answer.
    markdown_choice = (
        r"(?:\*\*|__)\s*\(?\s*([" + letters + r"])\s*\)?"
        r"(?:\s*[\.\):、，,:：;；]\s*\S.*?)?\s*(?:\*\*|__)"
    )
    m = re.fullmatch(r"\s*" + markdown_choice + r"\s*", upper, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    markdown_lines = re.findall(r"(?im)^\s*" + markdown_choice + r"\s*$", upper)
    nonempty_lines = [line.strip() for line in upper.splitlines() if line.strip()]
    if len(markdown_lines) == 1 and re.fullmatch(
        markdown_choice, nonempty_lines[-1], re.IGNORECASE
    ):
        return markdown_lines[0].upper()

    # 2) localized explicit answer marker followed by an option letter. Instruction-following
    # models often quote the letter ("the answer is \"B\""), so quotes are stripped too.
    tail = _explicit_answer_tail(upper)
    if tail is not None:
        tail = _strip_leading_markdown(tail.lstrip(" \t\r\n:：=.-\"'“”「」"))
    m = (re.match(r"^\s*[\"'“”「」]?\s*\(?\s*([" + letters + r"])\s*\)?(?![A-Za-z])",
                  tail, re.IGNORECASE) if tail is not None else None)
    if m:
        return m.group(1).upper()

    # 3) leading "X." / "X)" / "(X)" at the very start
    m = re.match(r"^\s*\(?\s*([" + letters + r"])\s*[\.\):]", upper, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 4) the entire (stripped) output is a single valid letter
    if len(upper) == 1 and upper.upper() in valid_letters:
        return upper.upper()

    bare_letter_line = r"\(?\s*([" + letters + r"])\s*\)?[。.!！]?"

    # 5) Models often reason first and put the final choice on its own last line. A bare
    # letter there is unambiguous even when the provider omits a closing ``</think>`` tag.
    if nonempty_lines and re.fullmatch(bare_letter_line, nonempty_lines[-1], re.IGNORECASE):
        return re.search(r"[" + letters + r"]", nonempty_lines[-1], re.IGNORECASE).group().upper()

    # 5b) The mirror case: the model obeyed "output one letter", then explained anyway. Without
    # this rule such answers were undecidable and silently scored as wrong.
    if len(nonempty_lines) >= 2 and re.fullmatch(
        bare_letter_line, nonempty_lines[0], re.IGNORECASE
    ):
        return re.search(r"[" + letters + r"]", nonempty_lines[0], re.IGNORECASE).group().upper()

    # 5c) A single emphasized choice inside the closing sentence ("…は**A. 経過観察**です。") is a
    # stated answer, not discussion. Requires the emphasis to name one letter across the whole
    # response and to appear on the last non-empty line, so a walk-through that bolds several
    # options ("**A. discussed**\n**B. also discussed**") stays ambiguous.
    emphasized = {letter.upper() for letter in re.findall(markdown_choice, upper, re.IGNORECASE)}
    if (len(emphasized) == 1 and nonempty_lines
            and re.search(markdown_choice, nonempty_lines[-1], re.IGNORECASE)):
        return next(iter(emphasized))

    # 6) Accept an option line such as ``B. answer text`` — models routinely state the choice
    # this way and then explain below it, so the line need not be last. What made this unsafe was
    # *enumeration*: a response walking through every option ends on a distractor, and which
    # option lands last is arbitrary. Require the option lines to name a single letter. Terms like
    # "A は誤り" are then not a veto but a confirmation: the rejected options are being discussed
    # in prose while only the chosen one is presented as an option line.
    option_lines = [
        letter.upper() for letter in re.findall(
            r"(?im)^\s*\(?\s*([" + letters + r"])\s*[\.\):]\s+\S.*$", upper
        )
    ]
    option_line_letters = set(option_lines)
    if len(option_line_letters) == 1:
        return next(iter(option_line_letters))

    # 7) The response enumerated several options. Position alone is not evidence — taking the
    # last enumerated line scored 25.4% on 566 real responses, barely above the ~20% chance rate
    # for five options. Two enumeration shapes do carry signal, and only these are accepted:
    #
    #   a) the choice is *restated* as a closing option line, so its letter appears twice
    #      (70.0% on 140 real responses);
    #   b) the choice is stated first and the remaining option lines then walk the rejected
    #      options in alphabetical order while the prose calls them wrong (65.2% on 187).
    #
    # Anything less certain is reported as undecidable rather than guessed.
    if option_lines and option_lines.count(option_lines[-1]) >= 2:
        return option_lines[-1]
    rest = option_lines[1:]
    if (rest and len(set(rest)) == len(rest) and rest == sorted(rest)
            and _OPTION_DISCUSSION_TERMS.search(upper)):
        return option_lines[0]

    # A lone isolated letter anywhere used to be accepted as a last resort. It reads the letter
    # out of ordinary prose ("Vitamin D deficiency" -> D, "hepatitis B" -> B) and never fired
    # legitimately in a 380-response audit, so an undecidable response is now reported as such.
    return None


def parse_multiple_choice_letters(text: str, valid_letters: list[str]) -> list[str] | None:
    """Extract one-or-more chosen option letters from free-form model output.

    Returns a sorted, de-duplicated list of letters drawn from ``valid_letters`` (upper-cased),
    or ``None`` if nothing could be decided. Handles multi-answer questions where the model
    may reply "B, D", "B and D", "\\boxed{BD}", "answer: B,D", etc. When the output looks like
    a single clean choice, this degrades to a one-element list.
    """
    if not text:
        return None
    letters = "".join(re.escape(letter) for letter in valid_letters)
    valid = set(valid_letters)
    # Blank out mixed letter+digit terms before any letter extraction: this extractor tokenizes
    # on script runs, which turned an option text such as ``E. C5a`` into the answer set A/C/E.
    upper = _mask_alphanumeric_terms(final_answer_region(text))
    if not upper:
        return None

    def _collect(segment: str, dense: bool = False) -> list[str]:
        # In an explicit answer region (dense=True) letters may run together ("BD", "ACE"),
        # so grab every valid letter. In the anywhere fallback (dense=False) require the letter
        # to be isolated, so option letters inside prose ("A patient...") aren't mistaken for answers.
        if dense:
            found = re.findall(r"[" + letters + r"]", segment, re.IGNORECASE)
        else:
            found = re.findall(r"(?<![A-Za-z])([" + letters + r"])(?![A-Za-z])", segment, re.IGNORECASE)
        picks = []
        for f in found:
            u = f.upper()
            if u in valid and u not in picks:
                picks.append(u)
        return sorted(picks)

    def _structured_region(segment: str) -> list[str] | None:
        """Parse a region that must contain only option tokens and list joiners."""
        segment = segment.strip()
        if (segment.isupper()
                and re.fullmatch(
                    r"[" + letters + r"]{1," + str(len(valid_letters)) + r"}", segment
                )):
            return _collect(segment, dense=True)

        joiners = {
            "and", "or", "et", "ou", "y", "o", "e", "och", "eller", "и", "или",
            "و", "أو", "与", "和", "以及", "或", "及", "と", "または", "및", "또는",
        }
        picks: list[str] = []
        # Keep script runs separate so compact multilingual forms such as ``A和C`` tokenize
        # into A / 和 / C instead of one mixed-script word.
        tokens = re.findall(
            r"[A-Za-z]+|[\u0600-\u06ff]+|[\u4e00-\u9fff]+|"
            r"[\u3040-\u30ff]+|[\uac00-\ud7af]+",
            segment,
        )
        for token in tokens:
            upper_token = token.upper()
            if len(token) == 1 and upper_token in valid:
                if upper_token not in picks:
                    picks.append(upper_token)
            elif token.lower() in joiners or token in joiners:
                continue
            else:
                return None
        return sorted(picks) or None

    # 1) \boxed{...} — require a structured option list inside the box. Dense scanning used
    # to turn ``\boxed{B and D}`` into A/B/D by reading the A in "and" as another choice.
    m = re.search(r"\\boxed\{([^}]*)\}", upper, re.IGNORECASE)
    if m:
        picks = _structured_region(m.group(1))
        if picks:
            return picks

    # 2) An explicit localized final-answer marker overrides option-like lines in the
    # explanation. Only the first line of the marker tail is parsed, so later discussion
    # cannot replace or add choices.
    for explicit_tail in _explicit_answer_tails(upper):
        tail = explicit_tail.lstrip(" \t\r\n:：=.-")
        tail = _strip_leading_markdown(tail.splitlines()[0].strip())
        picks: list[str] = []
        for tok in re.findall(
            r"[A-Za-z]+|[\u0600-\u06ff]+|[\u4e00-\u9fff]+|"
            r"[\u3040-\u30ff]+|[\uac00-\ud7af]+",
            tail,
        ):
            up = tok.upper()
            if len(tok) == 1 and up in valid:
                if up not in picks:
                    picks.append(up)
            elif tok.lower() in {
                "and", "or", "et", "ou", "y", "o", "e", "och", "eller", "и", "или",
            } or tok in {"و", "أو", "与", "和", "以及", "或", "及", "と", "または", "및", "또는"}:
                continue
            else:
                break
        if picks:
            return sorted(picks)

    # 3) A structured answer list on the first non-empty line remains unambiguous when the
    # following lines are explanatory prose. Requiring the complete first line to be a list
    # prevents option discussion from leaking letters into the answer.
    nonempty_lines = [line.strip() for line in upper.splitlines() if line.strip()]
    if len(nonempty_lines) >= 2:
        picks = _structured_region(_strip_leading_markdown(nonempty_lines[0]))
        if picks:
            return picks

    # 4) Some models enumerate only the selected options, one per line. Check the whole
    # response before localized markers because option text such as French ``bonne réponse``
    # contains a marker word of its own and would otherwise make only the last line visible.
    option_lines: list[str] = []
    all_option_lines = len(nonempty_lines) >= 2
    for line in nonempty_lines:
        match = re.match(
            r"^\(?\s*([" + letters + r"])\s*[\.\):、，,:：;；]\s*\S.*$",
            line,
            re.IGNORECASE,
        )
        if match is None:
            all_option_lines = False
            break
        option_lines.append(match.group(1).upper())
    if all_option_lines:
        if _OPTION_DISCUSSION_TERMS.search(upper):
            return None
        picks = sorted(set(option_lines))
        if picks:
            return picks

    # 5) A bare answer list (``B, D`` / ``B and D`` / ``BD``) is structured enough to parse
    # without a marker. Require the entire response to be the list so explanatory prose cannot
    # leak option letters into the answer.
    list_joiners = (
        r"(?:[,;/+&、，،]|\s+|\band\b|\bor\b|\bet\b|\bou\b|\by\b|\bo\b|"
        r"\boch\b|\beller\b|и|или|و|أو|与|和|以及|或|及|と|または|및|또는)"
    )
    bare_list = re.fullmatch(
        r"\s*\(?\s*([" + letters + r"])\s*\)?"
        r"(?:\s*" + list_joiners + r"\s*\(?\s*([" + letters + r"])\s*\)?)+\s*\.?\s*",
        upper,
        re.IGNORECASE,
    )
    if bare_list:
        picks = _collect(upper)
        if picks:
            return picks
    if (upper.isupper()
            and re.fullmatch(r"[" + letters + r"]{1," + str(len(valid_letters)) + r"}", upper)):
        picks = _collect(upper, dense=True)
        if len(picks) == len(upper):
            return picks

    # 5c) The whole response is a selection and nothing else: an optional "choose"/"答案"
    # lead-in, then option letters joined by separators or list words. ``_structured_region``
    # rejects the segment as soon as any other word appears, so this fires on "(A)(C)" and
    # "选 A 和 C" — separator shapes the joiner list above does not spell out — while prose such
    # as "hepatitis B infection" cannot reach it. It runs before the single-answer fallback
    # because that fallback reads "(A)(C)" as the single choice A, silently dropping C.
    lead_stripped = re.sub(
        r"^\s*(?:" + _CHOICE_LEAD_IN + r")\s*[:：=\-,，、]?\s*", "", upper,
        count=1, flags=re.IGNORECASE,
    )
    # An unbracketed leading letter followed by a label separator is the "chosen option plus
    # its own option text" form, not a list of picks. It must never reach _structured_region:
    # option E of a Chinese/African exam item is routinely written "E. B and D", and reading
    # its text as further selections turned a correct ['E'] into ['B','D','E']. Over the stored
    # corpus this shape accounted for 116 responses, every one of which the unguarded rule got
    # wrong. Bracketed letters are exempt so "(A)(C)" still parses.
    label_prefix = re.match(
        r"^\s*[" + letters + r"]\s*[\.\):、，,:：;；]", lead_stripped, re.IGNORECASE,
    )
    if not label_prefix:
        picks = _structured_region(lead_stripped)
        if picks:
            return picks

    # 6) fall back to the single-letter extractor for clean single answers.
    single = parse_multiple_choice_letter(text, valid_letters)
    if single is not None:
        return [single]

    # 7) Nothing decided. This used to return "any isolated valid letter appearing anywhere",
    # which mined an answer out of ordinary prose: "hepatitis B infection" -> ['B'],
    # "Vitamin D deficiency" -> ['D'], and most often the English article — "the analysis got
    # stuck in a loop" -> ['A']. Over 2,003,476 stored responses the rule fired 86 times and 53
    # of those (61.6%) disagreed with the reference, so it mostly invented a concrete answer for
    # a response that stated none, which set_match then scored as a real choice. An undecidable
    # response is reported as undecidable instead.
    return None


# Whole-word tokens per class. Single letters like "y"/"n" are deliberately excluded from
# the "anywhere" search because they match inside ordinary words (e.g. "n" in "answer").
_YES_WORDS = {
    "yes", "true", "correct", "是", "نعم", "はい", "예", "네", "oui", "sí",
    "ja", "да", "sì",
}
_NO_WORDS = {
    "no", "false", "incorrect", "否", "不是", "لا", "いいえ", "아니요", "아닙니다",
    "non", "nej", "нет",
}
_MAYBE_WORDS = {
    "maybe", "possibly", "unsure", "也许", "可能", "ربما", "おそらく", "아마도",
    "peut-être", "quizás", "quizais", "kanske", "возможно", "forse",
}
# Excluded from the unanchored "anywhere" search, though still accepted as a leading token or
# after an explicit marker: "correct"/"incorrect" appear inside ordinary judgements that state
# the opposite ("the lesion is not correct in size"), and bare "si" is Spanish "if" as well as
# the SI unit prefix, so it matched "si units of measurement".
_AMBIGUOUS_ANYWHERE = {"correct", "incorrect", "si"}
_LEADING = {
    "yes": "yes", "y": "yes", "true": "yes",
    "no": "no", "n": "no", "false": "no",
    "maybe": "maybe", "si": "yes",
}
_LEADING.update({word: "yes" for word in _YES_WORDS})
_LEADING.update({word: "no" for word in _NO_WORDS})
_LEADING.update({word: "maybe" for word in _MAYBE_WORDS})


def _word_present(word: str, text: str) -> bool:
    """Whole-word presence test, per the target script's own notion of a word.

    ASCII uses ``\\b``. Arabic and Hangul are space-separated but have no ``\\b`` support here, so
    a bare substring test matched inside longer words — Arabic ``لا`` ("no") inside ``الالتهاب``
    ("inflammation"), Hangul ``예`` ("yes") inside ``예를`` ("for example"); those get an explicit
    same-script boundary. Han and kana are *not* space-separated, so a boundary rule there would
    reject every legitimate match; nesting between classes (``是`` inside ``不是``) is resolved by
    longest-match precedence at the call site instead.
    """
    if word.isascii():
        return re.search(r"\b" + re.escape(word) + r"\b", text) is not None
    if re.search(r"[" + _SPACED_SCRIPT_CHARS + r"]", word):
        return re.search(
            r"(?<![" + _SPACED_SCRIPT_CHARS + r"])" + re.escape(word)
            + r"(?![" + _SPACED_SCRIPT_CHARS + r"])",
            text,
        ) is not None
    return word in text


def _longest_matches(words: set[str], text: str) -> set[str]:
    """Present forms with any that are a proper substring of another present form removed.

    Chinese offers no boundary to separate ``是`` ("yes") from ``不是`` ("is not"), so the shorter
    token must lose to the longer one or a negated sentence reads as an affirmation.
    """
    present = {word for word in words if _word_present(word, text)}
    return {
        word for word in present
        if not any(other != word and word in other for other in present)
    }


def parse_yes_no_maybe(text: str) -> str | None:
    """Parse a yes/no/maybe judgement. Returns 'yes'/'no'/'maybe' or None."""
    if not text:
        return None
    t = final_answer_region(text).lower()
    if not t:
        return None
    # boxed / explicit
    judgment_alt = "|".join(re.escape(x) for x in sorted(
        _YES_WORDS | _NO_WORDS | _MAYBE_WORDS, key=len, reverse=True
    ))
    m = re.search(r"\\boxed\{\s*(" + judgment_alt + r")\s*\}", t)
    if m:
        return _canon(m.group(1))
    # explicit "answer is X" marker, optionally quoted ("the answer is \"yes.\"")
    tail = _explicit_answer_tail(t)
    if tail is not None:
        tail = tail.lstrip(" \t\r\n:：=.-\"'“”「」")
    m = (re.match(r"^\s*\(?\s*(" + judgment_alt + r")(?:\b|[\"'”」。،,.!?！？])",
                  tail, re.IGNORECASE) if tail is not None else None)
    if m:
        return _canon(m.group(1))
    # Leading token. An ambiguous form only counts as the verdict when it stands *as* the answer:
    # bare "si" is Spanish "yes" but also the SI unit system, so "si units of measurement" was
    # read as a yes. What separates the two is what directly follows -- a clause separator or the
    # end of the text leaves "si" as the answer ("si, es correcto"), whereas another word makes it
    # a modifier of that word ("si units"). Unambiguous forms are unaffected.
    leading = re.match(r"\s*([^\s\.,:;!?،。！？：；]+)(.*)", t, re.S)
    if leading and leading.group(1) in _LEADING:
        first = leading.group(1)
        if first not in _AMBIGUOUS_ANYWHERE or not re.match(r"\s+\w", leading.group(2)):
            return _LEADING[first]
    # Anywhere, unambiguous (whole-word). Forms whose presence does not imply the verdict
    # ("not correct", "si units") are excluded from this unanchored pass, and a form nested in a
    # longer match from another class is dropped so ``不是`` beats ``是``.
    matched = _longest_matches(
        (_YES_WORDS | _NO_WORDS | _MAYBE_WORDS) - _AMBIGUOUS_ANYWHERE, t
    )
    picks = {
        label
        for label, words in (("yes", _YES_WORDS), ("no", _NO_WORDS), ("maybe", _MAYBE_WORDS))
        if matched & words
    }
    if len(picks) == 1:
        return next(iter(picks))
    return None


def _canon(tok: str) -> str:
    tok = tok.lower()
    if tok in _YES_WORDS:
        return "yes"
    if tok in _NO_WORDS:
        return "no"
    return "maybe"


def parse_label(text: str, labels: list[str], aliases: dict[str, str] | None = None) -> str | None:
    """Parse a single categorical label from free-form output against a fixed label set.

    Returns the canonical label (as given in ``labels``) or ``None`` if undecidable. Matching is
    case-insensitive and whole-word. ``aliases`` maps extra surface forms to a canonical label
    (e.g. {"entails": "entailment", "contradicts": "contradiction"}). An explicit
    "answer is X" / ``\\boxed{X}`` marker takes precedence; otherwise a label must appear
    unambiguously (exactly one of the label set present) to be returned — never a guess.
    """
    if not text or not labels:
        return None
    t = final_answer_region(text).lower()
    if not t:
        return None
    canon = {label.lower(): label for label in labels}
    alias_map = {a.lower(): canon.get(c.lower(), c) for a, c in (aliases or {}).items()}
    surface = dict(canon)
    surface.update(alias_map)
    # longest surface forms first so multi-word labels win over substrings.
    forms = sorted(surface, key=len, reverse=True)
    alt = "|".join(re.escape(f) for f in forms)

    # An answer consisting only of a label is the safest and most common classification
    # response. Handle labels ending in punctuation or parentheses before whole-word matching,
    # whose ``\b`` boundary is not defined at a trailing non-word character.
    direct = re.sub(r"^(?:\*\*|__)|(?:\*\*|__)$", "", t).strip()
    direct = direct.rstrip("。.!！").strip()
    if direct in surface:
        return surface[direct]

    # 1) explicit marker: \boxed{X} or "answer/diagnosis is X". Markdown emphasis is
    # accepted because instruction-following models commonly emit ``**appendicitis**``.
    label_marker = (
        _ANSWER_MARKER[:-1] +
        r"|label|prediction|conclusion|diagnosis|结论|标签|诊断|تصنيف|تشخيص|"
        r"分類|診断|분류|진단|étiquette|diagnostic|etiqueta|diagnóstico|"
        r"klass|diagnos|метка|диагноз|etichetta|diagnosi)"
    )
    for pat in (r"\\boxed\{\s*(" + alt + r")\s*\}",
                label_marker + _MARKER_LINK +
                r"[\"']?\s*(?:\*\*|__)?\s*(" + alt +
                r")(?:\*\*|__)?(?=$|[\s。.!！,，;；:：])"):
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            return surface[m.group(1).lower()]

    # 2) leading token equals a label/alias.
    first = re.split(r"[\s\.,:;!?\"']+", t, maxsplit=1)[0]
    if first in surface:
        return surface[first]

    # 3) Unambiguous whole-word presence anywhere. ``forms`` is longest-first, so when a label
    # set nests one label inside another (IOR-Bench has both ``风湿免疫科`` and ``风湿免疫科门诊``)
    # the longer surface form wins instead of the two cancelling each other out as "ambiguous".
    present = [f for f in forms if _word_present(f, t)]
    if not present:
        return None
    longest = present[0]
    if all(other in longest for other in present):
        return surface[longest]
    # Genuinely competing labels remain undecidable — never guess between them.
    if len({surface[f] for f in present}) == 1:
        return surface[longest]
    return None
