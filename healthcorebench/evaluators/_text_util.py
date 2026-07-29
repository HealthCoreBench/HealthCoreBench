"""Shared text normalization / tokenization for the generation-style metrics.

EM, token-F1, ROUGE and BLEU all need the same two decisions made consistently: how to
normalize a string, and how to split it into tokens — including for Chinese, where there are
no spaces and naive whitespace tokenization collapses a whole sentence into one token. These
helpers centralize that so ``text_f1_em`` / ``rouge`` / ``bleu`` score comparably.

Chinese (and other CJK) text is word-segmented with jieba; Latin text is lowercased and split
on word characters. Mixed strings are handled segment by segment.
"""

from __future__ import annotations

import re
from collections import Counter

# CJK ranges: CJK Unified, Hiragana/Katakana, Hangul. Enough to route zh/ja/ko text to a
# word segmenter rather than character-blind whitespace splitting.
_CJK = re.compile(r"[一-鿿぀-ヿ가-힣]")
_ARTICLES = {"a", "an", "the"}

_jieba = None


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def _jieba_cut(text: str) -> list[str]:
    global _jieba
    if _jieba is None:
        import jieba  # lazy: only paid when a Chinese sample is actually scored
        import logging
        jieba.setLogLevel(logging.ERROR)
        _jieba = jieba
    return [t for t in _jieba.cut(text) if t.strip()]


def word_tokens(text, *, drop_articles: bool = True) -> list[str]:
    """Tokens for EM / token-F1: lowercased, punctuation dropped, CJK word-segmented.

    English articles are dropped (SQuAD convention) so "the flu" and "flu" match. CJK spans
    are segmented with jieba; Latin/digit runs become word tokens.
    """
    if text is None:
        return []
    s = str(text).lower().strip()
    if not s:
        return []
    toks: list[str] = []
    if has_cjk(s):
        # segment the whole string with jieba, then keep only alnum/CJK content tokens.
        for t in _jieba_cut(s):
            t = t.strip()
            if not t:
                continue
            if _CJK.search(t) or t.isalnum():
                toks.append(t)
    else:
        # Unicode-aware word split: matches Latin (incl. accents é/ü), Arabic, Cyrillic, Greek,
        # etc. — NOT just [a-z0-9], which would silently drop whole non-Latin strings (making
        # token-F1 compare two empty bags → a false 1.0) and truncate accented Latin ("café").
        toks = re.findall(r"[^\W_]+", s, re.UNICODE)
    if drop_articles:
        toks = [t for t in toks if t not in _ARTICLES]
    return toks


def normalized_string(text) -> str:
    """Canonical form for exact-match comparison: normalized tokens joined by spaces."""
    return " ".join(word_tokens(text))


def rouge_tokens(text) -> list[str]:
    """Tokens for ROUGE (keeps articles; ROUGE counts all content words)."""
    return word_tokens(text, drop_articles=False)


def token_f1(pred_tokens: list[str], ref_tokens: list[str]) -> float | None:
    """SQuAD-style token-overlap F1 between two token bags.

    Returns ``None`` when the reference bag is empty: there is no overlap to measure, so the
    comparison is undefined rather than perfect. Returning 1.0 here used to credit a degenerate
    reference (``"-"`` tokenizes to nothing) even against an unparsed prediction.
    """
    if not ref_tokens:
        return None
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def reference_candidates(sample: dict) -> list[str]:
    """All acceptable reference strings for a sample.

    A short answer may have several accepted forms via an explicit alias list
    (``reference_aliases``) or a list-valued reference. Metrics take the best match over these
    so a correct-but-aliased answer is not penalized.

    We do NOT auto-split a reference on ``/``: that over-credited answers where ``/`` is content
    ("mg/dL", "120/80", "and/or", dates) — a prediction of just one side would falsely match.
    Datasets whose ``/`` genuinely separates alternates (e.g. RareBench disease names) populate
    ``reference_aliases`` explicitly in their adapter instead.
    """
    out: list[str] = []

    def _add(x):
        x = str(x).strip()
        if x and x not in out:
            out.append(x)

    aliases = sample.get("reference_aliases")
    if isinstance(aliases, (list, tuple)):
        for a in aliases:
            _add(a)

    ref = sample.get("reference_answer_normalized")
    if ref is None:
        ref = sample.get("reference_answer")
    if isinstance(ref, (list, tuple)):
        for x in ref:
            _add(x)
    elif ref is not None:
        _add(ref)
    return out
