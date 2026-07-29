"""Deterministic context budgeting for benchmarks with unusually long source records."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable


class ContextOverflowError(ValueError):
    """The configured model window cannot contain the fixed prompt and requested output."""


class UnbudgetedContextWarning(UserWarning):
    """No model window was configured, so no prompt trimming could be enforced."""


# Character classes a byte-pair tokenizer prices very differently. Calibrated against 8,748
# real prompts from runs/ with server-reported ``prompt_tokens`` (117 tasks, 15 languages):
# a marginal ASCII word costs ~1 token per 4 characters, digits ~1 per 3, punctuation ~1
# each, whitespace almost nothing (it merges into the following word), a CJK character
# ~1.35 tokens, and other non-ASCII (Arabic, Cyrillic, accented Latin) ~0.9. Unbroken
# alphabetic runs longer than a word (IDs, base64, DNA) match no merges and cost ~1 per 2.
_RUNS = re.compile(r"[A-Za-z]+|[0-9]+|\s+|[^A-Za-z0-9\s]", re.S)
_ASCII_WORD_CHARS_PER_TOKEN = 4.0
_UNBROKEN_RUN_CHARS_PER_TOKEN = 2.0
_UNBROKEN_RUN_THRESHOLD = 12
_ASCII_DIGITS_PER_TOKEN = 3.0
_TOKENS_PER_ASCII_PUNCT = 1.0
_TOKENS_PER_WHITESPACE = 0.05
_TOKENS_PER_CJK = 1.35
_TOKENS_PER_OTHER_NON_ASCII = 0.9
CALIBRATED_ESTIMATOR_NAME = "calibrated_script_aware_v2"


def _is_cjk(code_point: int) -> bool:
    return (0x3000 <= code_point <= 0x9FFF or 0xAC00 <= code_point <= 0xD7AF
            or 0xF900 <= code_point <= 0xFAFF or 0xFF00 <= code_point <= 0xFFEF)


def estimate_text_tokens(text: str, *, ascii_chars_per_token: float | None = None) -> int:
    """Return a tokenizer-independent token estimate for multilingual text.

    The default estimator prices each character class at its measured marginal cost (see the
    constants above). On the 1,383 real prompts of at least 1,500 characters it over-estimates
    by a median of 1.29x and under-estimates 0.9% of them, so it stays on the safe side of the
    window while leaving most of it usable.

    The previous fixed ratio — two ASCII characters per token, one token per non-ASCII
    character — was wrong in both directions: it over-estimated English clinical prose by
    1.90x (LongHealth documents run at 3.7 characters per token, so 64% of every source
    document was discarded while 75% of the window went unused) and *under*-estimated Chinese
    by up to 1.41x (a 3,230-character RJUA-QA record really costs 4,300 tokens, estimated at
    3,042). Passing ``ascii_chars_per_token`` explicitly selects that legacy fixed-ratio
    estimator, for callers that must reproduce an earlier run's budgeting exactly.
    """
    if ascii_chars_per_token is not None:
        ascii_count = sum(ord(ch) < 128 for ch in text)
        return math.ceil(ascii_count / ascii_chars_per_token) + (len(text) - ascii_count)

    tokens = 0.0
    for run in _RUNS.findall(text):
        first = run[0]
        if first.isascii():
            if first.isalpha():
                length = len(run)
                tokens += (math.ceil(length / _ASCII_WORD_CHARS_PER_TOKEN)
                           if length <= _UNBROKEN_RUN_THRESHOLD
                           else length / _UNBROKEN_RUN_CHARS_PER_TOKEN)
            elif first.isdigit():
                tokens += math.ceil(len(run) / _ASCII_DIGITS_PER_TOKEN)
            elif first.isspace():
                tokens += len(run) * _TOKENS_PER_WHITESPACE
            else:
                tokens += _TOKENS_PER_ASCII_PUNCT
        elif first.isspace():
            tokens += len(run) * _TOKENS_PER_WHITESPACE
        elif _is_cjk(ord(first)):
            tokens += _TOKENS_PER_CJK
        else:
            tokens += _TOKENS_PER_OTHER_NON_ASCII
    return math.ceil(tokens)


def _estimator_name(ascii_chars_per_token: float | None) -> str:
    if ascii_chars_per_token is None:
        return CALIBRATED_ESTIMATOR_NAME
    return f"conservative_multilingual_{ascii_chars_per_token}ascii_v1"


def fit_context_to_window(
    context: str,
    *,
    fixed_prompt: str,
    max_model_len: int | None,
    max_output_tokens: int | None,
    reserve_tokens: int,
    policy: str,
    ascii_chars_per_token: float | None = None,
    protected_spans: Iterable[tuple[int, int]] | None = None,
    protected_span_source: str | None = None,
) -> tuple[str, dict]:
    """Fit context to the configured window and return explicit provenance metadata."""
    original_estimate = estimate_text_tokens(
        context, ascii_chars_per_token=ascii_chars_per_token
    )
    metadata = {
        "context_truncated": False,
        "context_overflow_policy": policy,
        "context_estimator": _estimator_name(ascii_chars_per_token),
        "original_context_chars": len(context),
        "original_context_estimated_tokens": original_estimate,
    }
    normalized_spans = _normalize_spans(protected_spans or [], len(context))
    if normalized_spans:
        metadata.update({
            "protected_span_source": protected_span_source or "adapter",
            "protected_span_count": len(normalized_spans),
            "protected_spans_retained": True,
        })
    if max_model_len is None:
        # Without a window there is no budget to enforce, so nothing is trimmed. Say so
        # loudly and on the sample: silently skipping made a measured EHRBench/risk run
        # fail its 2,494 longest records ("your request has 27630 input tokens") and publish
        # a score over the remaining, systematically shorter 5,227 with no bias warning.
        metadata.update({
            "context_budget_enforced": False,
            "context_budget_skip_reason": "hardware.max_model_len_not_configured",
        })
        warnings.warn(
            "hardware.max_model_len is not set: no prompt trimming is enforced, so records "
            "longer than the served window will fail outright and the reported score will "
            "cover only the shorter records that fit. Set hardware.max_model_len to the "
            "served context length; context_overflow_policy and context_token_reserve have "
            "no effect until you do.",
            UnbudgetedContextWarning,
            stacklevel=2,
        )
        return context, metadata

    metadata["context_budget_enforced"] = True
    output_budget = max_output_tokens or 0
    context_budget = max_model_len - output_budget - reserve_tokens - estimate_text_tokens(
        fixed_prompt, ascii_chars_per_token=ascii_chars_per_token
    )
    metadata["context_token_budget"] = max(0, context_budget)
    if context_budget <= 0:
        raise ContextOverflowError(
            "No context budget remains after fixed prompt, output budget, and reserve: "
            f"max_model_len={max_model_len}, max_output_tokens={output_budget}, "
            f"reserve_tokens={reserve_tokens}. Reduce max_tokens or the fixed prompt."
        )
    if original_estimate <= context_budget:
        metadata["retained_context_chars"] = len(context)
        metadata["retained_context_estimated_tokens"] = original_estimate
        return context, metadata
    if policy == "error":
        raise ContextOverflowError(
            f"Estimated context length {original_estimate} exceeds available context budget "
            f"{context_budget}; set context_overflow_policy=head_tail or reduce max_tokens."
        )

    marker = "\n\n[... middle of source context omitted to fit the model window ...]\n\n"
    marker_tokens = estimate_text_tokens(marker, ascii_chars_per_token=ascii_chars_per_token)
    if marker_tokens > context_budget:
        raise ContextOverflowError(
            f"Context budget {context_budget} is too small even for the truncation marker."
        )
    if normalized_spans:
        fitted, protected_meta = _fit_protected_head_tail(
            context,
            protected_spans=normalized_spans,
            context_budget=context_budget,
            marker=marker,
            ascii_chars_per_token=ascii_chars_per_token,
        )
        metadata.update({
            "context_truncated": True,
            "context_truncation_strategy": "protected_head_tail",
            "retained_context_chars": len(fitted),
            "retained_context_estimated_tokens": estimate_text_tokens(
                fitted, ascii_chars_per_token=ascii_chars_per_token
            ),
            **protected_meta,
        })
        return fitted, metadata

    # Binary search the largest deterministic head+tail slice that fits. Keeping both ends
    # preserves record headers as well as recent events; the omission marker is visible to the
    # model and persisted with the sample rather than silently altering the source.
    low, high = 0, len(context)
    while low < high:
        keep = (low + high + 1) // 2
        head = (keep + 1) // 2
        candidate = context[:head] + marker + context[-(keep - head):] if keep > head else context[:head] + marker
        if estimate_text_tokens(candidate, ascii_chars_per_token=ascii_chars_per_token) <= context_budget:
            low = keep
        else:
            high = keep - 1
    head = (low + 1) // 2
    fitted = context[:head] + marker + context[-(low - head):] if low > head else context[:head] + marker
    retained_estimate = estimate_text_tokens(
        fitted, ascii_chars_per_token=ascii_chars_per_token
    )
    metadata.update({
        "context_truncated": True,
        "context_truncation_strategy": "head_tail",
        "retained_context_chars": len(fitted),
        "retained_context_estimated_tokens": retained_estimate,
        "omitted_context_chars": len(context) - low,
    })
    return fitted, metadata


def _normalize_spans(spans: Iterable[tuple[int, int]], context_length: int) -> list[tuple[int, int]]:
    normalized = []
    for start, end in spans:
        start = max(0, min(context_length, int(start)))
        end = max(start, min(context_length, int(end)))
        if end > start:
            normalized.append((start, end))
    return _merge_ranges(normalized)


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _render_ranges(context: str, ranges: Iterable[tuple[int, int]], marker: str) -> tuple[str, int]:
    merged = _merge_ranges(ranges)
    chunks = []
    retained_source_chars = 0
    for index, (start, end) in enumerate(merged):
        if index:
            chunks.append(marker)
        chunks.append(context[start:end])
        retained_source_chars += end - start
    return "".join(chunks), retained_source_chars


def _fit_protected_head_tail(
    context: str,
    *,
    protected_spans: list[tuple[int, int]],
    context_budget: int,
    marker: str,
    ascii_chars_per_token: float | None,
) -> tuple[str, dict]:
    """Retain benchmark-declared evidence plus as much surrounding/head/tail text as fits."""

    def render(evidence_margin: int, edge_chars: int) -> tuple[str, int]:
        ranges = [
            (max(0, start - evidence_margin), min(len(context), end + evidence_margin))
            for start, end in protected_spans
        ]
        head = (edge_chars + 1) // 2
        tail = edge_chars - head
        if head:
            ranges.append((0, head))
        if tail:
            ranges.append((len(context) - tail, len(context)))
        return _render_ranges(context, ranges, marker)

    # Keep useful local context around each answer-location span when possible. If evidence is
    # widely scattered, shrink the margin deterministically before sacrificing evidence itself.
    low, high = 0, 512
    while low < high:
        margin = (low + high + 1) // 2
        candidate, _ = render(margin, 0)
        if estimate_text_tokens(candidate, ascii_chars_per_token=ascii_chars_per_token) <= context_budget:
            low = margin
        else:
            high = margin - 1
    evidence_margin = low
    evidence_only, _ = render(evidence_margin, 0)
    if estimate_text_tokens(evidence_only, ascii_chars_per_token=ascii_chars_per_token) > context_budget:
        raise ContextOverflowError(
            "Protected evidence spans cannot fit in the available context budget; "
            "increase max_model_len or reduce the output/reserve budget."
        )

    low, high = 0, len(context)
    while low < high:
        edge_chars = (low + high + 1) // 2
        candidate, _ = render(evidence_margin, edge_chars)
        if estimate_text_tokens(candidate, ascii_chars_per_token=ascii_chars_per_token) <= context_budget:
            low = edge_chars
        else:
            high = edge_chars - 1
    fitted, retained_source_chars = render(evidence_margin, low)
    return fitted, {
        "protected_span_context_margin_chars": evidence_margin,
        "retained_source_context_chars": retained_source_chars,
        "omitted_context_chars": len(context) - retained_source_chars,
    }
