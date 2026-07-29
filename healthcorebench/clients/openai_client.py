"""Unified async OpenAI-compatible client.

This is the single inference entry point for the whole framework. It:

* talks to any OpenAI Chat Completions-compatible service (vLLM first, commercial APIs
  too) via the official ``openai`` SDK's ``AsyncOpenAI``;
* accepts already-built OpenAI ``messages`` (text and/or image parts) — message
  construction is the benchmark adapter's job, encoding is the media layer's job;
* returns a structured :class:`ModelResponse` (never a bare content string), capturing
  model identity, usage breakdown, finish reason, logprobs and precise timing;
* parses usage defensively: any field the provider does not return is ``None``, never 0.

Retry / concurrency / rate-limiting are handled by the runtime layer, not here — this
client performs exactly one request per call so every attempt can be logged individually.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from healthcorebench.clients.errors import classify_exception
from healthcorebench.utils.timestamps import utc_now_iso


@dataclass
class ModelResponse:
    """Structured result of a single chat completion request.

    Every field that a provider may omit defaults to ``None`` so missing information is
    explicit rather than guessed. ``raw_response`` holds the full provider payload (as a
    dict) so no information is lost to SDK field changes.
    """

    content: str | None
    model_requested: str
    refusal: str | None = None
    model_returned: str | None = None
    system_fingerprint: str | None = None
    provider_request_id: str | None = None
    finish_reason: str | None = None
    created: int | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    image_tokens: int | None = None
    audio_tokens: int | None = None

    logprobs: Any | None = None
    option_probabilities: dict | None = None

    request_start_time: str | None = None
    request_end_time: str | None = None
    latency_seconds: float | None = None
    time_to_first_token: float | None = None
    generation_time_seconds: float | None = None

    raw_response: dict | None = None
    raw_usage: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class OpenAICompatibleClient:
    """Thin async wrapper over ``AsyncOpenAI`` returning :class:`ModelResponse`.

    The client never retries and never swallows errors into empty strings: on failure it
    raises a classified :class:`ClientError` that the runtime records and decides about.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        requested_model_name: str,
        timeout: float = 180.0,
        default_headers: dict | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.base_url = base_url
        self.requested_model_name = requested_model_name
        self._timeout = timeout
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # retries handled by the runtime, not the SDK
            default_headers=default_headers or None,
        )

    # ------------------------------------------------------------------ #
    # model identity probe
    # ------------------------------------------------------------------ #
    async def list_models(self) -> list[str]:
        """Return served model names from ``/v1/models`` (empty list on any failure)."""
        try:
            resp = await self._client.models.list()
            return [m.id for m in resp.data]
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    # chat completion
    # ------------------------------------------------------------------ #
    async def chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        stop: Any | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        response_format: Any | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict | None = None,
        model: str | None = None,
    ) -> ModelResponse:
        """Perform one chat completion. Raises :class:`ClientError` on any failure."""
        model_name = model or self.requested_model_name
        params: dict = {"model": model_name, "messages": messages}
        # Evaluation must be deterministic: every request is sent with temperature 0. If the
        # caller leaves temperature unset (None), we still send 0.0 rather than omitting it, so
        # the provider never falls back to its own (non-zero) default. This applies to model
        # inference and to LLM-judge calls alike.
        params["temperature"] = 0.0 if temperature is None else temperature
        if top_p is not None:
            params["top_p"] = top_p
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if seed is not None:
            params["seed"] = seed
        if stop is not None:
            params["stop"] = stop
        if logprobs is not None:
            params["logprobs"] = logprobs
        if top_logprobs is not None:
            params["top_logprobs"] = top_logprobs
        if response_format is not None:
            params["response_format"] = response_format
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        if extra_body:
            params["extra_body"] = extra_body

        start = time.monotonic()
        start_iso = utc_now_iso()
        try:
            resp = await self._client.chat.completions.create(**params)
        except Exception as exc:
            raise classify_exception(exc)
        end = time.monotonic()
        end_iso = utc_now_iso()

        return self._parse_response(
            resp,
            model_requested=model_name,
            start_iso=start_iso,
            end_iso=end_iso,
            latency=end - start,
        )

    # ------------------------------------------------------------------ #
    # response parsing
    # ------------------------------------------------------------------ #
    def _parse_response(self, resp, *, model_requested, start_iso, end_iso, latency) -> ModelResponse:
        raw = self._to_dict(resp)
        choice0 = (raw.get("choices") or [{}])[0]
        message = choice0.get("message") or {}
        content = message.get("content")
        refusal = message.get("refusal")
        # OpenAI-compatible providers can return a refusal in a structured side field with
        # null content. It is still the model's answer and must reach safety evaluators.
        if (content is None or content == "") and isinstance(refusal, str) and refusal.strip():
            content = refusal
        finish_reason = choice0.get("finish_reason")
        logprobs = choice0.get("logprobs")

        usage = raw.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        # Some providers omit total_tokens; derive it from prompt+completion when both are present
        # so total is always available for aggregation (input/output/total all accounted for).
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cached_input_tokens = prompt_details.get("cached_tokens")
        image_tokens = prompt_details.get("image_tokens")
        audio_tokens = prompt_details.get("audio_tokens")
        reasoning_tokens = completion_details.get("reasoning_tokens")

        return ModelResponse(
            content=content,
            model_requested=model_requested,
            refusal=refusal if isinstance(refusal, str) else None,
            model_returned=raw.get("model"),
            system_fingerprint=raw.get("system_fingerprint"),
            provider_request_id=raw.get("id"),
            finish_reason=finish_reason,
            created=raw.get("created"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            logprobs=logprobs,
            option_probabilities=None,  # computed by evaluators when applicable
            request_start_time=start_iso,
            request_end_time=end_iso,
            latency_seconds=latency,
            time_to_first_token=None,  # only available with streaming
            generation_time_seconds=None,
            raw_response=raw,
            raw_usage=usage or None,
        )

    @staticmethod
    def _to_dict(resp) -> dict:
        """Best-effort conversion of an SDK response object to a plain dict."""
        for attr in ("model_dump", "to_dict", "dict"):
            fn = getattr(resp, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue
        if isinstance(resp, dict):
            return resp
        return {"_repr": str(resp)}

    async def aclose(self) -> None:
        try:
            await self._client.close()
        except Exception:
            pass
