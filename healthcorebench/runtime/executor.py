"""Single-sample execution: request → retries → attempt logs → result record.

The executor performs one logical inference for one ``(sample, repeat)``:

1. Build wire messages from the sample's logical messages (media encoded here).
2. Loop attempts under the retry policy; every attempt (success or failure) is written to
   ``attempts.jsonl`` immediately with its own ``attempt_id`` and shared
   ``request_group_id``.
3. On success, emit a success ``ResultRecord`` carrying the raw response, usage and
   timing. On exhausted retries, emit an error ``ResultRecord`` with null answers — never
   scored as a wrong answer.

The executor does not parse or score; that happens in the scoring stage so it can be
re-run without re-requesting the model.
"""

from __future__ import annotations

import re
import time
import uuid
import json

from healthcorebench.clients.errors import ClientError, ErrorType
from healthcorebench.clients.messages import build_messages
from healthcorebench.clients.openai_client import OpenAICompatibleClient, ModelResponse
from healthcorebench.benchmarks.context_window import estimate_text_tokens
from healthcorebench.runtime.retry import RetryPolicy
from healthcorebench.schemas.request import AttemptRecord, UsageInfo
from healthcorebench.schemas.result import ResultRecord
from healthcorebench.utils.hashing import hash_json
from healthcorebench.utils.timestamps import utc_now_iso
import asyncio


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Executor:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        run_id: str,
        provider: str,
        generation,          # GenerationConfig
        media,               # MediaConfig
        output,              # OutputConfig
        retry_policy: RetryPolicy,
        recorder,
        rate_limiter=None,
        request_purpose: str = "model_inference",
        model_role: str = "evaluation",
        same_budget_error_retries: int = 2,
        same_budget_timeout_retries: int = 1,
        transient_error_ladder_steps: int = 2,
        max_model_len: int | None = None,
        store_full_input_allowed: bool = True,
    ) -> None:
        self.client = client
        self.run_id = run_id
        self.provider = provider
        self.generation = generation
        self.media = media
        self.output = output
        self.retry_policy = retry_policy
        self.recorder = recorder
        self.rate_limiter = rate_limiter
        self.request_purpose = request_purpose
        self.model_role = model_role
        self.same_budget_error_retries = max(0, same_budget_error_retries)
        self.same_budget_timeout_retries = max(0, same_budget_timeout_retries)
        self.transient_error_ladder_steps = max(0, transient_error_ladder_steps)
        self.max_model_len = max_model_len
        self.store_full_input_allowed = store_full_input_allowed

    def _requested_max_tokens(self, sample: dict) -> int | None:
        configured = self.generation.max_tokens
        override = (sample.get("metadata") or {}).get("request_max_tokens")
        if not isinstance(override, int) or isinstance(override, bool) or override <= 0:
            return configured
        return min(configured, override) if configured is not None else override

    def _max_token_budgets(self, requested: int | None) -> list[int | None]:
        if not self.generation.adaptive_max_tokens or requested is None:
            return [requested]
        values = [requested, *self.generation.max_tokens_candidates]
        return sorted({v for v in values if v is not None and v <= requested}, reverse=True)

    @staticmethod
    def _context_limit(error: ClientError) -> tuple[int | None, int | None]:
        text = error.message.lower().replace(",", "")
        if not any(term in text for term in (
            "maximum context", "max context", "context length", "input tokens",
            "max_tokens", "maximum sequence length",
        )):
            return None, None
        max_match = re.search(r"(?:maximum context length|max(?:imum)? context(?: length)?|maximum sequence length)\D{0,30}(\d+)", text)
        input_match = re.search(r"(?:request has|input(?: has| length)?|prompt(?: has| length)?)\D{0,20}(\d+)\s*(?:input\s+)?tokens", text)
        return (int(max_match.group(1)) if max_match else None,
                int(input_match.group(1)) if input_match else None)

    @staticmethod
    def _is_context_error(error: ClientError) -> bool:
        if error.error_type not in {ErrorType.INVALID_REQUEST, ErrorType.MAX_OUTPUT_LENGTH}:
            return False
        text = error.message.lower()
        return any(term in text for term in (
            "maximum context", "max context", "context length", "input tokens",
            "max_tokens", "maximum sequence length",
        ))

    def _next_budget_index(self, budgets, index: int, error: ClientError) -> int | None:
        if index + 1 >= len(budgets):
            return None
        # Preserve the configured retry ladder exactly. Skipping directly to a server-estimated
        # feasible budget makes intermediate tiers impossible to try and can over-reduce output.
        return index + 1

    def _ladder_remedies(self, error: ClientError) -> bool:
        """Whether stepping down the output-token ladder can plausibly fix ``error``.

        The ladder only changes the requested output budget. A server that rejected that
        budget (context error) or ran out of time producing it (timeout) may succeed with a
        smaller one; every other deterministic failure — auth, permission, a malformed
        request — fails identically at every tier, so walking the ladder only multiplies
        futile requests.
        """
        if self._is_context_error(error):
            return bool(self.generation.reduce_max_tokens_on_context_error)
        if error.error_type == ErrorType.API_TIMEOUT:
            return bool(self.generation.reduce_max_tokens_on_timeout)
        # A transient non-timeout failure (5xx / rate limit / connection) may still be
        # resource pressure, so a bounded number of steps is allowed; see execute().
        return bool(error.retryable)

    @staticmethod
    def _joined_reasons(reasons: list[str]) -> str | None:
        """Preserve every tier-step cause in order, collapsing consecutive repeats."""
        collapsed: list[str] = []
        for reason in reasons:
            if not collapsed or collapsed[-1] != reason:
                collapsed.append(reason)
        return ",".join(collapsed) or None

    def _effective_seed(self, repeat_index: int) -> int | None:
        """Give repeated generations independent deterministic seeds when configured."""
        return (self.generation.seed + repeat_index
                if isinstance(self.generation.seed, int) else self.generation.seed)

    @staticmethod
    def _estimate_prompt_tokens(logged_messages: list[dict]) -> int:
        prompt = json.dumps(logged_messages, ensure_ascii=False, separators=(",", ":"))
        return estimate_text_tokens(prompt)

    @classmethod
    def _estimate_request_tokens(cls, logged_messages: list[dict], max_tokens: int | None) -> int:
        return cls._estimate_prompt_tokens(logged_messages) + max(0, max_tokens or 0)

    async def execute(self, sample: dict, logical_messages: list[dict], repeat_index: int) -> ResultRecord:
        """Run one logical inference and return its ResultRecord (also records attempts)."""
        request_group_id = _new_id("rg")
        sample_id = sample["sample_id"]
        requested_max_tokens = self._requested_max_tokens(sample)

        # --- build messages (media encoded here). A build failure is a deterministic,
        #     non-retryable error recorded as a failed result, not a wrong answer. ---
        try:
            built = build_messages(
                logical_messages,
                image_detail=self.media.image_detail,
                image_format=self.media.image_format,
                max_pixels=self.media.max_pixels,
                max_image_size_mb=self.media.max_image_size_mb,
                max_images=self.media.max_images,
                allow_image_urls=self.media.allow_image_urls,
                max_video_frames=self.media.max_video_frames,
                video_frame_sampling_strategy=self.media.video_frame_sampling_strategy,
            )
        except ClientError as ce:
            return self._error_result(sample, repeat_index, request_group_id, ce, retry_count=0,
                                      formatted_prompt=None, prompt_hash=None, media_hashes=[],
                                      requested_max_tokens=requested_max_tokens)

        # A benchmark that forbids storing its full input must not leak it through the
        # result record either; ``save_formatted_prompt`` alone is not a licence gate.
        formatted_prompt = (built.logged_messages
                            if self.output.save_formatted_prompt and self.store_full_input_allowed
                            else None)
        prompt_hash = hash_json(built.logged_messages)
        media_hashes = [i.media_hash for i in built.image_infos if i.media_hash]
        media_hashes.extend(
            info["media_hash"] for info in built.video_infos if info.get("media_hash")
        )
        media_provenance = {
            "images": [
                info.model_dump(
                    exclude={"source_path", "source_uri"},
                    exclude_none=True,
                )
                for info in built.image_infos
            ],
            "videos": built.video_infos,
        }

        budgets = self._max_token_budgets(requested_max_tokens)
        budget_index = 0
        adaptive_retry_count = 0
        adjustment_reasons: list[str] = []
        adjustment_reason = None
        same_budget_error_count = 0
        timeout_retries_at_budget = 0
        transient_ladder_steps = 0
        normal_retry_count = 0
        attempt_number = 0
        while True:
            attempt_number += 1
            if self.rate_limiter is not None:
                reservation_id = await self.rate_limiter.acquire(
                    self._estimate_request_tokens(built.logged_messages, budgets[budget_index])
                )
            else:
                reservation_id = None
            effective_max_tokens = budgets[budget_index]
            effective_seed = self._effective_seed(repeat_index)
            call_start = utc_now_iso()
            call_clock = time.monotonic()
            try:
                resp = await self.client.chat_completion(
                    built.wire_messages,
                    temperature=self.generation.temperature,
                    top_p=self.generation.top_p,
                    max_tokens=effective_max_tokens,
                    seed=effective_seed,
                    stop=self.generation.stop,
                    logprobs=self.generation.logprobs,
                    top_logprobs=self.generation.top_logprobs,
                    response_format=self.generation.response_format,
                    extra_body=self.generation.extra_body or None,
                )
            except ClientError as ce:
                call_end = utc_now_iso()
                latency = time.monotonic() - call_clock
                max_context, input_tokens = self._context_limit(ce)
                self._record_failed_attempt(
                    sample_id, request_group_id, repeat_index, attempt_number, ce,
                    request_start_time=call_start, request_end_time=call_end,
                    latency_seconds=latency, requested_max_tokens=requested_max_tokens,
                    effective_max_tokens=effective_max_tokens, fallback_index=budget_index,
                    adjustment_reason=adjustment_reason, adaptive_retry_count=adaptive_retry_count,
                    server_max_context=max_context, server_input_tokens=input_tokens,
                    generation_seed=effective_seed,
                    logged_messages=built.logged_messages,
                )
                # In adaptive mode the ladder is a remedy for exactly one thing: the server
                # rejecting the requested output budget. Retrying a deterministic failure at
                # the same budget, or walking the ladder for an error the ladder cannot fix,
                # only multiplies futile requests (measured: 95.7% of one task's 121k attempts).
                if self.generation.adaptive_max_tokens:
                    ladder_remedies = self._ladder_remedies(ce)
                    if ce.retryable and same_budget_error_count < self.same_budget_error_retries:
                        same_budget_error_count += 1
                        await asyncio.sleep(
                            self.retry_policy.backoff_seconds(same_budget_error_count, ce)
                        )
                        continue
                    # Repeated transient failures that survive a budget cut are a sick server,
                    # not an output-budget problem; bound how far the ladder is walked for them.
                    transient = not self._is_context_error(ce)
                    if transient and transient_ladder_steps >= self.transient_error_ladder_steps:
                        ladder_remedies = False
                    next_index = (self._next_budget_index(budgets, budget_index, ce)
                                  if ladder_remedies else None)
                    if next_index is not None:
                        budget_index = next_index
                        adaptive_retry_count += 1
                        transient_ladder_steps = transient_ladder_steps + 1 if transient else 0
                        adjustment_reasons.append(
                            "context_error" if self._is_context_error(ce)
                            else "timeout" if ce.error_type == ErrorType.API_TIMEOUT
                            else "api_error"
                        )
                        adjustment_reason = self._joined_reasons(adjustment_reasons)
                        same_budget_error_count = 0
                        timeout_retries_at_budget = 0
                        continue
                    return self._error_result(
                        sample, repeat_index, request_group_id, ce,
                        retry_count=attempt_number - 1,
                        formatted_prompt=formatted_prompt,
                        prompt_hash=prompt_hash,
                        media_hashes=media_hashes,
                        requested_max_tokens=requested_max_tokens,
                        effective_max_tokens=effective_max_tokens,
                        fallback_index=budget_index,
                        adjustment_reason=adjustment_reason,
                        adaptive_retry_count=adaptive_retry_count,
                        generation_seed=effective_seed,
                    )
                if ce.error_type == ErrorType.API_TIMEOUT and self.generation.reduce_max_tokens_on_timeout:
                    timeout_retries_at_budget += 1
                    if timeout_retries_at_budget <= self.same_budget_timeout_retries:
                        continue
                if self.retry_policy.should_retry(ce, normal_retry_count + 1):
                    normal_retry_count += 1
                    await asyncio.sleep(self.retry_policy.backoff_seconds(normal_retry_count, ce))
                    continue
                return self._error_result(sample, repeat_index, request_group_id, ce,
                                          retry_count=attempt_number - 1,
                                          formatted_prompt=formatted_prompt, prompt_hash=prompt_hash,
                                          media_hashes=media_hashes,
                                          requested_max_tokens=requested_max_tokens,
                                          effective_max_tokens=effective_max_tokens,
                                          fallback_index=budget_index, adjustment_reason=adjustment_reason,
                                          adaptive_retry_count=adaptive_retry_count,
                                          generation_seed=effective_seed)

            if self.rate_limiter is not None and resp.total_tokens is not None:
                await self.rate_limiter.record_tokens(resp.total_tokens, reservation_id)

            if resp.finish_reason == "content_filter":
                filtered = ClientError(
                    ErrorType.CONTENT_FILTER,
                    "Provider marked the response as content-filtered.",
                    retryable=False,
                )
                self._record_failed_attempt(
                    sample_id, request_group_id, repeat_index, attempt_number, filtered, resp=resp,
                    requested_max_tokens=requested_max_tokens,
                    effective_max_tokens=effective_max_tokens, fallback_index=budget_index,
                    adjustment_reason=adjustment_reason, adaptive_retry_count=adaptive_retry_count,
                    generation_seed=effective_seed,
                )
                return self._error_result(
                    sample, repeat_index, request_group_id, filtered, retry_count=attempt_number - 1,
                    formatted_prompt=formatted_prompt, prompt_hash=prompt_hash, media_hashes=media_hashes,
                    requested_max_tokens=requested_max_tokens,
                    effective_max_tokens=effective_max_tokens, fallback_index=budget_index,
                    adjustment_reason=adjustment_reason, adaptive_retry_count=adaptive_retry_count,
                    generation_seed=effective_seed,
                )

            # success path — but an empty content is treated as a retryable empty_response
            if resp.content is None or resp.content == "":
                empty = ClientError(ErrorType.EMPTY_RESPONSE, "Model returned empty content.")
                self._record_failed_attempt(sample_id, request_group_id, repeat_index, attempt_number, empty, resp=resp,
                                            requested_max_tokens=requested_max_tokens,
                                            effective_max_tokens=effective_max_tokens, fallback_index=budget_index,
                                            adjustment_reason=adjustment_reason,
                                            adaptive_retry_count=adaptive_retry_count,
                                            generation_seed=effective_seed)
                if self.retry_policy.should_retry(empty, normal_retry_count + 1):
                    normal_retry_count += 1
                    await asyncio.sleep(self.retry_policy.backoff_seconds(normal_retry_count, empty))
                    continue
                return self._error_result(sample, repeat_index, request_group_id, empty,
                                          retry_count=attempt_number - 1,
                                          formatted_prompt=formatted_prompt, prompt_hash=prompt_hash,
                                          media_hashes=media_hashes,
                                          requested_max_tokens=requested_max_tokens,
                                          effective_max_tokens=effective_max_tokens,
                                          fallback_index=budget_index, adjustment_reason=adjustment_reason,
                                          adaptive_retry_count=adaptive_retry_count,
                                          generation_seed=effective_seed)

            attempt_id = self._record_success_attempt(
                sample_id, request_group_id, repeat_index, attempt_number, resp,
                requested_max_tokens=requested_max_tokens,
                effective_max_tokens=effective_max_tokens, fallback_index=budget_index,
                adjustment_reason=adjustment_reason, adaptive_retry_count=adaptive_retry_count,
                generation_seed=effective_seed,
            )
            return self._success_result(
                sample, repeat_index, request_group_id, attempt_id, resp,
                formatted_prompt=formatted_prompt, prompt_hash=prompt_hash, media_hashes=media_hashes,
                media_provenance=media_provenance,
                retry_count=attempt_number - 1,
                requested_max_tokens=requested_max_tokens,
                effective_max_tokens=effective_max_tokens, fallback_index=budget_index,
                adjustment_reason=adjustment_reason, adaptive_retry_count=adaptive_retry_count,
                generation_seed=effective_seed,
            )

    # ------------------------------------------------------------------ #
    def _record_success_attempt(self, sample_id, rg, repeat, number, resp: ModelResponse, *,
                                requested_max_tokens=None,
                                effective_max_tokens=None, fallback_index=0,
                                adjustment_reason=None, adaptive_retry_count=0,
                                generation_seed=None) -> str:
        attempt_id = _new_id("att")
        rec = AttemptRecord(
            attempt_id=attempt_id, request_group_id=rg, run_id=self.run_id,
            sample_id=sample_id, sample_repeat_index=repeat, attempt_number=number,
            request_purpose=self.request_purpose,
            request_start_time=resp.request_start_time, request_end_time=resp.request_end_time,
            latency_seconds=resp.latency_seconds, provider=self.provider,
            requested_model_name=resp.model_requested, returned_model_name=resp.model_returned,
            system_fingerprint=resp.system_fingerprint, provider_request_id=resp.provider_request_id,
            status="success", http_status=200, finish_reason=resp.finish_reason,
            requested_max_tokens=requested_max_tokens,
            effective_max_tokens=effective_max_tokens,
            max_tokens_fallback_index=fallback_index,
            max_tokens_adjustment_reason=adjustment_reason,
            adaptive_retry_count=adaptive_retry_count,
            usage=UsageInfo(prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                            total_tokens=resp.total_tokens, cached_input_tokens=resp.cached_input_tokens,
                            reasoning_tokens=resp.reasoning_tokens, image_tokens=resp.image_tokens,
                            audio_tokens=resp.audio_tokens),
            raw_response=resp.raw_response if self.output.save_raw_response_object else None,
            provider_metadata={"generation_seed": generation_seed},
            timestamp=utc_now_iso(),
        )
        self.recorder.record_attempt(rec)
        return attempt_id

    def _record_failed_attempt(self, sample_id, rg, repeat, number, err: ClientError,
                               resp: ModelResponse | None = None, *, request_start_time=None,
                               request_end_time=None, latency_seconds=None,
                               requested_max_tokens=None, effective_max_tokens=None,
                               fallback_index=0, adjustment_reason=None,
                               adaptive_retry_count=0, server_max_context=None,
                               server_input_tokens=None, generation_seed=None,
                               logged_messages: list[dict] | None = None) -> str:
        attempt_id = _new_id("att")
        # A timeout can hide usage even though the provider may have generated output.
        usage_status = "unknown_due_to_timeout" if err.error_type == ErrorType.API_TIMEOUT else "known"
        usage = (UsageInfo(prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                           total_tokens=resp.total_tokens, cached_input_tokens=resp.cached_input_tokens,
                           reasoning_tokens=resp.reasoning_tokens, image_tokens=resp.image_tokens,
                           audio_tokens=resp.audio_tokens) if resp is not None
                 else self._failed_request_usage(logged_messages, server_input_tokens))
        rec = AttemptRecord(
            attempt_id=attempt_id, request_group_id=rg, run_id=self.run_id,
            sample_id=sample_id, sample_repeat_index=repeat, attempt_number=number,
            request_purpose=self.request_purpose,
            request_start_time=resp.request_start_time if resp else request_start_time,
            request_end_time=resp.request_end_time if resp else request_end_time,
            latency_seconds=resp.latency_seconds if resp else latency_seconds,
            provider=self.provider, requested_model_name=self.client.requested_model_name,
            returned_model_name=resp.model_returned if resp else None,
            status="error", http_status=err.http_status, error_type=err.error_type.value,
            error_message=err.message, exception_class=err.exception_class, retryable=err.retryable,
            finish_reason=resp.finish_reason if resp else None,
            requested_max_tokens=(requested_max_tokens if requested_max_tokens is not None
                                  else self.generation.max_tokens),
            effective_max_tokens=effective_max_tokens,
            max_tokens_fallback_index=fallback_index,
            max_tokens_adjustment_reason=adjustment_reason,
            adaptive_retry_count=adaptive_retry_count,
            usage=usage,
            usage_status=usage_status,
            raw_response=(resp.raw_response if (resp and self.output.save_raw_response_object) else None),
            provider_metadata={
                "server_reported_max_context": server_max_context,
                "server_reported_input_tokens": server_input_tokens,
                "generation_seed": generation_seed,
            },
            timestamp=utc_now_iso(),
        )
        self.recorder.record_attempt(rec)
        return attempt_id

    # ------------------------------------------------------------------ #
    def _failed_request_usage(self, logged_messages: list[dict] | None,
                              server_input_tokens: int | None) -> UsageInfo:
        """Account for the prompt a failed request already paid for.

        Providers return no usage block with an error, so recording an empty ``UsageInfo``
        makes retries look free: one measured task summed 32,780,413 tokens while 115,914
        failed requests contributed nothing. The prompt cost is taken from the server's own
        error text when it reports one ("your request has N input tokens") and otherwise
        estimated from the logged request payload. Both are flagged so aggregation can keep
        estimates out of metered totals; the estimate is text-only, so it is a lower bound
        for image/video requests.
        """
        if isinstance(server_input_tokens, int) and server_input_tokens > 0:
            prompt_tokens, source = server_input_tokens, "server_reported_error"
        elif logged_messages is not None:
            prompt_tokens, source = self._estimate_prompt_tokens(logged_messages), "prompt_estimate"
        else:
            return UsageInfo()
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
            prompt_tokens_source=source,
            prompt_tokens_are_estimated=source != "server_reported_error",
            completion_tokens_metered=False,
        )

    # ------------------------------------------------------------------ #
    def _success_result(self, sample, repeat, rg, attempt_id, resp: ModelResponse, *,
                        formatted_prompt, prompt_hash, media_hashes, retry_count,
                        media_provenance=None,
                        requested_max_tokens=None,
                        effective_max_tokens=None, fallback_index=0,
                        adjustment_reason=None, adaptive_retry_count=0,
                        generation_seed=None) -> ResultRecord:
        return ResultRecord(
            result_id=_new_id("res"), run_id=self.run_id, sample_id=sample["sample_id"],
            sample_repeat_index=repeat, request_group_id=rg, successful_attempt_id=attempt_id,
            benchmark_name=sample.get("benchmark_name"), benchmark_version=sample.get("benchmark_version"),
            benchmark_split=sample.get("benchmark_split"),
            model_name=resp.model_returned or resp.model_requested, model_role=self.model_role,
            difficulty=sample.get("difficulty"), component=sample.get("component"),
            capability=sample.get("capability"), specialty=sample.get("specialty"),
            language=sample.get("language"), modality=sample.get("modality"),
            task_type=sample.get("task_type"),
            formatted_prompt=formatted_prompt, prompt_hash=prompt_hash, media_hashes=media_hashes,
            raw_response=resp.content, raw_response_object=(resp.raw_response if self.output.save_raw_response_object else None),
            parsed_answer=None, normalized_answer=None, reference_answer=sample.get("reference_answer"),
            finish_reason=resp.finish_reason, logprobs=resp.logprobs, option_probabilities=resp.option_probabilities,
            requested_max_tokens=requested_max_tokens, effective_max_tokens=effective_max_tokens,
            max_tokens_fallback_index=fallback_index,
            max_tokens_adjustment_reason=adjustment_reason,
            adaptive_retry_count=adaptive_retry_count,
            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens, total_tokens=resp.total_tokens,
            cached_input_tokens=resp.cached_input_tokens, reasoning_tokens=resp.reasoning_tokens,
            image_tokens=resp.image_tokens, audio_tokens=resp.audio_tokens,
            request_start_time=resp.request_start_time, request_end_time=resp.request_end_time,
            latency_seconds=resp.latency_seconds, time_to_first_token=resp.time_to_first_token,
            generation_time_seconds=resp.generation_time_seconds,
            status="success", retry_count=retry_count,
            parsing_status="pending", evaluation_status="pending", timestamp=utc_now_iso(),
            provider_metadata={
                "generation_seed": generation_seed,
                "native_refusal": bool(resp.refusal),
                "media_provenance": media_provenance or {"images": [], "videos": []},
            },
        )

    def _error_result(self, sample, repeat, rg, err: ClientError, *, retry_count,
                      formatted_prompt, prompt_hash, media_hashes, effective_max_tokens=None,
                      requested_max_tokens=None,
                      fallback_index=0, adjustment_reason=None, adaptive_retry_count=0,
                      generation_seed=None) -> ResultRecord:
        return ResultRecord(
            result_id=_new_id("res"), run_id=self.run_id, sample_id=sample["sample_id"],
            sample_repeat_index=repeat, request_group_id=rg, successful_attempt_id=None,
            benchmark_name=sample.get("benchmark_name"), benchmark_version=sample.get("benchmark_version"),
            benchmark_split=sample.get("benchmark_split"),
            model_name=self.client.requested_model_name, model_role=self.model_role,
            difficulty=sample.get("difficulty"), component=sample.get("component"),
            capability=sample.get("capability"), specialty=sample.get("specialty"),
            language=sample.get("language"), modality=sample.get("modality"), task_type=sample.get("task_type"),
            formatted_prompt=formatted_prompt, prompt_hash=prompt_hash, media_hashes=media_hashes,
            raw_response=None, parsed_answer=None, normalized_answer=None,
            reference_answer=sample.get("reference_answer"),
            status="error", error_type=err.error_type.value, error_message=err.message,
            retry_count=retry_count,
            requested_max_tokens=requested_max_tokens, effective_max_tokens=effective_max_tokens,
            max_tokens_fallback_index=fallback_index,
            max_tokens_adjustment_reason=adjustment_reason,
            adaptive_retry_count=adaptive_retry_count,
            parsing_status="not_applicable", evaluation_status="not_applicable",
            timestamp=utc_now_iso(), provider_metadata={"generation_seed": generation_seed},
        )
