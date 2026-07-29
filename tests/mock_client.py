"""In-process mock of OpenAICompatibleClient for tests.

Drives the executor/runner without a network. A scripted queue of behaviours per call lets
tests exercise success, transient-then-success, terminal failures, empty content, missing
usage, reasoning tokens, differing returned model names, etc. Every ``chat_completion``
call is counted so tests can assert that each attempt was actually made and logged.
"""

from __future__ import annotations

import time

from healthcorebench.clients.errors import ClientError, ErrorType
from healthcorebench.clients.openai_client import ModelResponse
from healthcorebench.utils.timestamps import utc_now_iso


class MockClient:
    def __init__(self, requested_model_name="mock-model", behaviours=None, served_model="mock-served"):
        self.requested_model_name = requested_model_name
        self.base_url = "http://mock/v1"
        self.served_model = served_model
        # behaviours: list of dicts or ClientError; consumed per call. If exhausted, the
        # last behaviour repeats.
        self._behaviours = behaviours or [{"content": "The answer is A"}]
        self.calls = 0
        self.call_kwargs: list[dict] = []
        self._models = [served_model]

    async def list_models(self):
        return list(self._models)

    async def chat_completion(self, messages, **kwargs):
        idx = min(self.calls, len(self._behaviours) - 1)
        beh = self._behaviours[idx]
        self.calls += 1
        self.call_kwargs.append(dict(kwargs))
        if isinstance(beh, ClientError):
            raise beh
        if isinstance(beh, Exception):
            raise beh

        start = utc_now_iso()
        time.sleep(0.001)
        end = utc_now_iso()
        usage = beh.get("usage", {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})
        content = beh.get("content", "A")
        raw = {
            "id": beh.get("request_id", "req-mock"),
            "model": beh.get("model", self.served_model),
            "system_fingerprint": beh.get("fingerprint", "fp_mock"),
            "created": 1710000000,
            "choices": [{"index": 0, "message": {"content": content, "refusal": beh.get("refusal")},
                         "finish_reason": beh.get("finish_reason", "stop"), "logprobs": beh.get("logprobs")}],
            "usage": usage,
        }
        return ModelResponse(
            content=content,
            model_requested=kwargs.get("model") or self.requested_model_name,
            refusal=beh.get("refusal"),
            model_returned=raw["model"], system_fingerprint=raw["system_fingerprint"],
            provider_request_id=raw["id"], finish_reason=raw["choices"][0]["finish_reason"],
            created=raw["created"],
            prompt_tokens=(usage or {}).get("prompt_tokens") if usage else None,
            completion_tokens=(usage or {}).get("completion_tokens") if usage else None,
            total_tokens=(usage or {}).get("total_tokens") if usage else None,
            reasoning_tokens=((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") if usage else None,
            logprobs=beh.get("logprobs"),
            request_start_time=start, request_end_time=end, latency_seconds=0.001,
            raw_response=raw, raw_usage=usage,
        )

    async def aclose(self):
        pass


def err(error_type: ErrorType, msg="mock error", http_status=None):
    return ClientError(error_type, msg, http_status=http_status)
