"""Unit tests: error classification, redaction, usage parsing, message building."""

import pytest
from PIL import Image

from healthcorebench.clients.errors import classify_exception, redact_secrets, ErrorType
from healthcorebench.clients.openai_client import OpenAICompatibleClient
from healthcorebench.clients.messages import build_messages


def test_redact_secrets():
    r = redact_secrets("Authorization: Bearer sk-abc12345defg and api_key=TOPSECRET")
    assert "sk-abc12345defg" not in r
    assert "TOPSECRET" not in r


class _Resp:
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = headers or {}


def _exc_with_status(status, headers=None):
    e = Exception("boom")
    e.response = _Resp(status, headers)
    return e


def test_classify_rate_limit_with_retry_after():
    ce = classify_exception(_exc_with_status(429, {"retry-after": "12"}))
    assert ce.error_type == ErrorType.RATE_LIMIT
    assert ce.retryable is True
    assert ce.retry_after_seconds == 12.0


def test_classify_auth_not_retryable():
    ce = classify_exception(_exc_with_status(401))
    assert ce.error_type == ErrorType.AUTHENTICATION_ERROR
    assert ce.retryable is False


def test_classify_server_error_retryable():
    ce = classify_exception(_exc_with_status(500))
    assert ce.error_type == ErrorType.SERVER_ERROR
    assert ce.retryable is True


def test_classify_invalid_request_not_retryable():
    ce = classify_exception(_exc_with_status(400))
    assert ce.error_type == ErrorType.INVALID_REQUEST
    assert ce.retryable is False


def test_classify_timeout():
    assert classify_exception(TimeoutError("read timed out")).error_type == ErrorType.API_TIMEOUT


def _fake_completion(usage=None, model="served-model", content="hello", fingerprint="fp_1", finish="stop", refusal=None):
    choice = {"index": 0, "message": {"role": "assistant", "content": content, "refusal": refusal}, "finish_reason": finish, "logprobs": None}
    return {
        "id": "req-123",
        "model": model,
        "created": 1710000000,
        "system_fingerprint": fingerprint,
        "choices": [choice],
        "usage": usage,
    }


def _parse(raw):
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    return client._parse_response(raw, model_requested="req-model", start_iso="2026-01-01T00:00:00Z",
                                  end_iso="2026-01-01T00:00:01Z", latency=1.0)


def test_usage_parsing_full():
    usage = {
        "prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125,
        "prompt_tokens_details": {"cached_tokens": 10, "image_tokens": 5},
        "completion_tokens_details": {"reasoning_tokens": 8},
    }
    mr = _parse(_fake_completion(usage=usage))
    assert mr.prompt_tokens == 100 and mr.completion_tokens == 25 and mr.total_tokens == 125
    assert mr.cached_input_tokens == 10 and mr.image_tokens == 5 and mr.reasoning_tokens == 8
    assert mr.model_returned == "served-model" and mr.system_fingerprint == "fp_1"
    assert mr.provider_request_id == "req-123" and mr.finish_reason == "stop"


def test_usage_parsing_missing_is_none_not_zero():
    mr = _parse(_fake_completion(usage=None))
    assert mr.prompt_tokens is None and mr.completion_tokens is None and mr.total_tokens is None
    assert mr.reasoning_tokens is None and mr.cached_input_tokens is None


def test_structured_refusal_becomes_response_content():
    refusal = "I cannot provide instructions for that request."
    mr = _parse(_fake_completion(content=None, refusal=refusal))

    assert mr.content == refusal
    assert mr.refusal == refusal


def test_build_messages_text_only():
    bm = build_messages([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert bm.wire_messages == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert bm.image_infos == []


def test_build_messages_multi_image_order_preserved():
    i1 = Image.new("RGB", (16, 16), (1, 1, 1))
    i2 = Image.new("RGB", (16, 16), (2, 2, 2))
    bm = build_messages([{
        "role": "user",
        "content": [
            {"type": "text", "text": "a"},
            {"type": "image", "source": i1, "media_id": "img_0"},
            {"type": "text", "text": "b"},
            {"type": "image", "source": i2, "media_id": "img_1"},
        ],
    }])
    parts = bm.wire_messages[0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text", "image_url"]
    assert len(bm.image_infos) == 2
    # logged form must not embed base64
    assert "base64" not in str(bm.logged_messages)


def test_build_messages_respects_max_images():
    img = Image.new("RGB", (8, 8))
    with pytest.raises(Exception):
        build_messages([{
            "role": "user",
            "content": [
                {"type": "image", "source": img, "media_id": "a"},
                {"type": "image", "source": img, "media_id": "b"},
            ],
        }], max_images=1)
