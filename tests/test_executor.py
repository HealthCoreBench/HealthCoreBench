"""Integration tests for the Executor against the mock client.

Verifies: success path, retry-then-success (all attempts logged), terminal failure
recorded as an error result (not a wrong answer), empty-content retry, timeout usage_status,
and token-only usage persistence.
"""

import json

from PIL import Image

from healthcorebench.runtime.executor import Executor
from healthcorebench.runtime.recorder import Recorder
from healthcorebench.runtime.retry import RetryPolicy
from healthcorebench.schemas.config import GenerationConfig, MediaConfig, OutputConfig
from healthcorebench.utils.jsonl import read_jsonl
from healthcorebench.clients.errors import ErrorType
from healthcorebench.runtime.rate_limiter import RateLimiter
from tests.mock_client import MockClient, err


def _executor(tmp_path, client, retry_policy=None):
    rec = Recorder(tmp_path)
    return Executor(
        client=client, run_id="run1", provider="mock",
        generation=GenerationConfig(adaptive_max_tokens=False),
        media=MediaConfig(), output=OutputConfig(),
        retry_policy=retry_policy or RetryPolicy(max_retries=3, initial_seconds=0.001, max_seconds=0.01),
        recorder=rec,
    ), rec


SAMPLE = {"sample_id": "urn:s1", "benchmark_name": "MMLU", "benchmark_split": "test", "reference_answer": "A"}
MSG = [{"role": "user", "content": "Q?"}]


async def test_success(tmp_path):
    ex, rec = _executor(tmp_path, MockClient(behaviours=[{"content": "The answer is A"}]))
    res = await ex.execute(SAMPLE, MSG, 0)
    assert res.status == "success"
    assert res.raw_response == "The answer is A"
    assert res.parsed_answer is None  # executor does not parse
    attempts = read_jsonl(rec.path("attempts.jsonl"))
    assert len(attempts) == 1 and attempts[0]["status"] == "success"


async def test_success_persists_media_provenance_without_source_path(tmp_path):
    image_path = tmp_path / "private-study.png"
    Image.new("RGB", (8, 6), (12, 34, 56)).save(image_path)
    messages = [{"role": "user", "content": [
        {"type": "image", "source": image_path, "media_id": "study"},
        {"type": "text", "text": "Question"},
    ]}]
    executor, _ = _executor(tmp_path, MockClient())

    result = await executor.execute(SAMPLE, messages, 0)

    provenance = result.provider_metadata["media_provenance"]
    assert provenance["images"][0]["media_id"] == "study"
    assert provenance["images"][0]["media_hash"] in result.media_hashes
    assert "source_path" not in provenance["images"][0]
    assert "source_uri" not in provenance["images"][0]
    assert str(image_path) not in json.dumps(result.model_dump())


async def test_retry_then_success_logs_all_attempts(tmp_path):
    client = MockClient(behaviours=[err(ErrorType.SERVER_ERROR, http_status=500),
                                    err(ErrorType.RATE_LIMIT, http_status=429),
                                    {"content": "A"}])
    ex, rec = _executor(tmp_path, client)
    res = await ex.execute(SAMPLE, MSG, 0)
    assert res.status == "success"
    assert res.retry_count == 2
    attempts = read_jsonl(rec.path("attempts.jsonl"))
    assert len(attempts) == 3
    assert [a["status"] for a in attempts] == ["error", "error", "success"]
    # all share one request_group_id, all have distinct attempt_ids
    assert len({a["request_group_id"] for a in attempts}) == 1
    assert len({a["attempt_id"] for a in attempts}) == 3


async def test_terminal_failure_is_error_result_not_wrong_answer(tmp_path):
    client = MockClient(behaviours=[err(ErrorType.SERVER_ERROR, http_status=500)])
    ex, rec = _executor(tmp_path, client, RetryPolicy(max_retries=2, initial_seconds=0.001, max_seconds=0.01))
    res = await ex.execute(SAMPLE, MSG, 0)
    assert res.status == "error"
    assert res.error_type == "server_error"
    assert res.parsed_answer is None and res.normalized_answer is None
    assert res.evaluation_status == "not_applicable"
    # 1 initial + 2 retries = 3 attempts
    assert len(read_jsonl(rec.path("attempts.jsonl"))) == 3


async def test_auth_error_not_retried(tmp_path):
    client = MockClient(behaviours=[err(ErrorType.AUTHENTICATION_ERROR, http_status=401)])
    ex, rec = _executor(tmp_path, client)
    res = await ex.execute(SAMPLE, MSG, 0)
    assert res.status == "error" and res.error_type == "authentication_error"
    assert len(read_jsonl(rec.path("attempts.jsonl"))) == 1  # no retry


async def test_empty_content_retried_then_failed(tmp_path):
    client = MockClient(behaviours=[{"content": ""}])
    ex, rec = _executor(tmp_path, client, RetryPolicy(max_retries=1, initial_seconds=0.001, max_seconds=0.01))
    res = await ex.execute(SAMPLE, MSG, 0)
    assert res.status == "error" and res.error_type == "empty_response"
    assert len(read_jsonl(rec.path("attempts.jsonl"))) == 2


async def test_timeout_marks_usage_unknown(tmp_path):
    client = MockClient(behaviours=[err(ErrorType.API_TIMEOUT)])
    ex, rec = _executor(tmp_path, client, RetryPolicy(max_retries=0, initial_seconds=0.001, max_seconds=0.01))
    res = await ex.execute(SAMPLE, MSG, 0)
    attempts = read_jsonl(rec.path("attempts.jsonl"))
    assert attempts[0]["usage_status"] == "unknown_due_to_timeout"
    assert "cost" not in attempts[0]
    assert not any("cost" in key for key in res.model_dump())


async def test_usage_is_persisted_without_monetary_fields(tmp_path):
    ex, rec = _executor(tmp_path, MockClient())
    res = await ex.execute(SAMPLE, MSG, 0)
    attempt = read_jsonl(rec.path("attempts.jsonl"))[0]
    assert attempt["usage"]["prompt_tokens"] == 10
    assert attempt["usage"]["completion_tokens"] == 3
    assert attempt["usage"]["total_tokens"] == 13
    assert (res.prompt_tokens, res.completion_tokens, res.total_tokens) == (10, 3, 13)
    assert "cost" not in attempt
    assert not any("cost" in key for key in res.model_dump())


async def test_content_filter_is_terminal_and_preserves_provider_usage(tmp_path):
    client = MockClient(behaviours=[{"content": "", "finish_reason": "content_filter"}])
    ex, rec = _executor(tmp_path, client)

    result = await ex.execute(SAMPLE, MSG, 0)

    assert result.status == "error"
    assert result.error_type == "content_filter"
    assert client.calls == 1
    attempt = read_jsonl(rec.path("attempts.jsonl"))[0]
    assert attempt["status"] == "error"
    assert attempt["finish_reason"] == "content_filter"
    assert attempt["usage"]["total_tokens"] == 13


async def test_native_refusal_is_retained_as_successful_response(tmp_path):
    refusal = "I cannot help with that request."
    client = MockClient(behaviours=[{"content": refusal, "refusal": refusal}])
    ex, _ = _executor(tmp_path, client)

    result = await ex.execute(SAMPLE, MSG, 0)

    assert result.status == "success"
    assert result.raw_response == refusal
    assert result.provider_metadata["native_refusal"] is True


async def test_repeat_index_derives_distinct_deterministic_seed(tmp_path):
    client = MockClient()
    ex, _ = _executor(tmp_path, client)

    await ex.execute(SAMPLE, MSG, 0)
    await ex.execute(SAMPLE, MSG, 1)

    assert [call["seed"] for call in client.call_kwargs] == [42, 43]


async def test_tpm_reservation_is_reconciled_with_actual_usage():
    limiter = RateLimiter(tokens_per_minute=100)

    reservation_id = await limiter.acquire(estimated_tokens=80)
    assert reservation_id is not None
    assert limiter._current_tokens() == 80
    await limiter.record_tokens(23, reservation_id)

    assert limiter._current_tokens() == 23
