"""Regression tests for direct and environment-backed API credentials."""

from __future__ import annotations

import json

from healthcorebench.config import (
    config_hash,
    load_config,
    redact_config_for_persistence,
    resolve_api_key,
)
from healthcorebench.schemas.config import JudgeConfig, RunConfig
import pytest


def test_direct_judge_api_key_is_accepted_and_not_serialized(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
experiment:
  experiment_id: secret-test
  run_name: secret-test
benchmark:
  name: MMLU/mcqa
model:
  base_url: http://localhost:8000/v1
  requested_model_name: test-model
evaluation:
  judge:
    base_url: https://judge.example/v1
    api_key: sk-direct-secret
    requested_model_name: test-judge
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert resolve_api_key(
        config.evaluation.judge.api_key_env,
        config.evaluation.judge.api_key,
    ) == "sk-direct-secret"
    assert "sk-direct-secret" not in json.dumps(config.model_dump())
    assert "api_key" not in config.model_dump()["evaluation"]["judge"]
    assert "sk-direct-secret" not in json.dumps(config.config_hash_payload())


def test_direct_api_key_takes_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "sk-from-environment")
    judge = JudgeConfig(
        base_url="https://judge.example/v1",
        api_key="sk-direct",
        api_key_env="JUDGE_API_KEY",
        requested_model_name="test-judge",
    )

    assert resolve_api_key(judge.api_key_env, judge.api_key) == "sk-direct"


def test_environment_api_key_and_empty_fallback(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "sk-from-environment")
    assert resolve_api_key("JUDGE_API_KEY") == "sk-from-environment"

    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    assert resolve_api_key("MISSING_API_KEY") == "EMPTY"


def test_judge_scoring_settings_are_part_of_resume_identity(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
experiment: {experiment_id: hash-test, run_name: hash-test}
benchmark: {name: MMLU/mcqa}
model:
  base_url: http://localhost:8000/v1
  requested_model_name: test-model
evaluation:
  use_llm_judge: true
  judge:
    base_url: https://judge.example/v1
    api_key: sk-secret
    requested_model_name: judge-a
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    original_hash = config_hash(config)

    config.evaluation.judge.requested_model_name = "judge-b"
    assert config_hash(config) != original_hash

    semantic_hash = config_hash(config)
    config.evaluation.judge.request_timeout_seconds += 10
    config.evaluation.judge.concurrency += 1
    assert config_hash(config) == semantic_hash

    config.evaluation.use_llm_judge = False
    rule_based_hash = config_hash(config)
    config.evaluation.judge.requested_model_name = "unused-judge"
    assert config_hash(config) == rule_based_hash


def test_manifest_config_redacts_headers_extra_body_and_url_userinfo():
    config = RunConfig.model_validate({
        "experiment": {"experiment_id": "redaction", "run_name": "redaction"},
        "benchmark": {"name": "MMLU/mcqa"},
        "model": {
            "base_url": "https://user:password@provider.example/v1",
            "api_key": "sk-model-secret",
            "requested_model_name": "test-model",
            "default_headers": {
                "Authorization": "Bearer header-secret",
                "X-Provider-Routing": "strict",
            },
        },
        "generation": {
            "extra_body": {
                "session_token": "body-secret",
                "x_api_token": "other-body-secret",
                "callback_url": "https://callback-user:callback-password@callback.example/hook",
                "auditURL": "https://audit-user:audit-password@audit.example/endpoint",
                "provider_option": "enabled",
            },
        },
    })

    persisted = redact_config_for_persistence(config)
    serialized = json.dumps(persisted)

    for secret in (
        "sk-model-secret", "header-secret", "body-secret", "other-body-secret", "user:password",
        "callback-user:callback-password", "audit-user:audit-password",
    ):
        assert secret not in serialized
    assert persisted["model"]["base_url"] == "https://provider.example/v1"
    assert persisted["model"]["default_headers"]["Authorization"] == "***REDACTED***"
    assert persisted["model"]["default_headers"]["X-Provider-Routing"] == "strict"
    assert persisted["generation"]["extra_body"]["session_token"] == "***REDACTED***"
    assert persisted["generation"]["extra_body"]["x_api_token"] == "***REDACTED***"
    assert persisted["generation"]["extra_body"]["callback_url"] == "https://callback.example/hook"
    assert persisted["generation"]["extra_body"]["auditURL"] == "https://audit.example/endpoint"
    assert persisted["generation"]["extra_body"]["provider_option"] == "enabled"
    assert persisted["generation"]["max_tokens"] == config.generation.max_tokens
    assert persisted["generation"]["max_tokens_candidates"] == config.generation.max_tokens_candidates
    assert persisted["generation"]["context_token_reserve"] == config.generation.context_token_reserve
    assert persisted["runtime"]["tokens_per_minute"] is None


@pytest.mark.parametrize("value", [0, -1, [1024, 0], [1024, -1]])
def test_generation_rejects_non_positive_token_budgets(value):
    generation = {"max_tokens": value} if isinstance(value, int) else {"max_tokens_candidates": value}
    with pytest.raises(ValueError):
        RunConfig.model_validate({
            "experiment": {"experiment_id": "invalid", "run_name": "invalid"},
            "benchmark": {"name": "MMLU/mcqa"},
            "model": {"base_url": "http://localhost:8000/v1", "requested_model_name": "test"},
            "generation": generation,
        })


def test_removed_pricing_section_is_rejected():
    with pytest.raises(ValueError, match="pricing"):
        RunConfig.model_validate({
            "experiment": {"experiment_id": "invalid", "run_name": "invalid"},
            "benchmark": {"name": "MMLU/mcqa"},
            "model": {"base_url": "http://localhost:8000/v1", "requested_model_name": "test"},
            "pricing": {"pricing_mode": "not_applicable"},
        })
