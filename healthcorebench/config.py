"""Config loading, project-root discovery, and CLI overrides.

The project root is discovered deterministically (the directory containing the
``healthcorebench`` package), so benchmark data under ``benchmarks/medical_llm_benchmarks/`` is
located identically regardless of the caller's working directory. It can be overridden by
``HEALTHCOREBENCH_PROJECT_ROOT`` for unusual deployments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from pydantic import SecretStr

from healthcorebench.schemas.config import RunConfig
from healthcorebench.utils.hashing import hash_json


_SENSITIVE_CONFIG_KEYS = {
    "api_key", "apikey", "authorization", "access_token", "auth_token",
    "token", "api_token", "bearer_token", "session_token", "refresh_token", "id_token", "bearer",
    "secret", "client_secret", "password", "credential", "credentials",
}
_SENSITIVE_CONFIG_KEY_SUFFIXES = (
    "_api_key", "_token", "_secret", "_password", "_credential",
    "_credentials",
)


def _is_sensitive_config_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SENSITIVE_CONFIG_KEYS or normalized.endswith(
        _SENSITIVE_CONFIG_KEY_SUFFIXES
    )


def get_project_root() -> Path:
    """Return the project root (parent of the ``healthcorebench`` package)."""
    env = os.environ.get("HEALTHCOREBENCH_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # this file is <root>/healthcorebench/config.py
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Load a YAML run config, apply flat dotted overrides, and validate it."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(raw, dotted, value)
    return RunConfig.model_validate(raw)


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def config_hash(config: RunConfig) -> str:
    """Stable hash of the identity-defining subset of the config (for resume validation)."""
    return hash_json(config.config_hash_payload())


def redact_config_for_persistence(config: RunConfig) -> dict:
    """Return a manifest-safe config dump without changing the runtime config.

    ``SecretStr`` fields are excluded by the schema already. This additionally covers
    credentials supplied through arbitrary default headers / provider-specific bodies,
    which are ordinary dictionaries and therefore need recursive key-aware redaction.
    """
    return _redact_config_value(config.model_dump())


def _redact_config_value(value: Any, *, key: str | None = None) -> Any:
    if key and _is_sensitive_config_key(key):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {str(k): _redact_config_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_config_value(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [_redact_config_value(item, key=key) for item in value]
    normalized_key = key.lower().replace("-", "_") if key else ""
    if isinstance(value, str) and (
        normalized_key in {"base_url", "url", "endpoint", "endpoint_url"}
        or normalized_key.endswith("_url")
        or normalized_key.endswith("url")
    ):
        return redact_base_url(value)
    return value


def redact_base_url(base_url: str) -> str:
    """Strip any embedded credentials from a base URL before logging."""
    # base URLs generally don't carry creds, but guard against user@host forms.
    if "@" in base_url:
        scheme, _, rest = base_url.partition("://")
        if "@" in rest:
            rest = rest.split("@", 1)[1]
        return f"{scheme}://{rest}" if scheme else rest
    return base_url


def resolve_api_key(api_key_env: str, api_key: SecretStr | None = None) -> str:
    """Resolve a direct or environment-backed API key without logging it.

    A directly configured key takes precedence. This supports self-contained configs while
    keeping the value excluded from Pydantic dumps and persisted run artifacts.
    """
    if api_key is not None:
        return api_key.get_secret_value()
    key = os.environ.get(api_key_env)
    if not key:
        # vLLM commonly ignores the key; provide a harmless placeholder so the SDK is happy.
        return "EMPTY"
    return key
