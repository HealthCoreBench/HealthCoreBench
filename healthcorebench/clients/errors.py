"""Error taxonomy, classification and secret redaction.

Core principle: an API / parse / evaluation failure must never be silently converted into
a wrong model answer. Every failure is classified into a stable ``ErrorType`` and recorded;
retryability is derived from the type. Secrets (API keys, auth headers) are stripped from
any message before it is logged.
"""

from __future__ import annotations

import re
from enum import Enum


class ErrorType(str, Enum):
    """Stable error taxonomy. Values are persisted verbatim into logs."""

    # --- transport / API ---
    API_TIMEOUT = "api_timeout"
    CONNECTION_ERROR = "connection_error"
    DNS_ERROR = "dns_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    INVALID_REQUEST = "invalid_request"
    SERVER_ERROR = "server_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    CONTENT_FILTER = "content_filter"
    MODEL_REFUSAL = "model_refusal"
    MAX_OUTPUT_LENGTH = "max_output_length"
    EMPTY_RESPONSE = "empty_response"
    MALFORMED_RESPONSE = "malformed_response"
    # --- media ---
    MEDIA_NOT_FOUND = "media_not_found"
    MEDIA_DECODE_ERROR = "media_decode_error"
    MEDIA_TOO_LARGE = "media_too_large"
    UNSUPPORTED_MEDIA = "unsupported_media"
    # --- pipeline ---
    PROMPT_BUILD_ERROR = "prompt_build_error"
    RESPONSE_PARSE_ERROR = "response_parse_error"
    EVALUATION_ERROR = "evaluation_error"
    JUDGE_ERROR = "judge_error"
    SERIALIZATION_ERROR = "serialization_error"
    UNKNOWN_ERROR = "unknown_error"


# Error types that are transient and may be retried. Everything else (auth, invalid
# request, deterministic media/parse errors) is not retried by default.
RETRYABLE_ERROR_TYPES: frozenset[ErrorType] = frozenset(
    {
        ErrorType.API_TIMEOUT,
        ErrorType.CONNECTION_ERROR,
        ErrorType.DNS_ERROR,
        ErrorType.RATE_LIMIT,
        ErrorType.SERVER_ERROR,
        ErrorType.SERVICE_UNAVAILABLE,
        ErrorType.EMPTY_RESPONSE,
    }
)


class ClientError(Exception):
    """Structured client error carrying a classified type and metadata.

    ``retryable`` defaults to the taxonomy classification but can be overridden (e.g. a
    5xx that a provider marks non-retryable).
    """

    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        *,
        http_status: int | None = None,
        exception_class: str | None = None,
        retryable: bool | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.error_type = error_type
        self.message = redact_secrets(message)
        self.http_status = http_status
        self.exception_class = exception_class
        self.retryable = (
            error_type in RETRYABLE_ERROR_TYPES if retryable is None else retryable
        )
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"[{error_type.value}] {self.message}")

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type.value,
            "error_message": self.message,
            "exception_class": self.exception_class,
            "http_status": self.http_status,
            "retryable": self.retryable,
        }


# Patterns that may contain secrets. Applied to every message before logging.
_REDACT_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[=:]\s*)([^\s,&'\"]+)", re.IGNORECASE),
    re.compile(r"(authorization\s*[=:]\s*)([^\s,&'\"]+)", re.IGNORECASE),
]


def redact_secrets(text: str | None) -> str:
    """Remove API keys and auth tokens from a string before it is persisted."""
    if not text:
        return "" if text is None else text
    redacted = text
    redacted = _REDACT_PATTERNS[0].sub("sk-***REDACTED***", redacted)
    redacted = _REDACT_PATTERNS[1].sub(r"\1***REDACTED***", redacted)
    redacted = _REDACT_PATTERNS[2].sub(r"\1***REDACTED***", redacted)
    redacted = _REDACT_PATTERNS[3].sub(r"\1***REDACTED***", redacted)
    return redacted


def classify_exception(exc: BaseException) -> ClientError:
    """Map an arbitrary exception (openai SDK, httpx, stdlib) to a ``ClientError``.

    The mapping is defensive and import-light: it inspects class names, HTTP status codes
    and message text rather than hard-importing every possible SDK exception type, so it
    keeps working across SDK versions.
    """
    if isinstance(exc, ClientError):
        return exc

    cls_name = type(exc).__name__
    msg = str(exc)
    http_status = getattr(exc, "status_code", None)
    if http_status is None:
        response = getattr(exc, "response", None)
        http_status = getattr(response, "status_code", None)

    retry_after = _extract_retry_after(exc)

    # --- status-code driven classification (most reliable) ---
    if http_status is not None:
        if http_status == 429:
            return ClientError(ErrorType.RATE_LIMIT, msg, http_status=http_status,
                               exception_class=cls_name, retry_after_seconds=retry_after)
        if http_status in (401,):
            return ClientError(ErrorType.AUTHENTICATION_ERROR, msg, http_status=http_status,
                               exception_class=cls_name)
        if http_status in (403,):
            return ClientError(ErrorType.PERMISSION_ERROR, msg, http_status=http_status,
                               exception_class=cls_name)
        if http_status in (400, 404, 405, 422):
            return ClientError(ErrorType.INVALID_REQUEST, msg, http_status=http_status,
                               exception_class=cls_name)
        if http_status == 503:
            return ClientError(ErrorType.SERVICE_UNAVAILABLE, msg, http_status=http_status,
                               exception_class=cls_name, retry_after_seconds=retry_after)
        if 500 <= http_status < 600:
            return ClientError(ErrorType.SERVER_ERROR, msg, http_status=http_status,
                               exception_class=cls_name, retry_after_seconds=retry_after)

    # --- class-name / message driven fallback ---
    lname = cls_name.lower()
    lmsg = msg.lower()
    if "timeout" in lname or "timeout" in lmsg:
        return ClientError(ErrorType.API_TIMEOUT, msg, http_status=http_status, exception_class=cls_name)
    if "authentication" in lname or "unauthorized" in lmsg:
        return ClientError(ErrorType.AUTHENTICATION_ERROR, msg, http_status=http_status, exception_class=cls_name)
    if "permission" in lname or "forbidden" in lmsg:
        return ClientError(ErrorType.PERMISSION_ERROR, msg, http_status=http_status, exception_class=cls_name)
    if "ratelimit" in lname or "rate limit" in lmsg or "rate_limit" in lmsg:
        return ClientError(ErrorType.RATE_LIMIT, msg, http_status=http_status,
                           exception_class=cls_name, retry_after_seconds=retry_after)
    if "getaddrinfo" in lmsg or "name or service not known" in lmsg or "dns" in lmsg:
        return ClientError(ErrorType.DNS_ERROR, msg, http_status=http_status, exception_class=cls_name)
    if "connection" in lname or "connect" in lmsg:
        return ClientError(ErrorType.CONNECTION_ERROR, msg, http_status=http_status, exception_class=cls_name)
    if "badrequest" in lname or "invalid" in lname:
        return ClientError(ErrorType.INVALID_REQUEST, msg, http_status=http_status, exception_class=cls_name)
    if isinstance(exc, (TimeoutError,)):
        return ClientError(ErrorType.API_TIMEOUT, msg, exception_class=cls_name)
    if isinstance(exc, (ConnectionError,)):
        return ClientError(ErrorType.CONNECTION_ERROR, msg, exception_class=cls_name)

    return ClientError(ErrorType.UNKNOWN_ERROR, msg, http_status=http_status, exception_class=cls_name)


# Failures that mean the endpoint itself is unusable, as opposed to one bad request or one
# poor response. A run that keeps hitting these is not producing evaluation data.
INFRASTRUCTURE_ERROR_TYPES: frozenset[ErrorType] = frozenset(
    {
        ErrorType.AUTHENTICATION_ERROR,
        ErrorType.PERMISSION_ERROR,
        ErrorType.RATE_LIMIT,
        ErrorType.SERVICE_UNAVAILABLE,
        ErrorType.SERVER_ERROR,
    }
)

_STATUS_IN_MESSAGE = re.compile(
    r"\b(?:error code|status(?:\s+code)?|http)\b\D{0,3}(\d{3})\b", re.IGNORECASE
)


class _RecordedError(Exception):
    """Shim so a persisted error string can reuse the live classification rules."""

    def __init__(self, message: str, status_code: int | None) -> None:
        super().__init__(message)
        self.status_code = status_code


def classify_error_message(message: str | None) -> ErrorType | None:
    """Recover an ``ErrorType`` from an error string that was persisted without one.

    Judgment records keep only the message text, so callers that must react to *why* a
    judge failed (a dead endpoint versus one unparseable verdict) re-derive the type here
    instead of matching ad-hoc substrings. Returns ``None`` for an empty message.
    """
    if not message:
        return None
    match = _STATUS_IN_MESSAGE.search(message)
    status = int(match.group(1)) if match else None
    return classify_exception(_RecordedError(message, status)).error_type


def _extract_retry_after(exc: BaseException) -> float | None:
    """Pull a ``Retry-After`` header value (seconds) from an exception if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
