"""Timezone-aware UTC timestamp helpers.

All persisted timestamps use ISO-8601 UTC with microseconds and a trailing ``Z``, e.g.
``2026-07-17T09:30:12.123456Z``. All durations are numeric seconds (float).
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with microseconds and a ``Z`` suffix."""
    return _to_iso(datetime.now(timezone.utc))


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    # isoformat() yields +00:00; normalize to the canonical trailing Z.
    return dt.isoformat().replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    """Parse an ISO-8601 timestamp (tolerating a trailing ``Z``) into an aware datetime."""
    normalized = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def duration_seconds(start_iso: str, end_iso: str) -> float:
    """Seconds elapsed between two ISO-8601 timestamps (float)."""
    return (parse_iso(end_iso) - parse_iso(start_iso)).total_seconds()
