"""Optional RPM / TPM rate limiter for async requests.

Token-bucket style: separate buckets for requests-per-minute and tokens-per-minute. When a
limit is ``None`` that dimension is unbounded. ``acquire`` awaits until capacity is
available. Token accounting is approximate (tokens are only known after a response), so the
caller reserves an estimate up front and reconciles afterwards via ``record_tokens``.
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    def __init__(self, requests_per_minute: float | None = None, tokens_per_minute: float | None = None) -> None:
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._req_times: list[float] = []
        self._tok_events: list[tuple[float, int, int]] = []
        self._lock = asyncio.Lock()
        self._next_reservation_id = 0

    async def acquire(self, estimated_tokens: int = 0) -> int | None:
        if self._rpm is None and self._tpm is None:
            return None
        estimated_tokens = max(0, int(estimated_tokens))
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                rpm_ok = self._rpm is None or len(self._req_times) < self._rpm
                # A single request can legitimately exceed a provider's per-minute budget. Let
                # it occupy the full bucket rather than waiting forever for impossible capacity.
                reserved = min(estimated_tokens, int(self._tpm)) if self._tpm is not None else 0
                tpm_ok = self._tpm is None or (self._current_tokens() + reserved) <= self._tpm
                if rpm_ok and tpm_ok:
                    self._req_times.append(now)
                    if self._tpm is None:
                        return None
                    self._next_reservation_id += 1
                    reservation_id = self._next_reservation_id
                    self._tok_events.append((now, reserved, reservation_id))
                    return reservation_id
                wait = self._time_until_capacity(
                    now, rpm_blocked=not rpm_ok, tpm_blocked=not tpm_ok,
                )
            await asyncio.sleep(max(0.05, wait))

    async def record_tokens(self, actual_tokens: int, reservation_id: int | None = None) -> None:
        """Reconcile an acquired token reservation with provider-reported usage."""
        if self._tpm is None or actual_tokens <= 0:
            if self._tpm is not None and reservation_id is not None and actual_tokens == 0:
                async with self._lock:
                    self._replace_reservation(reservation_id, 0)
            return
        async with self._lock:
            if reservation_id is not None and self._replace_reservation(reservation_id, actual_tokens):
                return
            self._next_reservation_id += 1
            self._tok_events.append((time.monotonic(), actual_tokens, self._next_reservation_id))

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        self._req_times = [t for t in self._req_times if t > cutoff]
        self._tok_events = [(t, n, reservation_id) for t, n, reservation_id in self._tok_events
                            if t > cutoff]

    def _current_tokens(self) -> int:
        return sum(n for _, n, _ in self._tok_events)

    def _time_until_capacity(self, now: float, *, rpm_blocked: bool,
                             tpm_blocked: bool) -> float:
        waits = []
        if rpm_blocked and self._req_times:
            waits.append(self._req_times[0] + 60.0 - now)
        if tpm_blocked and self._tok_events:
            waits.append(self._tok_events[0][0] + 60.0 - now)
        return min(waits) if waits else 0.1

    def _replace_reservation(self, reservation_id: int, actual_tokens: int) -> bool:
        for index, (timestamp, _tokens, current_id) in enumerate(self._tok_events):
            if current_id == reservation_id:
                self._tok_events[index] = (timestamp, actual_tokens, current_id)
                return True
        return False
