"""In-memory sliding-window rate limiting (single process).

Fine for one uvicorn worker. With several workers or replicas each process
keeps its own counters; move to Redis at that point.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Request

from app.core.exception import AppError


class RateLimitError(AppError):
    status_code = 429
    error_code = "rate_limited"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _enabled() -> bool:
    return os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class SlidingWindowLimiter:
    _SWEEP_THRESHOLD = 10_000

    def __init__(self, max_events: int, window_seconds: int) -> None:
        self.max_events = max_events
        self.window = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float] | None:
        q = self._events.get(key)
        if q is None:
            return None
        cutoff = now - self.window
        while q and q[0] <= cutoff:
            q.popleft()
        if not q:
            del self._events[key]
            return None
        return q

    def _sweep(self, now: float) -> None:
        for key in list(self._events):
            self._prune(key, now)

    def _wait_seconds(self, key: str, now: float) -> int:
        q = self._prune(key, now)
        if q is None or len(q) < self.max_events:
            return 0
        return max(1, int(q[0] + self.window - now) + 1)

    def check(self, key: str) -> None:
        """Raise RateLimitError if the key is currently over its limit."""
        if not _enabled():
            return
        with self._lock:
            wait = self._wait_seconds(key, time.monotonic())
        if wait:
            raise RateLimitError(
                f"Too many attempts. Try again in {wait} seconds.",
                details={"retry_after_seconds": wait},
            )

    def hit(self, key: str) -> None:
        """Record one event for the key."""
        if not _enabled():
            return
        now = time.monotonic()
        with self._lock:
            if len(self._events) > self._SWEEP_THRESHOLD:
                self._sweep(now)
            self._events[key].append(now)

    def consume(self, key: str) -> None:
        """check() then hit(): count every attempt, reject once over the limit."""
        self.check(key)
        self.hit(key)

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)


_LOGIN_WINDOW = _env_int("RATE_LIMIT_LOGIN_WINDOW_SECONDS", 900)

# Failed logins per (client ip + username).
login_failure_limiter = SlidingWindowLimiter(
    _env_int("RATE_LIMIT_LOGIN_MAX_FAILURES", 5), _LOGIN_WINDOW
)
# Registration attempts per client ip.
register_limiter = SlidingWindowLimiter(
    _env_int("RATE_LIMIT_REGISTER_MAX_PER_HOUR", 10), 3600
)