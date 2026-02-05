"""Rate limiting middleware for the Stella Sentinel API.

Uses an in-memory sliding window counter per client key.
Suitable for single-process deployments (Docker Compose / uvicorn).

Configuration via environment variables:
    RATE_LIMIT_ENABLED=true          (default: true)
    RATE_LIMIT_REQUESTS=100          (default: 100 per window)
    RATE_LIMIT_WINDOW_SECONDS=60     (default: 60)
    RATE_LIMIT_BURST=20              (default: 20 — extra burst allowance)
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Paths exempt from rate limiting
_BYPASS_PATHS = frozenset(
    {
        "/",
        "/health",
        "/health/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/metrics",
    }
)


class _SlidingWindowCounter:
    """Thread-safe sliding window rate limiter.

    Tracks request timestamps per key and counts how many fall within the
    current window.  Periodically evicts expired entries to bound memory.
    """

    def __init__(self, max_requests: int, window_seconds: int, burst: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst = burst
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = max(window_seconds * 2, 120)

    def is_allowed(self, key: str) -> tuple[bool, int, int]:
        """Check if a request is allowed.

        Returns:
            (allowed, remaining, retry_after_seconds)
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Lazy cleanup of stale keys
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup(cutoff)

        timestamps = self._requests[key]
        # Remove expired timestamps for this key
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

        limit = self.max_requests + self.burst
        if len(timestamps) >= limit:
            retry_after = int(timestamps[0] - cutoff) + 1
            return False, 0, max(retry_after, 1)

        timestamps.append(now)
        remaining = max(0, limit - len(timestamps))
        return True, remaining, 0

    def _cleanup(self, cutoff: float) -> None:
        """Remove keys with no recent activity."""
        stale_keys = [k for k, v in self._requests.items() if not v or v[-1] < cutoff]
        for k in stale_keys:
            del self._requests[k]
        self._last_cleanup = time.monotonic()


def _get_client_key(request: Request) -> str:
    """Build a rate-limit key from client IP + tenant."""
    client_ip = "unknown"
    if request.client:
        client_ip = request.client.host

    # Use forwarded IP if behind a reverse proxy
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    tenant = request.headers.get("X-Tenant-Id", "default")
    return f"{client_ip}:{tenant}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that applies sliding-window rate limiting."""

    def __init__(self, app, **kwargs) -> None:  # noqa: ANN001
        super().__init__(app)
        self.enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.max_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        self.burst = int(os.getenv("RATE_LIMIT_BURST", "20"))
        self._counter = _SlidingWindowCounter(
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
            burst=self.burst,
        )
        if self.enabled:
            logger.info(
                "Rate limiting enabled: %d req/%ds (burst +%d)",
                self.max_requests,
                self.window_seconds,
                self.burst,
            )

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        if not self.enabled:
            return await call_next(request)

        if request.url.path in _BYPASS_PATHS:
            return await call_next(request)

        key = _get_client_key(request)
        allowed, remaining, retry_after = self._counter.is_allowed(key)

        if not allowed:
            logger.warning("Rate limit exceeded for %s on %s", key, request.url.path)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please retry later.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.max_requests + self.burst),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests + self.burst)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
