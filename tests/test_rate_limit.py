"""Tests for the sliding-window rate limiter."""

import time

from device_anomaly.api.rate_limit import _SlidingWindowCounter


class TestSlidingWindowCounter:
    def test_allows_requests_under_limit(self):
        counter = _SlidingWindowCounter(max_requests=5, window_seconds=60, burst=0)
        for _ in range(5):
            allowed, remaining, retry = counter.is_allowed("client-1")
            assert allowed is True
            assert retry == 0

    def test_rejects_over_limit(self):
        counter = _SlidingWindowCounter(max_requests=3, window_seconds=60, burst=0)
        for _ in range(3):
            counter.is_allowed("client-1")

        allowed, remaining, retry = counter.is_allowed("client-1")
        assert allowed is False
        assert remaining == 0
        assert retry >= 1

    def test_burst_extends_limit(self):
        counter = _SlidingWindowCounter(max_requests=3, window_seconds=60, burst=2)
        # Should allow 3 + 2 = 5 requests
        for i in range(5):
            allowed, _, _ = counter.is_allowed("client-1")
            assert allowed is True, f"Request {i + 1} should be allowed"

        allowed, _, _ = counter.is_allowed("client-1")
        assert allowed is False

    def test_different_keys_independent(self):
        counter = _SlidingWindowCounter(max_requests=2, window_seconds=60, burst=0)
        counter.is_allowed("client-a")
        counter.is_allowed("client-a")

        # client-a is at limit
        allowed_a, _, _ = counter.is_allowed("client-a")
        assert allowed_a is False

        # client-b should still be fine
        allowed_b, _, _ = counter.is_allowed("client-b")
        assert allowed_b is True

    def test_remaining_decreases(self):
        counter = _SlidingWindowCounter(max_requests=5, window_seconds=60, burst=0)
        _, remaining1, _ = counter.is_allowed("k")
        _, remaining2, _ = counter.is_allowed("k")
        assert remaining1 > remaining2

    def test_window_expiry(self):
        # Use a very short window to test expiry
        counter = _SlidingWindowCounter(max_requests=1, window_seconds=1, burst=0)
        counter.is_allowed("k")
        allowed, _, _ = counter.is_allowed("k")
        assert allowed is False

        # Wait for window to expire
        time.sleep(1.1)
        allowed, _, _ = counter.is_allowed("k")
        assert allowed is True

    def test_cleanup_removes_stale_keys(self):
        counter = _SlidingWindowCounter(max_requests=10, window_seconds=1, burst=0)
        counter.is_allowed("stale-key")
        # Force cleanup after window expires
        time.sleep(1.1)
        counter._cleanup(time.monotonic() - 1)
        assert "stale-key" not in counter._requests
