from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _parse_int(value: Optional[str], default: int) -> int:
    try:
        if value is None:
            return default
        parsed = int(value)
        return parsed
    except (TypeError, ValueError):
        return default


def _parse_float(value: Optional[str], default: float) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        return parsed
    except (TypeError, ValueError):
        return default


@dataclass
class DriftMonitorConfig:
    window_size: int
    bins: int
    warn_psi: float
    log_interval_sec: int
    min_samples: int

    @classmethod
    def from_env(cls) -> "DriftMonitorConfig":
        window_size = _parse_int(os.getenv("STREAMING_DRIFT_WINDOW_SIZE"), 200)
        bins = _parse_int(os.getenv("STREAMING_DRIFT_BINS"), 10)
        warn_psi = _parse_float(os.getenv("STREAMING_DRIFT_WARN_PSI"), 0.2)
        log_interval_sec = _parse_int(os.getenv("STREAMING_DRIFT_INTERVAL_SEC"), 300)
        min_samples = _parse_int(os.getenv("STREAMING_DRIFT_MIN_SAMPLES"), max(20, bins * 2))
        return cls(
            window_size=max(1, window_size),
            bins=max(2, bins),
            warn_psi=max(0.0, warn_psi),
            log_interval_sec=max(1, log_interval_sec),
            min_samples=max(5, min_samples),
        )


class StreamingDriftMonitor:
    def __init__(self, feature_names: list[str], config: DriftMonitorConfig):
        self.feature_names = feature_names
        self.config = config
        self._baseline_ready = False
        self._baseline_values: dict[str, list[float]] = {name: [] for name in feature_names}
        self._baseline_bins: dict[str, np.ndarray] = {}
        self._baseline_counts: dict[str, np.ndarray] = {}
        self._current_counts: dict[str, np.ndarray] = {}
        self._baseline_events = 0
        self._current_events = 0
        self._short_history_events = 0
        self._missing_norms_events = 0
        self._missing_feature_counts: dict[str, int] = {name: 0 for name in feature_names}
        self._last_emit = 0.0

    def update(
        self,
        features: dict[str, float],
        short_history: bool = False,
        missing_norms: bool = False,
    ) -> Optional[dict[str, Any]]:
        if short_history:
            self._short_history_events += 1
        if missing_norms:
            self._missing_norms_events += 1

        if not self._baseline_ready:
            self._baseline_events += 1
            for name in self.feature_names:
                value = self._coerce_value(features.get(name))
                if value is None:
                    self._missing_feature_counts[name] += 1
                    continue
                self._baseline_values[name].append(value)

            if self._baseline_events < self.config.window_size:
                return None

            return self._initialize_baseline()

        self._current_events += 1
        for name in self.feature_names:
            value = self._coerce_value(features.get(name))
            if value is None:
                self._missing_feature_counts[name] += 1
                continue
            bins = self._baseline_bins.get(name)
            if bins is None:
                continue
            idx = int(np.searchsorted(bins, value, side="right") - 1)
            idx = max(0, min(idx, len(bins) - 2))
            self._current_counts[name][idx] += 1

        now = time.time()
        if self._current_events < self.config.window_size and (now - self._last_emit) < self.config.log_interval_sec:
            return None

        metrics = self._emit_metrics()
        self._last_emit = now
        return metrics

    def _initialize_baseline(self) -> Optional[dict[str, Any]]:
        active_features = []
        for name, values in list(self._baseline_values.items()):
            series = np.array(values, dtype=float)
            series = series[np.isfinite(series)]
            if len(series) < self.config.min_samples:
                self._baseline_values.pop(name, None)
                self._missing_feature_counts.pop(name, None)
                continue
            bins = self._build_bins(series)
            if bins is None:
                self._baseline_values.pop(name, None)
                self._missing_feature_counts.pop(name, None)
                continue
            counts = np.histogram(series, bins=bins)[0].astype(float)
            self._baseline_bins[name] = bins
            self._baseline_counts[name] = counts
            self._current_counts[name] = np.zeros_like(counts)
            active_features.append(name)

        self._baseline_values.clear()
        self._baseline_ready = True
        self._baseline_events = 0
        self._current_events = 0
        self._short_history_events = 0
        self._missing_norms_events = 0
        self._missing_feature_counts = {name: 0 for name in self._baseline_counts.keys()}
        # Reset last_emit to current time so the first window starts fresh
        self._last_emit = time.time()

        if not active_features:
            logger.warning("Streaming drift monitor could not initialize baseline (no usable features).")
            return None

        return {
            "event": "streaming_drift_baseline_ready",
            "feature_count": len(active_features),
            "window_size": self.config.window_size,
        }

    def _emit_metrics(self) -> Optional[dict[str, Any]]:
        if not self._baseline_counts or self._current_events == 0:
            return None

        psi_scores: dict[str, float] = {}
        warn_features: list[str] = []
        missing_rates: dict[str, float] = {}

        for name, baseline_counts in self._baseline_counts.items():
            current_counts = self._current_counts.get(name)
            if current_counts is None:
                continue
            # Always report missing rates for all tracked features
            missing = self._missing_feature_counts.get(name, 0)
            missing_rates[name] = missing / max(1, self._current_events)
            # PSI may be None if no observations in current window
            psi = self._psi(baseline_counts, current_counts)
            if psi is None:
                continue
            psi_scores[name] = float(psi)
            if psi >= self.config.warn_psi:
                warn_features.append(name)

        metrics = {
            "event": "streaming_feature_drift",
            "feature_count": len(psi_scores),
            "window_size": self.config.window_size,
            "events_in_window": self._current_events,
            "short_history_events": self._short_history_events,
            "missing_norms_events": self._missing_norms_events,
            "psi": psi_scores,
            "missing_feature_rates": missing_rates,
            "warn_psi": self.config.warn_psi,
            "warn_features": warn_features,
        }

        self._baseline_counts = {name: counts.copy() for name, counts in self._current_counts.items()}
        self._current_counts = {name: np.zeros_like(counts) for name, counts in self._baseline_counts.items()}
        self._current_events = 0
        self._short_history_events = 0
        self._missing_norms_events = 0
        self._missing_feature_counts = {name: 0 for name in self._baseline_counts.keys()}

        return metrics

    @staticmethod
    def _coerce_value(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (np.integer, np.floating)):
            value = float(value)
        if not isinstance(value, (int, float)):
            return None
        if not np.isfinite(value):
            return None
        return float(value)

    def _build_bins(self, values: np.ndarray) -> Optional[np.ndarray]:
        if values.size < self.config.min_samples:
            return None
        quantiles = np.linspace(0, 1, self.config.bins + 1)
        edges = np.quantile(values, quantiles)
        edges = np.unique(edges)
        if edges.size < 2:
            return None
        if edges.size < self.config.bins + 1:
            min_val = float(np.min(values))
            max_val = float(np.max(values))
            if min_val == max_val:
                return None
            edges = np.linspace(min_val, max_val, self.config.bins + 1)
        return edges

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray) -> Optional[float]:
        expected_total = float(np.sum(expected))
        actual_total = float(np.sum(actual))
        if expected_total <= 0 or actual_total <= 0:
            return None
        expected_pct = expected / expected_total
        actual_pct = actual / actual_total
        epsilon = 1e-6
        return float(np.sum((actual_pct - expected_pct) * np.log((actual_pct + epsilon) / (expected_pct + epsilon))))

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "baseline_ready": self._baseline_ready,
            "feature_count": len(self._baseline_counts) if self._baseline_ready else 0,
            "window_size": self.config.window_size,
            "bins": self.config.bins,
            "warn_psi": self.config.warn_psi,
            "log_interval_sec": self.config.log_interval_sec,
        }


def resolve_drift_features(defaults: list[str]) -> list[str]:
    raw = os.getenv("STREAMING_DRIFT_FEATURES")
    if raw:
        features = [item.strip() for item in raw.split(",") if item.strip()]
        return features or defaults
    return defaults
