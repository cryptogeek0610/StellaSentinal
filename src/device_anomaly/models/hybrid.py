from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from device_anomaly.models.anomaly_detector import (
    AnomalyDetectorConfig,
    AnomalyDetectorIsolationForest,
)
from device_anomaly.models.heuristics import summarize_heuristics

LOGGER = logging.getLogger(__name__)


@dataclass
class TemporalResidualConfig:
    window: int = 7
    min_mad: float = 1e-6


class TemporalResidualDetector:
    """
    Simple temporal residual scorer that looks at consecutive daily deltas per device.
    Large jumps relative to historical deltas push the score negative (more anomalous).
    """

    def __init__(self, config: TemporalResidualConfig | None = None):
        self.config = config or TemporalResidualConfig()
        self.feature_cols: list[str] = []
        self.delta_median: pd.Series | None = None
        self.delta_mad: pd.Series | None = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        if "Timestamp" not in df.columns or "DeviceId" not in df.columns:
            self.feature_cols = []
            self.delta_median = None
            self.delta_mad = None
            LOGGER.warning(
                "TemporalResidualDetector requires Timestamp and DeviceId; skipping temporal fit."
            )
            return

        self.feature_cols = [
            c for c in feature_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)
        ]
        if not self.feature_cols:
            LOGGER.warning(
                "TemporalResidualDetector found no overlapping numeric columns; skipping temporal fit."
            )
            return

        df_sorted = df.sort_values(["DeviceId", "Timestamp"])
        deltas = df_sorted.groupby("DeviceId")[self.feature_cols].diff()

        self.delta_median = deltas.median(skipna=True)
        self.delta_mad = deltas.apply(lambda s: (s - s.median()).abs().median()).fillna(
            self.config.min_mad
        )

    def score(self, df: pd.DataFrame) -> pd.Series:
        if not self.feature_cols or self.delta_median is None or self.delta_mad is None:
            return pd.Series(np.zeros(len(df)), index=df.index, name="temporal_score")

        if "Timestamp" not in df.columns or "DeviceId" not in df.columns:
            return pd.Series(np.zeros(len(df)), index=df.index, name="temporal_score")

        df_sorted = df.sort_values(["DeviceId", "Timestamp"])
        deltas = df_sorted.groupby("DeviceId")[self.feature_cols].diff()
        deltas = deltas.reindex(df.index)

        z = (deltas - self.delta_median) / (self.delta_mad + self.config.min_mad)
        abs_z = z.abs()
        score = -abs_z.mean(axis=1).fillna(0.0)
        score.name = "temporal_score"
        return score


@dataclass
class HybridAnomalyDetectorConfig:
    iso_config: AnomalyDetectorConfig = field(default_factory=AnomalyDetectorConfig)
    temporal_config: TemporalResidualConfig = field(default_factory=TemporalResidualConfig)
    iso_weight: float = 0.6
    temporal_weight: float = 0.4
    use_cohort_models: bool = True
    min_cohort_rows: int = 200
    feature_overrides: list[str] | None = None
    heuristic_weight: float = 0.2


class HybridAnomalyDetector:
    """
    Combines the IsolationForest detector with a temporal residual scorer.
    Optionally trains per-cohort IsolationForest models for cohorts with enough data.
    """

    def __init__(self, config: HybridAnomalyDetectorConfig | None = None):
        self.config = config or HybridAnomalyDetectorConfig()
        self.global_detector = AnomalyDetectorIsolationForest(
            config=self.config.iso_config,
            feature_overrides=self.config.feature_overrides,
        )
        self.temporal_detector = TemporalResidualDetector(config=self.config.temporal_config)
        self.cohort_models: dict[str, AnomalyDetectorIsolationForest] = {}
        self.iso_score_mean: float = 0.0
        self.iso_score_std: float = 1.0
        self.threshold: float = 0.0
        self._score_scale: float = 1.0

    @property
    def feature_columns(self) -> list[str]:
        return self.global_detector.feature_cols

    def fit(self, df: pd.DataFrame) -> None:
        self.global_detector.fit(df)
        iso_scores = self.global_detector.score(df)
        self.iso_score_mean = float(np.mean(iso_scores))
        self.iso_score_std = float(np.std(iso_scores) or 1.0)

        self.temporal_detector.fit(df, self.global_detector.feature_cols)

        temporal_scores = self.temporal_detector.score(df)
        combined_raw = self._combine_raw_scores(
            pd.Series(iso_scores, index=df.index), temporal_scores
        )
        self._score_scale = float(np.nanmax(np.abs(combined_raw)) or 1.0)
        combined = np.clip(combined_raw / max(self._score_scale, 1e-6), -1.0, 1.0)
        contamination = np.clip(self.config.iso_config.contamination, 0.001, 0.5)
        cutoff_idx = max(int(len(combined) * contamination), 1)
        sorted_scores = np.sort(combined)
        self.threshold = float(sorted_scores[cutoff_idx - 1])

        self.cohort_models = {}
        if self.config.use_cohort_models and "cohort_id" in df.columns:
            for cohort_id, grp in df.groupby("cohort_id"):
                if len(grp) < self.config.min_cohort_rows:
                    continue
                model = AnomalyDetectorIsolationForest(
                    config=self.config.iso_config,
                    feature_overrides=self.config.feature_overrides,
                )
                try:
                    model.fit(grp)
                    self.cohort_models[str(cohort_id)] = model
                except ValueError:
                    continue
            if self.cohort_models:
                LOGGER.info(
                    "Trained %d cohort-specific IsolationForest models.", len(self.cohort_models)
                )

    def score_dataframe(
        self, df: pd.DataFrame, heuristic_flags: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        df_scored = df.copy()
        iso_scores = pd.Series(self.global_detector.score(df), index=df.index, name="iforest_score")
        iso_preds = pd.Series(self.global_detector.predict(df), index=df.index)
        cohort_used = pd.Series(False, index=df.index, name="cohort_model_used")

        if self.cohort_models and "cohort_id" in df.columns:
            for cohort_id, model in self.cohort_models.items():
                mask = df["cohort_id"] == cohort_id
                if not mask.any():
                    continue
                try:
                    iso_scores.loc[mask] = model.score(df.loc[mask])
                    iso_preds.loc[mask] = model.predict(df.loc[mask])
                    cohort_used.loc[mask] = True
                except ValueError:
                    LOGGER.warning(
                        "Cohort model %s missing features; falling back to global model.", cohort_id
                    )

        temporal_scores = self.temporal_detector.score(df)
        combined_raw = self._combine_raw_scores(iso_scores, temporal_scores)

        heuristic_score = pd.Series(0.0, index=df.index, name="heuristic_score")
        heuristic_reasons = pd.Series("", index=df.index, name="heuristic_reasons")
        if (
            heuristic_flags is not None
            and not heuristic_flags.empty
            and "DeviceId" in df_scored.columns
        ):
            summary = summarize_heuristics(heuristic_flags)
            if not summary.empty:
                summary_indexed = summary.set_index("DeviceId")
                heuristic_score = (
                    df_scored["DeviceId"].map(summary_indexed["HeuristicScore"]).fillna(0.0)
                )
                heuristic_reasons = (
                    df_scored["DeviceId"].map(summary_indexed["HeuristicReasons"]).fillna("")
                )
                combined_raw = (
                    combined_raw - self.config.heuristic_weight * heuristic_score.to_numpy()
                )

        combined = np.clip(combined_raw / max(self._score_scale, 1e-6), -1.0, 1.0)

        labels = np.where(combined < self.threshold, -1, 1)
        df_scored["iforest_score"] = iso_scores
        df_scored["temporal_score"] = temporal_scores
        df_scored["hybrid_score"] = combined
        df_scored["anomaly_score"] = combined
        df_scored["anomaly_label"] = labels
        df_scored["cohort_model_used"] = cohort_used
        df_scored["heuristic_score"] = heuristic_score
        df_scored["heuristic_reasons"] = heuristic_reasons
        return df_scored

    def _combine_raw_scores(self, iso_scores: pd.Series, temporal_scores: pd.Series) -> np.ndarray:
        iso_norm = (iso_scores - self.iso_score_mean) / max(self.iso_score_std, 1e-6)
        iso_norm = iso_norm.fillna(0.0)
        temporal_norm = temporal_scores.fillna(temporal_scores.median() or 0.0)

        combined = (
            self.config.iso_weight * iso_norm.to_numpy()
            + self.config.temporal_weight * temporal_norm.to_numpy()
        )
        return combined
