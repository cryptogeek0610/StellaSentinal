from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DRIVEN_SCHEMA_VERSION = "data_driven_v1"


@dataclass
class BaselineLevel:
    name: str
    group_columns: list[str]
    min_rows: int = 25


@dataclass
class BaselineFeedback:
    level: str
    group_key: dict[str, object] | str
    feature: str
    adjustment: float
    reason: str | None = None


def _group_key(df: pd.DataFrame, level: BaselineLevel) -> pd.Series:
    missing = [c for c in level.group_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Baseline level '{level.name}' missing grouping cols: {missing}")
    return df[level.group_columns].astype(str).agg("|".join, axis=1)


def compute_baselines(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    levels: list[BaselineLevel],
) -> dict[str, pd.DataFrame]:
    baselines: dict[str, pd.DataFrame] = {}
    for level in levels:
        try:
            key = _group_key(df, level)
        except ValueError:
            continue

        df_level = df.copy()
        df_level["__group_key__"] = key

        agg_rows = []
        for group_val, grp in df_level.groupby("__group_key__"):
            if len(grp) < level.min_rows:
                continue
            for col in feature_cols:
                if col not in grp.columns:
                    continue
                series = pd.to_numeric(grp[col], errors="coerce")
                median = series.median(skipna=True)
                mad = (series - median).abs().median()
                agg_rows.append(
                    {
                        "__group_key__": group_val,
                        "feature": col,
                        "median": float(median),
                        "mad": float(mad if mad != 0 else 1e-6),
                    }
                )

        if not agg_rows:
            continue
        baselines[level.name] = pd.DataFrame(agg_rows)
    return baselines


def apply_baselines(
    df: pd.DataFrame,
    baselines: dict[str, pd.DataFrame],
    levels: list[BaselineLevel],
) -> pd.DataFrame:
    if not baselines:
        return df

    out = df.copy()
    for level in levels:
        level_stats = baselines.get(level.name)
        if level_stats is None or level_stats.empty:
            continue

        try:
            key = _group_key(out, level)
        except ValueError:
            continue

        out["__group_key__"] = key
        merged = (
            out[["__group_key__"]]
            .drop_duplicates()
            .merge(
                level_stats,
                on="__group_key__",
                how="left",
            )
        )

        for _, row in merged.iterrows():
            gkey = row["__group_key__"]
            feature = row["feature"]
            median = row["median"]
            mad = row["mad"] or 1e-6
            mask = out["__group_key__"] == gkey
            if feature in out.columns:
                zcol = f"{feature}_z_{level.name}"
                out.loc[mask, zcol] = (out.loc[mask, feature] - median) / mad

        out = out.drop(columns=["__group_key__"])

    return out


def save_baselines(baselines: dict[str, pd.DataFrame], path: Path) -> None:
    payload = {level: df.to_dict(orient="records") for level, df in baselines.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))


def load_baselines(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    parsed = {}
    for level, rows in raw.items():
        parsed[level] = pd.DataFrame(rows)
    return parsed


def save_baselines_versioned(baselines: dict[str, pd.DataFrame], path: Path) -> Path | None:
    backup_path = None
    if path.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        backup_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        path.replace(backup_path)
    save_baselines(baselines, path)
    return backup_path


def apply_feedback(
    baselines: dict[str, pd.DataFrame],
    feedback_items: list[BaselineFeedback],
    learning_rate: float = 0.3,
) -> dict[str, pd.DataFrame]:
    if not baselines or not feedback_items:
        return baselines

    updated = {}
    for level_name, df in baselines.items():
        df_level = df.copy()
        known_keys = set(df_level.get("__group_key__", pd.Series(dtype=str)).dropna().astype(str))
        for fb in feedback_items:
            if fb.level != level_name:
                continue
            if isinstance(fb.group_key, str):
                gkey = fb.group_key
            elif isinstance(fb.group_key, dict):
                ordered_key = "|".join(str(v) for v in fb.group_key.values())
                if ordered_key in known_keys:
                    gkey = ordered_key
                else:
                    sorted_key = "|".join(str(v) for _, v in sorted(fb.group_key.items()))
                    gkey = sorted_key if sorted_key in known_keys else ordered_key
            else:
                gkey = str(fb.group_key)
            mask = (df_level["__group_key__"] == gkey) & (df_level["feature"] == fb.feature)
            if not mask.any():
                continue
            df_level.loc[mask, "median"] = (
                df_level.loc[mask, "median"] + learning_rate * fb.adjustment
            )
        updated[level_name] = df_level
    return updated


def suggest_baseline_adjustments(
    anomalies_df: pd.DataFrame,
    baselines: dict[str, pd.DataFrame],
    levels: list[BaselineLevel],
    z_threshold: float = 3.0,
) -> list[dict]:
    """
    Propose baseline adjustments (do not apply automatically) based on repeated anomalies.
    """
    suggestions: list[dict] = []
    if anomalies_df.empty or not baselines:
        return suggestions

    for level in levels:
        stats = baselines.get(level.name)
        if stats is None or stats.empty:
            continue
        try:
            key_series = _group_key(anomalies_df, level)
        except ValueError:
            continue
        anomalies_df = anomalies_df.copy()
        anomalies_df["__group_key__"] = key_series

        for _, stat_row in stats.iterrows():
            gkey = stat_row["__group_key__"]
            feature = stat_row["feature"]
            baseline_median = stat_row["median"]
            mad = stat_row["mad"] or 1e-6

            mask = anomalies_df["__group_key__"] == gkey
            if not mask.any() or feature not in anomalies_df.columns:
                continue

            current_median = pd.to_numeric(
                anomalies_df.loc[mask, feature], errors="coerce"
            ).median()
            z = abs(current_median - baseline_median) / mad
            if z >= z_threshold:
                suggestions.append(
                    {
                        "level": level.name,
                        "group_key": gkey,
                        "feature": feature,
                        "baseline_median": baseline_median,
                        "observed_median": current_median,
                        "proposed_new_median": float(
                            baseline_median + 0.5 * (current_median - baseline_median)
                        ),
                        "rationale": f"Median drifted {z:.1f} MADs above baseline; consider lifting baseline.",
                    }
                )
        anomalies_df = anomalies_df.drop(columns=["__group_key__"])
    return suggestions


# =============================================================================
# Enhanced Baseline Statistics (Phase 2)
# =============================================================================


@dataclass
class EnhancedBaselineStats:
    """Extended statistics beyond median/MAD for a single metric."""

    median: float
    mad: float
    mean: float
    std: float
    sample_count: int
    percentiles: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_series(cls, series: pd.Series) -> EnhancedBaselineStats:
        """Compute enhanced statistics from a pandas Series."""
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) == 0:
            return cls(
                median=0.0,
                mad=1e-6,
                mean=0.0,
                std=0.0,
                sample_count=0,
                percentiles={},
            )

        median = float(series.median())
        mad = float((series - median).abs().median())
        if mad == 0:
            mad = 1e-6

        return cls(
            median=median,
            mad=mad,
            mean=float(series.mean()),
            std=float(series.std()),
            sample_count=len(series),
            percentiles={
                "p5": float(np.percentile(series, 5)),
                "p25": float(np.percentile(series, 25)),
                "p50": float(np.percentile(series, 50)),
                "p75": float(np.percentile(series, 75)),
                "p95": float(np.percentile(series, 95)),
                "p99": float(np.percentile(series, 99)),
            },
        )


@dataclass
class TemporalBaseline:
    """Time-segmented baselines for a metric."""

    metric_name: str
    hourly_stats: dict[int, EnhancedBaselineStats] = field(default_factory=dict)
    daily_stats: dict[int, EnhancedBaselineStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "hourly_stats": {str(h): stats.to_dict() for h, stats in self.hourly_stats.items()},
            "daily_stats": {str(d): stats.to_dict() for d, stats in self.daily_stats.items()},
        }


@dataclass
class AnomalyThresholds:
    """Threshold definitions for anomaly detection."""

    warning_lower: float
    warning_upper: float
    critical_lower: float
    critical_upper: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "warning": {"lower": self.warning_lower, "upper": self.warning_upper},
            "critical": {"lower": self.critical_lower, "upper": self.critical_upper},
        }

    @classmethod
    def from_percentiles(
        cls,
        percentiles: dict[str, float],
        warning_percentile: tuple[str, str] = ("p5", "p95"),
        critical_percentile: tuple[str, str] = ("p5", "p99"),
    ) -> AnomalyThresholds:
        """Compute thresholds from percentiles."""
        return cls(
            warning_lower=percentiles.get(warning_percentile[0], 0.0),
            warning_upper=percentiles.get(warning_percentile[1], 100.0),
            critical_lower=percentiles.get(critical_percentile[0], 0.0),
            critical_upper=percentiles.get(critical_percentile[1], 100.0),
        )


@dataclass
class DataDrivenBaseline:
    """Complete data-driven baseline for a single metric."""

    metric_name: str
    global_stats: EnhancedBaselineStats
    by_hour: dict[int, EnhancedBaselineStats] | None = None
    by_device_type: dict[str, EnhancedBaselineStats] | None = None
    thresholds: AnomalyThresholds | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "metric_name": self.metric_name,
            "global": self.global_stats.to_dict(),
        }
        if self.by_hour:
            result["by_hour"] = [
                {"hour": h, **stats.to_dict()} for h, stats in sorted(self.by_hour.items())
            ]
        if self.by_device_type:
            result["by_device_type"] = {
                dtype: stats.to_dict() for dtype, stats in self.by_device_type.items()
            }
        if self.thresholds:
            result["anomaly_thresholds"] = self.thresholds.to_dict()
        return result


def compute_enhanced_baselines(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    levels: list[BaselineLevel],
    include_percentiles: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Compute enhanced baselines with percentiles for each feature and level.

    Args:
        df: Input DataFrame with telemetry data
        feature_cols: Columns to compute baselines for
        levels: Baseline levels (global, store, customer, etc.)
        include_percentiles: Whether to compute percentile statistics

    Returns:
        Dictionary mapping level names to DataFrames with enhanced statistics
    """
    baselines: dict[str, pd.DataFrame] = {}

    for level in levels:
        try:
            key = _group_key(df, level)
        except ValueError:
            continue

        df_level = df.copy()
        df_level["__group_key__"] = key

        agg_rows = []
        for group_val, grp in df_level.groupby("__group_key__"):
            if len(grp) < level.min_rows:
                continue

            for col in feature_cols:
                if col not in grp.columns:
                    continue

                stats = EnhancedBaselineStats.from_series(grp[col])
                row = {
                    "__group_key__": group_val,
                    "feature": col,
                    "median": stats.median,
                    "mad": stats.mad,
                    "mean": stats.mean,
                    "std": stats.std,
                    "sample_count": stats.sample_count,
                }
                if include_percentiles:
                    row.update({f"percentile_{k}": v for k, v in stats.percentiles.items()})
                agg_rows.append(row)

        if agg_rows:
            baselines[level.name] = pd.DataFrame(agg_rows)

    return baselines


def compute_temporal_baselines(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    timestamp_col: str = "Timestamp",
    min_samples_per_segment: int = 10,
) -> dict[str, TemporalBaseline]:
    """
    Compute time-segmented baselines (hourly and daily patterns).

    Args:
        df: Input DataFrame with timestamp and feature columns
        feature_cols: Columns to compute baselines for
        timestamp_col: Name of the timestamp column
        min_samples_per_segment: Minimum samples required per time segment

    Returns:
        Dictionary mapping metric names to TemporalBaseline objects
    """
    if df.empty or timestamp_col not in df.columns:
        return {}

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["_hour"] = df[timestamp_col].dt.hour
    df["_dayofweek"] = df[timestamp_col].dt.dayofweek

    temporal_baselines: dict[str, TemporalBaseline] = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        temporal = TemporalBaseline(metric_name=col)

        # Hourly patterns (0-23)
        for hour in range(24):
            hour_data = df[df["_hour"] == hour][col]
            if len(hour_data) >= min_samples_per_segment:
                temporal.hourly_stats[hour] = EnhancedBaselineStats.from_series(hour_data)

        # Daily patterns (0=Monday, 6=Sunday)
        for day in range(7):
            day_data = df[df["_dayofweek"] == day][col]
            if len(day_data) >= min_samples_per_segment:
                temporal.daily_stats[day] = EnhancedBaselineStats.from_series(day_data)

        if temporal.hourly_stats or temporal.daily_stats:
            temporal_baselines[col] = temporal

    return temporal_baselines


def compute_data_driven_baselines(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    timestamp_col: str = "Timestamp",
    device_type_col: str | None = None,
    include_temporal: bool = True,
    min_samples: int = 25,
) -> dict[str, DataDrivenBaseline]:
    """
    Compute comprehensive data-driven baselines for each metric.

    This function generates the baseline format requested:
    {
        "metric_name": {
            "global": {"mean": x, "std": y, "percentiles": {...}},
            "by_hour": [...],
            "by_device_type": {...},
            "anomaly_thresholds": {...}
        }
    }

    Args:
        df: Input DataFrame with telemetry data
        feature_cols: Columns to compute baselines for
        timestamp_col: Name of the timestamp column
        device_type_col: Optional column for device type grouping
        include_temporal: Whether to include hourly/daily patterns
        min_samples: Minimum samples required for valid statistics

    Returns:
        Dictionary mapping metric names to DataDrivenBaseline objects
    """
    baselines: dict[str, DataDrivenBaseline] = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Global statistics
        global_stats = EnhancedBaselineStats.from_series(df[col])
        if global_stats.sample_count < min_samples:
            continue

        # Compute thresholds from percentiles
        thresholds = AnomalyThresholds.from_percentiles(global_stats.percentiles)

        # Device type specific baselines
        by_device_type: dict[str, EnhancedBaselineStats] | None = None
        if device_type_col and device_type_col in df.columns:
            by_device_type = {}
            for dtype, grp in df.groupby(device_type_col):
                if len(grp) >= min_samples:
                    by_device_type[str(dtype)] = EnhancedBaselineStats.from_series(grp[col])

        # Temporal baselines
        by_hour: dict[int, EnhancedBaselineStats] | None = None
        if include_temporal and timestamp_col in df.columns:
            temporal = compute_temporal_baselines(df, [col], timestamp_col, min_samples // 2)
            if col in temporal:
                by_hour = temporal[col].hourly_stats

        baselines[col] = DataDrivenBaseline(
            metric_name=col,
            global_stats=global_stats,
            by_hour=by_hour,
            by_device_type=by_device_type,
            thresholds=thresholds,
        )

    return baselines


def save_data_driven_baselines(
    baselines: dict[str, DataDrivenBaseline],
    path: Path,
) -> None:
    """Save data-driven baselines to JSON file in the requested format."""
    payload = {
        "schema_version": DATA_DRIVEN_SCHEMA_VERSION,
        "baseline_type": "data_driven",
        "generated_at": datetime.now(UTC).isoformat(),
        "baselines": {name: baseline.to_dict() for name, baseline in baselines.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    logger.info(f"Saved data-driven baselines to {path}")


def load_data_driven_baselines(path: Path) -> dict[str, dict[str, Any]]:
    """Load data-driven baselines from JSON file."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "baselines" in payload:
        return payload.get("baselines", {})
    return payload


def load_data_driven_baselines_payload(path: Path) -> dict[str, Any]:
    """Load the full data-driven baseline payload (including schema metadata)."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "baselines" in payload:
        return payload
    return {
        "schema_version": "data_driven_v0",
        "baseline_type": "data_driven",
        "baselines": payload,
    }


def compute_cohort_baselines_from_real_data(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    cohort_columns: list[str],
    min_devices_per_cohort: int = 25,
) -> dict[str, pd.DataFrame]:
    """
    Compute per-cohort baselines from real telemetry data.

    Cohorts can be defined by any combination of columns like:
    - ManufacturerId, ModelId, OsVersionId, FirmwareVersion

    Args:
        df: Input DataFrame with telemetry data
        feature_cols: Columns to compute baselines for
        cohort_columns: Columns defining the cohort grouping
        min_devices_per_cohort: Minimum devices required per cohort

    Returns:
        Dictionary with cohort baselines DataFrames
    """
    # Create a cohort level
    cohort_level = BaselineLevel(
        name="cohort",
        group_columns=cohort_columns,
        min_rows=min_devices_per_cohort,
    )

    return compute_enhanced_baselines(
        df=df,
        feature_cols=feature_cols,
        levels=[cohort_level],
        include_percentiles=True,
    )
