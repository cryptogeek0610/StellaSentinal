from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from device_anomaly.config.feature_config import FeatureConfig
from device_anomaly.models.model_registry import resolve_artifact_path, resolve_model_artifacts

COHORT_STATS_SCHEMA_VERSION = "cohort_stats_v2"
_REQUIRED_COHORT_COLUMNS = ("ManufacturerId", "ModelId", "OsVersionId")

# Extended cohort dimensions available for richer segmentation
EXTENDED_COHORT_COLUMNS = (
    "ManufacturerId",
    "ModelId",
    "OsVersionId",
    "FirmwareVersion",
    "Carrier",
    "CarrierCode",
    "LocationRegion",
    "BusinessUnit",
    "SiteId",
    "StoreId",
)


def build_cohort_id(df: pd.DataFrame, cohort_columns: list[str] | None = None) -> pd.Series | None:
    columns = cohort_columns or FeatureConfig.cohort_columns
    if not all(col in df.columns for col in _REQUIRED_COHORT_COLUMNS):
        return None

    parts = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if col == "FirmwareVersion":
            series = series.fillna("na")
        parts.append(series.astype(str))

    if not parts:
        return None

    cohort_id = parts[0]
    for part in parts[1:]:
        cohort_id = cohort_id + "_" + part
    return cohort_id


def select_cohort_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude_cols = FeatureConfig.excluded_columns | {"cohort_id"}
    return [
        col
        for col in df.columns
        if col not in exclude_cols
        and np.issubdtype(df[col].dtype, np.number)
        and not col.endswith("_cohort_z")
    ]


def compute_cohort_stats(
    df: pd.DataFrame,
    cohort_columns: list[str] | None = None,
    feature_cols: list[str] | None = None,
    min_samples: int = 25,
    min_mad: float = 1e-6,
) -> dict[str, Any]:
    feature_cols = list(feature_cols or select_cohort_feature_cols(df))
    payload = {
        "schema_version": COHORT_STATS_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort_columns": cohort_columns or FeatureConfig.cohort_columns,
        "feature_columns": feature_cols,
        "min_samples": min_samples,
        "min_mad": min_mad,
        "stats": {},
        "global": {},
    }

    if df.empty:
        return payload

    cohort_id = build_cohort_id(df, cohort_columns)
    if cohort_id is None:
        return payload

    df_local = df.copy()
    df_local["cohort_id"] = cohort_id.astype(str)

    stats: dict[str, dict[str, dict[str, float]]] = {}
    for cohort, grp in df_local.groupby("cohort_id"):
        cohort_stats: dict[str, dict[str, float]] = {}
        for col in feature_cols:
            if col not in grp.columns:
                continue
            series = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(series) < min_samples:
                continue
            median = float(series.median())
            mad = float((series - median).abs().median())
            if mad < min_mad:
                continue
            cohort_stats[col] = {"median": median, "mad": mad, "count": int(len(series))}
        if cohort_stats:
            stats[str(cohort)] = cohort_stats

    global_stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        series = pd.to_numeric(df_local[col], errors="coerce").dropna()
        if len(series) < min_samples:
            continue
        median = float(series.median())
        mad = float((series - median).abs().median())
        if mad < min_mad:
            continue
        global_stats[col] = {"median": median, "mad": mad, "count": int(len(series))}

    payload["stats"] = stats
    payload["global"] = global_stats
    return payload


def save_cohort_stats(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))


def load_cohort_stats_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@dataclass
class CohortStatsStore:
    payload: dict[str, Any]
    _by_feature: dict[str, dict[str, dict[str, float]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        stats = self.payload.get("stats") or {}
        for cohort_id, metrics in stats.items():
            if not isinstance(metrics, dict):
                continue
            for metric, values in metrics.items():
                if not isinstance(values, dict):
                    continue
                self._by_feature.setdefault(metric, {})[cohort_id] = values

    @property
    def cohort_columns(self) -> list[str]:
        return list(self.payload.get("cohort_columns") or FeatureConfig.cohort_columns)

    @property
    def feature_columns(self) -> list[str]:
        return list(self.payload.get("feature_columns") or self._by_feature.keys())

    @property
    def min_samples(self) -> int:
        return int(self.payload.get("min_samples", 1))

    @property
    def min_mad(self) -> float:
        return float(self.payload.get("min_mad", 1e-6))

    @property
    def global_stats(self) -> dict[str, dict[str, float]]:
        return self.payload.get("global") or {}

    def get_feature_maps(self, feature: str) -> tuple[dict[str, float], dict[str, float]]:
        stats = self._by_feature.get(feature, {})
        medians = {cohort: float(values.get("median", 0.0)) for cohort, values in stats.items()}
        mads = {cohort: float(values.get("mad", 0.0)) for cohort, values in stats.items()}
        return medians, mads

    def get_z_score(self, cohort_id: str, metric: str, value: float | None) -> float | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None

        stats = (self.payload.get("stats") or {}).get(cohort_id, {}).get(metric)
        if not stats:
            stats = (self.payload.get("global") or {}).get(metric)
        if not stats:
            return None

        count = stats.get("count", self.min_samples)
        mad = float(stats.get("mad", 0.0))
        if count < self.min_samples or mad < self.min_mad:
            return None

        median = float(stats.get("median", 0.0))
        return (float(value) - median) / (mad + 1e-6)

    def get_cohort_stats(self, cohort_id: str) -> dict[str, Any] | None:
        stats = (self.payload.get("stats") or {}).get(cohort_id)
        if not stats:
            return None
        return {
            "cohort_id": cohort_id,
            "metrics": stats,
            "last_update": self.payload.get("generated_at"),
        }

    def get_all_cohort_ids(self) -> list[str]:
        return list((self.payload.get("stats") or {}).keys())


def apply_cohort_stats(df: pd.DataFrame, store: CohortStatsStore | None) -> pd.DataFrame:
    if store is None or df.empty:
        return df

    df_local = df.copy()
    if "cohort_id" not in df_local.columns:
        cohort_id = build_cohort_id(df_local, store.cohort_columns)
        if cohort_id is None:
            return df
        df_local["cohort_id"] = cohort_id.astype(str)

    feature_cols = [c for c in store.feature_columns if c in df_local.columns]
    global_stats = store.global_stats

    for col in feature_cols:
        medians, mads = store.get_feature_maps(col)
        if not medians and col not in global_stats:
            continue

        cohort_series = df_local["cohort_id"].map(medians)
        mad_series = df_local["cohort_id"].map(mads)

        global_stat = global_stats.get(col)
        if global_stat:
            if "median" in global_stat:
                cohort_series = cohort_series.fillna(float(global_stat["median"]))
            if "mad" in global_stat:
                mad_series = mad_series.fillna(float(global_stat["mad"]))

        value_series = pd.to_numeric(df_local[col], errors="coerce")
        df_local[f"{col}_cohort_z"] = (value_series - cohort_series) / (mad_series + 1e-6)

    return df_local


def load_latest_cohort_stats(models_dir: Path | None = None) -> CohortStatsStore | None:
    artifacts = resolve_model_artifacts(models_dir)
    metadata = artifacts.metadata or {}
    stats_path = resolve_artifact_path(
        artifacts.model_dir,
        (metadata.get("artifacts") or {}).get("cohort_stats_path"),
    )
    if stats_path is None:
        candidate = artifacts.model_dir / "cohort_stats.json"
        stats_path = candidate if candidate.exists() else None

    if stats_path is None or not stats_path.exists():
        return None

    payload = load_cohort_stats_payload(stats_path)
    if not payload:
        return None
    return CohortStatsStore(payload)


# =============================================================================
# EXTENDED COHORT FUNCTIONS
# =============================================================================

def get_available_cohort_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of cohort columns that are available in the dataframe.

    Checks both required and extended cohort columns.
    """
    available = []
    for col in EXTENDED_COHORT_COLUMNS:
        if col in df.columns:
            available.append(col)
    return available


def build_extended_cohort_id(
    df: pd.DataFrame,
    include_carrier: bool = True,
    include_location: bool = True,
    include_business_unit: bool = True,
) -> pd.Series | None:
    """
    Build cohort ID with extended dimensions.

    Args:
        df: DataFrame with cohort columns
        include_carrier: Include Carrier/CarrierCode in cohort ID
        include_location: Include LocationRegion/SiteId in cohort ID
        include_business_unit: Include BusinessUnit in cohort ID

    Returns:
        Series of cohort IDs, or None if required columns missing
    """
    # Start with required columns
    if not all(col in df.columns for col in _REQUIRED_COHORT_COLUMNS):
        return None

    columns_to_use = list(_REQUIRED_COHORT_COLUMNS)

    # Add FirmwareVersion if available
    if "FirmwareVersion" in df.columns:
        columns_to_use.append("FirmwareVersion")

    # Add carrier if requested and available
    if include_carrier:
        if "CarrierCode" in df.columns:
            columns_to_use.append("CarrierCode")
        elif "Carrier" in df.columns:
            columns_to_use.append("Carrier")

    # Add location region if requested and available
    if include_location:
        if "LocationRegion" in df.columns:
            columns_to_use.append("LocationRegion")
        elif "SiteId" in df.columns:
            columns_to_use.append("SiteId")

    # Add business unit if requested and available
    if include_business_unit:
        if "BusinessUnit" in df.columns:
            columns_to_use.append("BusinessUnit")
        elif "StoreId" in df.columns:
            columns_to_use.append("StoreId")

    # Build cohort ID from available columns
    parts = []
    for col in columns_to_use:
        if col not in df.columns:
            continue
        series = df[col]
        if col == "FirmwareVersion":
            series = series.fillna("na")
        parts.append(series.astype(str))

    if not parts:
        return None

    cohort_id = parts[0]
    for part in parts[1:]:
        cohort_id = cohort_id + "_" + part
    return cohort_id


def compute_extended_cohort_stats(
    df: pd.DataFrame,
    include_carrier: bool = True,
    include_location: bool = True,
    include_business_unit: bool = True,
    feature_cols: list[str] | None = None,
    min_samples: int = 25,
    min_mad: float = 1e-6,
) -> dict[str, Any]:
    """
    Compute cohort statistics with extended dimensions.

    This enables richer device segmentation including:
    - Network operator (Carrier)
    - Geographic region (LocationRegion, SiteId)
    - Organizational unit (BusinessUnit, StoreId)

    Args:
        df: DataFrame with features and cohort columns
        include_carrier: Include carrier in cohort segmentation
        include_location: Include location region in cohort segmentation
        include_business_unit: Include business unit in cohort segmentation
        feature_cols: Features to compute stats for (auto-detected if None)
        min_samples: Minimum samples for a cohort to be included
        min_mad: Minimum MAD to avoid division by zero

    Returns:
        Cohort stats payload with extended dimensions
    """
    feature_cols = list(feature_cols or select_cohort_feature_cols(df))

    # Determine which cohort columns are available
    available_cohort_cols = get_available_cohort_columns(df)
    used_cohort_cols = list(_REQUIRED_COHORT_COLUMNS)

    if include_carrier:
        if "CarrierCode" in available_cohort_cols:
            used_cohort_cols.append("CarrierCode")
        elif "Carrier" in available_cohort_cols:
            used_cohort_cols.append("Carrier")

    if include_location:
        if "LocationRegion" in available_cohort_cols:
            used_cohort_cols.append("LocationRegion")
        elif "SiteId" in available_cohort_cols:
            used_cohort_cols.append("SiteId")

    if include_business_unit:
        if "BusinessUnit" in available_cohort_cols:
            used_cohort_cols.append("BusinessUnit")
        elif "StoreId" in available_cohort_cols:
            used_cohort_cols.append("StoreId")

    # Filter to columns that exist
    used_cohort_cols = [c for c in used_cohort_cols if c in df.columns]

    payload = {
        "schema_version": COHORT_STATS_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort_columns": used_cohort_cols,
        "available_columns": available_cohort_cols,
        "feature_columns": feature_cols,
        "min_samples": min_samples,
        "min_mad": min_mad,
        "stats": {},
        "global": {},
        "cohort_summary": {},
    }

    if df.empty:
        return payload

    # Build extended cohort ID
    cohort_id = build_extended_cohort_id(
        df,
        include_carrier=include_carrier,
        include_location=include_location,
        include_business_unit=include_business_unit,
    )

    if cohort_id is None:
        # Fall back to basic cohort ID
        cohort_id = build_cohort_id(df)
        if cohort_id is None:
            return payload

    df_local = df.copy()
    df_local["cohort_id"] = cohort_id.astype(str)

    # Compute per-cohort statistics
    stats: dict[str, dict[str, dict[str, float]]] = {}
    cohort_counts: dict[str, int] = {}

    for cohort, grp in df_local.groupby("cohort_id"):
        cohort_counts[str(cohort)] = len(grp)
        cohort_stats: dict[str, dict[str, float]] = {}

        for col in feature_cols:
            if col not in grp.columns:
                continue
            series = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(series) < min_samples:
                continue
            median = float(series.median())
            mad = float((series - median).abs().median())
            if mad < min_mad:
                continue
            cohort_stats[col] = {
                "median": median,
                "mad": mad,
                "count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "p25": float(series.quantile(0.25)),
                "p75": float(series.quantile(0.75)),
            }

        if cohort_stats:
            stats[str(cohort)] = cohort_stats

    # Compute global statistics
    global_stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        series = pd.to_numeric(df_local[col], errors="coerce").dropna()
        if len(series) < min_samples:
            continue
        median = float(series.median())
        mad = float((series - median).abs().median())
        if mad < min_mad:
            continue
        global_stats[col] = {
            "median": median,
            "mad": mad,
            "count": int(len(series)),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
        }

    # Cohort summary
    payload["stats"] = stats
    payload["global"] = global_stats
    payload["cohort_summary"] = {
        "total_cohorts": len(stats),
        "cohort_sizes": cohort_counts,
        "min_cohort_size": min(cohort_counts.values()) if cohort_counts else 0,
        "max_cohort_size": max(cohort_counts.values()) if cohort_counts else 0,
        "median_cohort_size": float(np.median(list(cohort_counts.values()))) if cohort_counts else 0,
    }

    return payload


def compute_cohort_fairness(
    df: pd.DataFrame,
    anomaly_col: str = "anomaly_label",
    cohort_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute fairness metrics across cohorts.

    Measures if anomaly detection rate is consistent across device segments.
    Large disparities may indicate bias in the model.

    Args:
        df: DataFrame with anomaly labels and cohort columns
        anomaly_col: Column containing anomaly labels (-1 = anomaly)
        cohort_columns: Cohort columns to use

    Returns:
        Fairness metrics including parity scores
    """
    if anomaly_col not in df.columns:
        return {"error": f"Column {anomaly_col} not found"}

    cohort_id = build_extended_cohort_id(df) if cohort_columns is None else build_cohort_id(df, cohort_columns)
    if cohort_id is None:
        return {"error": "Could not build cohort ID"}

    df_local = df.copy()
    df_local["cohort_id"] = cohort_id.astype(str)

    # Compute anomaly rate per cohort
    cohort_rates = {}
    for cohort, grp in df_local.groupby("cohort_id"):
        if len(grp) < 10:
            continue
        anomaly_rate = (grp[anomaly_col] == -1).mean()
        cohort_rates[str(cohort)] = {
            "anomaly_rate": float(anomaly_rate),
            "sample_count": len(grp),
        }

    if not cohort_rates:
        return {"error": "No cohorts with sufficient samples"}

    # Global anomaly rate
    global_rate = (df_local[anomaly_col] == -1).mean()

    # Compute disparity metrics
    rates = [v["anomaly_rate"] for v in cohort_rates.values()]
    max_rate = max(rates)
    min_rate = min(rates)

    return {
        "global_anomaly_rate": float(global_rate),
        "cohort_rates": cohort_rates,
        "disparity_ratio": float(max_rate / min_rate) if min_rate > 0 else float("inf"),
        "disparity_diff": float(max_rate - min_rate),
        "rate_std": float(np.std(rates)),
        "cohort_count": len(cohort_rates),
    }


# =============================================================================
# FLEET-LEVEL COMPARISON FUNCTIONS
# =============================================================================


def compute_cohort_fleet_comparison(
    df: pd.DataFrame,
    cohort_columns: list[str] | None = None,
    feature_cols: list[str] | None = None,
    min_cohort_size: int = 20,
    z_threshold: float = 2.0,
) -> dict[str, Any]:
    """
    Compare each cohort's aggregate metrics to fleet-wide baselines.

    This function identifies cohorts that are systematically deviating
    from fleet norms, enabling detection of systemic issues like:
    - "All Samsung S21 + Android 13 devices have 2.3x higher crash rate"
    - "Devices at Warehouse B have 3x more network disconnects"

    Args:
        df: DataFrame with device features and cohort columns
        cohort_columns: Columns to use for cohort segmentation (auto-detected if None)
        feature_cols: Features to analyze (auto-detected if None)
        min_cohort_size: Minimum devices for a cohort to be analyzed
        z_threshold: Z-score threshold for flagging systemic issues

    Returns:
        Dictionary containing:
        - cohort_deviations: Per-cohort Z-scores for each metric
        - systemic_issues: Cohorts with significant deviations (|Z| > threshold)
        - fleet_baselines: Fleet-wide statistics for reference
        - summary: High-level summary statistics
    """
    feature_cols = list(feature_cols or select_cohort_feature_cols(df))
    cohort_columns = cohort_columns or list(_REQUIRED_COHORT_COLUMNS)

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort_columns": cohort_columns,
        "feature_columns": feature_cols,
        "min_cohort_size": min_cohort_size,
        "z_threshold": z_threshold,
        "fleet_baselines": {},
        "cohort_deviations": {},
        "systemic_issues": [],
        "summary": {},
    }

    if df.empty:
        return result

    # Build cohort ID
    cohort_id = build_cohort_id(df, cohort_columns)
    if cohort_id is None:
        # Try extended cohort ID
        cohort_id = build_extended_cohort_id(df)
        if cohort_id is None:
            return result

    df_local = df.copy()
    df_local["cohort_id"] = cohort_id.astype(str)

    # Compute fleet-wide baselines
    fleet_baselines: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        if col not in df_local.columns:
            continue
        series = pd.to_numeric(df_local[col], errors="coerce").dropna()
        if len(series) < min_cohort_size:
            continue

        fleet_baselines[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()) if series.std() > 0 else 1e-6,
            "median": float(series.median()),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
            "count": int(len(series)),
        }

    result["fleet_baselines"] = fleet_baselines

    # Compute per-cohort deviations from fleet
    cohort_deviations: dict[str, dict[str, Any]] = {}
    systemic_issues: list[dict[str, Any]] = []

    for cohort, grp in df_local.groupby("cohort_id"):
        cohort_str = str(cohort)
        if len(grp) < min_cohort_size:
            continue

        cohort_stats: dict[str, Any] = {
            "device_count": len(grp),
            "fleet_percentage": round(100 * len(grp) / len(df_local), 2),
            "metrics": {},
        }

        for col in feature_cols:
            if col not in grp.columns or col not in fleet_baselines:
                continue

            series = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(series) < 5:
                continue

            cohort_mean = float(series.mean())
            fleet_mean = fleet_baselines[col]["mean"]
            fleet_std = fleet_baselines[col]["std"]

            # Compute Z-score (cohort mean vs fleet mean)
            z_score = (cohort_mean - fleet_mean) / fleet_std if fleet_std > 0 else 0

            # Compute multiplier
            multiplier = cohort_mean / fleet_mean if fleet_mean != 0 else 1.0

            cohort_stats["metrics"][col] = {
                "cohort_mean": round(cohort_mean, 2),
                "fleet_mean": round(fleet_mean, 2),
                "z_score": round(z_score, 2),
                "vs_fleet_multiplier": round(multiplier, 2),
                "is_anomalous": abs(z_score) > z_threshold,
            }

            # Track systemic issues
            if abs(z_score) > z_threshold:
                direction = "elevated" if z_score > 0 else "reduced"
                systemic_issues.append({
                    "cohort_id": cohort_str,
                    "device_count": len(grp),
                    "fleet_percentage": cohort_stats["fleet_percentage"],
                    "metric": col,
                    "cohort_mean": round(cohort_mean, 2),
                    "fleet_mean": round(fleet_mean, 2),
                    "z_score": round(z_score, 2),
                    "vs_fleet_multiplier": round(multiplier, 2),
                    "direction": direction,
                    "severity": _compute_issue_severity(z_score),
                })

        if cohort_stats["metrics"]:
            cohort_deviations[cohort_str] = cohort_stats

    result["cohort_deviations"] = cohort_deviations
    result["systemic_issues"] = sorted(
        systemic_issues,
        key=lambda x: abs(x["z_score"]),
        reverse=True,
    )

    # Summary statistics
    result["summary"] = {
        "total_cohorts": len(cohort_deviations),
        "cohorts_with_issues": len({i["cohort_id"] for i in systemic_issues}),
        "total_issues": len(systemic_issues),
        "critical_issues": len([i for i in systemic_issues if i["severity"] == "critical"]),
        "high_issues": len([i for i in systemic_issues if i["severity"] == "high"]),
        "medium_issues": len([i for i in systemic_issues if i["severity"] == "medium"]),
    }

    return result


def _compute_issue_severity(z_score: float) -> str:
    """Compute severity based on Z-score magnitude."""
    abs_z = abs(z_score)
    if abs_z >= 3.0:
        return "critical"
    elif abs_z >= 2.5:
        return "high"
    elif abs_z >= 2.0:
        return "medium"
    return "low"


def get_cohort_systemic_issues(
    df: pd.DataFrame,
    min_cohort_size: int = 20,
    min_z: float = 2.0,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """
    Get top systemic issues affecting device cohorts.

    Convenience function that returns a simplified list of systemic issues
    suitable for API responses.

    Args:
        df: DataFrame with device features
        min_cohort_size: Minimum devices per cohort
        min_z: Minimum Z-score to flag
        top_n: Maximum issues to return

    Returns:
        List of systemic issues sorted by severity/impact
    """
    comparison = compute_cohort_fleet_comparison(
        df,
        min_cohort_size=min_cohort_size,
        z_threshold=min_z,
    )

    issues = comparison.get("systemic_issues", [])
    return issues[:top_n]


def build_cohort_name_from_id(
    cohort_id: str,
    df: pd.DataFrame | None = None,
) -> str:
    """
    Build human-readable cohort name from cohort ID.

    Args:
        cohort_id: Underscore-separated cohort ID (e.g., "Samsung_SM-G991B_13")
        df: Optional DataFrame to look up human-readable names

    Returns:
        Human-readable cohort name
    """
    parts = cohort_id.split("_")
    if len(parts) >= 3:
        manufacturer = parts[0]
        model = parts[1]
        os_version = parts[2]
        return f"{manufacturer} {model} (Android {os_version})"
    elif len(parts) == 2:
        return f"{parts[0]} {parts[1]}"
    return cohort_id
