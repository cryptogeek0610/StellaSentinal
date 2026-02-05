"""API routes for data discovery and profiling endpoints."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from device_anomaly.api.dependencies import get_mock_mode, require_role
from device_anomaly.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-discovery", tags=["data-discovery"])


# ============================================================================
# Pydantic Response Models
# ============================================================================


class ColumnStatsResponse(BaseModel):
    """Statistical summary for a single column."""

    column_name: str
    dtype: str
    null_count: int = 0
    null_percent: float = 0.0
    unique_count: int = 0
    min_val: float | None = None
    max_val: float | None = None
    mean: float | None = None
    std: float | None = None
    percentiles: dict[str, float] = Field(default_factory=dict)


class TableProfileResponse(BaseModel):
    """Complete statistical profile of a database table."""

    table_name: str
    row_count: int
    date_range: tuple[str | None, str | None]
    device_count: int
    column_stats: dict[str, ColumnStatsResponse] = Field(default_factory=dict)
    profiled_at: str | None = None
    source_db: str | None = None  # "xsight" or "mobicontrol"


class TemporalPatternResponse(BaseModel):
    """Time-based pattern for a metric."""

    metric_name: str
    hourly_stats: dict[int, dict[str, float]] = Field(default_factory=dict)
    daily_stats: dict[int, dict[str, float]] = Field(default_factory=dict)


class MetricDistributionResponse(BaseModel):
    """Histogram distribution for a metric."""

    bins: list[float]
    counts: list[int]
    stats: dict[str, Any]


class AvailableMetricResponse(BaseModel):
    """Information about an available metric."""

    table: str
    column: str
    dtype: str
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    category: str | None = None  # 'raw', 'rolling', 'derived', 'temporal', 'cohort', 'volatility'
    domain: str | None = None  # 'battery', 'rf', 'throughput', 'usage', 'storage', etc.
    description: str | None = None  # Human-readable description


class DiscoverySummaryResponse(BaseModel):
    """Summary of data discovery results."""

    total_tables_profiled: int
    total_rows: int
    total_devices: int
    metrics_discovered: int
    patterns_analyzed: int
    date_range: dict[str, str] | None = None
    discovery_completed: str | None = None


class DataDiscoveryStatusResponse(BaseModel):
    """Status of a data discovery job."""

    status: str  # 'idle', 'running', 'completed', 'failed'
    progress: float = 0.0  # 0-100
    message: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    results_available: bool = False


# ============================================================================
# In-memory storage for discovery results (in production, use Redis or DB)
# ============================================================================

_discovery_cache: dict[str, Any] = {
    "status": "idle",
    "progress": 0.0,
    "message": None,
    "started_at": None,
    "completed_at": None,
    "results": None,
}


# ============================================================================
# Mock Data for Development
# ============================================================================


def get_mock_table_profiles() -> list[TableProfileResponse]:
    """Generate mock table profiles for development."""
    return [
        TableProfileResponse(
            table_name="cs_BatteryStat",
            row_count=1_250_000,
            date_range=("2024-01-01", "2024-12-28"),
            device_count=4_500,
            column_stats={
                "TotalBatteryLevelDrop": ColumnStatsResponse(
                    column_name="TotalBatteryLevelDrop",
                    dtype="int",
                    null_percent=0.5,
                    min_val=0,
                    max_val=100,
                    mean=35.2,
                    std=18.5,
                    percentiles={"p5": 5, "p25": 20, "p50": 32, "p75": 48, "p95": 72, "p99": 88},
                ),
                "TotalDischargeTime_Sec": ColumnStatsResponse(
                    column_name="TotalDischargeTime_Sec",
                    dtype="int",
                    null_percent=0.2,
                    min_val=0,
                    max_val=86400,
                    mean=28800,
                    std=12000,
                    percentiles={
                        "p5": 3600,
                        "p25": 18000,
                        "p50": 28800,
                        "p75": 39600,
                        "p95": 57600,
                        "p99": 72000,
                    },
                ),
            },
            profiled_at=datetime.now(UTC).isoformat(),
        ),
        TableProfileResponse(
            table_name="cs_AppUsage",
            row_count=3_500_000,
            date_range=("2024-01-01", "2024-12-28"),
            device_count=4_500,
            column_stats={
                "VisitCount": ColumnStatsResponse(
                    column_name="VisitCount",
                    dtype="int",
                    null_percent=0.1,
                    min_val=0,
                    max_val=1000,
                    mean=45,
                    std=32,
                    percentiles={"p5": 2, "p25": 18, "p50": 38, "p75": 65, "p95": 120, "p99": 200},
                ),
            },
            profiled_at=datetime.now(UTC).isoformat(),
        ),
        TableProfileResponse(
            table_name="cs_DataUsage",
            row_count=2_800_000,
            date_range=("2024-01-01", "2024-12-28"),
            device_count=4_500,
            column_stats={
                "Download": ColumnStatsResponse(
                    column_name="Download",
                    dtype="bigint",
                    null_percent=0.1,
                    min_val=0,
                    max_val=10_000_000_000,
                    mean=150_000_000,
                    std=250_000_000,
                    percentiles={
                        "p5": 1000,
                        "p25": 10_000_000,
                        "p50": 80_000_000,
                        "p75": 200_000_000,
                        "p95": 600_000_000,
                        "p99": 1_500_000_000,
                    },
                ),
            },
            profiled_at=datetime.now(UTC).isoformat(),
        ),
        TableProfileResponse(
            table_name="cs_Heatmap",
            row_count=5_200_000,
            date_range=("2024-01-01", "2024-12-28"),
            device_count=4_500,
            column_stats={
                "SignalStrength": ColumnStatsResponse(
                    column_name="SignalStrength",
                    dtype="int",
                    null_percent=2.5,
                    min_val=-120,
                    max_val=-40,
                    mean=-72,
                    std=15,
                    percentiles={
                        "p5": -98,
                        "p25": -82,
                        "p50": -72,
                        "p75": -62,
                        "p95": -50,
                        "p99": -44,
                    },
                ),
            },
            profiled_at=datetime.now(UTC).isoformat(),
        ),
    ]


def get_mock_available_metrics() -> list[AvailableMetricResponse]:
    """Generate mock available metrics including raw and engineered features."""
    metrics = [
        # Raw database metrics
        AvailableMetricResponse(
            table="cs_BatteryStat",
            column="TotalBatteryLevelDrop",
            dtype="int",
            mean=35.2,
            std=18.5,
            min=0,
            max=100,
            category="raw",
            domain="battery",
        ),
        AvailableMetricResponse(
            table="cs_BatteryStat",
            column="TotalDischargeTime_Sec",
            dtype="int",
            mean=28800,
            std=12000,
            min=0,
            max=86400,
            category="raw",
            domain="battery",
        ),
        AvailableMetricResponse(
            table="cs_BatteryStat",
            column="TotalFreeStorageKb",
            dtype="bigint",
            mean=2_000_000,
            std=1_500_000,
            min=0,
            max=16_000_000,
            category="raw",
            domain="storage",
        ),
        AvailableMetricResponse(
            table="cs_AppUsage",
            column="VisitCount",
            dtype="int",
            mean=45,
            std=32,
            min=0,
            max=1000,
            category="raw",
            domain="usage",
        ),
        AvailableMetricResponse(
            table="cs_AppUsage",
            column="TotalForegroundTime",
            dtype="int",
            mean=3600,
            std=2400,
            min=0,
            max=28800,
            category="raw",
            domain="usage",
        ),
        AvailableMetricResponse(
            table="cs_DataUsage",
            column="Download",
            dtype="bigint",
            mean=150_000_000,
            std=250_000_000,
            min=0,
            max=10_000_000_000,
            category="raw",
            domain="throughput",
        ),
        AvailableMetricResponse(
            table="cs_DataUsage",
            column="Upload",
            dtype="bigint",
            mean=25_000_000,
            std=50_000_000,
            min=0,
            max=2_000_000_000,
            category="raw",
            domain="throughput",
        ),
        AvailableMetricResponse(
            table="cs_Heatmap",
            column="SignalStrength",
            dtype="int",
            mean=-72,
            std=15,
            min=-120,
            max=-40,
            category="raw",
            domain="rf",
        ),
        AvailableMetricResponse(
            table="cs_Heatmap",
            column="DropCnt",
            dtype="int",
            mean=2.5,
            std=5.2,
            min=0,
            max=100,
            category="raw",
            domain="rf",
        ),
    ]

    # Add some engineered features as examples
    metrics.extend(
        [
            AvailableMetricResponse(
                table="feature_engineered",
                column="TotalBatteryLevelDrop_roll_mean",
                dtype="float",
                category="rolling",
                domain="battery",
                description="Mean of TotalBatteryLevelDrop over 14 days",
            ),
            AvailableMetricResponse(
                table="feature_engineered",
                column="BatteryDrainPerHour",
                dtype="float",
                category="derived",
                domain="battery",
                description="Derived: TotalBatteryLevelDrop / (TotalDischargeTime_Sec / 3600 + 1)",
            ),
            AvailableMetricResponse(
                table="feature_engineered",
                column="Download_delta",
                dtype="float",
                category="delta",
                domain="throughput",
                description="Day-over-day change for Download",
            ),
            AvailableMetricResponse(
                table="feature_engineered",
                column="hour_of_day",
                dtype="int",
                category="temporal",
                domain="temporal",
                description="Temporal feature: hour of day",
            ),
        ]
    )

    return metrics


def get_mock_temporal_patterns() -> list[TemporalPatternResponse]:
    """Generate mock temporal patterns."""
    import random

    patterns = []
    for metric in ["TotalBatteryLevelDrop", "AppVisitCount", "Download"]:
        hourly = {}
        for hour in range(24):
            # Simulate higher activity during business hours
            base = 50 if 8 <= hour <= 18 else 30
            hourly[hour] = {
                "mean": base + random.uniform(-10, 10),
                "std": random.uniform(5, 15),
                "count": random.randint(1000, 5000),
            }

        daily = {}
        for day in range(7):
            # Lower activity on weekends
            base = 40 if day < 5 else 25
            daily[day] = {
                "mean": base + random.uniform(-5, 5),
                "std": random.uniform(8, 12),
                "count": random.randint(5000, 20000),
            }

        patterns.append(
            TemporalPatternResponse(
                metric_name=metric,
                hourly_stats=hourly,
                daily_stats=daily,
            )
        )

    return patterns


def get_mock_metric_distribution(metric_name: str) -> MetricDistributionResponse:
    """Generate mock histogram for a metric."""
    import numpy as np

    # Generate mock normal-ish distribution
    np.random.seed(hash(metric_name) % 2**32)
    data = np.random.normal(50, 15, 10000)
    data = np.clip(data, 0, 100)

    counts, bin_edges = np.histogram(data, bins=30)

    return MetricDistributionResponse(
        bins=[float(b) for b in bin_edges],
        counts=[int(c) for c in counts],
        stats={
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "median": float(np.median(data)),
            "total_samples": len(data),
        },
    )


# ============================================================================
# API Endpoints
# ============================================================================


class AvailableTableResponse(BaseModel):
    """Information about an available table in the database."""

    table_name: str
    exists: bool = True
    source_db: str | None = None  # "xsight" or "mobicontrol"
    row_count: int | None = None  # Approximate row count if available


@router.get("/available-tables", response_model=list[AvailableTableResponse])
def get_available_tables(
    pattern: str = Query(default="cs_%", description="SQL LIKE pattern to filter tables"),
    exclude_time_slices: bool = Query(
        default=True, description="Exclude time-sliced views like _Last7, _LastMonth"
    ),
    include_mc: bool = Query(default=True, description="Include MobiControl database tables"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Discover available tables in both XSight and MobiControl databases.

    This is a fast endpoint that lists table names from both databases.
    Use /tables for full statistical profiles.
    """
    if mock_mode:
        # Return known table names in mock mode
        return [
            AvailableTableResponse(table_name="cs_BatteryStat", source_db="xsight"),
            AvailableTableResponse(table_name="cs_AppUsage", source_db="xsight"),
            AvailableTableResponse(table_name="cs_DataUsage", source_db="xsight"),
            AvailableTableResponse(table_name="cs_BatteryAppDrain", source_db="xsight"),
            AvailableTableResponse(table_name="cs_Heatmap", source_db="xsight"),
            AvailableTableResponse(table_name="DeviceStatInt", source_db="mobicontrol"),
            AvailableTableResponse(table_name="DeviceStatLocation", source_db="mobicontrol"),
            AvailableTableResponse(table_name="MainLog", source_db="mobicontrol"),
        ]

    results = []

    # Discover XSight tables
    try:
        from device_anomaly.data_access.data_profiler import discover_dw_tables

        tables = discover_dw_tables(pattern=pattern, exclude_time_slices=exclude_time_slices)
        results.extend(
            [AvailableTableResponse(table_name=name, source_db="xsight") for name in tables]
        )
    except Exception as e:
        logger.warning(f"Failed to discover XSight tables: {e}")

    # Discover MobiControl tables
    if include_mc:
        try:
            from device_anomaly.data_access.schema_discovery import (
                discover_mobicontrol_schema,
            )

            mc_schema = discover_mobicontrol_schema(use_cache=True)
            for table_name, table_info in mc_schema.tables.items():
                results.append(
                    AvailableTableResponse(
                        table_name=table_name,
                        source_db="mobicontrol",
                        row_count=table_info.row_count,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to discover MobiControl tables: {e}")

    if not results:
        raise HTTPException(status_code=503, detail="Failed to connect to any database")

    return results


@router.get("/tables", response_model=list[TableProfileResponse])
def get_table_profiles(
    include_mc: bool = Query(default=True, description="Include MobiControl database tables"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get statistical profiles for all telemetry tables.

    Returns row counts, date ranges, device counts, and column statistics
    for each profiled table in both XSight and MobiControl databases.
    """
    if mock_mode:
        return get_mock_table_profiles()

    results = []

    # Check if we have cached results first
    if _discovery_cache.get("results"):
        cache = _discovery_cache["results"]

        # Add XSight profiles from cache
        if cache.get("dw_profiles"):
            for name, p in cache["dw_profiles"].items():
                results.append(
                    TableProfileResponse(
                        table_name=name,
                        row_count=p.get("row_count", 0),
                        date_range=tuple(p.get("date_range", (None, None))),
                        device_count=p.get("device_count", 0),
                        column_stats={
                            col_name: ColumnStatsResponse(**col_data)
                            for col_name, col_data in p.get("column_stats", {}).items()
                        },
                        profiled_at=p.get("profiled_at"),
                        source_db="xsight",
                    )
                )

        # Add MobiControl profiles from cache
        if include_mc and cache.get("mc_profiles"):
            for name, p in cache["mc_profiles"].items():
                results.append(
                    TableProfileResponse(
                        table_name=name,
                        row_count=p.get("row_count", 0),
                        date_range=tuple(p.get("date_range", (None, None))),
                        device_count=p.get("device_count", 0),
                        column_stats={
                            col_name: ColumnStatsResponse(**col_data)
                            for col_name, col_data in p.get("column_stats", {}).items()
                        },
                        profiled_at=p.get("profiled_at"),
                        source_db="mobicontrol",
                    )
                )

        if results:
            return results

    # No cached results - profile tables directly
    # Profile XSight tables
    try:
        from device_anomaly.data_access.data_profiler import profile_dw_tables

        dw_profiles = profile_dw_tables(sample_limit=50_000)
        for name, p in dw_profiles.items():
            results.append(
                TableProfileResponse(
                    table_name=name,
                    row_count=p.row_count,
                    date_range=p.date_range,
                    device_count=p.device_count,
                    column_stats={
                        col_name: ColumnStatsResponse(**col_stats.to_dict())
                        for col_name, col_stats in p.column_stats.items()
                    },
                    profiled_at=p.profiled_at,
                    source_db="xsight",
                )
            )
    except Exception as e:
        logger.warning(f"Failed to profile XSight tables: {e}")

    # Profile MobiControl tables
    if include_mc:
        try:
            from device_anomaly.data_access.data_profiler import (
                get_curated_mc_tables,
                profile_mc_tables,
            )

            mc_tables = get_curated_mc_tables(include_discovered=True)
            mc_profiles = profile_mc_tables(tables=mc_tables, sample_limit=50_000)
            for name, p in mc_profiles.items():
                results.append(
                    TableProfileResponse(
                        table_name=name,
                        row_count=p.row_count,
                        date_range=p.date_range,
                        device_count=p.device_count,
                        column_stats={
                            col_name: ColumnStatsResponse(**col_stats.to_dict())
                            for col_name, col_stats in p.column_stats.items()
                        },
                        profiled_at=p.profiled_at,
                        source_db="mobicontrol",
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to profile MobiControl tables: {e}")

    if not results:
        raise HTTPException(status_code=503, detail="Failed to connect to any database")

    return results


@router.get("/tables/{table_name}/stats", response_model=TableProfileResponse)
def get_table_stats(
    table_name: str,
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get detailed statistics for a specific table."""
    if mock_mode:
        mock_profiles = get_mock_table_profiles()
        for profile in mock_profiles:
            if profile.table_name == table_name:
                return profile
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

    # Check cache first
    if _discovery_cache.get("results") and _discovery_cache.get("results", {}).get("dw_profiles"):
        profiles = _discovery_cache["results"]["dw_profiles"]
        if table_name in profiles:
            p = profiles[table_name]
            return TableProfileResponse(
                table_name=table_name,
                row_count=p.get("row_count", 0),
                date_range=tuple(p.get("date_range", (None, None))),
                device_count=p.get("device_count", 0),
                column_stats={
                    col_name: ColumnStatsResponse(**col_data)
                    for col_name, col_data in p.get("column_stats", {}).items()
                },
                profiled_at=p.get("profiled_at"),
            )

    # Profile the specific table
    try:
        from device_anomaly.data_access.data_profiler import profile_table
        from device_anomaly.data_access.db_connection import create_dw_engine

        engine = create_dw_engine()
        profile = profile_table(engine, table_name, sample_limit=100_000)

        return TableProfileResponse(
            table_name=profile.table_name,
            row_count=profile.row_count,
            date_range=profile.date_range,
            device_count=profile.device_count,
            column_stats={
                col_name: ColumnStatsResponse(**col_stats.to_dict())
                for col_name, col_stats in profile.column_stats.items()
            },
            profiled_at=profile.profiled_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to profile table {table_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to profile table: {str(e)}")


@router.get("/metrics", response_model=list[AvailableMetricResponse])
def get_available_metrics(
    mock_mode: bool = Depends(get_mock_mode),
    include_engineered: bool = Query(
        True, description="Include engineered features from FeatureConfig"
    ),
):
    """Get list of all available numeric metrics across all tables.

    Returns metrics that can be used for analysis, training, and visualization.
    Includes both raw database columns and engineered features (rolling stats,
    derived features, temporal features, etc.).
    """
    if mock_mode:
        return get_mock_available_metrics()

    metrics = []
    raw_metrics_found = False

    # Try to import FeatureConfig early for domain inference
    try:
        from device_anomaly.config.feature_config import FeatureConfig

        has_feature_config = True
    except Exception as e:
        logger.warning(f"Failed to import FeatureConfig: {e}")
        has_feature_config = False
        FeatureConfig = None

    # Helper to get domain
    def get_domain(column_name: str) -> str:
        if has_feature_config:
            try:
                return FeatureConfig.get_domain_for_feature(column_name)
            except Exception:
                pass
        # Infer from column name
        col_lower = column_name.lower()
        if any(x in col_lower for x in ["battery", "charge", "discharge", "drain"]):
            return "battery"
        if any(x in col_lower for x in ["signal", "drop", "wifi", "cell", "rf"]):
            return "rf"
        if any(x in col_lower for x in ["download", "upload", "data", "throughput"]):
            return "throughput"
        if any(x in col_lower for x in ["app", "usage", "visit", "session", "crash"]):
            return "usage"
        if any(x in col_lower for x in ["storage", "ram", "memory", "disk"]):
            return "storage"
        return "unknown"

    # 1. First try to get metrics from cached discovery results
    if _discovery_cache.get("results"):
        cache = _discovery_cache["results"]

        # Extract from XSight profiles cache
        if cache.get("dw_profiles"):
            for table_name, profile in cache["dw_profiles"].items():
                for col_name, col_stats in profile.get("column_stats", {}).items():
                    # Only include numeric columns (those with mean/std)
                    if col_stats.get("mean") is not None:
                        metrics.append(
                            AvailableMetricResponse(
                                table=table_name,
                                column=col_name,
                                dtype=col_stats.get("dtype", "float"),
                                mean=col_stats.get("mean"),
                                std=col_stats.get("std"),
                                min=col_stats.get("min_val"),
                                max=col_stats.get("max_val"),
                                category="raw",
                                domain=get_domain(col_name),
                            )
                        )
                        raw_metrics_found = True

        # Extract from MobiControl profiles cache
        if cache.get("mc_profiles"):
            for table_name, profile in cache["mc_profiles"].items():
                for col_name, col_stats in profile.get("column_stats", {}).items():
                    if col_stats.get("mean") is not None:
                        metrics.append(
                            AvailableMetricResponse(
                                table=table_name,
                                column=col_name,
                                dtype=col_stats.get("dtype", "float"),
                                mean=col_stats.get("mean"),
                                std=col_stats.get("std"),
                                min=col_stats.get("min_val"),
                                max=col_stats.get("max_val"),
                                category="raw",
                                domain=get_domain(col_name),
                            )
                        )
                        raw_metrics_found = True

    # 2. If no cached metrics, try profiling directly
    if not raw_metrics_found:
        try:
            from device_anomaly.data_access.data_profiler import (
                get_available_metrics as _get_metrics,
            )
            from device_anomaly.data_access.data_profiler import profile_dw_tables

            profiles = profile_dw_tables(sample_limit=50_000)
            raw_metrics = _get_metrics(profiles)

            # Add category and domain info for raw metrics
            for m in raw_metrics:
                metric = AvailableMetricResponse(**m)
                metric.category = "raw"
                metric.domain = get_domain(m["column"])
                metrics.append(metric)
                raw_metrics_found = True
        except Exception as e:
            logger.warning(f"Failed to get raw metrics from profiler: {e}")

    # 3. Add engineered features from FeatureConfig (if available)
    if include_engineered and has_feature_config:
        try:
            # Add all raw features that could be engineered
            all_raw_features = FeatureConfig.get_all_raw_features()
            for feature_name in all_raw_features:
                # Skip if already in raw metrics
                if any(m.column == feature_name and m.category == "raw" for m in metrics):
                    continue

                # Check if this is a potential feature
                domain = FeatureConfig.get_domain_for_feature(feature_name)
                if domain != "unknown":
                    metrics.append(
                        AvailableMetricResponse(
                            table="feature_config",
                            column=feature_name,
                            dtype="float",
                            category="raw",
                            domain=domain,
                            description=f"Raw feature from {domain} domain",
                        )
                    )

            # Add rolling window features for key metrics
            for base_feature in FeatureConfig.rolling_feature_candidates:
                for window in FeatureConfig.rolling_windows:
                    for agg in ["mean", "std", "min", "max", "median"]:
                        feature_name = (
                            f"{base_feature}_roll_{window}d_{agg}"
                            if window != 14
                            else f"{base_feature}_roll_{agg}"
                        )
                        domain = FeatureConfig.get_domain_for_feature(base_feature)
                        metrics.append(
                            AvailableMetricResponse(
                                table="feature_engineered",
                                column=feature_name,
                                dtype="float",
                                category="rolling",
                                domain=domain,
                                description=f"{agg.title()} of {base_feature} over {window} days",
                            )
                        )

            # Add delta features
            for base_feature in FeatureConfig.rolling_feature_candidates:
                for delta_type in ["delta", "pct_change", "trend_7d"]:
                    feature_name = f"{base_feature}_{delta_type}"
                    domain = FeatureConfig.get_domain_for_feature(base_feature)
                    metrics.append(
                        AvailableMetricResponse(
                            table="feature_engineered",
                            column=feature_name,
                            dtype="float",
                            category="delta",
                            domain=domain,
                            description=f"{delta_type.replace('_', ' ').title()} for {base_feature}",
                        )
                    )

            # Add derived features
            for feature_name, defn in FeatureConfig.derived_feature_definitions.items():
                domain = defn.get("domain", "unknown")
                formula = defn.get("formula", "")
                metrics.append(
                    AvailableMetricResponse(
                        table="feature_engineered",
                        column=feature_name,
                        dtype="float",
                        category="derived",
                        domain=domain,
                        description=f"Derived: {formula}",
                    )
                )

            # Add temporal features
            for feature_name in FeatureConfig.temporal_features:
                metrics.append(
                    AvailableMetricResponse(
                        table="feature_engineered",
                        column=feature_name,
                        dtype="float",
                        category="temporal",
                        domain="temporal",
                        description=f"Temporal feature: {feature_name}",
                    )
                )

            # Add cohort z-score features (for all numeric features)
            for base_feature in all_raw_features:
                feature_name = f"{base_feature}_cohort_z"
                domain = FeatureConfig.get_domain_for_feature(base_feature)
                metrics.append(
                    AvailableMetricResponse(
                        table="feature_engineered",
                        column=feature_name,
                        dtype="float",
                        category="cohort",
                        domain=domain,
                        description=f"Cohort-normalized z-score for {base_feature}",
                    )
                )

            # Add volatility (CV) features
            for base_feature in FeatureConfig.rolling_feature_candidates:
                feature_name = f"{base_feature}_cv"
                domain = FeatureConfig.get_domain_for_feature(base_feature)
                metrics.append(
                    AvailableMetricResponse(
                        table="feature_engineered",
                        column=feature_name,
                        dtype="float",
                        category="volatility",
                        domain=domain,
                        description=f"Coefficient of variation (volatility) for {base_feature}",
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to add engineered features: {e}")

    # Sort by category, then domain, then name
    metrics.sort(key=lambda m: (m.category or "", m.domain or "", m.column))

    return metrics


@router.get("/metrics/{metric_name}/distribution", response_model=MetricDistributionResponse)
def get_metric_distribution(
    metric_name: str,
    table_name: str | None = Query(None, description="Table containing the metric"),
    bins: int = Query(50, ge=10, le=100, description="Number of histogram bins"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get histogram distribution for a specific metric.

    Useful for visualizing data distributions and identifying outliers.
    """
    if mock_mode:
        return get_mock_metric_distribution(metric_name)

    # Determine which table to query
    if not table_name:
        # Try to find the table containing this metric
        try:
            from device_anomaly.data_access.data_profiler import (
                get_available_metrics as _get_metrics,
            )
            from device_anomaly.data_access.data_profiler import profile_dw_tables

            profiles = profile_dw_tables(sample_limit=10_000)
            metrics = _get_metrics(profiles)
            matching = [m for m in metrics if m["column"] == metric_name]
            if not matching:
                raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
            table_name = matching[0]["table"]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to find metric: {str(e)}")

    # Compute distribution
    try:
        from device_anomaly.data_access.data_profiler import compute_metric_distribution
        from device_anomaly.data_access.db_connection import create_dw_engine

        engine = create_dw_engine()
        distribution = compute_metric_distribution(engine, table_name, metric_name, bins=bins)

        return MetricDistributionResponse(**distribution)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to compute distribution for {metric_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to compute distribution: {str(e)}")


@router.get("/temporal-patterns", response_model=list[TemporalPatternResponse])
def get_temporal_patterns(
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get time-of-day and day-of-week patterns for key metrics.

    Shows how metrics vary by hour (0-23) and day of week (Mon-Sun).
    Useful for understanding normal patterns and setting time-aware thresholds.
    """
    if mock_mode:
        return get_mock_temporal_patterns()

    # Check cache
    if _discovery_cache.get("results") and _discovery_cache.get("results", {}).get(
        "temporal_patterns"
    ):
        patterns = _discovery_cache["results"]["temporal_patterns"]
        return [
            TemporalPatternResponse(
                metric_name=name,
                hourly_stats=p.get("hourly_stats", {}),
                daily_stats=p.get("daily_stats", {}),
            )
            for name, p in patterns.items()
        ]

    # Need to run discovery to get temporal patterns
    raise HTTPException(
        status_code=404, detail="Temporal patterns not available. Run a full data discovery first."
    )


@router.get("/summary", response_model=DiscoverySummaryResponse)
def get_discovery_summary(
    include_mc: bool = Query(default=True, description="Include MobiControl database"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get summary of data discovery results from both XSight and MobiControl databases."""
    if mock_mode:
        return DiscoverySummaryResponse(
            total_tables_profiled=8,
            total_rows=162_750_000,
            total_devices=4_500,
            metrics_discovered=45,
            patterns_analyzed=8,
            date_range={"start": "2024-01-01", "end": "2024-12-28"},
            discovery_completed=datetime.now(UTC).isoformat(),
        )

    # Check cache first
    if _discovery_cache.get("results") and _discovery_cache.get("results", {}).get("summary"):
        summary = _discovery_cache["results"]["summary"]
        return DiscoverySummaryResponse(
            total_tables_profiled=summary.get("total_tables_profiled", 0),
            total_rows=summary.get("total_rows", 0),
            total_devices=summary.get("total_devices", 0),
            metrics_discovered=summary.get("metrics_discovered", 0),
            patterns_analyzed=summary.get("patterns_analyzed", 0),
            date_range=_discovery_cache["results"].get("date_range"),
            discovery_completed=_discovery_cache.get("completed_at"),
        )

    # No cache - compute summary on-the-fly using schema discovery (fast, no table scans)
    try:
        from device_anomaly.data_access.schema_discovery import (
            discover_mobicontrol_schema,
            discover_xsight_schema,
        )

        total_tables = 0
        total_rows = 0
        total_devices = 0
        metrics_count = 0

        # XSight schema
        try:
            xsight_schema = discover_xsight_schema(use_cache=True)
            total_tables += len(xsight_schema.tables)
            total_rows += sum(t.row_count for t in xsight_schema.tables.values())
            # Count unique devices from high-value tables
            for t in xsight_schema.high_value_tables:
                if t.has_device_id and t.row_count > 0:
                    # Approximate device count from the table with most rows
                    pass
            # Count numeric columns as potential metrics
            for t in xsight_schema.tables.values():
                metrics_count += sum(1 for c in t.columns if c.is_numeric)
        except Exception as e:
            logger.warning(f"Failed to get XSight schema: {e}")

        # MobiControl schema
        if include_mc:
            try:
                mc_schema = discover_mobicontrol_schema(use_cache=True)
                total_tables += len(mc_schema.tables)
                total_rows += sum(t.row_count for t in mc_schema.tables.values())
                for t in mc_schema.tables.values():
                    metrics_count += sum(1 for c in t.columns if c.is_numeric)
            except Exception as e:
                logger.warning(f"Failed to get MobiControl schema: {e}")

        # Get device count and date range from a quick query if possible
        date_range = None
        try:
            from device_anomaly.data_access.db_connection import create_dw_engine

            engine = create_dw_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(DISTINCT DeviceId) FROM cs_BatteryStat"))
                row = result.fetchone()
                if row:
                    total_devices = row[0]

                # Get date range from telemetry data
                date_result = conn.execute(
                    text("SELECT MIN(Timestamp), MAX(Timestamp) FROM cs_BatteryStat")
                )
                date_row = date_result.fetchone()
                if date_row and date_row[0] and date_row[1]:
                    date_range = {
                        "start": date_row[0].strftime("%Y-%m-%d")
                        if hasattr(date_row[0], "strftime")
                        else str(date_row[0])[:10],
                        "end": date_row[1].strftime("%Y-%m-%d")
                        if hasattr(date_row[1], "strftime")
                        else str(date_row[1])[:10],
                    }
        except Exception as e:
            logger.debug(f"Could not get device count or date range: {e}")

        # patterns_analyzed is meaningful when temporal/correlation analysis has run
        # For schema-only discovery, we count high-value tables as analyzed patterns
        patterns_analyzed = 0
        try:
            if "xsight_schema" in dir() and xsight_schema:
                patterns_analyzed += len(xsight_schema.high_value_tables)
            if include_mc and "mc_schema" in dir() and mc_schema:
                patterns_analyzed += len(getattr(mc_schema, "high_value_tables", []))
        except Exception:
            pass

        return DiscoverySummaryResponse(
            total_tables_profiled=total_tables,
            total_rows=total_rows,
            total_devices=total_devices,
            metrics_discovered=metrics_count,
            patterns_analyzed=patterns_analyzed,
            date_range=date_range,
            discovery_completed=datetime.now(UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to compute discovery summary: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to compute summary: {str(e)}")


@router.get("/status", response_model=DataDiscoveryStatusResponse)
def get_discovery_status():
    """Get status of the current or last data discovery job."""
    return DataDiscoveryStatusResponse(
        status=_discovery_cache.get("status", "idle"),
        progress=_discovery_cache.get("progress", 0.0),
        message=_discovery_cache.get("message"),
        started_at=_discovery_cache.get("started_at"),
        completed_at=_discovery_cache.get("completed_at"),
        results_available=_discovery_cache.get("results") is not None,
    )


def _run_discovery_job(start_date: str, end_date: str, include_mc: bool, analyze_patterns: bool):
    """Background task to run data discovery."""
    global _discovery_cache

    try:
        _discovery_cache["status"] = "running"
        _discovery_cache["progress"] = 10.0
        _discovery_cache["message"] = "Loading data profiler..."

        from device_anomaly.cli.data_discovery import run_data_discovery

        _discovery_cache["progress"] = 20.0
        _discovery_cache["message"] = "Profiling database tables..."

        results = run_data_discovery(
            start_date=start_date,
            end_date=end_date,
            include_mc=include_mc,
            analyze_patterns=analyze_patterns,
            load_sample_data=analyze_patterns,
        )

        _discovery_cache["status"] = "completed"
        _discovery_cache["progress"] = 100.0
        _discovery_cache["message"] = "Discovery completed successfully"
        _discovery_cache["completed_at"] = datetime.now(UTC).isoformat()
        _discovery_cache["results"] = results

    except Exception as e:
        logger.error(f"Data discovery failed: {e}")
        _discovery_cache["status"] = "failed"
        _discovery_cache["message"] = f"Discovery failed: {str(e)}"


@router.post("/run", response_model=DataDiscoveryStatusResponse)
def run_data_discovery_job(
    background_tasks: BackgroundTasks,
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    include_mc: bool = Query(True, description="Include MobiControl database"),
    analyze_patterns: bool = Query(True, description="Analyze temporal patterns"),
    _: None = Depends(require_role(["analyst", "admin"])),
):
    """Start a data discovery job in the background.

    This endpoint triggers a comprehensive analysis of all telemetry tables,
    computing statistics, distributions, and temporal patterns.
    """
    global _discovery_cache

    if _discovery_cache.get("status") == "running":
        raise HTTPException(status_code=409, detail="A discovery job is already running")

    # Set default dates if not provided
    now = datetime.now()
    if not end_date:
        end_date = now.strftime("%Y-%m-%d")
    if not start_date:
        start_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")

    # Reset cache and start job
    _discovery_cache = {
        "status": "running",
        "progress": 0.0,
        "message": "Starting discovery...",
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "results": None,
    }

    background_tasks.add_task(
        _run_discovery_job,
        start_date=start_date,
        end_date=end_date,
        include_mc=include_mc,
        analyze_patterns=analyze_patterns,
    )

    return DataDiscoveryStatusResponse(
        status="running",
        progress=0.0,
        message="Discovery job started",
        started_at=_discovery_cache["started_at"],
        results_available=False,
    )


# ============================================================================
# MobiControl Integration Health
# ============================================================================


class MCIntegrationStatusResponse(BaseModel):
    """Status of MobiControl database integration."""

    connected: bool = False
    credentials_configured: bool = False
    host: str | None = None
    database: str | None = None
    streaming_cache_size: int = 0
    streaming_cache_loaded_at: str | None = None
    batch_last_load_rows: int | None = None
    error: str | None = None


@router.get("/mc-status", response_model=MCIntegrationStatusResponse)
def get_mc_integration_status():
    """
    Get MobiControl database integration status.

    Shows whether MC credentials are configured, if connection is working,
    and whether the streaming enrichment cache is populated.
    """

    settings = get_settings()
    mc_settings = settings.mc

    response = MCIntegrationStatusResponse(
        credentials_configured=bool(mc_settings.host and mc_settings.user and mc_settings.password),
        host=mc_settings.host or None,
        database=mc_settings.database or None,
    )

    if not response.credentials_configured:
        response.error = "MC_DB_HOST, MC_DB_USER, or MC_DB_PASS not configured"
        return response

    # Check streaming cache
    try:
        from device_anomaly.services.device_metadata_sync import get_mc_cache_stats

        cache_stats = get_mc_cache_stats()
        response.streaming_cache_size = cache_stats.get("cache_size", 0)
        response.streaming_cache_loaded_at = cache_stats.get("loaded_at")
    except Exception as e:
        logger.debug("Failed to get MC cache stats: %s", e)

    # Test connection
    try:
        from device_anomaly.data_access.db_connection import create_mc_engine

        engine = create_mc_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 AS test"))
            row = result.fetchone()
            if row and row[0] == 1:
                response.connected = True

                # Try to get device count
                try:
                    count_result = conn.execute(text("SELECT COUNT(*) FROM dbo.DevInfo"))
                    response.batch_last_load_rows = count_result.fetchone()[0]
                except Exception:
                    pass

    except Exception as e:
        response.connected = False
        response.error = f"Connection failed: {str(e)}"

    return response
