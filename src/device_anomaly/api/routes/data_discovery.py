"""API routes for data discovery and profiling endpoints."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

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
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    percentiles: Dict[str, float] = Field(default_factory=dict)


class TableProfileResponse(BaseModel):
    """Complete statistical profile of a database table."""

    table_name: str
    row_count: int
    date_range: tuple[Optional[str], Optional[str]]
    device_count: int
    column_stats: Dict[str, ColumnStatsResponse] = Field(default_factory=dict)
    profiled_at: Optional[str] = None


class TemporalPatternResponse(BaseModel):
    """Time-based pattern for a metric."""

    metric_name: str
    hourly_stats: Dict[int, Dict[str, float]] = Field(default_factory=dict)
    daily_stats: Dict[int, Dict[str, float]] = Field(default_factory=dict)


class MetricDistributionResponse(BaseModel):
    """Histogram distribution for a metric."""

    bins: List[float]
    counts: List[int]
    stats: Dict[str, Any]


class AvailableMetricResponse(BaseModel):
    """Information about an available metric."""

    table: str
    column: str
    dtype: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    category: Optional[str] = None  # 'raw', 'rolling', 'derived', 'temporal', 'cohort', 'volatility'
    domain: Optional[str] = None  # 'battery', 'rf', 'throughput', 'usage', 'storage', etc.
    description: Optional[str] = None  # Human-readable description


class DiscoverySummaryResponse(BaseModel):
    """Summary of data discovery results."""

    total_tables_profiled: int
    total_rows: int
    total_devices: int
    metrics_discovered: int
    patterns_analyzed: int
    date_range: Optional[Dict[str, str]] = None
    discovery_completed: Optional[str] = None


class DataDiscoveryStatusResponse(BaseModel):
    """Status of a data discovery job."""

    status: str  # 'idle', 'running', 'completed', 'failed'
    progress: float = 0.0  # 0-100
    message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results_available: bool = False


# ============================================================================
# In-memory storage for discovery results (in production, use Redis or DB)
# ============================================================================

_discovery_cache: Dict[str, Any] = {
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


def get_mock_table_profiles() -> List[TableProfileResponse]:
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
                    percentiles={"p5": 3600, "p25": 18000, "p50": 28800, "p75": 39600, "p95": 57600, "p99": 72000},
                ),
            },
            profiled_at=datetime.now(timezone.utc).isoformat(),
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
            profiled_at=datetime.now(timezone.utc).isoformat(),
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
                    percentiles={"p5": 1000, "p25": 10_000_000, "p50": 80_000_000, "p75": 200_000_000, "p95": 600_000_000, "p99": 1_500_000_000},
                ),
            },
            profiled_at=datetime.now(timezone.utc).isoformat(),
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
                    percentiles={"p5": -98, "p25": -82, "p50": -72, "p75": -62, "p95": -50, "p99": -44},
                ),
            },
            profiled_at=datetime.now(timezone.utc).isoformat(),
        ),
    ]


def get_mock_available_metrics() -> List[AvailableMetricResponse]:
    """Generate mock available metrics including raw and engineered features."""
    metrics = [
        # Raw database metrics
        AvailableMetricResponse(
            table="cs_BatteryStat", column="TotalBatteryLevelDrop", dtype="int",
            mean=35.2, std=18.5, min=0, max=100,
            category="raw", domain="battery"
        ),
        AvailableMetricResponse(
            table="cs_BatteryStat", column="TotalDischargeTime_Sec", dtype="int",
            mean=28800, std=12000, min=0, max=86400,
            category="raw", domain="battery"
        ),
        AvailableMetricResponse(
            table="cs_BatteryStat", column="TotalFreeStorageKb", dtype="bigint",
            mean=2_000_000, std=1_500_000, min=0, max=16_000_000,
            category="raw", domain="storage"
        ),
        AvailableMetricResponse(
            table="cs_AppUsage", column="VisitCount", dtype="int",
            mean=45, std=32, min=0, max=1000,
            category="raw", domain="usage"
        ),
        AvailableMetricResponse(
            table="cs_AppUsage", column="TotalForegroundTime", dtype="int",
            mean=3600, std=2400, min=0, max=28800,
            category="raw", domain="usage"
        ),
        AvailableMetricResponse(
            table="cs_DataUsage", column="Download", dtype="bigint",
            mean=150_000_000, std=250_000_000, min=0, max=10_000_000_000,
            category="raw", domain="throughput"
        ),
        AvailableMetricResponse(
            table="cs_DataUsage", column="Upload", dtype="bigint",
            mean=25_000_000, std=50_000_000, min=0, max=2_000_000_000,
            category="raw", domain="throughput"
        ),
        AvailableMetricResponse(
            table="cs_Heatmap", column="SignalStrength", dtype="int",
            mean=-72, std=15, min=-120, max=-40,
            category="raw", domain="rf"
        ),
        AvailableMetricResponse(
            table="cs_Heatmap", column="DropCnt", dtype="int",
            mean=2.5, std=5.2, min=0, max=100,
            category="raw", domain="rf"
        ),
    ]
    
    # Add some engineered features as examples
    metrics.extend([
        AvailableMetricResponse(
            table="feature_engineered", column="TotalBatteryLevelDrop_roll_mean",
            dtype="float", category="rolling", domain="battery",
            description="Mean of TotalBatteryLevelDrop over 14 days"
        ),
        AvailableMetricResponse(
            table="feature_engineered", column="BatteryDrainPerHour",
            dtype="float", category="derived", domain="battery",
            description="Derived: TotalBatteryLevelDrop / (TotalDischargeTime_Sec / 3600 + 1)"
        ),
        AvailableMetricResponse(
            table="feature_engineered", column="Download_delta",
            dtype="float", category="delta", domain="throughput",
            description="Day-over-day change for Download"
        ),
        AvailableMetricResponse(
            table="feature_engineered", column="hour_of_day",
            dtype="int", category="temporal", domain="temporal",
            description="Temporal feature: hour of day"
        ),
    ])
    
    return metrics


def get_mock_temporal_patterns() -> List[TemporalPatternResponse]:
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

        patterns.append(TemporalPatternResponse(
            metric_name=metric,
            hourly_stats=hourly,
            daily_stats=daily,
        ))

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


@router.get("/available-tables", response_model=List[AvailableTableResponse])
def get_available_tables(
    pattern: str = Query(default="cs_%", description="SQL LIKE pattern to filter tables"),
    exclude_time_slices: bool = Query(default=True, description="Exclude time-sliced views like _Last7, _LastMonth"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Discover available tables in the data warehouse without full profiling.

    This is a fast endpoint that just lists table names matching the pattern.
    Use /tables for full statistical profiles.
    """
    if mock_mode:
        # Return known table names in mock mode
        return [
            AvailableTableResponse(table_name="cs_BatteryStat"),
            AvailableTableResponse(table_name="cs_AppUsage"),
            AvailableTableResponse(table_name="cs_DataUsage"),
            AvailableTableResponse(table_name="cs_BatteryAppDrain"),
            AvailableTableResponse(table_name="cs_Heatmap"),
        ]

    try:
        from device_anomaly.data_access.data_profiler import discover_dw_tables

        tables = discover_dw_tables(pattern=pattern, exclude_time_slices=exclude_time_slices)
        return [AvailableTableResponse(table_name=name) for name in tables]
    except Exception as e:
        logger.error(f"Failed to discover tables: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to database: {str(e)}")


@router.get("/tables", response_model=List[TableProfileResponse])
def get_table_profiles(
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get statistical profiles for all telemetry tables.

    Returns row counts, date ranges, device counts, and column statistics
    for each profiled table in the XSight Database.
    """
    if mock_mode:
        return get_mock_table_profiles()

    # Check if we have cached results
    if _discovery_cache.get("results") and _discovery_cache.get("results", {}).get("dw_profiles"):
        profiles = _discovery_cache["results"]["dw_profiles"]
        return [
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
            )
            for name, p in profiles.items()
        ]

    # No cached results - try to profile tables directly
    try:
        from device_anomaly.data_access.data_profiler import profile_dw_tables

        profiles = profile_dw_tables(sample_limit=50_000)
        return [
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
            )
            for name, p in profiles.items()
        ]
    except Exception as e:
        logger.error(f"Failed to profile tables: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to database: {str(e)}")


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


@router.get("/metrics", response_model=List[AvailableMetricResponse])
def get_available_metrics(
    mock_mode: bool = Depends(get_mock_mode),
    include_engineered: bool = Query(True, description="Include engineered features from FeatureConfig"),
):
    """Get list of all available numeric metrics across all tables.

    Returns metrics that can be used for analysis, training, and visualization.
    Includes both raw database columns and engineered features (rolling stats,
    derived features, temporal features, etc.).
    """
    if mock_mode:
        return get_mock_available_metrics()

    metrics = []

    # 1. Get raw database metrics
    try:
        from device_anomaly.data_access.data_profiler import profile_dw_tables, get_available_metrics as _get_metrics

        profiles = profile_dw_tables(sample_limit=50_000)
        raw_metrics = _get_metrics(profiles)
        
        # Add category and domain info for raw metrics
        for m in raw_metrics:
            metric = AvailableMetricResponse(**m)
            metric.category = "raw"
            # Try to infer domain from FeatureConfig
            from device_anomaly.config.feature_config import FeatureConfig
            metric.domain = FeatureConfig.get_domain_for_feature(m["column"])
            metrics.append(metric)
    except Exception as e:
        logger.warning(f"Failed to get raw metrics: {e}")

    # 2. Add engineered features from FeatureConfig
    if include_engineered:
        from device_anomaly.config.feature_config import FeatureConfig

        # Add all raw features that could be engineered
        all_raw_features = FeatureConfig.get_all_raw_features()
        for feature_name in all_raw_features:
            # Skip if already in raw metrics
            if any(m.column == feature_name and m.category == "raw" for m in metrics):
                continue
            
            # Check if this is a potential feature
            domain = FeatureConfig.get_domain_for_feature(feature_name)
            if domain != "unknown":
                metrics.append(AvailableMetricResponse(
                    table="feature_config",
                    column=feature_name,
                    dtype="float",
                    category="raw",
                    domain=domain,
                    description=f"Raw feature from {domain} domain"
                ))

        # Add rolling window features for key metrics
        for base_feature in FeatureConfig.rolling_feature_candidates:
            for window in FeatureConfig.rolling_windows:
                for agg in ["mean", "std", "min", "max", "median"]:
                    feature_name = f"{base_feature}_roll_{window}d_{agg}" if window != 14 else f"{base_feature}_roll_{agg}"
                    domain = FeatureConfig.get_domain_for_feature(base_feature)
                    metrics.append(AvailableMetricResponse(
                        table="feature_engineered",
                        column=feature_name,
                        dtype="float",
                        category="rolling",
                        domain=domain,
                        description=f"{agg.title()} of {base_feature} over {window} days"
                    ))

        # Add delta features
        for base_feature in FeatureConfig.rolling_feature_candidates:
            for delta_type in ["delta", "pct_change", "trend_7d"]:
                feature_name = f"{base_feature}_{delta_type}"
                domain = FeatureConfig.get_domain_for_feature(base_feature)
                metrics.append(AvailableMetricResponse(
                    table="feature_engineered",
                    column=feature_name,
                    dtype="float",
                    category="delta",
                    domain=domain,
                    description=f"{delta_type.replace('_', ' ').title()} for {base_feature}"
                ))

        # Add derived features
        for feature_name, defn in FeatureConfig.derived_feature_definitions.items():
            domain = defn.get("domain", "unknown")
            formula = defn.get("formula", "")
            metrics.append(AvailableMetricResponse(
                table="feature_engineered",
                column=feature_name,
                dtype="float",
                category="derived",
                domain=domain,
                description=f"Derived: {formula}"
            ))

        # Add temporal features
        for feature_name in FeatureConfig.temporal_features:
            metrics.append(AvailableMetricResponse(
                table="feature_engineered",
                column=feature_name,
                dtype="float",
                category="temporal",
                domain="temporal",
                description=f"Temporal feature: {feature_name}"
            ))

        # Add cohort z-score features (for all numeric features)
        for base_feature in all_raw_features:
            feature_name = f"{base_feature}_cohort_z"
            domain = FeatureConfig.get_domain_for_feature(base_feature)
            metrics.append(AvailableMetricResponse(
                table="feature_engineered",
                column=feature_name,
                dtype="float",
                category="cohort",
                domain=domain,
                description=f"Cohort-normalized z-score for {base_feature}"
            ))

        # Add volatility (CV) features
        for base_feature in FeatureConfig.rolling_feature_candidates:
            feature_name = f"{base_feature}_cv"
            domain = FeatureConfig.get_domain_for_feature(base_feature)
            metrics.append(AvailableMetricResponse(
                table="feature_engineered",
                column=feature_name,
                dtype="float",
                category="volatility",
                domain=domain,
                description=f"Coefficient of variation (volatility) for {base_feature}"
            ))

    # Sort by category, then domain, then name
    metrics.sort(key=lambda m: (m.category or "", m.domain or "", m.column))

    return metrics


@router.get("/metrics/{metric_name}/distribution", response_model=MetricDistributionResponse)
def get_metric_distribution(
    metric_name: str,
    table_name: Optional[str] = Query(None, description="Table containing the metric"),
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
            from device_anomaly.data_access.data_profiler import profile_dw_tables, get_available_metrics as _get_metrics

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


@router.get("/temporal-patterns", response_model=List[TemporalPatternResponse])
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
    if _discovery_cache.get("results") and _discovery_cache.get("results", {}).get("temporal_patterns"):
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
        status_code=404,
        detail="Temporal patterns not available. Run a full data discovery first."
    )


@router.get("/summary", response_model=DiscoverySummaryResponse)
def get_discovery_summary(
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get summary of the latest data discovery results."""
    if mock_mode:
        return DiscoverySummaryResponse(
            total_tables_profiled=5,
            total_rows=12_750_000,
            total_devices=4_500,
            metrics_discovered=15,
            patterns_analyzed=8,
            date_range={"start": "2024-01-01", "end": "2024-12-28"},
            discovery_completed=datetime.now(timezone.utc).isoformat(),
        )

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

    raise HTTPException(status_code=404, detail="No discovery results available. Run a data discovery first.")


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
        _discovery_cache["completed_at"] = datetime.now(timezone.utc).isoformat()
        _discovery_cache["results"] = results

    except Exception as e:
        logger.error(f"Data discovery failed: {e}")
        _discovery_cache["status"] = "failed"
        _discovery_cache["message"] = f"Discovery failed: {str(e)}"


@router.post("/run", response_model=DataDiscoveryStatusResponse)
def run_data_discovery_job(
    background_tasks: BackgroundTasks,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
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
        "started_at": datetime.now(timezone.utc).isoformat(),
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
