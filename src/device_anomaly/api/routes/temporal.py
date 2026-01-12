"""API routes for temporal analysis."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights/temporal", tags=["temporal-analysis"])


# ============================================================================
# Response Models
# ============================================================================

class HourlyDataPointResponse(BaseModel):
    """Single hour's aggregated data."""
    hour: int
    avg_value: float
    min_value: float
    max_value: float
    std_value: float
    sample_count: int


class HourlyBreakdownResponse(BaseModel):
    """Hourly breakdown analysis."""
    tenant_id: str
    metric: str
    hourly_data: List[HourlyDataPointResponse]
    peak_hours: List[int]
    low_hours: List[int]
    day_night_ratio: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PeakDetectionResponse(BaseModel):
    """Detected usage peak."""
    timestamp: datetime
    value: float
    z_score: float
    is_significant: bool


class PeakDetectionListResponse(BaseModel):
    """List of detected peaks."""
    tenant_id: str
    metric: str
    peaks: List[PeakDetectionResponse]
    total_peaks: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PeriodStatsResponse(BaseModel):
    """Statistics for a time period."""
    start: datetime
    end: datetime
    avg: float
    median: float
    std: float
    sample_count: int


class TemporalComparisonResponse(BaseModel):
    """Comparison between two time periods."""
    tenant_id: str
    metric: str
    period_a: PeriodStatsResponse
    period_b: PeriodStatsResponse
    change_percent: float
    is_significant: bool
    p_value: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DailyComparisonPoint(BaseModel):
    """Single day in day-over-day comparison."""
    date: str
    value: float
    sample_count: int
    change_percent: float


class DayOverDayResponse(BaseModel):
    """Day over day comparison."""
    tenant_id: str
    metric: str
    comparisons: List[DailyComparisonPoint]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WeeklyComparisonPoint(BaseModel):
    """Single week in week-over-week comparison."""
    year: int
    week: int
    value: float
    sample_count: int
    change_percent: float


class WeekOverWeekResponse(BaseModel):
    """Week over week comparison."""
    tenant_id: str
    metric: str
    comparisons: List[WeeklyComparisonPoint]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Mock Data Functions
# ============================================================================

def get_mock_hourly_breakdown(metric: str) -> HourlyBreakdownResponse:
    """Generate mock hourly breakdown."""
    import random
    random.seed(hash(metric) % 100)

    base_values = {
        "data_usage": 50.0,
        "battery_drain": 5.0,
        "app_usage": 30.0,
    }
    base = base_values.get(metric, 40.0)

    hourly_data = []
    for hour in range(24):
        # Add daily pattern - higher during work hours
        is_work_hour = 8 <= hour <= 18
        multiplier = 1.5 if is_work_hour else 0.7

        avg = base * multiplier + random.uniform(-5, 5)
        std = avg * 0.2

        hourly_data.append(HourlyDataPointResponse(
            hour=hour,
            avg_value=avg,
            min_value=avg - std * 2,
            max_value=avg + std * 2,
            std_value=std,
            sample_count=random.randint(500, 2000),
        ))

    return HourlyBreakdownResponse(
        tenant_id="default",
        metric=metric,
        hourly_data=hourly_data,
        peak_hours=[9, 10, 11, 14, 15, 16],
        low_hours=[2, 3, 4, 5],
        day_night_ratio=1.8,
    )


def get_mock_peak_detection(metric: str, period_days: int) -> PeakDetectionListResponse:
    """Generate mock peak detection."""
    import random
    random.seed(42)

    now = datetime.now(timezone.utc)
    peaks = []

    for i in range(8):
        ts = now - timedelta(days=random.uniform(0, period_days), hours=random.randint(0, 23))
        z_score = random.uniform(2.0, 4.5)

        peaks.append(PeakDetectionResponse(
            timestamp=ts,
            value=random.uniform(100, 500),
            z_score=z_score,
            is_significant=True,
        ))

    peaks.sort(key=lambda p: p.z_score, reverse=True)

    return PeakDetectionListResponse(
        tenant_id="default",
        metric=metric,
        peaks=peaks,
        total_peaks=len(peaks),
    )


def get_mock_comparison(
    metric: str,
    period_a_start: datetime,
    period_a_end: datetime,
    period_b_start: datetime,
    period_b_end: datetime,
) -> TemporalComparisonResponse:
    """Generate mock period comparison."""
    import random
    random.seed(42)

    avg_a = random.uniform(40, 60)
    avg_b = avg_a * random.uniform(0.8, 1.2)

    return TemporalComparisonResponse(
        tenant_id="default",
        metric=metric,
        period_a=PeriodStatsResponse(
            start=period_a_start,
            end=period_a_end,
            avg=avg_a,
            median=avg_a * 0.95,
            std=avg_a * 0.2,
            sample_count=random.randint(5000, 15000),
        ),
        period_b=PeriodStatsResponse(
            start=period_b_start,
            end=period_b_end,
            avg=avg_b,
            median=avg_b * 0.95,
            std=avg_b * 0.2,
            sample_count=random.randint(5000, 15000),
        ),
        change_percent=((avg_b - avg_a) / avg_a) * 100,
        is_significant=abs(avg_b - avg_a) / avg_a > 0.1,
        p_value=random.uniform(0.001, 0.2),
    )


def get_mock_day_over_day(metric: str, lookback_days: int) -> DayOverDayResponse:
    """Generate mock day-over-day comparison."""
    import random
    random.seed(42)

    now = datetime.now(timezone.utc)
    comparisons = []
    prev_value = None

    for i in range(lookback_days):
        date = (now - timedelta(days=lookback_days - i - 1)).strftime("%Y-%m-%d")
        value = random.uniform(40, 60)

        if prev_value:
            change = ((value - prev_value) / prev_value) * 100
        else:
            change = 0

        comparisons.append(DailyComparisonPoint(
            date=date,
            value=value,
            sample_count=random.randint(1000, 3000),
            change_percent=change,
        ))
        prev_value = value

    return DayOverDayResponse(
        tenant_id="default",
        metric=metric,
        comparisons=comparisons,
    )


def get_mock_week_over_week(metric: str, lookback_weeks: int) -> WeekOverWeekResponse:
    """Generate mock week-over-week comparison."""
    import random
    random.seed(42)

    now = datetime.now(timezone.utc)
    comparisons = []
    prev_value = None

    for i in range(lookback_weeks):
        week_date = now - timedelta(weeks=lookback_weeks - i - 1)
        year = week_date.year
        week = week_date.isocalendar()[1]
        value = random.uniform(280, 420)

        if prev_value:
            change = ((value - prev_value) / prev_value) * 100
        else:
            change = 0

        comparisons.append(WeeklyComparisonPoint(
            year=year,
            week=week,
            value=value,
            sample_count=random.randint(7000, 21000),
            change_percent=change,
        ))
        prev_value = value

    return WeekOverWeekResponse(
        tenant_id="default",
        metric=metric,
        comparisons=comparisons,
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/hourly-breakdown", response_model=HourlyBreakdownResponse)
def get_hourly_breakdown(
    metric: str = Query(..., description="Metric: data_usage, battery_drain, app_usage"),
    period_days: int = Query(7, ge=1, le=30, description="Analysis period in days"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get hour-of-day analysis for a metric.

    Returns average values by hour with peak/low hour detection.
    Useful for understanding daily usage patterns.
    """
    if mock_mode:
        return get_mock_hourly_breakdown(metric)

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.temporal_analysis_loader import load_hourly_breakdown

        data = load_hourly_breakdown(metric=metric, period_days=period_days)

        hourly = [
            HourlyDataPointResponse(
                hour=h.hour,
                avg_value=h.avg_value,
                min_value=h.min_value,
                max_value=h.max_value,
                std_value=h.std_value,
                sample_count=h.sample_count,
            )
            for h in data.hourly_data
        ]

        return HourlyBreakdownResponse(
            tenant_id=tenant_id,
            metric=metric,
            hourly_data=hourly,
            peak_hours=data.peak_hours,
            low_hours=data.low_hours,
            day_night_ratio=data.day_night_ratio,
        )

    except Exception as e:
        logger.error(f"Failed to get hourly breakdown: {e}")
        return HourlyBreakdownResponse(
            tenant_id=tenant_id,
            metric=metric,
            hourly_data=[],
            peak_hours=[],
            low_hours=[],
            day_night_ratio=1.0,
        )


@router.get("/peak-detection", response_model=PeakDetectionListResponse)
def get_peak_detection(
    metric: str = Query(..., description="Metric: data_usage, battery_drain"),
    period_days: int = Query(7, ge=1, le=30, description="Analysis period"),
    std_threshold: float = Query(2.0, ge=1.0, le=5.0, description="Z-score threshold"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Detect statistically significant usage peaks.

    Identifies time periods where usage significantly deviates from normal.
    """
    if mock_mode:
        return get_mock_peak_detection(metric, period_days)

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.temporal_analysis_loader import detect_peaks

        peaks_data = detect_peaks(
            metric=metric,
            period_days=period_days,
            std_threshold=std_threshold,
        )

        peaks = [
            PeakDetectionResponse(
                timestamp=p.timestamp,
                value=p.value,
                z_score=p.z_score,
                is_significant=p.is_significant,
            )
            for p in peaks_data
        ]

        return PeakDetectionListResponse(
            tenant_id=tenant_id,
            metric=metric,
            peaks=peaks,
            total_peaks=len(peaks),
        )

    except Exception as e:
        logger.error(f"Failed to detect peaks: {e}")
        return PeakDetectionListResponse(
            tenant_id=tenant_id,
            metric=metric,
            peaks=[],
            total_peaks=0,
        )


@router.get("/comparison", response_model=TemporalComparisonResponse)
def get_temporal_comparison(
    metric: str = Query(..., description="Metric to compare"),
    period_a_start: datetime = Query(..., description="Start of period A"),
    period_a_end: datetime = Query(..., description="End of period A"),
    period_b_start: datetime = Query(..., description="Start of period B"),
    period_b_end: datetime = Query(..., description="End of period B"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Compare metrics between two time periods.

    Performs statistical comparison (t-test) to determine if change is significant.
    """
    if mock_mode:
        return get_mock_comparison(metric, period_a_start, period_a_end, period_b_start, period_b_end)

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.temporal_analysis_loader import compare_periods

        data = compare_periods(
            metric=metric,
            period_a_start=period_a_start,
            period_a_end=period_a_end,
            period_b_start=period_b_start,
            period_b_end=period_b_end,
        )

        return TemporalComparisonResponse(
            tenant_id=tenant_id,
            metric=metric,
            period_a=PeriodStatsResponse(
                start=data.period_a.start,
                end=data.period_a.end,
                avg=data.period_a.avg,
                median=data.period_a.median,
                std=data.period_a.std,
                sample_count=data.period_a.sample_count,
            ),
            period_b=PeriodStatsResponse(
                start=data.period_b.start,
                end=data.period_b.end,
                avg=data.period_b.avg,
                median=data.period_b.median,
                std=data.period_b.std,
                sample_count=data.period_b.sample_count,
            ),
            change_percent=data.change_percent,
            is_significant=data.is_significant,
            p_value=data.p_value,
        )

    except Exception as e:
        logger.error(f"Failed to compare periods: {e}")
        return TemporalComparisonResponse(
            tenant_id=tenant_id,
            metric=metric,
            period_a=PeriodStatsResponse(
                start=period_a_start, end=period_a_end, avg=0, median=0, std=0, sample_count=0
            ),
            period_b=PeriodStatsResponse(
                start=period_b_start, end=period_b_end, avg=0, median=0, std=0, sample_count=0
            ),
            change_percent=0,
            is_significant=False,
            p_value=1.0,
        )


@router.get("/day-over-day", response_model=DayOverDayResponse)
def get_day_over_day(
    metric: str = Query(..., description="Metric: data_usage, battery_drain"),
    lookback_days: int = Query(7, ge=1, le=30, description="Number of days to compare"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get day-over-day comparison.

    Shows daily values with change percentage from previous day.
    """
    if mock_mode:
        return get_mock_day_over_day(metric, lookback_days)

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.temporal_analysis_loader import get_day_over_day_comparison

        data = get_day_over_day_comparison(metric=metric, lookback_days=lookback_days)

        comparisons = [
            DailyComparisonPoint(
                date=d["date"],
                value=d["value"],
                sample_count=d["sample_count"],
                change_percent=d["change_percent"],
            )
            for d in data
        ]

        return DayOverDayResponse(
            tenant_id=tenant_id,
            metric=metric,
            comparisons=comparisons,
        )

    except Exception as e:
        logger.error(f"Failed to get day over day: {e}")
        return DayOverDayResponse(
            tenant_id=tenant_id,
            metric=metric,
            comparisons=[],
        )


@router.get("/week-over-week", response_model=WeekOverWeekResponse)
def get_week_over_week(
    metric: str = Query(..., description="Metric: data_usage, battery_drain"),
    lookback_weeks: int = Query(4, ge=2, le=12, description="Number of weeks to compare"),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get week-over-week comparison.

    Shows weekly values with change percentage from previous week.
    """
    if mock_mode:
        return get_mock_week_over_week(metric, lookback_weeks)

    tenant_id = get_tenant_id()

    try:
        from device_anomaly.data_access.temporal_analysis_loader import get_week_over_week_comparison

        data = get_week_over_week_comparison(metric=metric, lookback_weeks=lookback_weeks)

        comparisons = [
            WeeklyComparisonPoint(
                year=w["year"],
                week=w["week"],
                value=w["value"],
                sample_count=w["sample_count"],
                change_percent=w["change_percent"],
            )
            for w in data
        ]

        return WeekOverWeekResponse(
            tenant_id=tenant_id,
            metric=metric,
            comparisons=comparisons,
        )

    except Exception as e:
        logger.error(f"Failed to get week over week: {e}")
        return WeekOverWeekResponse(
            tenant_id=tenant_id,
            metric=metric,
            comparisons=[],
        )
