"""
API routes for Correlation Intelligence endpoints.

Provides correlation analysis including:
- Correlation matrix computation
- Scatter plot data
- Causal graph visualization
- Auto-generated insights
- Cohort correlation patterns
- Time-lagged correlations
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from device_anomaly.api.dependencies import get_mock_mode
from device_anomaly.api.mocks.correlation_mocks import (
    get_mock_causal_graph,
    get_mock_cohort_patterns,
    get_mock_correlation_insights,
    get_mock_correlation_matrix,
    get_mock_scatter_data,
    get_mock_scatter_explanation,
    get_mock_time_lagged_correlations,
)
from device_anomaly.api.schemas.correlations import (
    CausalGraphResponse,
    CohortCorrelationPattern,
    CohortCorrelationPatternsResponse,
    CorrelationCell,
    CorrelationErrorDetail,
    CorrelationInsight,
    CorrelationInsightsResponse,
    CorrelationMatrixResponse,
    FilterStats,
    ScatterAnomalyExplainRequest,
    ScatterAnomalyExplanation,
    ScatterDataPoint,
    ScatterPlotResponse,
    TimeLagCorrelation,
    TimeLagCorrelationsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/correlations", tags=["correlations"])


# =============================================================================
# Error Helpers
# =============================================================================


def _raise_data_unavailable_error(context: str = "correlation analysis") -> None:
    """Raise HTTPException for unavailable data."""
    raise HTTPException(
        status_code=503,
        detail=CorrelationErrorDetail(
            error_type="data_unavailable",
            message=f"No telemetry data available for {context}. The database may be empty or services may be offline.",
            recommendations=[
                "Check the System page to verify database connections are online",
                "Verify data sync/ingestion is running",
                "Ensure the date range contains data",
            ],
        ).model_dump(),
    )


def _raise_computation_error(error: Exception, context: str = "correlation computation") -> None:
    """Raise HTTPException for computation failures."""
    raise HTTPException(
        status_code=500,
        detail=CorrelationErrorDetail(
            error_type="computation_error",
            message=f"Failed to compute {context}: {str(error)}",
            recommendations=[
                "Check server logs for detailed error information",
                "Verify service health on the System page",
                "Try again with different parameters",
            ],
        ).model_dump(),
    )


def _raise_insufficient_data_error(context: str, details: str = "") -> None:
    """Raise HTTPException for insufficient data."""
    raise HTTPException(
        status_code=422,
        detail=CorrelationErrorDetail(
            error_type="insufficient_data",
            message=f"Insufficient data for {context}. {details}".strip(),
            recommendations=[
                "Try a broader date range",
                "Select different metrics",
                "Verify data quality in the source database",
            ],
        ).model_dump(),
    )


def _raise_database_error(error: Exception, database: str = "telemetry") -> None:
    """Raise HTTPException for database connection failures."""
    error_str = str(error)[:200]  # Truncate long error messages
    raise HTTPException(
        status_code=503,
        detail=CorrelationErrorDetail(
            error_type="database_unavailable",
            message=f"Cannot connect to {database} database: {error_str}",
            recommendations=[
                f"Check {database} database server is running and accessible",
                "Verify database credentials in environment variables",
                "Check network connectivity to database host",
                "Review the System page for database health status",
            ],
        ).model_dump(),
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/matrix", response_model=CorrelationMatrixResponse)
def get_correlation_matrix(
    domain: str | None = Query(None, description="Filter by domain (battery, rf, throughput, usage, storage, system)"),
    method: str = Query("pearson", description="Correlation method: pearson or spearman"),
    threshold: float = Query(0.6, description="Minimum |r| for strong correlations"),
    max_metrics: int = Query(50, description="Maximum metrics to include"),
    min_variance: float = Query(0.001, description="Minimum variance to include a metric (filters constant columns)"),
    min_unique_values: int = Query(3, description="Minimum unique values required (filters binary/low-cardinality columns)"),
    min_non_null_ratio: float = Query(0.1, description="Minimum ratio of non-null values required"),
    mock_mode: bool = Depends(get_mock_mode),
) -> CorrelationMatrixResponse:
    """
    Get correlation matrix for numeric metrics.

    Returns N x N correlation matrix and list of strong correlations.
    Can be filtered by metric domain.

    Quality filters remove metrics that would produce meaningless correlations:
    - Low variance (constant or near-constant values)
    - Low cardinality (binary flags, categorical codes)
    - High null ratio (mostly missing data)
    """
    if mock_mode:
        return get_mock_correlation_matrix(domain, method)

    # Real implementation using CorrelationService
    try:
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from device_anomaly.services.correlation_service import CorrelationService

        # Load recent telemetry data
        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=30)

        df = load_unified_device_dataset(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        if df is None or df.empty:
            logger.warning("No data available for correlation matrix")
            _raise_data_unavailable_error("correlation matrix computation")

        service = CorrelationService()
        result = service.compute_correlation_matrix(
            df=df,
            method=method,
            domain_filter=domain,
            min_variance=min_variance,
            min_unique_values=min_unique_values,
            min_non_null_ratio=min_non_null_ratio,
        )

        # Check if service returned an error (e.g., insufficient metrics after filtering)
        if result.get("error"):
            filter_stats = result.get("filter_stats", {})
            details = f"After quality filtering: {filter_stats.get('high_null', 0)} metrics had too many nulls, {filter_stats.get('low_cardinality', 0)} had low cardinality, {filter_stats.get('low_variance', 0)} had low variance."
            _raise_insufficient_data_error("correlation matrix", details)

        # Convert strong correlations to response format
        strong = [
            CorrelationCell(
                metric_x=c["metric_x"],
                metric_y=c["metric_y"],
                correlation=c["correlation"],
                p_value=c.get("p_value"),
                sample_count=c.get("sample_count", 0),
                method=method,
                is_significant=c.get("is_significant", False),
            )
            for c in result.get("strong_correlations", [])
            if abs(c["correlation"]) >= threshold
        ]

        # Build filter_stats from service result
        raw_filter_stats = result.get("filter_stats")
        filter_stats_response = None
        if raw_filter_stats:
            filter_stats_response = FilterStats(
                total_input=raw_filter_stats.get("total_input", 0),
                low_variance=raw_filter_stats.get("low_variance", 0),
                low_cardinality=raw_filter_stats.get("low_cardinality", 0),
                high_null=raw_filter_stats.get("high_null", 0),
                passed=raw_filter_stats.get("passed", 0),
            )

        return CorrelationMatrixResponse(
            metrics=result.get("metrics", []),
            matrix=result.get("matrix", []),
            p_values=result.get("p_values"),
            strong_correlations=strong,
            method=method,
            computed_at=result.get("computed_at", datetime.now(UTC).isoformat()),
            total_samples=result.get("sample_count", 0),
            domain_filter=domain,
            date_range={"start": start_date.isoformat(), "end": end_date.isoformat()},
            filter_stats=filter_stats_response,
        )

    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Real correlation computation failed: {e}")
        # Check if it's a database connection error
        error_str = str(e).lower()
        if any(
            keyword in error_str
            for keyword in ["connection", "timeout", "refused", "network", "login", "authentication", "odbc", "pyodbc", "sqlalchemy"]
        ):
            _raise_database_error(e, "telemetry")
        _raise_computation_error(e, "correlation matrix")


@router.get("/scatter", response_model=ScatterPlotResponse)
def get_scatter_data(
    metric_x: str = Query(..., description="First metric name"),
    metric_y: str = Query(..., description="Second metric name"),
    color_by: str = Query("anomaly", description="Color by: anomaly or cohort"),
    limit: int = Query(500, description="Max data points"),
    mock_mode: bool = Depends(get_mock_mode),
) -> ScatterPlotResponse:
    """
    Get scatter plot data for two metrics.

    Returns data points, correlation coefficient, and regression line parameters.
    """
    if mock_mode:
        return get_mock_scatter_data(metric_x, metric_y, limit)

    # Real implementation
    try:
        import warnings

        from scipy import stats as scipy_stats

        from device_anomaly.data_access.unified_loader import load_unified_device_dataset

        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=30)

        df = load_unified_device_dataset(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        if df is None or df.empty:
            logger.warning("No data available for scatter plot")
            _raise_data_unavailable_error("scatter plot")

        if metric_x not in df.columns or metric_y not in df.columns:
            missing = []
            if metric_x not in df.columns:
                missing.append(metric_x)
            if metric_y not in df.columns:
                missing.append(metric_y)
            _raise_insufficient_data_error("scatter plot", f"Metrics not found in data: {', '.join(missing)}. Available metrics may differ from selection.")

        # Build columns list for scatter data
        cols_needed = [metric_x, metric_y]
        if "DeviceId" in df.columns:
            cols_needed.append("DeviceId")
        if "is_anomaly" in df.columns:
            cols_needed.append("is_anomaly")
        if "cohort_id" in df.columns:
            cols_needed.append("cohort_id")

        # Get valid data points (drop rows where either metric is NaN)
        valid_df = df[cols_needed].dropna(subset=[metric_x, metric_y])
        if len(valid_df) < 10:
            logger.warning(f"Insufficient data points for scatter plot: {len(valid_df)} points")
            _raise_insufficient_data_error(
                "scatter plot", f"Only {len(valid_df)} valid data points found (minimum 10 required). The selected metrics may have too many missing values."
            )

        # Sample if too many points
        if len(valid_df) > limit:
            valid_df = valid_df.sample(n=limit, random_state=42)

        # Compute correlation (handle constant input)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                corr, p_value = scipy_stats.pearsonr(valid_df[metric_x], valid_df[metric_y])
                # Handle NaN correlation (constant input)
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0

        # Compute regression (handle edge cases)
        try:
            slope, intercept, r_value, _, _ = scipy_stats.linregress(valid_df[metric_x], valid_df[metric_y])
            # Handle NaN values
            if np.isnan(slope):
                slope = 0.0
            if np.isnan(intercept):
                intercept = 0.0
            if np.isnan(r_value):
                r_value = 0.0
        except Exception:
            slope, intercept, r_value = 0.0, 0.0, 0.0

        # Build points
        points = []
        for idx, row in valid_df.iterrows():
            device_id = row.get("DeviceId", idx) if "DeviceId" in valid_df.columns else idx
            is_anomaly = row.get("is_anomaly", False) if "is_anomaly" in valid_df.columns else False
            cohort = row.get("cohort_id") if "cohort_id" in valid_df.columns else None

            points.append(
                ScatterDataPoint(
                    device_id=int(device_id) if not isinstance(device_id, int) else device_id,
                    x_value=round(float(row[metric_x]), 2),
                    y_value=round(float(row[metric_y]), 2),
                    is_anomaly=bool(is_anomaly),
                    cohort=str(cohort) if cohort else None,
                )
            )

        anomaly_count = sum(1 for p in points if p.is_anomaly)

        return ScatterPlotResponse(
            metric_x=metric_x,
            metric_y=metric_y,
            points=points,
            correlation=round(corr, 3),
            regression_slope=round(slope, 4),
            regression_intercept=round(intercept, 4),
            r_squared=round(r_value**2, 4),
            total_points=len(points),
            anomaly_count=anomaly_count,
        )

    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Real scatter data failed: {e}")
        _raise_computation_error(e, "scatter plot")


@router.get("/causal-graph", response_model=CausalGraphResponse)
def get_causal_graph(
    include_inferred: bool = Query(True, description="Include correlation-inferred edges"),
    mock_mode: bool = Depends(get_mock_mode),
) -> CausalGraphResponse:
    """
    Get causal relationship network.

    Returns nodes and edges representing known causal relationships
    from domain knowledge (RootCauseAnalyzer).
    """
    if mock_mode:
        return get_mock_causal_graph()

    # Real implementation would use RootCauseAnalyzer's causal graph
    try:
        from device_anomaly.insights.root_cause import RootCauseAnalyzer

        RootCauseAnalyzer()
        # Build graph from analyzer._causal_graph
        # For now, mock data matches the actual causal graph
    except Exception as e:
        logger.warning(f"Failed to load RootCauseAnalyzer: {e}")

    return get_mock_causal_graph()


@router.get("/insights", response_model=CorrelationInsightsResponse)
def get_correlation_insights(
    top_k: int = Query(10, description="Number of top insights to return"),
    min_strength: float = Query(0.5, description="Minimum correlation strength"),
    mock_mode: bool = Depends(get_mock_mode),
) -> CorrelationInsightsResponse:
    """
    Get auto-discovered correlation insights.

    Returns ranked list of insights about metric relationships,
    including strength, direction, and recommendations.
    """
    if mock_mode:
        return get_mock_correlation_insights()

    # Real implementation
    try:
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from device_anomaly.services.correlation_service import CorrelationService

        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=30)

        df = load_unified_device_dataset(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        if df is None or df.empty:
            logger.warning("No data available for correlation insights")
            _raise_data_unavailable_error("correlation insights")

        service = CorrelationService()
        insights_data = service.generate_correlation_insights(
            df=df,
            top_k=top_k,
            min_strength=min_strength,
        )

        insights = [
            CorrelationInsight(
                insight_id=i.get("insight_id", f"ins_{idx}"),
                headline=i.get("headline", ""),
                description=i.get("description", ""),
                metrics_involved=i.get("metrics_involved", []),
                correlation_value=i.get("correlation_value", 0),
                strength=i.get("strength", "weak"),
                direction=i.get("direction", "positive"),
                novelty_score=i.get("novelty_score", 0.5),
                # Use p_value directly if available, otherwise use confidence field
                # p-value < 0.05 means high confidence, so confidence = 1 - p_value
                confidence=1 - i.get("p_value", i.get("confidence", 0.05)),
                recommendation=i.get("recommendation"),
            )
            for idx, i in enumerate(insights_data)
        ]

        return CorrelationInsightsResponse(
            insights=insights,
            total_correlations_analyzed=len(df.columns) * (len(df.columns) - 1) // 2,
            generated_at=datetime.now(UTC).isoformat(),
        )

    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Real insights generation failed: {e}")
        _raise_computation_error(e, "correlation insights")


@router.get("/cohort-patterns", response_model=CohortCorrelationPatternsResponse)
def get_cohort_correlation_patterns(
    metric_pair: str | None = Query(None, description="Specific metric pair (comma-separated)"),
    mock_mode: bool = Depends(get_mock_mode),
) -> CohortCorrelationPatternsResponse:
    """
    Get cohort-specific correlation patterns.

    Identifies cohorts with unusual correlation patterns compared to fleet average.
    """
    if mock_mode:
        return get_mock_cohort_patterns()

    # Real implementation
    try:
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from device_anomaly.services.correlation_service import CorrelationService

        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=30)

        df = load_unified_device_dataset(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        if df is None or df.empty:
            logger.warning("No data available for cohort pattern analysis")
            _raise_data_unavailable_error("cohort correlation patterns")

        service = CorrelationService()

        # Default metric pairs to analyze
        # Use metrics that actually exist in the database
        metric_pairs_to_analyze = [
            ("TotalBatteryLevelDrop", "TotalDischargeTime_Sec"),
            ("StorageUtilizationPct", "RAMUtilizationPct"),
            ("TotalBatteryLevelDrop", "StorageUtilizationPct"),
            ("SecurityScore", "CompositeSecurityScore"),
        ]

        # Use specific pair if provided
        if metric_pair:
            parts = metric_pair.split(",")
            if len(parts) == 2:
                metric_pairs_to_analyze = [(parts[0].strip(), parts[1].strip())]

        all_patterns = []
        for pair in metric_pairs_to_analyze:
            if pair[0] not in df.columns or pair[1] not in df.columns:
                continue

            patterns = service.compute_cohort_correlations(
                df=df,
                metric_pair=pair,
            )
            all_patterns.extend(patterns)

        # Convert to response format
        response_patterns = [
            CohortCorrelationPattern(
                cohort_id=p.cohort_id,
                cohort_name=p.cohort_name,
                metric_pair=p.metric_pair,
                cohort_correlation=p.cohort_correlation,
                fleet_correlation=p.fleet_correlation,
                deviation=p.deviation,
                device_count=p.device_count,
                is_anomalous=p.is_anomalous,
                insight=p.insight,
            )
            for p in all_patterns
        ]

        anomalous_count = sum(1 for p in response_patterns if p.is_anomalous)

        return CohortCorrelationPatternsResponse(
            patterns=response_patterns,
            anomalous_cohorts=anomalous_count,
            generated_at=datetime.now(UTC).isoformat(),
        )

    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Real cohort pattern analysis failed: {e}")
        _raise_computation_error(e, "cohort correlation patterns")


# =============================================================================
# Scatter Anomaly Explanation Endpoint
# =============================================================================


def _generate_scatter_explanation_prompt(
    device_id: int,
    metric_x: str,
    metric_y: str,
    x_value: float,
    y_value: float,
    regression_slope: float | None = None,
    regression_intercept: float | None = None,
    correlation: float | None = None,
    x_mean: float | None = None,
    y_mean: float | None = None,
    x_std: float | None = None,
    y_std: float | None = None,
) -> str:
    """Build the LLM prompt for scatter anomaly explanation."""
    # Calculate expected Y from regression if available
    expected_y = None
    residual = None
    if regression_slope is not None and regression_intercept is not None:
        expected_y = regression_intercept + regression_slope * x_value
        residual = y_value - expected_y

    # Calculate z-scores if stats available
    x_z = None
    y_z = None
    if x_mean is not None and x_std is not None and x_std > 0:
        x_z = (x_value - x_mean) / x_std
    if y_mean is not None and y_std is not None and y_std > 0:
        y_z = (y_value - y_mean) / y_std

    context_parts = [
        f"Device ID: {device_id}",
        f"Metric X: {metric_x} = {x_value:.2f}",
        f"Metric Y: {metric_y} = {y_value:.2f}",
    ]
    if correlation is not None:
        context_parts.append(f"Correlation (r): {correlation:.3f}")
    if expected_y is not None:
        context_parts.append(f"Expected Y (from regression): {expected_y:.2f}")
    if residual is not None:
        context_parts.append(f"Residual (deviation): {residual:.2f}")
    if x_z is not None:
        context_parts.append(f"X z-score: {x_z:.2f}")
    if y_z is not None:
        context_parts.append(f"Y z-score: {y_z:.2f}")
    if x_mean is not None and x_std is not None:
        context_parts.append(f"X typical range: {x_mean:.2f} ± {x_std:.2f}")
    if y_mean is not None and y_std is not None:
        context_parts.append(f"Y typical range: {y_mean:.2f} ± {y_std:.2f}")

    context_str = "\n".join(context_parts)

    prompt = f"""<role>
You are a device health analyst explaining why a specific device was flagged as an anomaly in a scatter plot visualization. You are helping warehouse supervisors and IT staff understand device behavior.
</role>

<output_format>
Provide your response in exactly this structure:

WHAT HAPPENED
[2-3 sentences describing what the data shows in plain English]

KEY CONCERNS
- [First concern]
- [Second concern]
- [Third concern if applicable]

LIKELY EXPLANATION
[2-3 sentences suggesting what might have caused this anomaly]

SUGGESTED ACTION
[1-2 specific, actionable recommendations]
</output_format>

<anomaly_context>
{context_str}

This device appears as an outlier in the scatter plot comparing {metric_x} and {metric_y}.
The scatter plot shows how these two metrics typically correlate across the device fleet.
This device's data point deviates significantly from the expected pattern.
</anomaly_context>

<instructions>
1. Explain why this particular combination of {metric_x} and {metric_y} values is unusual
2. Consider what device conditions could cause these specific metric values together
3. Focus on practical insights an IT administrator would find useful
4. Keep your response under 250 words total
5. Be specific about the metrics involved, not generic
</instructions>"""

    return prompt


@router.post("/scatter/explain", response_model=ScatterAnomalyExplanation)
def explain_scatter_anomaly(
    request: ScatterAnomalyExplainRequest,
    mock_mode: bool = Depends(get_mock_mode),
) -> ScatterAnomalyExplanation:
    """
    Generate an LLM explanation for a scatter plot anomaly point.

    Provides contextual analysis of why a specific device appears as an
    anomaly in the correlation scatter plot, including what happened,
    key concerns, likely explanation, and suggested action.
    """
    if mock_mode:
        return get_mock_scatter_explanation(
            request.device_id,
            request.metric_x,
            request.metric_y,
            request.x_value,
            request.y_value,
        )

    # Real implementation: get additional context and generate explanation
    try:
        from device_anomaly.llm.client import get_default_llm_client, strip_thinking_tags

        # Try to get scatter data for context (regression line, stats)
        regression_slope = None
        regression_intercept = None
        correlation = None
        x_mean = None
        y_mean = None
        x_std = None
        y_std = None

        try:
            scatter_response = get_mock_scatter_data(request.metric_x, request.metric_y, 500)
            regression_slope = scatter_response.regression_slope
            regression_intercept = scatter_response.regression_intercept
            correlation = scatter_response.correlation

            # Calculate basic stats from points
            x_values = [p.x_value for p in scatter_response.points]
            y_values = [p.y_value for p in scatter_response.points]
            if x_values:
                x_mean = np.mean(x_values)
                x_std = np.std(x_values)
            if y_values:
                y_mean = np.mean(y_values)
                y_std = np.std(y_values)
        except Exception as e:
            logger.warning(f"Could not load scatter context: {e}")

        # Build prompt
        prompt = _generate_scatter_explanation_prompt(
            device_id=request.device_id,
            metric_x=request.metric_x,
            metric_y=request.metric_y,
            x_value=request.x_value,
            y_value=request.y_value,
            regression_slope=regression_slope,
            regression_intercept=regression_intercept,
            correlation=correlation,
            x_mean=x_mean,
            y_mean=y_mean,
            x_std=x_std,
            y_std=y_std,
        )

        # Generate explanation
        llm_client = get_default_llm_client()
        raw_response = llm_client.generate(prompt, max_tokens=500, temperature=0.3)
        response_text = strip_thinking_tags(raw_response)

        # Parse the response into structured sections
        what_happened = ""
        key_concerns: list[str] = []
        likely_explanation = ""
        suggested_action = ""

        # Simple parsing of structured response
        current_section = ""
        for line in response_text.split("\n"):
            line = line.strip()
            if line.upper().startswith("WHAT HAPPENED"):
                current_section = "what"
            elif line.upper().startswith("KEY CONCERNS"):
                current_section = "concerns"
            elif line.upper().startswith("LIKELY EXPLANATION"):
                current_section = "explanation"
            elif line.upper().startswith("SUGGESTED ACTION"):
                current_section = "action"
            elif line:
                if current_section == "what":
                    what_happened += " " + line
                elif current_section == "concerns":
                    if line.startswith("-") or line.startswith("•"):
                        key_concerns.append(line.lstrip("-•").strip())
                    elif key_concerns:
                        key_concerns[-1] += " " + line
                    else:
                        key_concerns.append(line)
                elif current_section == "explanation":
                    likely_explanation += " " + line
                elif current_section == "action":
                    suggested_action += " " + line

        # Fallback if parsing failed
        if not what_happened:
            what_happened = response_text[:300] if len(response_text) > 300 else response_text
        if not key_concerns:
            key_concerns = [
                f"Unusual {request.metric_x} value: {request.x_value:.2f}",
                f"Unusual {request.metric_y} value: {request.y_value:.2f}",
            ]
        if not likely_explanation:
            likely_explanation = "Unable to determine specific cause. Review device logs for more context."
        if not suggested_action:
            suggested_action = "Investigate device activity and compare with similar devices."

        return ScatterAnomalyExplanation(
            explanation=response_text,
            what_happened=what_happened.strip(),
            key_concerns=key_concerns[:5],  # Limit to 5 concerns
            likely_explanation=likely_explanation.strip(),
            suggested_action=suggested_action.strip(),
        )

    except Exception as e:
        logger.warning(f"LLM scatter explanation failed: {e}, returning mock")
        return get_mock_scatter_explanation(
            request.device_id,
            request.metric_x,
            request.metric_y,
            request.x_value,
            request.y_value,
        )


@router.get("/time-lagged", response_model=TimeLagCorrelationsResponse)
def get_time_lagged_correlations(
    max_lag: int = Query(7, description="Maximum lag in days"),
    min_correlation: float = Query(0.3, description="Minimum correlation threshold"),
    mock_mode: bool = Depends(get_mock_mode),
) -> TimeLagCorrelationsResponse:
    """
    Get time-lagged correlations for predictive insights.

    Analyzes how metrics at time T correlate with other metrics at time T+lag.
    Useful for predictive alerting.
    """
    if mock_mode:
        return get_mock_time_lagged_correlations()

    # Real implementation
    try:
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from device_anomaly.services.correlation_service import CorrelationService

        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=60)  # Need more data for lag analysis

        df = load_unified_device_dataset(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        if df is None or df.empty:
            logger.warning("No data available for time-lagged correlation analysis")
            _raise_data_unavailable_error("time-lagged correlations")

        service = CorrelationService()
        lagged_results = service.compute_time_lagged_correlations(
            df=df,
            max_lag_days=max_lag,
        )

        # Filter by minimum correlation
        filtered = [r for r in lagged_results if abs(r.correlation) >= min_correlation]

        # Convert to response format
        correlations = [
            TimeLagCorrelation(
                metric_a=r.metric_a,
                metric_b=r.metric_b,
                lag_days=r.lag_days,
                correlation=r.correlation,
                p_value=r.p_value,
                direction=r.direction,
                insight=r.insight,
            )
            for r in filtered
        ]

        return TimeLagCorrelationsResponse(
            correlations=correlations,
            max_lag_analyzed=max_lag,
            generated_at=datetime.now(UTC).isoformat(),
        )

    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"Real time-lagged analysis failed: {e}")
        _raise_computation_error(e, "time-lagged correlations")
