"""
Pydantic models for Correlation Intelligence API endpoints.
"""

from pydantic import BaseModel, Field

# =============================================================================
# Error Response Model
# =============================================================================


class CorrelationErrorDetail(BaseModel):
    """Structured error detail for correlation endpoints."""

    error_type: (
        str  # "data_unavailable", "service_offline", "computation_error", "insufficient_data"
    )
    message: str
    recommendations: list[str] = []


# =============================================================================
# Response Models
# =============================================================================


class FilterStats(BaseModel):
    """Statistics about metrics filtered during correlation computation."""

    total_input: int = Field(..., description="Total metrics before filtering")
    low_variance: int = Field(0, description="Metrics removed due to low variance")
    low_cardinality: int = Field(0, description="Metrics removed due to low cardinality")
    high_null: int = Field(0, description="Metrics removed due to high null ratio")
    passed: int = Field(..., description="Metrics that passed quality filters")


class CorrelationCell(BaseModel):
    """Single cell in correlation matrix."""

    metric_x: str
    metric_y: str
    correlation: float
    p_value: float | None = None
    sample_count: int
    method: str = "pearson"
    is_significant: bool = Field(
        False, description="Whether correlation is statistically significant (p < 0.05)"
    )


class CorrelationMatrixResponse(BaseModel):
    """Full correlation matrix response."""

    metrics: list[str]
    matrix: list[list[float]]
    p_values: list[list[float]] | None = Field(
        None, description="Matrix of p-values for each correlation"
    )
    strong_correlations: list[CorrelationCell]
    method: str
    computed_at: str
    total_samples: int
    domain_filter: str | None = None
    date_range: dict[str, str] | None = Field(
        None, description="Start and end dates used for computation"
    )
    filter_stats: FilterStats | None = Field(None, description="Statistics about filtered metrics")


class ScatterDataPoint(BaseModel):
    """Single point for scatter plot."""

    device_id: int
    x_value: float
    y_value: float
    is_anomaly: bool
    cohort: str | None = None
    timestamp: str | None = None


class ScatterPlotResponse(BaseModel):
    """Scatter plot data for two metrics."""

    metric_x: str
    metric_y: str
    points: list[ScatterDataPoint]
    correlation: float
    regression_slope: float | None = None
    regression_intercept: float | None = None
    r_squared: float | None = None
    total_points: int
    anomaly_count: int


class CausalNode(BaseModel):
    """Node in causal graph."""

    metric: str
    domain: str
    is_cause: bool
    is_effect: bool
    connection_count: int


class CausalEdge(BaseModel):
    """Edge in causal graph."""

    source: str
    target: str
    relationship: str
    strength: float
    evidence: str | None = None


class CausalGraphResponse(BaseModel):
    """Causal relationship network."""

    nodes: list[CausalNode]
    edges: list[CausalEdge]
    generated_at: str


class CorrelationInsight(BaseModel):
    """Auto-discovered correlation insight."""

    insight_id: str
    headline: str
    description: str
    metrics_involved: list[str]
    correlation_value: float
    strength: str
    direction: str
    novelty_score: float
    confidence: float
    recommendation: str | None = None


class CorrelationInsightsResponse(BaseModel):
    """Auto-generated correlation insights."""

    insights: list[CorrelationInsight]
    total_correlations_analyzed: int
    generated_at: str


class CohortCorrelationPattern(BaseModel):
    """Correlation pattern for a specific cohort."""

    cohort_id: str
    cohort_name: str
    metric_pair: list[str]
    cohort_correlation: float
    fleet_correlation: float
    deviation: float
    device_count: int
    is_anomalous: bool
    insight: str | None = None


class CohortCorrelationPatternsResponse(BaseModel):
    """Cohort-specific correlation patterns."""

    patterns: list[CohortCorrelationPattern]
    anomalous_cohorts: int
    generated_at: str


class TimeLagCorrelation(BaseModel):
    """Time-lagged correlation result."""

    metric_a: str
    metric_b: str
    lag_days: int
    correlation: float
    p_value: float | None = None
    direction: str
    insight: str


class TimeLagCorrelationsResponse(BaseModel):
    """Time-lagged correlation analysis."""

    correlations: list[TimeLagCorrelation]
    max_lag_analyzed: int
    generated_at: str


class ScatterAnomalyExplainRequest(BaseModel):
    """Request body for scatter anomaly explanation."""

    device_id: int
    metric_x: str
    metric_y: str
    x_value: float
    y_value: float


class ScatterAnomalyExplanation(BaseModel):
    """LLM-generated explanation for a scatter plot anomaly."""

    explanation: str = Field(..., description="Full explanation text")
    what_happened: str = Field(..., description="Description of what occurred")
    key_concerns: list[str] = Field(..., description="List of key concerns")
    likely_explanation: str = Field(..., description="Probable cause")
    suggested_action: str = Field(..., description="Recommended next step")
