"""Pydantic models for API requests and responses."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class AnomalyResponse(BaseModel):
    """Response model for a single anomaly."""

    id: int
    device_id: int
    timestamp: datetime
    anomaly_score: float
    anomaly_label: int
    status: str
    assigned_to: Optional[str] = None

    # Metric values
    total_battery_level_drop: Optional[float] = None
    total_free_storage_kb: Optional[float] = None
    download: Optional[float] = None
    upload: Optional[float] = None
    offline_time: Optional[float] = None
    disconnect_count: Optional[float] = None
    wifi_signal_strength: Optional[float] = None
    connection_time: Optional[float] = None

    # Feature values (as JSON string)
    feature_values_json: Optional[str] = None

    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AnomalyListResponse(BaseModel):
    """Response model for paginated anomaly list."""

    anomalies: List[AnomalyResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class AnomalyDetailResponse(AnomalyResponse):
    """Extended response model for anomaly detail page."""

    notes: Optional[str] = None
    investigation_notes: List[dict] = Field(default_factory=list)


class DeviceResponse(BaseModel):
    """Response model for device metadata."""

    device_id: int
    device_model: Optional[str] = None
    device_name: Optional[str] = None
    location: Optional[str] = None
    status: str
    last_seen: Optional[datetime] = None
    os_version: Optional[str] = None
    agent_version: Optional[str] = None

    model_config = {"from_attributes": True}


class DeviceDetailResponse(DeviceResponse):
    """Extended response model for device detail page."""

    anomaly_count: int = 0
    recent_anomalies: List[AnomalyResponse] = Field(default_factory=list)


class DashboardStatsResponse(BaseModel):
    """Response model for dashboard statistics."""

    anomalies_today: int
    devices_monitored: int
    critical_issues: int
    resolved_today: int
    open_cases: int = 0


class DashboardTrendResponse(BaseModel):
    """Response model for anomaly trends."""

    date: datetime
    anomaly_count: int


class AnomalyFilters(BaseModel):
    """Filter parameters for anomaly list."""

    device_id: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[str] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)


class ResolveAnomalyRequest(BaseModel):
    """Request model for resolving an anomaly."""

    status: str = "resolved"
    notes: Optional[str] = None


class AddNoteRequest(BaseModel):
    """Request model for adding investigation notes."""

    note: str
    action_type: Optional[str] = None


class ConnectionStatusResponse(BaseModel):
    """Response model for connection status."""

    connected: bool
    server: str
    error: Optional[str] = None
    status: str = "unknown"  # connected, disconnected, error, not_configured


class AllConnectionsStatusResponse(BaseModel):
    """Response model for all connection statuses."""

    # Core services
    backend_db: ConnectionStatusResponse  # PostgreSQL
    dw_sql: ConnectionStatusResponse  # XSight SQL Server
    mc_sql: ConnectionStatusResponse  # MobiControl SQL Server
    mobicontrol_api: ConnectionStatusResponse  # MobiControl REST API
    llm: ConnectionStatusResponse  # LLM service
    redis: ConnectionStatusResponse  # Redis for streaming
    qdrant: ConnectionStatusResponse  # Qdrant vector database
    last_checked: datetime


class TroubleshootingAdviceResponse(BaseModel):
    """Response model for LLM-generated troubleshooting advice."""

    advice: str
    summary: Optional[str] = None


class IsolationForestConfigResponse(BaseModel):
    """Response model for Isolation Forest model configuration."""

    n_estimators: int
    contamination: float
    random_state: int
    scale_features: bool
    min_variance: float
    feature_count: Optional[int] = None
    model_type: str = "isolation_forest"


class ScoreDistributionBin(BaseModel):
    """Response model for a single histogram bin."""

    bin_start: float
    bin_end: float
    count: int
    is_anomaly: bool


class ScoreDistributionResponse(BaseModel):
    """Response model for anomaly score distribution."""

    bins: List[ScoreDistributionBin]
    total_normal: int
    total_anomalies: int
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    std_score: float


class IsolationForestStatsResponse(BaseModel):
    """Response model for Isolation Forest statistics."""

    config: IsolationForestConfigResponse
    score_distribution: ScoreDistributionResponse
    total_predictions: int
    anomaly_rate: float


class LocationDataResponse(BaseModel):
    """Response model for a single location in the heatmap."""

    id: str
    name: str
    utilization: float
    baseline: float
    deviceCount: int
    activeDeviceCount: int
    region: Optional[str] = None
    anomalyCount: Optional[int] = None


class LocationHeatmapResponse(BaseModel):
    """Response model for location heatmap data."""

    locations: List[LocationDataResponse]
    attributeName: str
    totalLocations: int
    totalDevices: int


# ============================================
# Investigation Panel Types
# ============================================


class FeatureContribution(BaseModel):
    """Single feature contribution to anomaly detection."""

    feature_name: str
    feature_display_name: str
    contribution_percentage: float  # 0-100
    contribution_direction: str  # 'positive' or 'negative'
    current_value: float
    current_value_display: str
    baseline_value: float
    baseline_value_display: str
    deviation_sigma: float  # Standard deviations from mean
    percentile: float  # 0-100
    plain_text_explanation: str


class AnomalyExplanation(BaseModel):
    """Explanation data for an anomaly."""

    summary_text: str  # Human-readable one-liner
    detailed_explanation: str  # 2-3 paragraph explanation
    feature_contributions: List[FeatureContribution]
    top_contributing_features: List[str]  # Top 3-5 feature names
    explanation_method: str = "z_score"  # z_score, shap, lime, etc.
    explanation_generated_at: Optional[datetime] = None


class BaselineMetric(BaseModel):
    """Single metric comparison against baseline."""

    metric_name: str
    metric_display_name: str
    metric_unit: str
    current_value: float
    current_value_display: str
    baseline_mean: float
    baseline_std: float
    baseline_min: float
    baseline_max: float
    deviation_sigma: float
    deviation_percentage: float
    percentile_rank: float
    is_anomalous: bool
    anomaly_direction: str  # 'above', 'below', 'normal'


class BaselineConfig(BaseModel):
    """Configuration used for baseline comparison."""

    baseline_type: str  # 'rolling_average', 'fixed_period', 'peer_group'
    baseline_period_days: int
    comparison_window_hours: int
    statistical_method: str  # 'z_score', 'mad', 'iqr'
    peer_group_name: Optional[str] = None
    peer_group_size: Optional[int] = None
    baseline_calculated_at: datetime


class BaselineComparison(BaseModel):
    """Complete baseline comparison data."""

    baseline_config: BaselineConfig
    metrics: List[BaselineMetric]
    overall_deviation_score: float


class TimeSeriesDataPoint(BaseModel):
    """Single point in time series data."""

    timestamp: datetime
    value: float
    is_anomalous: bool = False


class HistoricalTimelineResponse(BaseModel):
    """Response for historical metric timeline."""

    metric_name: str
    data_points: List[TimeSeriesDataPoint]
    baseline_mean: float
    baseline_std: float
    baseline_upper: float  # μ + 2σ
    baseline_lower: float  # μ - 2σ


class EvidenceEvent(BaseModel):
    """Single event in evidence timeline."""

    event_id: str
    timestamp: datetime
    event_type: str  # app_installed, storage_change, network_connection, etc.
    event_category: str  # apps, storage, battery, network, security, system
    severity: str  # critical, high, medium, low, info
    title: str
    description: str
    details: Optional[dict] = None
    is_contributing_event: bool = False
    contribution_note: Optional[str] = None


class EvidenceHypothesis(BaseModel):
    """Evidence point for/against a hypothesis."""

    statement: str
    strength: str  # strong, moderate, weak
    source: str  # telemetry, pattern_match, inference
    linked_event_id: Optional[str] = None


class RootCauseHypothesis(BaseModel):
    """Single root cause hypothesis from AI analysis."""

    hypothesis_id: str
    title: str
    description: str
    likelihood: float  # 0-1
    evidence_for: List[EvidenceHypothesis]
    evidence_against: List[EvidenceHypothesis]
    recommended_actions: List[str]


class AIAnalysisResponse(BaseModel):
    """AI-generated root cause analysis."""

    analysis_id: str
    generated_at: datetime
    model_used: str
    primary_hypothesis: RootCauseHypothesis
    alternative_hypotheses: List[RootCauseHypothesis]
    confidence_score: float  # 0-1
    confidence_level: str  # high, medium, low, uncertain
    confidence_explanation: str
    similar_cases_analyzed: int
    feedback_received: bool = False
    feedback_rating: Optional[str] = None  # helpful, not_helpful


class RemediationSuggestion(BaseModel):
    """Suggested remediation action."""

    remediation_id: str
    title: str
    description: str
    detailed_steps: List[str]
    priority: int  # 1 = highest
    confidence_score: float
    confidence_level: str
    source: str  # learned, ai_generated, policy
    source_details: str
    historical_success_rate: Optional[float] = None
    historical_sample_size: Optional[int] = None
    estimated_impact: Optional[str] = None
    is_automated: bool = False
    automation_type: Optional[str] = None  # mobicontrol_action, script, notification


class SimilarCase(BaseModel):
    """Similar historical anomaly case."""

    case_id: str
    anomaly_id: int
    device_id: int
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    similarity_score: float
    similarity_factors: List[str]
    anomaly_type: str
    severity: str
    resolution_status: str
    resolution_summary: Optional[str] = None
    successful_remediation: Optional[str] = None
    time_to_resolution_hours: Optional[float] = None


class InvestigationPanelResponse(BaseModel):
    """Complete investigation panel data."""

    anomaly_id: int
    device_id: int

    # Core detection info
    anomaly_score: float
    severity: str
    confidence_score: float
    detected_at: datetime

    # Explanation (the "Why")
    explanation: AnomalyExplanation

    # Baseline comparison (the "How it deviates")
    baseline_comparison: Optional[BaselineComparison] = None

    # Evidence timeline
    evidence_events: List[EvidenceEvent]
    evidence_event_count: int

    # AI Analysis (if available)
    ai_analysis: Optional[AIAnalysisResponse] = None

    # Remediation suggestions
    suggested_remediations: List[RemediationSuggestion]

    # Similar cases
    similar_cases: List[SimilarCase]


class AIAnalysisFeedbackRequest(BaseModel):
    """Request to submit feedback on AI analysis."""

    rating: str  # helpful, not_helpful
    feedback_text: Optional[str] = None
    actual_root_cause: Optional[str] = None


class RemediationExecuteRequest(BaseModel):
    """Request to execute a remediation action."""

    confirm: bool = False
    custom_params: Optional[dict] = None


class RemediationOutcomeRequest(BaseModel):
    """Request to record remediation outcome."""

    outcome: str  # resolved, partially_resolved, no_effect, made_worse
    notes: Optional[str] = None


class LearnFromFixRequest(BaseModel):
    """Request to learn from a successful fix."""

    remediation_description: str
    tags: List[str] = Field(default_factory=list)

