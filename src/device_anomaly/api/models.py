"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AnomalyResponse(BaseModel):
    """Response model for a single anomaly."""

    id: int
    device_id: int
    device_name: str | None = None
    timestamp: datetime
    anomaly_score: float
    anomaly_label: int
    status: str
    assigned_to: str | None = None

    # Metric values
    total_battery_level_drop: float | None = None
    total_free_storage_kb: float | None = None
    download: float | None = None
    upload: float | None = None
    offline_time: float | None = None
    disconnect_count: float | None = None
    wifi_signal_strength: float | None = None
    connection_time: float | None = None

    # Feature values (as JSON string)
    feature_values_json: str | None = None

    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AnomalyListResponse(BaseModel):
    """Response model for paginated anomaly list."""

    anomalies: list[AnomalyResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class AnomalyDetailResponse(AnomalyResponse):
    """Extended response model for anomaly detail page."""

    notes: str | None = None
    investigation_notes: list[dict] = Field(default_factory=list)


class DeviceResponse(BaseModel):
    """Response model for device metadata."""

    device_id: int
    device_model: str | None = None
    device_name: str | None = None
    location: str | None = None
    store_id: str | None = None
    status: str
    last_seen: datetime | None = None
    os_version: str | None = None
    agent_version: str | None = None
    custom_attributes: dict[str, str] | None = None

    model_config = {"from_attributes": True}


class DeviceDetailResponse(DeviceResponse):
    """Extended response model for device detail page."""

    anomaly_count: int = 0
    recent_anomalies: list[AnomalyResponse] = Field(default_factory=list)


class DashboardStatsResponse(BaseModel):
    """Response model for dashboard statistics."""

    anomalies_today: int
    devices_monitored: int
    critical_issues: int
    resolved_today: int
    open_cases: int = 0
    total_anomalies: int = 0  # Total count of all anomalies in database
    # Fleet size breakdown
    total_devices: int = 0  # Total devices in database (all time)
    active_devices: int = 0  # Devices active in last 30 days


class DashboardTrendResponse(BaseModel):
    """Response model for anomaly trends."""

    date: datetime
    anomaly_count: int


class AnomalyFilters(BaseModel):
    """Filter parameters for anomaly list."""

    device_id: int | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    status: str | None = None
    min_score: float | None = None
    max_score: float | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)


class ResolveAnomalyRequest(BaseModel):
    """Request model for resolving an anomaly."""

    status: str = "resolved"
    notes: str | None = None


class AddNoteRequest(BaseModel):
    """Request model for adding investigation notes."""

    note: str
    action_type: str | None = None


class ConnectionStatusResponse(BaseModel):
    """Response model for connection status."""

    connected: bool
    server: str
    error: str | None = None
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
    summary: str | None = None


class IsolationForestConfigResponse(BaseModel):
    """Response model for Isolation Forest model configuration."""

    n_estimators: int
    contamination: float
    random_state: int
    scale_features: bool
    min_variance: float
    feature_count: int | None = None
    model_type: str = "isolation_forest"


class ScoreDistributionBin(BaseModel):
    """Response model for a single histogram bin."""

    bin_start: float
    bin_end: float
    count: int
    is_anomaly: bool


class ScoreDistributionResponse(BaseModel):
    """Response model for anomaly score distribution."""

    bins: list[ScoreDistributionBin]
    total_normal: int
    total_anomalies: int
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    std_score: float


class FeedbackStatsResponse(BaseModel):
    """Response model for model feedback statistics."""

    total_feedback: int
    false_positives: int
    confirmed_anomalies: int
    projected_accuracy_gain: float
    last_retrain: str | None = None


class IsolationForestStatsResponse(BaseModel):
    """Response model for Isolation Forest statistics."""

    config: IsolationForestConfigResponse
    defaults: IsolationForestConfigResponse | None = None
    score_distribution: ScoreDistributionResponse
    total_predictions: int
    anomaly_rate: float
    feedback_stats: FeedbackStatsResponse | None = None


class LocationDataResponse(BaseModel):
    """Response model for a single location in the heatmap."""

    id: str
    name: str
    utilization: float
    baseline: float
    deviceCount: int
    activeDeviceCount: int
    region: str | None = None
    anomalyCount: int | None = None


class LocationHeatmapResponse(BaseModel):
    """Response model for location heatmap data."""

    locations: list[LocationDataResponse]
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
    feature_contributions: list[FeatureContribution]
    top_contributing_features: list[str]  # Top 3-5 feature names
    explanation_method: str = "z_score"  # z_score, shap, lime, etc.
    explanation_generated_at: datetime | None = None


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
    peer_group_name: str | None = None
    peer_group_size: int | None = None
    baseline_calculated_at: datetime


class BaselineComparison(BaseModel):
    """Complete baseline comparison data."""

    baseline_config: BaselineConfig
    metrics: list[BaselineMetric]
    overall_deviation_score: float


class TimeSeriesDataPoint(BaseModel):
    """Single point in time series data."""

    timestamp: datetime
    value: float
    is_anomalous: bool = False


class HistoricalTimelineResponse(BaseModel):
    """Response for historical metric timeline."""

    metric_name: str
    data_points: list[TimeSeriesDataPoint]
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
    details: dict | None = None
    is_contributing_event: bool = False
    contribution_note: str | None = None


class EvidenceHypothesis(BaseModel):
    """Evidence point for/against a hypothesis."""

    statement: str
    strength: str  # strong, moderate, weak
    source: str  # telemetry, pattern_match, inference
    linked_event_id: str | None = None


class RootCauseHypothesis(BaseModel):
    """Single root cause hypothesis from AI analysis."""

    hypothesis_id: str
    title: str
    description: str
    likelihood: float  # 0-1
    evidence_for: list[EvidenceHypothesis]
    evidence_against: list[EvidenceHypothesis]
    recommended_actions: list[str]


class AIAnalysisResponse(BaseModel):
    """AI-generated root cause analysis."""

    analysis_id: str
    generated_at: datetime
    model_used: str
    analysis_source: str = "llm"  # llm, rule_based, unavailable
    primary_hypothesis: RootCauseHypothesis
    alternative_hypotheses: list[RootCauseHypothesis]
    confidence_score: float  # 0-1
    confidence_level: str  # high, medium, low, uncertain
    confidence_explanation: str
    similar_cases_analyzed: int
    feedback_received: bool = False
    feedback_rating: str | None = None  # helpful, not_helpful


class RemediationSuggestion(BaseModel):
    """Suggested remediation action."""

    remediation_id: str
    title: str
    description: str
    detailed_steps: list[str]
    priority: int  # 1 = highest
    confidence_score: float
    confidence_level: str
    source: str  # learned, ai_generated, policy
    source_details: str
    historical_success_rate: float | None = None
    historical_sample_size: int | None = None
    estimated_impact: str | None = None
    is_automated: bool = False
    automation_type: str | None = None  # mobicontrol_action, script, notification


class SimilarCase(BaseModel):
    """Similar historical anomaly case."""

    case_id: str
    anomaly_id: int
    device_id: int
    device_name: str | None = None
    detected_at: datetime
    resolved_at: datetime | None = None
    similarity_score: float
    similarity_factors: list[str]
    anomaly_type: str
    severity: str
    resolution_status: str
    resolution_summary: str | None = None
    successful_remediation: str | None = None
    time_to_resolution_hours: float | None = None


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
    baseline_comparison: BaselineComparison | None = None

    # Evidence timeline
    evidence_events: list[EvidenceEvent]
    evidence_event_count: int

    # AI Analysis (if available)
    ai_analysis: AIAnalysisResponse | None = None

    # Remediation suggestions
    suggested_remediations: list[RemediationSuggestion]

    # Similar cases
    similar_cases: list[SimilarCase]


class AIAnalysisFeedbackRequest(BaseModel):
    """Request to submit feedback on AI analysis."""

    rating: str  # helpful, not_helpful
    feedback_text: str | None = None
    actual_root_cause: str | None = None


class RemediationExecuteRequest(BaseModel):
    """Request to execute a remediation action."""

    confirm: bool = False
    custom_params: dict | None = None


class RemediationOutcomeRequest(BaseModel):
    """Request to record remediation outcome."""

    outcome: str  # resolved, partially_resolved, no_effect, made_worse
    notes: str | None = None


class LearnFromFixRequest(BaseModel):
    """Request to learn from a successful fix."""

    remediation_description: str
    tags: list[str] = Field(default_factory=list)


class SuccessResponse(BaseModel):
    """Generic success response with message."""

    success: bool = True
    message: str


class FeedbackResponse(SuccessResponse):
    """Response after submitting AI feedback."""


class RemediationOutcomeResponse(SuccessResponse):
    """Response after recording a remediation outcome."""

    outcome_id: int


class LearnRemediationResponse(SuccessResponse):
    """Response after learning from a fix."""

    learned_remediation_id: int
    current_confidence: float | None = None
    initial_confidence: float | None = None


# ============================================
# Smart Anomaly Grouping Types
# ============================================


class AnomalyGroupMember(BaseModel):
    """Single anomaly within a group."""

    anomaly_id: int
    device_id: int
    anomaly_score: float
    severity: str  # critical, high, medium, low
    status: str  # open, investigating, resolved, false_positive
    timestamp: datetime
    device_name: str | None = None
    device_model: str | None = None
    location: str | None = None
    primary_metric: str | None = None  # Main contributing factor


class AnomalyGroup(BaseModel):
    """Smart-grouped collection of related anomalies."""

    group_id: str  # Unique identifier for the group
    group_name: str  # Human-readable name, e.g., "Battery Drain Issues (5 devices)"
    group_category: str  # InsightCategory value or custom category
    group_type: str  # category_match, remediation_match, similarity_cluster, temporal_cluster
    severity: str  # Worst severity in the group
    total_count: int  # Total anomalies in group
    open_count: int  # Open + investigating count
    device_count: int  # Unique devices affected

    # Optional group context
    suggested_remediation: RemediationSuggestion | None = None
    common_location: str | None = None
    common_device_model: str | None = None
    time_range_start: datetime
    time_range_end: datetime

    # Sample anomalies for preview (first 5)
    sample_anomalies: list[AnomalyGroupMember]

    # Explanation of why these are grouped
    grouping_factors: list[
        str
    ]  # e.g., ["Same category: BATTERY_RAPID_DRAIN", "Similar score range"]
    avg_similarity_score: float | None = None


class GroupedAnomaliesResponse(BaseModel):
    """Response containing grouped anomalies."""

    groups: list[AnomalyGroup]
    total_anomalies: int
    total_groups: int
    ungrouped_count: int  # Anomalies that don't fit any group
    ungrouped_anomalies: list[AnomalyGroupMember] = Field(default_factory=list)
    grouping_method: str = "smart_auto"
    computed_at: datetime

    # Impact metrics for hero card
    coverage_percent: float = 0.0  # % of anomalies in groups
    top_impact_group_id: str | None = None  # Group with highest impact (count * severity)
    top_impact_group_name: str | None = None  # Name of top impact group


class BulkActionRequest(BaseModel):
    """Request for bulk status changes on anomalies."""

    action: str  # resolve, dismiss, investigate, false_positive
    anomaly_ids: list[int]  # IDs of anomalies to update
    notes: str | None = None  # Optional note to add


class BulkActionResponse(BaseModel):
    """Response for bulk action."""

    success: bool
    affected_count: int
    failed_ids: list[int] = Field(default_factory=list)
    message: str


# ============================================
# Insight Impacted Devices Types
# ============================================


class ImpactedDeviceResponse(BaseModel):
    """Device affected by an insight."""

    device_id: int
    device_name: str | None = None
    device_model: str | None = None
    location: str | None = None
    status: str = "unknown"
    last_seen: datetime | None = None
    os_version: str | None = None
    anomaly_count: int = 0
    severity: str | None = None
    primary_metric: str | None = None


class DeviceGroupingResponse(BaseModel):
    """A grouping of devices."""

    group_key: str
    group_label: str
    device_count: int
    devices: list[ImpactedDeviceResponse]


class InsightDevicesResponse(BaseModel):
    """Response containing devices affected by an insight with grouping options."""

    insight_id: str
    insight_headline: str
    insight_category: str | None = None
    insight_severity: str | None = None
    total_devices: int
    devices: list[ImpactedDeviceResponse]
    groupings: dict[str, list[DeviceGroupingResponse]]
    ai_pattern_analysis: str | None = None
    generated_at: datetime
