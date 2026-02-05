"""Database schema for storing anomaly detection results."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class AnomalyStatus(StrEnum):
    """Status of an anomaly investigation."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class AnomalyResult(Base):
    """Table for storing anomaly detection results."""

    __tablename__ = "anomaly_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    device_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    anomaly_score = Column(Float, nullable=False)
    anomaly_label = Column(Integer, nullable=False)  # -1 for anomaly, 1 for normal

    # Metric values at anomaly time
    total_battery_level_drop = Column(Float)
    total_free_storage_kb = Column(Float)
    download = Column(Float)
    upload = Column(Float)
    offline_time = Column(Float)
    disconnect_count = Column(Float)
    wifi_signal_strength = Column(Float)
    connection_time = Column(Float)

    # Feature values (for reference)
    feature_values_json = Column(Text)  # JSON string of feature values

    # Status and investigation
    status = Column(String(20), default=AnomalyStatus.OPEN.value, index=True)
    assigned_to = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Unique constraint: prevent duplicate anomalies for same device/timestamp
    # This ensures re-scoring the same data updates existing records rather than
    # creating duplicates, which prevents anomaly count inflation over time.
    __table_args__ = (
        Index(
            "idx_anomaly_unique_device_timestamp",
            "tenant_id",
            "device_id",
            "timestamp",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<AnomalyResult(id={self.id}, device_id={self.device_id}, timestamp={self.timestamp}, score={self.anomaly_score})>"


class DeviceMetadata(Base):
    """Table for storing device metadata."""

    __tablename__ = "device_metadata"

    device_id = Column(Integer, primary_key=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    device_model = Column(String(100), nullable=True)
    device_name = Column(String(200), nullable=True)
    location = Column(String(200), nullable=True)
    status = Column(String(20), default="unknown")  # online, offline, unknown
    last_seen = Column(DateTime, nullable=True)

    # Software versions
    os_version = Column(String(50), nullable=True)
    agent_version = Column(String(50), nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<DeviceMetadata(device_id={self.device_id}, model={self.device_model})>"


class InvestigationNote(Base):
    """Table for storing investigation notes and actions."""

    __tablename__ = "investigation_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    anomaly_id = Column(Integer, nullable=False, index=True)
    user = Column(String(100), nullable=False)
    note = Column(Text, nullable=False)
    action_type = Column(String(50), nullable=True)  # e.g., "status_change", "assignment", "note"

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<InvestigationNote(id={self.id}, anomaly_id={self.anomaly_id}, user={self.user})>"


class TroubleshootingCache(Base):
    """Table for caching LLM-generated troubleshooting advice."""

    __tablename__ = "troubleshooting_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    error_signature = Column(String(255), nullable=False, index=True)  # Hash of error pattern
    error_pattern = Column(Text, nullable=False)  # JSON of error details for reference
    advice = Column(Text, nullable=False)  # LLM-generated troubleshooting advice
    summary = Column(String(500), nullable=True)  # Brief summary
    service_type = Column(String(50), nullable=True, index=True)  # e.g., "sql", "api", "llm"
    use_count = Column(Integer, default=0, nullable=False)  # Track how often this advice is reused
    last_used = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_tc_tenant_signature", "tenant_id", "error_signature", unique=True),
    )

    def __repr__(self) -> str:
        return f"<TroubleshootingCache(id={self.id}, signature={self.error_signature[:20]}..., uses={self.use_count})>"


class TrainingRun(Base):
    """Table for persisting training run metadata."""

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False, unique=True, index=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)
    status = Column(String(20), nullable=False)
    config_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    artifacts_json = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Training data versioning
    dataset_version_id = Column(
        Integer, nullable=True, index=True
    )  # FK to training_dataset_versions

    def __repr__(self) -> str:
        return f"<TrainingRun(run_id={self.run_id}, status={self.status})>"


class TrainingDatasetVersion(Base):
    """Table for versioning training datasets."""

    __tablename__ = "training_dataset_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    version_tag = Column(String(100), nullable=False, index=True)  # e.g., "v20241230_120000"

    # Data source info
    data_source = Column(String(50), nullable=False)  # e.g., "xsight", "mobicontrol", "unified"
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)

    # Dataset statistics
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    device_count = Column(Integer, nullable=True)
    feature_columns_json = Column(Text, nullable=True)  # JSON array of column names

    # Data integrity
    data_hash = Column(String(64), nullable=True)  # SHA256 of dataset for integrity check
    schema_hash = Column(String(64), nullable=True)  # SHA256 of column names + types

    # Metadata
    description = Column(Text, nullable=True)
    query_used = Column(Text, nullable=True)  # SQL query or parameters used to fetch data
    parameters_json = Column(Text, nullable=True)  # JSON object with loading parameters

    # Status tracking
    is_active = Column(Boolean, default=True, nullable=False)  # False if superseded or deprecated
    superseded_by_id = Column(Integer, nullable=True)  # FK to newer version

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(100), nullable=True)

    __table_args__ = (
        Index("idx_dataset_version_tag", "tenant_id", "version_tag", unique=True),
        Index("idx_dataset_version_dates", "start_date", "end_date"),
    )

    def __repr__(self) -> str:
        return f"<TrainingDatasetVersion(id={self.id}, version={self.version_tag}, rows={self.row_count})>"


class ModelRegistry(Base):
    """Table for registering trained models with versioning support."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)  # e.g., "anomaly_detector_v1"
    model_version = Column(String(50), nullable=False)  # e.g., "v20241230_120000"

    # Associations
    training_run_id = Column(String(50), nullable=True, index=True)  # FK to training_runs.run_id
    dataset_version_id = Column(
        Integer, nullable=True, index=True
    )  # FK to training_dataset_versions

    # Model info
    model_type = Column(String(50), nullable=False)  # e.g., "isolation_forest", "autoencoder"
    algorithm = Column(String(100), nullable=True)  # e.g., "IsolationForest_sklearn"
    framework = Column(String(50), nullable=True)  # e.g., "sklearn", "pytorch", "onnx"

    # File paths (relative to model storage root)
    model_path = Column(String(500), nullable=True)  # Primary model file
    onnx_path = Column(String(500), nullable=True)  # ONNX export
    config_path = Column(String(500), nullable=True)  # Model config
    baselines_path = Column(String(500), nullable=True)  # Baselines file

    # Performance metrics (for quick comparison)
    validation_auc = Column(Float, nullable=True)
    anomaly_rate = Column(Float, nullable=True)
    feature_count = Column(Integer, nullable=True)
    train_rows = Column(Integer, nullable=True)

    # Deployment status
    stage = Column(
        String(20), default="development", nullable=False
    )  # development, staging, production, archived
    is_active = Column(Boolean, default=False, nullable=False)  # Currently deployed model
    deployed_at = Column(DateTime, nullable=True)
    deployed_by = Column(String(100), nullable=True)

    # Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(100), nullable=True)
    archived_at = Column(DateTime, nullable=True)
    archive_reason = Column(Text, nullable=True)

    __table_args__ = (
        Index(
            "idx_model_registry_name_version",
            "tenant_id",
            "model_name",
            "model_version",
            unique=True,
        ),
        Index("idx_model_registry_active", "tenant_id", "is_active"),
        Index("idx_model_registry_stage", "tenant_id", "stage"),
    )

    def __repr__(self) -> str:
        return f"<ModelRegistry(id={self.id}, name={self.model_name}, version={self.model_version}, stage={self.stage})>"


class AnomalyExplanationCache(Base):
    """Table for caching AI-generated anomaly explanations."""

    __tablename__ = "anomaly_explanation_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    anomaly_id = Column(Integer, nullable=False, index=True)

    # Explanation content
    summary_text = Column(Text, nullable=False)
    detailed_explanation = Column(Text, nullable=False)
    feature_contributions_json = Column(Text, nullable=False)  # JSON array
    top_contributing_features = Column(Text, nullable=False)  # JSON array

    # AI Analysis (root cause)
    ai_analysis_json = Column(Text, nullable=True)  # Full AI analysis JSON
    ai_model_used = Column(String(100), nullable=True)

    # Feedback tracking
    feedback_rating = Column(String(20), nullable=True)  # helpful, not_helpful
    feedback_text = Column(Text, nullable=True)
    actual_root_cause = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_explanation_tenant_anomaly", "tenant_id", "anomaly_id", unique=True),
    )

    def __repr__(self) -> str:
        return f"<AnomalyExplanationCache(id={self.id}, anomaly_id={self.anomaly_id})>"


class LearnedRemediation(Base):
    """Table for storing learned remediation patterns."""

    __tablename__ = "learned_remediations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)

    # Pattern identification
    pattern_name = Column(String(255), nullable=False)
    pattern_hash = Column(String(64), nullable=False, index=True)  # For fast matching
    anomaly_types = Column(Text, nullable=False)  # JSON array
    severity_range = Column(Text, nullable=False)  # JSON array

    # Pattern matching criteria
    feature_conditions_json = Column(Text, nullable=True)  # JSON array of conditions
    event_patterns_json = Column(Text, nullable=True)  # JSON array of event patterns

    # The remediation
    remediation_title = Column(String(255), nullable=False)
    remediation_description = Column(Text, nullable=False)
    remediation_steps_json = Column(Text, nullable=False)  # JSON array
    automation_config_json = Column(Text, nullable=True)  # JSON object

    # Effectiveness tracking
    times_suggested = Column(Integer, default=0, nullable=False)
    times_applied = Column(Integer, default=0, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)

    # Confidence tracking
    initial_confidence = Column(Float, nullable=False, default=0.5)
    current_confidence = Column(Float, nullable=False, default=0.5)
    confidence_history_json = Column(Text, nullable=True)  # JSON array of {timestamp, confidence}

    # Source tracking
    learned_from_cases_json = Column(Text, nullable=True)  # JSON array of anomaly_ids
    last_successful_case_id = Column(Integer, nullable=True)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    deactivation_reason = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (Index("idx_learned_pattern_hash", "tenant_id", "pattern_hash"),)

    def __repr__(self) -> str:
        return f"<LearnedRemediation(id={self.id}, pattern={self.pattern_name})>"


class DeviceActionLog(Base):
    """Table for logging device control actions (SOTI MobiControl actions)."""

    __tablename__ = "device_action_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    device_id = Column(Integer, nullable=False, index=True)

    # Action details
    action_type = Column(
        String(50), nullable=False, index=True
    )  # lock, restart, wipe, message, locate, sync
    initiated_by = Column(String(100), nullable=False, default="system")
    reason = Column(Text, nullable=True)

    # Result tracking
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    mobicontrol_action_id = Column(String(100), nullable=True)  # Action ID from MobiControl API

    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_device_action_tenant_device", "tenant_id", "device_id"),
        Index("idx_device_action_type_time", "action_type", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<DeviceActionLog(id={self.id}, device_id={self.device_id}, action={self.action_type})>"


class RemediationOutcome(Base):
    """Table for tracking remediation outcomes for learning."""

    __tablename__ = "remediation_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    anomaly_id = Column(Integer, nullable=False, index=True)
    learned_remediation_id = Column(
        Integer, nullable=True, index=True
    )  # FK to learned_remediations

    # What was applied
    remediation_title = Column(String(255), nullable=False)
    remediation_source = Column(String(50), nullable=False)  # learned, ai_generated, policy, manual

    # Outcome tracking
    applied_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    applied_by = Column(String(100), nullable=True)
    outcome = Column(
        String(50), nullable=True
    )  # resolved, partially_resolved, no_effect, made_worse
    outcome_recorded_at = Column(DateTime, nullable=True)
    outcome_notes = Column(Text, nullable=True)

    # Context for learning
    anomaly_context_json = Column(Text, nullable=True)  # Snapshot of anomaly state when applied

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<RemediationOutcome(id={self.id}, anomaly_id={self.anomaly_id}, outcome={self.outcome})>"


# ============================================================================
# CEO Requirements: Location-Based Insights Tables
# ============================================================================


class LocationMappingType(StrEnum):
    """How devices are mapped to locations."""

    CUSTOM_ATTRIBUTE = "custom_attribute"  # Match device CustomAttributes[attribute_name] == value
    LABEL = "label"  # Match LabelDevice entries
    DEVICE_GROUP = "device_group"  # Match DeviceGroupId
    GEO_FENCE = "geo_fence"  # Match lat/lon within radius


class InsightSeverity(StrEnum):
    """Severity level for customer-facing insights."""

    CRITICAL = "critical"  # Immediate action required
    WARNING = "warning"  # Should address soon
    INFO = "info"  # FYI / monitoring


class TrendDirection(StrEnum):
    """Trend direction for metrics."""

    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"


class LocationMetadata(Base):
    """Location configuration for aggregation and shift schedules.

    Enables Carl's requirement: "Relate any anomalies to location (warehouse 1 vs warehouse 2)"
    """

    __tablename__ = "location_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    location_id = Column(String(100), nullable=False)  # e.g., "warehouse-1", "store-a101"
    location_name = Column(String(255), nullable=False)
    parent_region = Column(String(100), nullable=True)  # e.g., "Northeast", "Region-1"
    timezone = Column(String(50), default="UTC")

    # Mapping configuration - how to identify devices at this location
    mapping_type = Column(String(50), nullable=False)  # LocationMappingType value
    mapping_attribute = Column(String(100), nullable=True)  # e.g., "Store", "Warehouse"
    mapping_value = Column(String(255), nullable=True)  # e.g., "A101", "WH-North"
    device_group_id = Column(Integer, nullable=True)  # For device_group mapping
    geo_fence_json = Column(
        Text, nullable=True
    )  # For geo_fence: {"lat": 0, "lon": 0, "radius_m": 100}

    # Shift schedules (JSON array)
    # Example: [{"name": "Morning", "start": "06:00", "end": "14:00"}, {"name": "Afternoon", ...}]
    shift_schedules_json = Column(Text, nullable=True)

    # Location baselines (computed periodically)
    baseline_battery_drain_per_hour = Column(Float, nullable=True)
    baseline_disconnect_rate = Column(Float, nullable=True)
    baseline_drop_rate = Column(Float, nullable=True)
    baseline_computed_at = Column(DateTime, nullable=True)

    # Metadata
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_location_tenant_id", "tenant_id", "location_id", unique=True),
        Index("idx_location_active", "tenant_id", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<LocationMetadata(id={self.id}, location_id={self.location_id}, name={self.location_name})>"


class AggregatedInsight(Base):
    """Pre-computed insights aggregated by entity (location, user, cohort).

    Stores customer-facing insights in plain language with business impact.
    Enables Carl's requirement for insights that "make sense customers could understand".
    """

    __tablename__ = "aggregated_insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)

    # Entity identification (what/who is this insight about)
    entity_type = Column(
        String(50), nullable=False
    )  # "location", "user", "cohort", "device_model", "app"
    entity_id = Column(String(100), nullable=False)
    entity_name = Column(String(255), nullable=True)

    # Insight classification (from InsightCategory enum)
    insight_category = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False)  # InsightSeverity value

    # Insight content (customer-facing)
    headline = Column(
        Text, nullable=False
    )  # e.g., "5 devices in Warehouse 1 won't last a full shift"
    impact_statement = Column(
        Text, nullable=True
    )  # e.g., "Workers may experience 2 hours of downtime"
    comparison_context = Column(Text, nullable=True)  # e.g., "30% worse than Warehouse 2"
    recommended_actions_json = Column(Text, nullable=True)  # JSON array of action strings

    # Full structured payload for detailed views
    insight_data_json = Column(Text, nullable=False)

    # Affected entities
    affected_device_count = Column(Integer, default=0)
    affected_devices_json = Column(Text, nullable=True)  # JSON array of device_ids

    # Trend tracking
    trend_direction = Column(String(20), nullable=True)  # TrendDirection value
    previous_value = Column(Float, nullable=True)
    current_value = Column(Float, nullable=True)
    change_percent = Column(Float, nullable=True)

    # Confidence and quality
    confidence_score = Column(Float, nullable=True)  # 0.0 to 1.0
    data_quality_score = Column(Float, nullable=True)  # 0.0 to 1.0

    # Validity window
    computed_at = Column(DateTime, nullable=False)
    valid_until = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Acknowledgement tracking
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)

    __table_args__ = (
        Index("idx_insight_entity", "tenant_id", "entity_type", "entity_id"),
        Index("idx_insight_category", "tenant_id", "insight_category", "is_active"),
        Index("idx_insight_severity", "tenant_id", "severity", "is_active"),
        Index("idx_insight_active", "tenant_id", "is_active", "computed_at"),
    )

    def __repr__(self) -> str:
        return f"<AggregatedInsight(id={self.id}, category={self.insight_category}, entity={self.entity_type}:{self.entity_id})>"


class DeviceFeature(Base):
    """Computed device features for insight generation and ML analysis.

    Stores per-device feature snapshots used by insight analyzers
    and the anomaly detection pipeline.
    """

    __tablename__ = "device_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    device_id = Column(Integer, nullable=False, index=True)

    # Feature data
    feature_values_json = Column(Text, nullable=True)  # JSON of computed features
    metadata_json = Column(
        Text, nullable=True
    )  # JSON of device metadata (location_id, device_name, etc.)

    # Timestamps
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_device_feature_tenant_device", "tenant_id", "device_id"),
        Index("idx_device_feature_computed", "tenant_id", "computed_at"),
    )

    def __repr__(self) -> str:
        return f"<DeviceFeature(id={self.id}, device_id={self.device_id}, computed_at={self.computed_at})>"


class ShiftPerformance(Base):
    """Per-shift performance tracking for battery and productivity analysis.

    Enables Carl's requirement: "Batteries that don't last a shift"
    """

    __tablename__ = "shift_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    device_id = Column(Integer, nullable=False, index=True)
    location_id = Column(String(100), nullable=True, index=True)

    # Shift identification
    shift_date = Column(DateTime, nullable=False)
    shift_name = Column(String(50), nullable=False)  # "Morning", "Afternoon", "Night"
    shift_start = Column(DateTime, nullable=False)
    shift_end = Column(DateTime, nullable=False)
    shift_duration_hours = Column(Float, nullable=False)

    # Battery metrics
    battery_start = Column(Float, nullable=True)
    battery_end = Column(Float, nullable=True)
    battery_drain_total = Column(Float, nullable=True)
    battery_drain_rate_per_hour = Column(Float, nullable=True)

    # Shift completion prediction
    will_complete_shift = Column(Boolean, nullable=True)  # Predicted at shift start
    estimated_dead_time = Column(DateTime, nullable=True)  # If predicted to die
    actual_completed_shift = Column(Boolean, nullable=True)  # Did it actually complete

    # Charging metrics
    was_fully_charged_at_start = Column(Boolean, nullable=True)
    charge_events_count = Column(Integer, default=0)
    total_charge_time_minutes = Column(Float, default=0)
    charge_received_during_shift = Column(Float, nullable=True)  # Battery % gained

    # Usage metrics
    screen_on_time_minutes = Column(Float, nullable=True)
    app_foreground_time_minutes = Column(Float, nullable=True)
    total_drops = Column(Integer, default=0)
    total_disconnects = Column(Integer, default=0)

    # Comparison to baselines
    drain_vs_location_baseline = Column(Float, nullable=True)  # % difference
    drain_vs_device_baseline = Column(Float, nullable=True)  # % difference

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_shift_device_date", "device_id", "shift_date"),
        Index("idx_shift_location_date", "location_id", "shift_date"),
        Index("idx_shift_completion", "tenant_id", "will_complete_shift", "shift_date"),
    )

    def __repr__(self) -> str:
        return f"<ShiftPerformance(id={self.id}, device_id={self.device_id}, shift={self.shift_name}, date={self.shift_date})>"


class DeviceAssignment(Base):
    """Device-to-user assignment tracking.

    Enables Carl's requirement: "People with excessive drops" - aggregate
    device abuse metrics by the person assigned to the device.

    Supports multiple assignment types (owner, user, operator) and
    temporal validity for historical analysis.
    """

    __tablename__ = "device_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(100), nullable=False, index=True)
    device_id = Column(String(100), nullable=False, index=True)

    # User identification
    user_id = Column(String(100), nullable=True, index=True)
    user_name = Column(String(200), nullable=True)
    user_email = Column(String(200), nullable=True)

    # Team/group identification (for team-level aggregation)
    team_id = Column(String(100), nullable=True, index=True)
    team_name = Column(String(200), nullable=True)

    # Assignment type: owner (primary), user (current), operator, manager
    assignment_type = Column(String(50), default="owner")

    # Temporal validity for historical tracking
    valid_from = Column(DateTime, default=datetime.utcnow, nullable=False)
    valid_to = Column(DateTime, nullable=True)  # NULL = currently valid

    # Source of assignment data
    source = Column(String(50), nullable=False)  # "mc_label", "mc_attribute", "manual", "ad_sync"
    source_label_type = Column(String(100), nullable=True)  # e.g., "Owner", "AssignedUser"

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_device_assignments_lookup", "tenant_id", "device_id", "valid_to"),
        Index("ix_device_assignments_user", "tenant_id", "user_id", "valid_to"),
        Index("ix_device_assignments_team", "tenant_id", "team_id", "valid_to"),
    )

    def __repr__(self) -> str:
        return f"<DeviceAssignment(id={self.id}, device_id={self.device_id}, user_id={self.user_id}, type={self.assignment_type})>"


def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def get_session_factory(database_url: str):
    """Create a session factory for the given database URL."""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
