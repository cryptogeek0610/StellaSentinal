"""Database schema for storing anomaly detection results."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    Index,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class AnomalyStatus(str, Enum):
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
    dataset_version_id = Column(Integer, nullable=True, index=True)  # FK to training_dataset_versions

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
    dataset_version_id = Column(Integer, nullable=True, index=True)  # FK to training_dataset_versions

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
    stage = Column(String(20), default="development", nullable=False)  # development, staging, production, archived
    is_active = Column(Boolean, default=False, nullable=False)  # Currently deployed model
    deployed_at = Column(DateTime, nullable=True)
    deployed_by = Column(String(100), nullable=True)

    # Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(100), nullable=True)
    archived_at = Column(DateTime, nullable=True)
    archive_reason = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_model_registry_name_version", "tenant_id", "model_name", "model_version", unique=True),
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

    __table_args__ = (
        Index("idx_learned_pattern_hash", "tenant_id", "pattern_hash"),
    )

    def __repr__(self) -> str:
        return f"<LearnedRemediation(id={self.id}, pattern={self.pattern_name})>"


class DeviceActionLog(Base):
    """Table for logging device control actions (SOTI MobiControl actions)."""

    __tablename__ = "device_action_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True)
    device_id = Column(Integer, nullable=False, index=True)

    # Action details
    action_type = Column(String(50), nullable=False, index=True)  # lock, restart, wipe, message, locate, sync
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
    learned_remediation_id = Column(Integer, nullable=True, index=True)  # FK to learned_remediations

    # What was applied
    remediation_title = Column(String(255), nullable=False)
    remediation_source = Column(String(50), nullable=False)  # learned, ai_generated, policy, manual

    # Outcome tracking
    applied_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    applied_by = Column(String(100), nullable=True)
    outcome = Column(String(50), nullable=True)  # resolved, partially_resolved, no_effect, made_worse
    outcome_recorded_at = Column(DateTime, nullable=True)
    outcome_notes = Column(Text, nullable=True)

    # Context for learning
    anomaly_context_json = Column(Text, nullable=True)  # Snapshot of anomaly state when applied

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<RemediationOutcome(id={self.id}, anomaly_id={self.anomaly_id}, outcome={self.outcome})>"


def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def get_session_factory(database_url: str):
    """Create a session factory for the given database URL."""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
