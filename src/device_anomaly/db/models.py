"""SQLAlchemy models for the SOTI Anomaly Detection backend database.

This module defines the canonical data model for storing:
- Tenants and devices (multi-source: XSight, MobiControl)
- Metric definitions (standard + custom datapoints)
- Telemetry points (time-series data)
- Baselines and anomalies
- Change logs and LLM explanations
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utc_now() -> datetime:
    """Return current UTC time. Used as default for DateTime columns."""
    return datetime.now(UTC)


class Tenant(Base):
    """Multi-tenant isolation table.

    Each tenant represents a separate customer/organization.
    All data access must be filtered by tenant_id.
    """

    __tablename__ = "tenants"

    tenant_id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    tier = Column(String(20), default="standard")  # free, standard, enterprise
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    extra_data = Column(Text)  # JSON stored as NVARCHAR(MAX)

    def __repr__(self):
        return f"<Tenant(tenant_id='{self.tenant_id}', name='{self.name}', tier='{self.tier}')>"


class Device(Base):
    """Device registry supporting multiple sources (XSight, MobiControl).

    Devices are identified by device_id (internal) and external_id (from source).
    The 'source' field indicates origin: 'xsight', 'mobicontrol', etc.
    """

    __tablename__ = "devices"

    device_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    source = Column(String(20), nullable=False)  # xsight, mobicontrol, synthetic
    external_id = Column(String(100), nullable=False)  # Original ID from source system
    name = Column(String(255))
    device_type = Column(String(50))  # phone, tablet, laptop, etc.
    os_version = Column(String(50))
    last_seen = Column(DateTime)
    device_group_id = Column(String(50))  # For future grouping support
    extra_data = Column(Text)  # JSON: additional device properties

    __table_args__ = (
        Index("idx_tenant_source", "tenant_id", "source"),
        Index("idx_external", "external_id", "source"),
        Index("idx_last_seen", "last_seen"),
        Index("idx_device_group", "device_group_id"),
    )

    def __repr__(self):
        return f"<Device(device_id='{self.device_id}', source='{self.source}', name='{self.name}')>"


class MetricDefinition(Base):
    """Catalog of all metrics (standard predefined + dynamically discovered custom).

    Standard metrics: TotalBatteryLevelDrop, Download, Upload, etc.
    Custom metrics: Discovered from data sources, registered here.
    """

    __tablename__ = "metric_definitions"

    metric_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50))  # battery, network, storage, cpu, memory, custom
    unit = Column(String(20))  # %, KB, MB, ms, etc.
    data_type = Column(String(20))  # int, float, string, bool
    source = Column(String(50))  # xsight, mobicontrol, custom
    is_standard = Column(Boolean, default=False)
    validation_rules = Column(Text)  # JSON: min, max, allowed_values, etc.

    __table_args__ = (
        Index("idx_md_source", "source"),
        Index("idx_md_category", "category"),
        Index("idx_md_standard", "is_standard"),
    )

    def __repr__(self):
        return f"<MetricDefinition(metric_id='{self.metric_id}', name='{self.name}', category='{self.category}')>"


class TelemetryPoint(Base):
    """Time-series telemetry data points.

    Stores individual metric measurements over time.
    Partitioned by timestamp for performance (see migration script).
    """

    __tablename__ = "telemetry_points"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    device_id = Column(String(50), ForeignKey("devices.device_id"), nullable=False)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    metric_id = Column(String(50), ForeignKey("metric_definitions.metric_id"), nullable=False)
    value = Column(Float)  # Numeric values
    value_str = Column(String(255))  # Non-numeric values (optional)
    quality = Column(Integer, default=100)  # Data quality score 0-100
    ingestion_time = Column(DateTime, default=_utc_now)
    source_batch_id = Column(String(50))  # For tracking ingestion batches

    __table_args__ = (
        Index("idx_tp_device_time", "device_id", "timestamp"),
        Index("idx_tp_tenant_time", "tenant_id", "timestamp"),
        Index("idx_tp_metric_time", "metric_id", "timestamp"),
        Index("idx_tp_ingestion", "ingestion_time"),
    )

    def __repr__(self):
        return f"<TelemetryPoint(device_id='{self.device_id}', timestamp='{self.timestamp}', metric_id='{self.metric_id}', value={self.value})>"


class Baseline(Base):
    """Baseline profiles for normal behavior.

    Baselines can be scoped at different levels:
    - tenant: All devices in a tenant
    - site: Devices at a specific site
    - device_group: A group of similar devices
    - device: Individual device baseline
    """

    __tablename__ = "baselines"

    baseline_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    name = Column(String(255), nullable=False)
    scope = Column(String(50), nullable=False)  # tenant, site, device_group, device
    scope_id = Column(String(50), nullable=False)  # ID of the scope entity
    metric_id = Column(String(50), ForeignKey("metric_definitions.metric_id"), nullable=False)
    window_config = Column(Text)  # JSON: window_days, update_frequency, time_segmentation
    stats = Column(Text)  # JSON: mean, std, median, p5, p95, etc.
    valid_from = Column(DateTime, nullable=False)
    valid_to = Column(DateTime)  # NULL = currently active
    created_by = Column(String(50))  # user_id or 'system'

    __table_args__ = (
        Index("idx_bl_tenant_scope", "tenant_id", "scope", "scope_id"),
        Index("idx_bl_valid", "valid_from", "valid_to"),
        Index("idx_bl_metric", "metric_id"),
    )

    def __repr__(self):
        return f"<Baseline(baseline_id='{self.baseline_id}', scope='{self.scope}', metric_id='{self.metric_id}')>"


class Anomaly(Base):
    """Detected anomalies with metadata.

    Each record represents one anomaly detection event.
    Includes severity, score, explanation, and user feedback.
    """

    __tablename__ = "anomalies"

    anomaly_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    device_id = Column(String(50), ForeignKey("devices.device_id"), nullable=False)
    event_id = Column(String(50), ForeignKey("anomaly_events.event_id"))  # Link to anomaly_events
    timestamp = Column(DateTime, nullable=False)
    detector_name = Column(String(100), nullable=False)  # isolation_forest, z_score, etc.
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    score = Column(Float, nullable=False)  # Anomaly score from detector
    metrics_involved = Column(Text)  # JSON: {metric_id: value}
    explanation = Column(Text)  # Human-readable explanation (may be LLM-generated)
    explanation_cache_key = Column(String(100))  # For LLM response caching
    user_feedback = Column(String(20))  # true_positive, false_positive, unknown
    status = Column(String(20), default="new")  # new, acknowledged, resolved, ignored
    created_at = Column(DateTime, default=_utc_now)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now)

    __table_args__ = (
        Index("idx_an_tenant_time", "tenant_id", "timestamp"),
        Index("idx_an_device_time", "device_id", "timestamp"),
        Index("idx_an_severity", "severity", "timestamp"),
        Index("idx_an_status", "status", "timestamp"),
        Index("idx_an_detector", "detector_name"),
        Index("idx_an_feedback", "user_feedback"),
        Index("idx_event_id", "event_id"),
    )

    def __repr__(self):
        return f"<Anomaly(anomaly_id='{self.anomaly_id}', device_id='{self.device_id}', severity='{self.severity}', detector='{self.detector_name}')>"


class ChangeLog(Base):
    """Log of environmental changes that may explain anomalies.

    Tracks changes like:
    - Policy updates
    - OS version rollouts
    - App version deployments
    - Access Point additions/removals
    - Configuration changes
    """

    __tablename__ = "change_log"

    change_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    change_type = Column(
        String(50), nullable=False
    )  # policy, os_version, app_version, ap_added, etc.
    description = Column(Text)  # Detailed description
    affected_devices = Column(Text)  # JSON: list of device_ids or device_group_ids
    source = Column(String(50))  # xsight, mobicontrol, manual, api
    extra_data = Column(Text)  # JSON: additional change details

    __table_args__ = (
        Index("idx_cl_tenant_time", "tenant_id", "timestamp"),
        Index("idx_cl_type_time", "change_type", "timestamp"),
        Index("idx_cl_source", "source"),
    )

    def __repr__(self):
        return f"<ChangeLog(change_id='{self.change_id}', change_type='{self.change_type}', timestamp='{self.timestamp}')>"


class Explanation(Base):
    """LLM-generated explanations for anomalies.

    Stores natural language explanations with:
    - Which LLM model was used
    - Prompt version for reproducibility
    - Token usage for cost tracking
    - Retrieved context for transparency
    """

    __tablename__ = "explanations"

    explanation_id = Column(String(50), primary_key=True)
    anomaly_id = Column(String(50), ForeignKey("anomalies.anomaly_id"), nullable=False)
    llm_model = Column(String(100), nullable=False)  # ollama/llama3.2, claude-3-haiku, etc.
    prompt_version = Column(String(20))  # For tracking prompt changes
    generated_text = Column(Text, nullable=False)
    confidence = Column(Float)  # Optional: LLM confidence score 0-1
    tokens_used = Column(Integer)  # For cost tracking
    generation_time_ms = Column(Integer)  # Performance tracking
    context_used = Column(Text)  # JSON: Retrieved RAG context
    created_at = Column(DateTime, default=_utc_now)

    __table_args__ = (
        Index("idx_ex_anomaly", "anomaly_id"),
        Index("idx_ex_model", "llm_model"),
        Index("idx_ex_created", "created_at"),
    )

    def __repr__(self):
        return f"<Explanation(explanation_id='{self.explanation_id}', anomaly_id='{self.anomaly_id}', model='{self.llm_model}')>"


class User(Base):
    """Users for authentication and RBAC.

    Supports role-based access control:
    - viewer: Read-only access
    - analyst: Can provide feedback, manage baselines
    - admin: Full access including user management
    """

    __tablename__ = "users"

    user_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)  # viewer, analyst, admin
    password_hash = Column(String(255))  # Hashed password
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=_utc_now)
    extra_data = Column(Text)  # JSON: preferences, settings

    __table_args__ = (
        Index("idx_usr_tenant", "tenant_id"),
        Index("idx_usr_email", "email"),
        Index("idx_usr_role", "role"),
    )

    def __repr__(self):
        return f"<User(user_id='{self.user_id}', email='{self.email}', role='{self.role}')>"


class AuditLog(Base):
    """Audit trail for all data access and modifications.

    Required for compliance (GDPR, SOC2, etc.).
    Tracks who accessed what data when.
    """

    __tablename__ = "audit_logs"

    log_id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=_utc_now, nullable=False)
    user_id = Column(String(50))
    tenant_id = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)  # view, create, update, delete, export
    resource_type = Column(String(50), nullable=False)  # anomaly, device, baseline, etc.
    resource_id = Column(String(50))
    ip_address = Column(String(45))  # IPv6 max length
    extra_data = Column(Text)  # JSON: request details, changes made

    __table_args__ = (
        Index("idx_al_timestamp", "timestamp"),
        Index("idx_al_user", "user_id", "timestamp"),
        Index("idx_al_tenant", "tenant_id", "timestamp"),
        Index("idx_al_resource", "resource_type", "resource_id"),
    )

    def __repr__(self):
        return f"<AuditLog(log_id={self.log_id}, action='{self.action}', resource_type='{self.resource_type}', timestamp='{self.timestamp}')>"


class AnomalyEvent(Base):
    """Anomaly events grouping consecutive anomalies.

    Groups row-level anomalies into contiguous events per device.
    An event represents a sequence of anomalous rows where gaps
    between consecutive anomalies are within a threshold.
    """

    __tablename__ = "anomaly_events"

    event_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    device_id = Column(String(50), ForeignKey("devices.device_id"), nullable=False)
    event_start = Column(DateTime, nullable=False)
    event_end = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    anomaly_score_min = Column(Float, nullable=False)
    anomaly_score_max = Column(Float, nullable=False)
    anomaly_score_mean = Column(Float, nullable=False)
    row_count = Column(Integer, nullable=False)
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    metrics_json = Column(Text)  # JSON: aggregated metric values
    status = Column(String(20), default="new")  # new, acknowledged, resolved, ignored
    model_version = Column(String(50))
    created_at = Column(DateTime, default=_utc_now)

    __table_args__ = (
        Index("idx_ev_tenant_time", "tenant_id", "event_start"),
        Index("idx_ev_device_time", "device_id", "event_start"),
        Index("idx_ev_severity", "severity", "event_start"),
        Index("idx_ev_status", "status", "event_start"),
    )

    def __repr__(self):
        return f"<AnomalyEvent(event_id='{self.event_id}', device_id='{self.device_id}', severity='{self.severity}')>"


class DevicePattern(Base):
    """Long-term device behavior patterns.

    Stores aggregated pattern analysis for devices over time periods,
    including anomaly rates, event counts, and pattern summaries.
    """

    __tablename__ = "device_patterns"

    pattern_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    device_id = Column(String(50), ForeignKey("devices.device_id"), nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    total_points = Column(Integer, nullable=False)
    total_anomalies = Column(Integer, nullable=False)
    anomaly_rate = Column(Float, nullable=False)
    event_count = Column(Integer, nullable=False)
    worst_anomaly_score = Column(Float, nullable=False)
    mean_anomaly_score = Column(Float, nullable=False)
    pattern_json = Column(Text)  # JSON: detailed pattern data
    explanation = Column(Text)  # LLM-generated pattern explanation
    model_version = Column(String(50))
    created_at = Column(DateTime, default=_utc_now)

    __table_args__ = (
        Index("idx_pat_tenant_device", "tenant_id", "device_id", "period_start"),
        Index("idx_pat_period", "period_start", "period_end"),
        Index("idx_pat_anomaly_rate", "anomaly_rate"),
    )

    def __repr__(self):
        return f"<DevicePattern(pattern_id='{self.pattern_id}', device_id='{self.device_id}', anomaly_rate={self.anomaly_rate})>"


class MLModel(Base):
    """Registry of trained anomaly detection models.

    Tracks ML models used for anomaly detection, including configuration,
    feature columns, and deployment status.
    """

    __tablename__ = "ml_models"

    model_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)  # isolation_forest, z_score, hybrid, etc.
    version = Column(String(50), nullable=False)
    config_json = Column(Text)  # JSON: model configuration
    feature_cols_json = Column(Text)  # JSON: list of feature columns used
    model_artifact = Column(LargeBinary)  # Serialized model (optional, can store externally)
    status = Column(String(20), default="training")  # training, trained, deployed, archived
    trained_at = Column(DateTime)
    created_at = Column(DateTime, default=_utc_now)

    __table_args__ = (
        Index("idx_model_tenant", "tenant_id"),
        Index("idx_model_type", "model_type"),
        Index("idx_model_status", "status"),
        Index("idx_model_version", "version"),
    )

    def __repr__(self):
        return f"<MLModel(model_id='{self.model_id}', name='{self.name}', model_type='{self.model_type}', version='{self.version}')>"


class ModelDeployment(Base):
    """Track which models are deployed in which environments.

    Manages model deployment lifecycle across different environments.
    """

    __tablename__ = "model_deployments"

    deployment_id = Column(String(50), primary_key=True)
    model_id = Column(String(50), ForeignKey("ml_models.model_id"), nullable=False)
    environment = Column(String(50), nullable=False)  # production, staging, development
    is_active = Column(Boolean, default=True)
    deployed_at = Column(DateTime, default=_utc_now)
    deployed_by = Column(String(50))  # user_id

    __table_args__ = (
        Index("idx_deploy_model", "model_id"),
        Index("idx_deploy_env", "environment", "is_active"),
        Index("idx_deploy_active", "is_active", "deployed_at"),
    )

    def __repr__(self):
        return f"<ModelDeployment(deployment_id='{self.deployment_id}', model_id='{self.model_id}', environment='{self.environment}', is_active={self.is_active})>"


class ModelMetric(Base):
    """Performance metrics for deployed models.

    Tracks model performance over time including precision, recall, F1,
    and confusion matrix components.
    """

    __tablename__ = "model_metrics"

    metric_id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(String(50), ForeignKey("ml_models.model_id"), nullable=False)
    timestamp = Column(DateTime, default=_utc_now, nullable=False)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    true_positives = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    total_predictions = Column(Integer, default=0)
    extra_data = Column(Text)  # JSON: additional metrics

    __table_args__ = (
        Index("idx_metrics_model_time", "model_id", "timestamp"),
        Index("idx_metrics_timestamp", "timestamp"),
    )

    def __repr__(self):
        return f"<ModelMetric(metric_id={self.metric_id}, model_id='{self.model_id}', f1_score={self.f1_score})>"


class AlertRule(Base):
    """Configurable alerting conditions.

    Defines rules that trigger alerts based on anomaly characteristics,
    patterns, or other conditions.
    """

    __tablename__ = "alert_rules"

    rule_id = Column(String(50), primary_key=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    name = Column(String(255), nullable=False)
    rule_type = Column(
        String(50), nullable=False
    )  # anomaly_severity, anomaly_count, pattern_detected, etc.
    conditions_json = Column(Text, nullable=False)  # JSON: rule conditions
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    actions_json = Column(Text)  # JSON: notification actions (email, webhook, etc.)
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utc_now)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now)

    __table_args__ = (
        Index("idx_rule_tenant", "tenant_id"),
        Index("idx_rule_enabled", "is_enabled", "tenant_id"),
        Index("idx_rule_type", "rule_type"),
    )

    def __repr__(self):
        return f"<AlertRule(rule_id='{self.rule_id}', name='{self.name}', rule_type='{self.rule_type}', is_enabled={self.is_enabled})>"


class Alert(Base):
    """Generated alerts from alert rules.

    Stores alerts that have been triggered by alert rules,
    including status and resolution tracking.
    """

    __tablename__ = "alerts"

    alert_id = Column(String(50), primary_key=True)
    rule_id = Column(String(50), ForeignKey("alert_rules.rule_id"), nullable=False)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    anomaly_id = Column(
        String(50), ForeignKey("anomalies.anomaly_id")
    )  # Optional: link to specific anomaly
    device_id = Column(String(50), ForeignKey("devices.device_id"))  # Optional: link to device
    event_id = Column(
        String(50), ForeignKey("anomaly_events.event_id")
    )  # Optional: link to anomaly event
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    status = Column(String(20), default="open")  # open, acknowledged, resolved, suppressed
    message = Column(Text, nullable=False)
    triggered_at = Column(DateTime, default=_utc_now)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(50))  # user_id
    resolved_at = Column(DateTime)
    extra_data = Column(Text)  # JSON: additional alert context

    __table_args__ = (
        Index("idx_alert_tenant_time", "tenant_id", "triggered_at"),
        Index("idx_alert_status", "status", "triggered_at"),
        Index("idx_alert_severity", "severity", "triggered_at"),
        Index("idx_alert_rule", "rule_id"),
        Index("idx_alert_device", "device_id", "triggered_at"),
    )

    def __repr__(self):
        return f"<Alert(alert_id='{self.alert_id}', rule_id='{self.rule_id}', severity='{self.severity}', status='{self.status}')>"


class AlertNotification(Base):
    """Notification delivery tracking.

    Tracks individual notification attempts for alerts across
    different channels (email, webhook, SMS, etc.).
    """

    __tablename__ = "alert_notifications"

    notification_id = Column(BigInteger, primary_key=True, autoincrement=True)
    alert_id = Column(String(50), ForeignKey("alerts.alert_id"), nullable=False)
    channel = Column(String(50), nullable=False)  # email, webhook, sms, slack, etc.
    recipient = Column(
        String(255), nullable=False
    )  # email address, webhook URL, phone number, etc.
    status = Column(String(20), default="pending")  # pending, sent, failed, delivered
    response_json = Column(Text)  # JSON: response from notification service
    sent_at = Column(DateTime)

    __table_args__ = (
        Index("idx_notif_alert", "alert_id"),
        Index("idx_notif_status", "status", "sent_at"),
        Index("idx_notif_channel", "channel"),
    )

    def __repr__(self):
        return f"<AlertNotification(notification_id={self.notification_id}, alert_id='{self.alert_id}', channel='{self.channel}', status='{self.status}')>"
