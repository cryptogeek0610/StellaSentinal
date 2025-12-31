"""Database module for SOTI Anomaly Detection.

This module provides database connectivity and repository patterns
for accessing the backend SQL Server database.
"""
from device_anomaly.db.models import (
    Anomaly,
    AuditLog,
    Base,
    Baseline,
    ChangeLog,
    Device,
    Explanation,
    MetricDefinition,
    TelemetryPoint,
    Tenant,
    User,
)
from device_anomaly.db.session import DatabaseSession, get_session

__all__ = [
    # Models
    "Anomaly",
    "AuditLog",
    "Base",
    "Baseline",
    "ChangeLog",
    "Device",
    "Explanation",
    "MetricDefinition",
    "TelemetryPoint",
    "Tenant",
    "User",
    # Session
    "DatabaseSession",
    "get_session",
]
