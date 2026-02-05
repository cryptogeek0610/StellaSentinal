"""Repository for Anomaly entity operations.

This module provides data access methods for anomaly records,
including filtering by severity, status, time range, and device.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from device_anomaly.db.models import Anomaly
from device_anomaly.db.repositories.base import BaseRepository

# Maximum allowed page size to prevent DoS via large queries
MAX_PAGE_SIZE = 1000


class AnomalyRepository(BaseRepository[Anomaly]):
    """Repository for Anomaly CRUD operations.

    Extends BaseRepository with anomaly-specific query methods.
    """

    def __init__(self, session: Session):
        """Initialize the anomaly repository.

        Args:
            session: SQLAlchemy session
        """
        super().__init__(session, Anomaly)

    def get_by_tenant(
        self,
        tenant_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        severity: str | None = None,
        status: str | None = None,
        device_id: str | None = None,
        detector_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Anomaly]:
        """Get anomalies for a tenant with filters.

        Args:
            tenant_id: Tenant ID (required for multi-tenancy)
            start_date: Filter by timestamp >= start_date
            end_date: Filter by timestamp <= end_date
            severity: Filter by severity level
            status: Filter by status
            device_id: Filter by device ID
            detector_name: Filter by detector name
            limit: Maximum records to return
            offset: Records to skip

        Returns:
            List of matching anomalies, ordered by timestamp descending
        """
        # Enforce maximum page size to prevent DoS
        limit = min(limit, MAX_PAGE_SIZE)

        query = self.session.query(Anomaly).filter(Anomaly.tenant_id == tenant_id)

        if start_date:
            query = query.filter(Anomaly.timestamp >= start_date)
        if end_date:
            query = query.filter(Anomaly.timestamp <= end_date)
        if severity:
            query = query.filter(Anomaly.severity == severity)
        if status:
            query = query.filter(Anomaly.status == status)
        if device_id:
            query = query.filter(Anomaly.device_id == device_id)
        if detector_name:
            query = query.filter(Anomaly.detector_name == detector_name)

        return query.order_by(desc(Anomaly.timestamp)).offset(offset).limit(limit).all()

    def get_by_device(
        self,
        device_id: str,
        tenant_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
    ) -> list[Anomaly]:
        """Get anomalies for a specific device.

        Args:
            device_id: Device ID
            tenant_id: Tenant ID
            start_date: Filter by timestamp >= start_date
            end_date: Filter by timestamp <= end_date
            limit: Maximum records to return

        Returns:
            List of anomalies for the device
        """
        return self.get_by_tenant(
            tenant_id=tenant_id,
            device_id=device_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

    def get_by_severity(
        self,
        tenant_id: str,
        severity: str,
        limit: int = 100,
    ) -> list[Anomaly]:
        """Get anomalies by severity level.

        Args:
            tenant_id: Tenant ID
            severity: Severity level (low, medium, high, critical)
            limit: Maximum records to return

        Returns:
            List of anomalies with the specified severity
        """
        return self.get_by_tenant(tenant_id=tenant_id, severity=severity, limit=limit)

    def get_unresolved(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> list[Anomaly]:
        """Get unresolved anomalies (status = new or acknowledged).

        Args:
            tenant_id: Tenant ID
            limit: Maximum records to return

        Returns:
            List of unresolved anomalies
        """
        return (
            self.session.query(Anomaly)
            .filter(
                and_(
                    Anomaly.tenant_id == tenant_id,
                    Anomaly.status.in_(["new", "acknowledged"]),
                )
            )
            .order_by(desc(Anomaly.timestamp))
            .limit(limit)
            .all()
        )

    def update_status(
        self,
        anomaly_id: str,
        tenant_id: str,
        status: str,
    ) -> Anomaly | None:
        """Update anomaly status.

        Args:
            anomaly_id: Anomaly ID
            tenant_id: Tenant ID for validation
            status: New status (new, acknowledged, resolved, ignored)

        Returns:
            Updated anomaly if found, None otherwise
        """
        anomaly = self.get_by_id(anomaly_id, tenant_id)
        if anomaly:
            anomaly.status = status
            anomaly.updated_at = datetime.now(UTC)
            self.session.flush()
        return anomaly

    def record_feedback(
        self,
        anomaly_id: str,
        tenant_id: str,
        feedback: str,
    ) -> Anomaly | None:
        """Record user feedback for an anomaly.

        Args:
            anomaly_id: Anomaly ID
            tenant_id: Tenant ID for validation
            feedback: Feedback (true_positive, false_positive, unknown)

        Returns:
            Updated anomaly if found, None otherwise
        """
        anomaly = self.get_by_id(anomaly_id, tenant_id)
        if anomaly:
            anomaly.user_feedback = feedback
            anomaly.updated_at = datetime.now(UTC)
            self.session.flush()
        return anomaly

    def set_explanation(
        self,
        anomaly_id: str,
        tenant_id: str,
        explanation: str,
        cache_key: str | None = None,
    ) -> Anomaly | None:
        """Set the explanation for an anomaly.

        Args:
            anomaly_id: Anomaly ID
            tenant_id: Tenant ID for validation
            explanation: Human-readable explanation text
            cache_key: Optional cache key for LLM response caching

        Returns:
            Updated anomaly if found, None otherwise
        """
        anomaly = self.get_by_id(anomaly_id, tenant_id)
        if anomaly:
            anomaly.explanation = explanation
            if cache_key:
                anomaly.explanation_cache_key = cache_key
            anomaly.updated_at = datetime.now(UTC)
            self.session.flush()
        return anomaly

    def count_by_severity(self, tenant_id: str) -> dict[str, int]:
        """Count anomalies grouped by severity.

        Args:
            tenant_id: Tenant ID

        Returns:
            Dict mapping severity to count
        """
        results = (
            self.session.query(Anomaly.severity, func.count(Anomaly.anomaly_id))
            .filter(Anomaly.tenant_id == tenant_id)
            .group_by(Anomaly.severity)
            .all()
        )
        return dict(results)

    def count_by_status(self, tenant_id: str) -> dict[str, int]:
        """Count anomalies grouped by status.

        Args:
            tenant_id: Tenant ID

        Returns:
            Dict mapping status to count
        """
        results = (
            self.session.query(Anomaly.status, func.count(Anomaly.anomaly_id))
            .filter(Anomaly.tenant_id == tenant_id)
            .group_by(Anomaly.status)
            .all()
        )
        return dict(results)

    def count_by_date(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, int]:
        """Count anomalies per day in a date range.

        Args:
            tenant_id: Tenant ID
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Dict mapping date string (YYYY-MM-DD) to count
        """
        from sqlalchemy import Date, cast

        # Cast timestamp to date for grouping
        date_col = cast(Anomaly.timestamp, Date).label("date")

        results = (
            self.session.query(date_col, func.count(Anomaly.anomaly_id))
            .filter(
                and_(
                    Anomaly.tenant_id == tenant_id,
                    Anomaly.timestamp >= start_date,
                    Anomaly.timestamp <= end_date,
                )
            )
            .group_by(date_col)
            .all()
        )
        return {str(date): count for date, count in results}

    def get_top_devices(
        self,
        tenant_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get devices with most anomalies.

        Args:
            tenant_id: Tenant ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Number of top devices to return

        Returns:
            List of dicts with device_id and anomaly_count
        """
        query = self.session.query(
            Anomaly.device_id,
            func.count(Anomaly.anomaly_id).label("anomaly_count"),
        ).filter(Anomaly.tenant_id == tenant_id)

        if start_date:
            query = query.filter(Anomaly.timestamp >= start_date)
        if end_date:
            query = query.filter(Anomaly.timestamp <= end_date)

        results = (
            query.group_by(Anomaly.device_id).order_by(desc("anomaly_count")).limit(limit).all()
        )

        return [{"device_id": device_id, "anomaly_count": count} for device_id, count in results]

    def get_feedback_stats(
        self,
        tenant_id: str,
        detector_name: str | None = None,
    ) -> dict[str, int]:
        """Get feedback statistics for tuning detectors.

        Args:
            tenant_id: Tenant ID
            detector_name: Optional filter by detector

        Returns:
            Dict with feedback counts
        """
        query = self.session.query(Anomaly.user_feedback, func.count(Anomaly.anomaly_id)).filter(
            and_(
                Anomaly.tenant_id == tenant_id,
                Anomaly.user_feedback.isnot(None),
            )
        )

        if detector_name:
            query = query.filter(Anomaly.detector_name == detector_name)

        results = query.group_by(Anomaly.user_feedback).all()
        return dict(results)
