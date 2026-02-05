"""Repository for Baseline entity operations.

This module provides data access methods for baseline profiles,
supporting different scopes (tenant, site, device_group, device).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from device_anomaly.db.models import Baseline
from device_anomaly.db.repositories.base import BaseRepository


class BaselineRepository(BaseRepository[Baseline]):
    """Repository for Baseline CRUD operations.

    Extends BaseRepository with baseline-specific query methods.
    """

    def __init__(self, session: Session):
        """Initialize the baseline repository.

        Args:
            session: SQLAlchemy session
        """
        super().__init__(session, Baseline)

    def get_active_baselines(
        self,
        tenant_id: str,
        scope: str | None = None,
        metric_id: str | None = None,
    ) -> list[Baseline]:
        """Get currently active baselines.

        Args:
            tenant_id: Tenant ID
            scope: Optional scope filter (tenant, site, device_group, device)
            metric_id: Optional metric ID filter

        Returns:
            List of active baselines (valid_to is NULL)
        """
        query = self.session.query(Baseline).filter(
            and_(
                Baseline.tenant_id == tenant_id,
                Baseline.valid_to.is_(None),  # Currently active
            )
        )

        if scope:
            query = query.filter(Baseline.scope == scope)
        if metric_id:
            query = query.filter(Baseline.metric_id == metric_id)

        return query.all()

    def get_baseline_for_device(
        self,
        tenant_id: str,
        device_id: str,
        metric_id: str,
        as_of: datetime | None = None,
    ) -> Baseline | None:
        """Get the most specific applicable baseline for a device.

        Searches for baselines in order: device -> device_group -> site -> tenant.
        Returns the first (most specific) match.

        Args:
            tenant_id: Tenant ID
            device_id: Device ID
            metric_id: Metric ID
            as_of: Point in time for validity (default: now)

        Returns:
            Most specific applicable baseline, or None
        """
        if as_of is None:
            as_of = datetime.now(UTC)

        # Try device-specific first
        baseline = (
            self.session.query(Baseline)
            .filter(
                and_(
                    Baseline.tenant_id == tenant_id,
                    Baseline.metric_id == metric_id,
                    Baseline.scope == "device",
                    Baseline.scope_id == device_id,
                    Baseline.valid_from <= as_of,
                    or_(Baseline.valid_to.is_(None), Baseline.valid_to >= as_of),
                )
            )
            .first()
        )

        if baseline:
            return baseline

        # Fall back to tenant-level baseline
        return (
            self.session.query(Baseline)
            .filter(
                and_(
                    Baseline.tenant_id == tenant_id,
                    Baseline.metric_id == metric_id,
                    Baseline.scope == "tenant",
                    Baseline.scope_id == tenant_id,
                    Baseline.valid_from <= as_of,
                    or_(Baseline.valid_to.is_(None), Baseline.valid_to >= as_of),
                )
            )
            .first()
        )

    def get_baselines_for_metric(
        self,
        tenant_id: str,
        metric_id: str,
        include_expired: bool = False,
    ) -> list[Baseline]:
        """Get all baselines for a metric.

        Args:
            tenant_id: Tenant ID
            metric_id: Metric ID
            include_expired: Whether to include expired baselines

        Returns:
            List of baselines
        """
        query = self.session.query(Baseline).filter(
            and_(
                Baseline.tenant_id == tenant_id,
                Baseline.metric_id == metric_id,
            )
        )

        if not include_expired:
            query = query.filter(Baseline.valid_to.is_(None))

        return query.order_by(Baseline.valid_from.desc()).all()

    def create_baseline(
        self,
        baseline_id: str,
        tenant_id: str,
        name: str,
        scope: str,
        scope_id: str,
        metric_id: str,
        stats: dict[str, Any],
        window_config: dict[str, Any] | None = None,
        created_by: str = "system",
    ) -> Baseline:
        """Create a new baseline profile.

        Args:
            baseline_id: Unique baseline ID
            tenant_id: Tenant ID
            name: Human-readable name
            scope: Scope level (tenant, site, device_group, device)
            scope_id: ID of the scope entity
            metric_id: Metric this baseline applies to
            stats: Statistical profile (mean, std, percentiles, etc.)
            window_config: Configuration for the baseline window
            created_by: User or system that created this baseline

        Returns:
            Created baseline entity
        """
        baseline = Baseline(
            baseline_id=baseline_id,
            tenant_id=tenant_id,
            name=name,
            scope=scope,
            scope_id=scope_id,
            metric_id=metric_id,
            stats=json.dumps(stats),
            window_config=json.dumps(window_config) if window_config else None,
            valid_from=datetime.now(UTC),
            valid_to=None,
            created_by=created_by,
        )
        return self.create(baseline)

    def expire_baseline(
        self,
        baseline_id: str,
        tenant_id: str,
    ) -> Baseline | None:
        """Mark a baseline as expired.

        Sets valid_to to current time, making it no longer active.

        Args:
            baseline_id: Baseline ID
            tenant_id: Tenant ID for validation

        Returns:
            Updated baseline if found, None otherwise
        """
        baseline = self.get_by_id(baseline_id, tenant_id)
        if baseline:
            baseline.valid_to = datetime.now(UTC)
            self.session.flush()
        return baseline

    def update_stats(
        self,
        baseline_id: str,
        tenant_id: str,
        stats: dict[str, Any],
    ) -> Baseline | None:
        """Update baseline statistics.

        This creates a new version by expiring the current baseline
        and creating a new one with updated stats.

        Args:
            baseline_id: Baseline ID to update
            tenant_id: Tenant ID for validation
            stats: New statistical profile

        Returns:
            New baseline if original found, None otherwise
        """
        old_baseline = self.get_by_id(baseline_id, tenant_id)
        if not old_baseline:
            return None

        # Expire old baseline
        old_baseline.valid_to = datetime.now(UTC)

        # Create new baseline with same config but updated stats
        import uuid

        new_baseline_id = f"baseline_{uuid.uuid4().hex[:12]}"

        return self.create_baseline(
            baseline_id=new_baseline_id,
            tenant_id=old_baseline.tenant_id,
            name=old_baseline.name,
            scope=old_baseline.scope,
            scope_id=old_baseline.scope_id,
            metric_id=old_baseline.metric_id,
            stats=stats,
            window_config=json.loads(old_baseline.window_config)
            if old_baseline.window_config
            else None,
            created_by="system",
        )

    def get_stats(self, baseline: Baseline) -> dict[str, Any]:
        """Parse and return baseline statistics.

        Args:
            baseline: Baseline entity

        Returns:
            Parsed stats dictionary
        """
        if baseline.stats:
            return json.loads(baseline.stats)
        return {}

    def get_window_config(self, baseline: Baseline) -> dict[str, Any]:
        """Parse and return baseline window configuration.

        Args:
            baseline: Baseline entity

        Returns:
            Parsed window config dictionary
        """
        if baseline.window_config:
            return json.loads(baseline.window_config)
        return {}
