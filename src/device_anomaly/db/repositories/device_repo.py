"""Repository for Device entity operations.

This module provides data access methods for device records,
supporting multi-source devices (XSight, MobiControl, etc.).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from device_anomaly.db.models import Device
from device_anomaly.db.repositories.base import BaseRepository


class DeviceRepository(BaseRepository[Device]):
    """Repository for Device CRUD operations.

    Extends BaseRepository with device-specific query methods.
    """

    def __init__(self, session: Session):
        """Initialize the device repository.

        Args:
            session: SQLAlchemy session
        """
        super().__init__(session, Device)

    def get_by_tenant(
        self,
        tenant_id: str,
        source: str | None = None,
        device_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Device]:
        """Get devices for a tenant.

        Args:
            tenant_id: Tenant ID
            source: Optional source filter (xsight, mobicontrol, synthetic)
            device_type: Optional device type filter
            limit: Maximum records to return
            offset: Records to skip

        Returns:
            List of devices
        """
        query = self.session.query(Device).filter(Device.tenant_id == tenant_id)

        if source:
            query = query.filter(Device.source == source)
        if device_type:
            query = query.filter(Device.device_type == device_type)

        return query.order_by(Device.last_seen.desc()).offset(offset).limit(limit).all()

    def get_by_external_id(
        self,
        external_id: str,
        source: str,
        tenant_id: str | None = None,
    ) -> Device | None:
        """Get a device by its external source ID.

        Args:
            external_id: ID from the source system
            source: Source system (xsight, mobicontrol)
            tenant_id: Optional tenant filter

        Returns:
            Device if found, None otherwise
        """
        query = self.session.query(Device).filter(
            and_(
                Device.external_id == external_id,
                Device.source == source,
            )
        )

        if tenant_id:
            query = query.filter(Device.tenant_id == tenant_id)

        return query.first()

    def get_by_source(
        self,
        tenant_id: str,
        source: str,
        limit: int = 100,
    ) -> list[Device]:
        """Get all devices from a specific source.

        Args:
            tenant_id: Tenant ID
            source: Source system (xsight, mobicontrol, synthetic)
            limit: Maximum records to return

        Returns:
            List of devices from the source
        """
        return self.get_by_tenant(tenant_id=tenant_id, source=source, limit=limit)

    def upsert_device(
        self,
        device_id: str,
        tenant_id: str,
        source: str,
        external_id: str,
        name: str | None = None,
        device_type: str | None = None,
        os_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Device:
        """Create or update a device.

        Args:
            device_id: Internal device ID
            tenant_id: Tenant ID
            source: Source system
            external_id: ID from source system
            name: Device name
            device_type: Device type
            os_version: OS version
            metadata: Additional metadata

        Returns:
            Created or updated device
        """
        device = self.get_by_id(device_id, tenant_id)

        if device:
            # Update existing device
            device.external_id = external_id
            if name:
                device.name = name
            if device_type:
                device.device_type = device_type
            if os_version:
                device.os_version = os_version
            if metadata:
                device.metadata = json.dumps(metadata)
            device.last_seen = datetime.now(UTC)
            self.session.flush()
        else:
            # Create new device
            device = Device(
                device_id=device_id,
                tenant_id=tenant_id,
                source=source,
                external_id=external_id,
                name=name,
                device_type=device_type,
                os_version=os_version,
                metadata=json.dumps(metadata) if metadata else None,
                last_seen=datetime.now(UTC),
            )
            self.create(device)

        return device

    def update_last_seen(
        self,
        device_id: str,
        tenant_id: str,
    ) -> Device | None:
        """Update the last_seen timestamp for a device.

        Args:
            device_id: Device ID
            tenant_id: Tenant ID

        Returns:
            Updated device if found, None otherwise
        """
        device = self.get_by_id(device_id, tenant_id)
        if device:
            device.last_seen = datetime.now(UTC)
            self.session.flush()
        return device

    def count_by_source(self, tenant_id: str) -> dict[str, int]:
        """Count devices grouped by source.

        Args:
            tenant_id: Tenant ID

        Returns:
            Dict mapping source to count
        """
        results = (
            self.session.query(Device.source, func.count(Device.device_id))
            .filter(Device.tenant_id == tenant_id)
            .group_by(Device.source)
            .all()
        )
        return dict(results)

    def count_by_type(self, tenant_id: str) -> dict[str, int]:
        """Count devices grouped by device type.

        Args:
            tenant_id: Tenant ID

        Returns:
            Dict mapping device_type to count
        """
        results = (
            self.session.query(Device.device_type, func.count(Device.device_id))
            .filter(Device.tenant_id == tenant_id)
            .group_by(Device.device_type)
            .all()
        )
        return {device_type or "unknown": count for device_type, count in results}

    def get_stale_devices(
        self,
        tenant_id: str,
        stale_threshold_hours: int = 24,
        limit: int = 100,
    ) -> list[Device]:
        """Get devices that haven't been seen recently.

        Args:
            tenant_id: Tenant ID
            stale_threshold_hours: Hours since last_seen to consider stale
            limit: Maximum records to return

        Returns:
            List of stale devices
        """

        threshold = datetime.now(UTC) - timedelta(hours=stale_threshold_hours)

        return (
            self.session.query(Device)
            .filter(
                and_(
                    Device.tenant_id == tenant_id,
                    Device.last_seen < threshold,
                )
            )
            .order_by(Device.last_seen.asc())
            .limit(limit)
            .all()
        )

    def search_by_name(
        self,
        tenant_id: str,
        name_pattern: str,
        limit: int = 50,
    ) -> list[Device]:
        """Search devices by name pattern.

        Args:
            tenant_id: Tenant ID
            name_pattern: Pattern to match (supports SQL LIKE wildcards)
            limit: Maximum records to return

        Returns:
            List of matching devices
        """
        return (
            self.session.query(Device)
            .filter(
                and_(
                    Device.tenant_id == tenant_id,
                    Device.name.ilike(f"%{name_pattern}%"),
                )
            )
            .limit(limit)
            .all()
        )

    def get_metadata(self, device: Device) -> dict[str, Any]:
        """Parse and return device metadata.

        Args:
            device: Device entity

        Returns:
            Parsed metadata dictionary
        """
        if device.metadata:
            return json.loads(device.metadata)
        return {}
