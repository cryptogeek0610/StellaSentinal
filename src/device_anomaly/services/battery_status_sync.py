"""Battery Status Sync Service.

Syncs battery health data to SOTI MobiControl Custom Attributes,
enabling visibility of battery replacement status directly in the web console.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from device_anomaly.config.settings import get_settings
from device_anomaly.data_access.mobicontrol_client import MobiControlClient, MobiControlAPIError
from device_anomaly.database.schema import DeviceFeature, DeviceMetadata

logger = logging.getLogger(__name__)


@dataclass
class BatteryStatusUpdate:
    """Battery status data for a device."""

    device_id: str
    mc_device_id: str  # MobiControl device ID (may differ from internal ID)
    battery_health_percent: Optional[float] = None
    battery_status: str = "Unknown"
    replacement_due: bool = False
    replacement_urgency: str = "None"
    estimated_replacement_date: Optional[str] = None


class BatteryStatusSyncService:
    """Service for syncing battery status to MobiControl Custom Attributes."""

    # Battery health thresholds
    HEALTH_CRITICAL = 60  # Below this = Replace Now
    HEALTH_WARNING = 70   # Below this = Replace Soon
    HEALTH_CAUTION = 80   # Below this = Plan Replacement

    def __init__(
        self,
        results_db: Session,
        tenant_id: str,
        mc_client: Optional[MobiControlClient] = None,
    ):
        """Initialize the battery status sync service.

        Args:
            results_db: SQLAlchemy session for results database
            tenant_id: Tenant identifier
            mc_client: Optional MobiControl client (created if not provided)
        """
        self.results_db = results_db
        self.tenant_id = tenant_id
        self.mc_client = mc_client

    def _get_mc_client(self) -> MobiControlClient:
        """Get or create MobiControl client."""
        if self.mc_client is None:
            settings = get_settings()
            if not settings.mobicontrol.server_url:
                raise ValueError("MobiControl is not configured. Set MOBICONTROL_SERVER_URL.")
            self.mc_client = MobiControlClient()
        return self.mc_client

    def _determine_battery_status(self, health: Optional[float]) -> tuple[str, str, bool]:
        """Determine battery status, urgency, and replacement due flag.

        Args:
            health: Battery health percentage (0-100)

        Returns:
            Tuple of (status_string, urgency_string, replacement_due_bool)
        """
        if health is None:
            return ("Unknown", "Unknown", False)

        if health < self.HEALTH_CRITICAL:
            return ("Replace Now", "Immediate", True)
        elif health < self.HEALTH_WARNING:
            return ("Replace Soon", "High", True)
        elif health < self.HEALTH_CAUTION:
            return ("Plan Replacement", "Medium", False)
        else:
            return ("Good", "None", False)

    def get_devices_battery_status(
        self,
        device_ids: Optional[List[int]] = None,
    ) -> List[BatteryStatusUpdate]:
        """Get battery status for devices from feature data.

        Args:
            device_ids: Optional list of device IDs to filter (all if None)

        Returns:
            List of BatteryStatusUpdate objects
        """
        from sqlalchemy import func

        # Get device metadata for MC device ID mapping
        device_query = (
            self.results_db.query(DeviceMetadata)
            .filter(DeviceMetadata.tenant_id == self.tenant_id)
        )
        if device_ids:
            device_query = device_query.filter(DeviceMetadata.device_id.in_(device_ids))

        devices = {d.device_id: d for d in device_query.all()}

        if not devices:
            logger.warning("No devices found for tenant %s", self.tenant_id)
            return []

        # Get latest feature data per device
        subq = (
            self.results_db.query(
                DeviceFeature.device_id,
                func.max(DeviceFeature.computed_at).label("latest")
            )
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.device_id.in_(list(devices.keys()))
            )
            .group_by(DeviceFeature.device_id)
            .subquery()
        )

        latest_features = (
            self.results_db.query(DeviceFeature)
            .join(
                subq,
                (DeviceFeature.device_id == subq.c.device_id) &
                (DeviceFeature.computed_at == subq.c.latest)
            )
            .filter(DeviceFeature.tenant_id == self.tenant_id)
            .all()
        )

        # Build status updates
        updates = []
        for feature in latest_features:
            device = devices.get(feature.device_id)
            if not device:
                continue

            # Extract battery health from features
            health = None
            if feature.feature_values_json:
                try:
                    features = json.loads(feature.feature_values_json)
                    health = features.get("BatteryHealth") or features.get("battery_health")
                    if health is not None:
                        health = float(health)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            status, urgency, replacement_due = self._determine_battery_status(health)

            # Use device_id as MC device ID (they should match, but could be mapped differently)
            mc_device_id = str(device.device_id)

            updates.append(BatteryStatusUpdate(
                device_id=str(feature.device_id),
                mc_device_id=mc_device_id,
                battery_health_percent=health,
                battery_status=status,
                replacement_due=replacement_due,
                replacement_urgency=urgency,
            ))

        return updates

    def sync_battery_status(
        self,
        device_ids: Optional[List[int]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Sync battery status to MobiControl Custom Attributes.

        Args:
            device_ids: Optional list of device IDs to sync (all if None)
            dry_run: If True, don't actually update MobiControl, just return what would be updated

        Returns:
            Dictionary with sync results
        """
        start_time = datetime.now(timezone.utc)
        result = {
            "success": True,
            "devices_processed": 0,
            "devices_updated": 0,
            "devices_skipped": 0,
            "devices_failed": 0,
            "errors": [],
            "dry_run": dry_run,
            "started_at": start_time.isoformat(),
        }

        try:
            # Get battery status for devices
            updates = self.get_devices_battery_status(device_ids)
            result["devices_processed"] = len(updates)

            if not updates:
                result["message"] = "No devices with battery data found"
                return result

            if dry_run:
                # Just return what would be updated
                result["updates"] = [
                    {
                        "device_id": u.device_id,
                        "mc_device_id": u.mc_device_id,
                        "battery_health": u.battery_health_percent,
                        "status": u.battery_status,
                        "urgency": u.replacement_urgency,
                        "replacement_due": u.replacement_due,
                    }
                    for u in updates
                ]
                result["message"] = f"Dry run: {len(updates)} devices would be updated"
                return result

            # Get MobiControl client
            mc_client = self._get_mc_client()

            # Update each device
            for update in updates:
                try:
                    mc_client.set_battery_status_attributes(
                        device_id=update.mc_device_id,
                        battery_health_percent=update.battery_health_percent,
                        battery_status=update.battery_status,
                        replacement_due=update.replacement_due,
                        replacement_urgency=update.replacement_urgency,
                        estimated_replacement_date=update.estimated_replacement_date,
                    )
                    result["devices_updated"] += 1
                    logger.debug(
                        "Updated battery status for device %s: %s (%s)",
                        update.device_id,
                        update.battery_status,
                        update.battery_health_percent,
                    )
                except MobiControlAPIError as e:
                    result["devices_failed"] += 1
                    error_info = {
                        "device_id": update.device_id,
                        "error": str(e),
                        "status_code": e.status_code,
                    }
                    result["errors"].append(error_info)
                    logger.warning(
                        "Failed to update battery status for device %s: %s",
                        update.device_id,
                        e,
                    )
                except Exception as e:
                    result["devices_failed"] += 1
                    result["errors"].append({
                        "device_id": update.device_id,
                        "error": str(e),
                    })
                    logger.exception(
                        "Unexpected error updating battery status for device %s",
                        update.device_id,
                    )

            result["success"] = result["devices_failed"] == 0
            result["message"] = (
                f"Updated {result['devices_updated']} devices, "
                f"{result['devices_failed']} failed"
            )

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.exception("Battery status sync failed")

        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        return result

    def sync_single_device(
        self,
        device_id: int,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Sync battery status for a single device.

        Args:
            device_id: Device ID to sync
            dry_run: If True, don't actually update MobiControl

        Returns:
            Dictionary with sync result
        """
        return self.sync_battery_status(device_ids=[device_id], dry_run=dry_run)


def sync_battery_status_to_mobicontrol(
    results_db: Session,
    tenant_id: str,
    device_ids: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Convenience function to sync battery status to MobiControl.

    Args:
        results_db: SQLAlchemy session
        tenant_id: Tenant identifier
        device_ids: Optional list of device IDs (all if None)
        dry_run: If True, don't actually update

    Returns:
        Dictionary with sync results
    """
    service = BatteryStatusSyncService(results_db, tenant_id)
    return service.sync_battery_status(device_ids, dry_run)
