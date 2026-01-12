"""API routes for SOTI MobiControl device control actions.

These endpoints enable remote device management via the MobiControl API:
- Lock, restart, wipe devices
- Send messages to devices
- Locate devices
- Force sync/check-in
- Clear app data/cache
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from device_anomaly.api.dependencies import get_db, get_mock_mode, get_tenant_id
from device_anomaly.config.settings import get_settings
from device_anomaly.database.schema import DeviceActionLog

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/devices", tags=["device-actions"])


# =============================================================================
# Pydantic Models
# =============================================================================


class ActionType(str, Enum):
    """Available device control actions."""
    LOCK = "lock"
    RESTART = "restart"
    WIPE = "wipe"
    MESSAGE = "message"
    LOCATE = "locate"
    SYNC = "sync"
    CLEAR_PASSCODE = "clearPasscode"
    CLEAR_CACHE = "clearCache"


class DeviceActionRequest(BaseModel):
    """Base request for device actions."""
    reason: Optional[str] = Field(None, description="Reason for the action (for audit)")


class SendMessageRequest(DeviceActionRequest):
    """Request to send a message to a device."""
    title: str = Field(default="Message from Admin", max_length=100)
    message: str = Field(..., min_length=1, max_length=500)


class WipeDeviceRequest(DeviceActionRequest):
    """Request to wipe a device."""
    factory_reset: bool = Field(default=False, description="Perform factory reset instead of enterprise wipe")
    confirm: bool = Field(default=False, description="Confirmation required for wipe action")


class ClearAppDataRequest(DeviceActionRequest):
    """Request to clear app data."""
    package_name: str = Field(..., description="Package name of the app to clear")


class DeviceActionResponse(BaseModel):
    """Response for device actions."""
    success: bool
    action: str
    device_id: int
    message: str
    action_id: Optional[str] = None
    timestamp: str


class DeviceLocationResponse(BaseModel):
    """Response for device location."""
    success: bool
    device_id: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: Optional[str] = None
    message: str


# =============================================================================
# Helper Functions
# =============================================================================


def get_mobicontrol_client():
    """Get MobiControl API client instance."""
    settings = get_settings()

    if not settings.enable_mobicontrol:
        return None

    if not settings.mobicontrol.server_url:
        return None

    try:
        from device_anomaly.data_access.mobicontrol_client import MobiControlClient
        return MobiControlClient()
    except Exception as e:
        logger.error(f"Failed to create MobiControl client: {e}")
        return None


def log_device_action(
    db: Session,
    device_id: int,
    action_type: str,
    tenant_id: str,
    user: str = "system",
    reason: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
) -> DeviceActionLog:
    """Log a device action to the database for audit purposes."""
    try:
        log_entry = DeviceActionLog(
            device_id=device_id,
            tenant_id=tenant_id,
            action_type=action_type,
            initiated_by=user,
            reason=reason,
            success=success,
            error_message=error_message,
            timestamp=datetime.now(timezone.utc),
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        return log_entry
    except Exception as e:
        logger.error(f"Failed to log device action: {e}")
        db.rollback()
        # Don't fail the action just because logging failed
        return None


def create_mock_response(
    action: str,
    device_id: int,
    message: str = "Action executed successfully (mock mode)",
) -> DeviceActionResponse:
    """Create a mock response for testing."""
    return DeviceActionResponse(
        success=True,
        action=action,
        device_id=device_id,
        message=message,
        action_id=f"mock-{action}-{device_id}",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/{device_id}/actions/lock", response_model=DeviceActionResponse)
def lock_device(
    device_id: int,
    request: DeviceActionRequest = None,
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Lock a device remotely via MobiControl.

    This action will immediately lock the device screen, requiring the user
    to enter their PIN/password/biometric to unlock.
    """
    tenant_id = get_tenant_id()
    request = request or DeviceActionRequest()

    if mock_mode:
        log_device_action(db, device_id, "lock", tenant_id, reason=request.reason)
        return create_mock_response("lock", device_id, "Device locked successfully (mock mode)")

    client = get_mobicontrol_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="MobiControl integration not configured. Enable it in settings.",
        )

    try:
        result = client.lock_device(str(device_id))
        log_device_action(db, device_id, "lock", tenant_id, reason=request.reason)
        return DeviceActionResponse(
            success=True,
            action="lock",
            device_id=device_id,
            message="Device lock command sent successfully",
            action_id=result.get("actionId"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to lock device {device_id}: {e}")
        log_device_action(db, device_id, "lock", tenant_id, reason=request.reason, success=False, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to lock device: {e}")


@router.post("/{device_id}/actions/restart", response_model=DeviceActionResponse)
def restart_device(
    device_id: int,
    request: DeviceActionRequest = None,
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Restart a device remotely via MobiControl.

    This action will trigger a soft reboot of the device.
    """
    tenant_id = get_tenant_id()
    request = request or DeviceActionRequest()

    if mock_mode:
        log_device_action(db, device_id, "restart", tenant_id, reason=request.reason)
        return create_mock_response("restart", device_id, "Device restart initiated (mock mode)")

    client = get_mobicontrol_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="MobiControl integration not configured. Enable it in settings.",
        )

    try:
        result = client.restart_device(str(device_id))
        log_device_action(db, device_id, "restart", tenant_id, reason=request.reason)
        return DeviceActionResponse(
            success=True,
            action="restart",
            device_id=device_id,
            message="Device restart command sent successfully",
            action_id=result.get("actionId"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to restart device {device_id}: {e}")
        log_device_action(db, device_id, "restart", tenant_id, reason=request.reason, success=False, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to restart device: {e}")


@router.post("/{device_id}/actions/wipe", response_model=DeviceActionResponse)
def wipe_device(
    device_id: int,
    request: WipeDeviceRequest,
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Wipe a device remotely via MobiControl.

    WARNING: This is a destructive action that cannot be undone.

    - Enterprise wipe: Removes managed data and apps, keeps personal data
    - Factory reset: Complete device reset to factory state

    Requires explicit confirmation (confirm=true) to proceed.
    """
    tenant_id = get_tenant_id()

    if not request.confirm:
        raise HTTPException(
            status_code=400,
            detail="Wipe action requires explicit confirmation. Set confirm=true to proceed.",
        )

    wipe_type = "factory_reset" if request.factory_reset else "enterprise_wipe"

    if mock_mode:
        log_device_action(db, device_id, wipe_type, tenant_id, reason=request.reason)
        return create_mock_response(wipe_type, device_id, f"Device {wipe_type} initiated (mock mode)")

    client = get_mobicontrol_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="MobiControl integration not configured. Enable it in settings.",
        )

    try:
        result = client.wipe_device(str(device_id), factory_reset=request.factory_reset)
        log_device_action(db, device_id, wipe_type, tenant_id, reason=request.reason)
        return DeviceActionResponse(
            success=True,
            action=wipe_type,
            device_id=device_id,
            message=f"Device {wipe_type.replace('_', ' ')} command sent successfully",
            action_id=result.get("actionId"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to wipe device {device_id}: {e}")
        log_device_action(db, device_id, wipe_type, tenant_id, reason=request.reason, success=False, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to wipe device: {e}")


@router.post("/{device_id}/actions/message", response_model=DeviceActionResponse)
def send_message_to_device(
    device_id: int,
    request: SendMessageRequest,
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Send a message to a device via MobiControl.

    The message will appear as a notification on the device.
    """
    tenant_id = get_tenant_id()

    if mock_mode:
        log_device_action(db, device_id, "message", tenant_id, reason=f"Message: {request.message[:50]}...")
        return create_mock_response("message", device_id, "Message sent successfully (mock mode)")

    client = get_mobicontrol_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="MobiControl integration not configured. Enable it in settings.",
        )

    try:
        result = client.send_message(str(device_id), request.message, request.title)
        log_device_action(db, device_id, "message", tenant_id, reason=f"Message: {request.message[:50]}...")
        return DeviceActionResponse(
            success=True,
            action="message",
            device_id=device_id,
            message="Message sent to device successfully",
            action_id=result.get("actionId"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to send message to device {device_id}: {e}")
        log_device_action(db, device_id, "message", tenant_id, reason=request.reason, success=False, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to send message: {e}")


@router.post("/{device_id}/actions/locate", response_model=DeviceLocationResponse)
def locate_device(
    device_id: int,
    request: DeviceActionRequest = None,
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Request current location of a device via MobiControl.

    This triggers the device to report its current GPS location.
    """
    tenant_id = get_tenant_id()
    request = request or DeviceActionRequest()

    if mock_mode:
        log_device_action(db, device_id, "locate", tenant_id, reason=request.reason)
        return DeviceLocationResponse(
            success=True,
            device_id=device_id,
            latitude=52.3676,  # Example: Amsterdam
            longitude=4.9041,
            accuracy=10.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Location retrieved successfully (mock mode)",
        )

    client = get_mobicontrol_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="MobiControl integration not configured. Enable it in settings.",
        )

    try:
        result = client.locate_device(str(device_id))
        log_device_action(db, device_id, "locate", tenant_id, reason=request.reason)

        # Extract location data from response
        location = result.get("location", result)
        return DeviceLocationResponse(
            success=True,
            device_id=device_id,
            latitude=location.get("latitude"),
            longitude=location.get("longitude"),
            accuracy=location.get("accuracy"),
            timestamp=location.get("timestamp", datetime.now(timezone.utc).isoformat()),
            message="Location request sent. Device will report location when online.",
        )
    except Exception as e:
        logger.error(f"Failed to locate device {device_id}: {e}")
        log_device_action(db, device_id, "locate", tenant_id, reason=request.reason, success=False, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to locate device: {e}")


@router.post("/{device_id}/actions/sync", response_model=DeviceActionResponse)
def sync_device(
    device_id: int,
    request: DeviceActionRequest = None,
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Force a device to sync/check-in with MobiControl server.

    This triggers the device to immediately report its status and telemetry.
    """
    tenant_id = get_tenant_id()
    request = request or DeviceActionRequest()

    if mock_mode:
        log_device_action(db, device_id, "sync", tenant_id, reason=request.reason)
        return create_mock_response("sync", device_id, "Device sync initiated (mock mode)")

    client = get_mobicontrol_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="MobiControl integration not configured. Enable it in settings.",
        )

    try:
        result = client.sync_device(str(device_id))
        log_device_action(db, device_id, "sync", tenant_id, reason=request.reason)
        return DeviceActionResponse(
            success=True,
            action="sync",
            device_id=device_id,
            message="Device sync command sent successfully",
            action_id=result.get("actionId"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to sync device {device_id}: {e}")
        log_device_action(db, device_id, "sync", tenant_id, reason=request.reason, success=False, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to sync device: {e}")


@router.post("/{device_id}/actions/clear-cache", response_model=DeviceActionResponse)
def clear_device_cache(
    device_id: int,
    request: ClearAppDataRequest = None,
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Clear app cache/data on a device via MobiControl.

    If no package_name is provided, this clears general system cache.
    """
    tenant_id = get_tenant_id()

    if mock_mode:
        pkg = request.package_name if request else "system"
        log_device_action(db, device_id, "clearCache", tenant_id, reason=f"Clear cache for: {pkg}")
        return create_mock_response("clearCache", device_id, f"Cache cleared for {pkg} (mock mode)")

    client = get_mobicontrol_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="MobiControl integration not configured. Enable it in settings.",
        )

    try:
        if request and request.package_name:
            result = client.clear_app_data(str(device_id), request.package_name)
            message = f"Cache cleared for {request.package_name}"
        else:
            # General cache clear - may need different API endpoint
            result = client.clear_app_data(str(device_id), "*")
            message = "System cache cleared"

        log_device_action(db, device_id, "clearCache", tenant_id, reason=request.reason if request else None)
        return DeviceActionResponse(
            success=True,
            action="clearCache",
            device_id=device_id,
            message=message,
            action_id=result.get("actionId"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to clear cache on device {device_id}: {e}")
        log_device_action(db, device_id, "clearCache", tenant_id, reason=request.reason if request else None, success=False, error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")


@router.get("/{device_id}/actions/history")
def get_device_action_history(
    device_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Get action history for a device.

    Returns a list of all actions performed on this device.
    """
    tenant_id = get_tenant_id()

    try:
        actions = (
            db.query(DeviceActionLog)
            .filter(DeviceActionLog.device_id == device_id)
            .filter(DeviceActionLog.tenant_id == tenant_id)
            .order_by(DeviceActionLog.timestamp.desc())
            .limit(limit)
            .all()
        )

        return {
            "device_id": device_id,
            "total": len(actions),
            "actions": [
                {
                    "id": a.id,
                    "action_type": a.action_type,
                    "initiated_by": a.initiated_by,
                    "reason": a.reason,
                    "success": a.success,
                    "error_message": a.error_message,
                    "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                }
                for a in actions
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get action history for device {device_id}: {e}")
        # Return empty history if table doesn't exist yet
        return {"device_id": device_id, "total": 0, "actions": []}


# =============================================================================
# Battery Status Sync to MobiControl Custom Attributes
# =============================================================================


class BatterySyncRequest(BaseModel):
    """Request to sync battery status to MobiControl."""
    device_ids: Optional[list[int]] = Field(
        None,
        description="Specific device IDs to sync (all devices if not provided)"
    )
    dry_run: bool = Field(
        False,
        description="If true, return what would be updated without making changes"
    )


class BatterySyncResponse(BaseModel):
    """Response from battery status sync."""
    success: bool
    devices_processed: int
    devices_updated: int
    devices_failed: int
    message: str
    dry_run: bool
    duration_seconds: Optional[float] = None
    errors: list[dict] = []
    updates: Optional[list[dict]] = None


@router.post("/battery-status/sync", response_model=BatterySyncResponse)
def sync_battery_status_to_mobicontrol(
    request: BatterySyncRequest,
    db: Session = Depends(get_db),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Sync battery health status to MobiControl Custom Attributes.

    This endpoint pushes battery health data to SOTI MobiControl as Custom Attributes,
    enabling visibility of battery replacement status directly in the web console.

    Custom Attributes set:
    - BatteryHealthPercent: Battery health as percentage (e.g., "72%")
    - BatteryStatus: Status string ("Good", "Plan Replacement", "Replace Soon", "Replace Now")
    - BatteryReplacementDue: Whether replacement is needed ("Yes" or "No")
    - BatteryReplacementUrgency: Urgency level ("None", "Medium", "High", "Immediate")

    Use dry_run=true to preview changes without updating MobiControl.
    """
    tenant_id = get_tenant_id()

    if mock_mode:
        # Return mock response
        return BatterySyncResponse(
            success=True,
            devices_processed=5,
            devices_updated=5,
            devices_failed=0,
            message="Mock: 5 devices updated",
            dry_run=request.dry_run,
            duration_seconds=0.5,
            errors=[],
            updates=[
                {"device_id": "1001", "status": "Good", "health": 85},
                {"device_id": "1002", "status": "Plan Replacement", "health": 75},
                {"device_id": "1003", "status": "Replace Soon", "health": 65},
                {"device_id": "1004", "status": "Replace Now", "health": 55},
                {"device_id": "1005", "status": "Good", "health": 92},
            ] if request.dry_run else None,
        )

    try:
        from device_anomaly.services.battery_status_sync import BatteryStatusSyncService

        service = BatteryStatusSyncService(db, tenant_id)
        result = service.sync_battery_status(
            device_ids=request.device_ids,
            dry_run=request.dry_run,
        )

        return BatterySyncResponse(
            success=result.get("success", False),
            devices_processed=result.get("devices_processed", 0),
            devices_updated=result.get("devices_updated", 0),
            devices_failed=result.get("devices_failed", 0),
            message=result.get("message", ""),
            dry_run=request.dry_run,
            duration_seconds=result.get("duration_seconds"),
            errors=result.get("errors", []),
            updates=result.get("updates") if request.dry_run else None,
        )

    except ValueError as e:
        # Configuration error (e.g., MobiControl not configured)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Battery status sync failed")
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")


@router.post("/{device_id}/battery-status/sync", response_model=BatterySyncResponse)
def sync_single_device_battery_status(
    device_id: int,
    dry_run: bool = False,
    db: Session = Depends(get_db),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Sync battery status for a single device to MobiControl.

    Args:
        device_id: Device ID to sync
        dry_run: If true, return what would be updated without making changes
    """
    tenant_id = get_tenant_id()

    if mock_mode:
        return BatterySyncResponse(
            success=True,
            devices_processed=1,
            devices_updated=1,
            devices_failed=0,
            message=f"Mock: Device {device_id} updated",
            dry_run=dry_run,
            duration_seconds=0.1,
            errors=[],
        )

    try:
        from device_anomaly.services.battery_status_sync import BatteryStatusSyncService

        service = BatteryStatusSyncService(db, tenant_id)
        result = service.sync_single_device(device_id, dry_run=dry_run)

        return BatterySyncResponse(
            success=result.get("success", False),
            devices_processed=result.get("devices_processed", 0),
            devices_updated=result.get("devices_updated", 0),
            devices_failed=result.get("devices_failed", 0),
            message=result.get("message", ""),
            dry_run=dry_run,
            duration_seconds=result.get("duration_seconds"),
            errors=result.get("errors", []),
            updates=result.get("updates") if dry_run else None,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Battery status sync failed for device {device_id}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")
