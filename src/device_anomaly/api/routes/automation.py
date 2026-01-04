"""
Automation API Routes - Scheduler Configuration and Status.

This module provides REST endpoints for:
- Viewing and configuring the automation scheduler
- Triggering manual training/scoring jobs
- Monitoring real-time status
- Managing alerts and notifications
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import redis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/automation", tags=["Automation"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ScheduleInterval(str, Enum):
    """Supported scheduling intervals."""
    HOURLY = "hourly"
    EVERY_6_HOURS = "every_6_hours"
    EVERY_12_HOURS = "every_12_hours"
    DAILY = "daily"
    WEEKLY = "weekly"
    MANUAL = "manual"


class SchedulerConfigResponse(BaseModel):
    """Scheduler configuration response."""
    # Training settings
    training_enabled: bool = True
    training_interval: ScheduleInterval = ScheduleInterval.DAILY
    training_hour: int = Field(2, ge=0, le=23, description="Hour of day (0-23)")
    training_day_of_week: int = Field(0, ge=0, le=6, description="Day of week (0=Mon)")
    training_lookback_days: int = Field(90, ge=7, le=365)
    training_validation_days: int = Field(7, ge=1, le=30)

    # Scoring settings
    scoring_enabled: bool = True
    scoring_interval_minutes: int = Field(15, ge=5, le=1440)

    # Auto-retrain settings
    auto_retrain_enabled: bool = True
    auto_retrain_fp_threshold: float = Field(0.15, ge=0.01, le=0.50)
    auto_retrain_min_feedback: int = Field(50, ge=10, le=1000)
    auto_retrain_cooldown_hours: int = Field(24, ge=1, le=168)

    # Alerting
    alerting_enabled: bool = True
    alert_on_high_anomaly_rate: bool = True
    high_anomaly_rate_threshold: float = Field(0.10, ge=0.01, le=0.50)

    # Insight generation settings
    insights_enabled: bool = True
    daily_digest_hour: int = Field(5, ge=0, le=23, description="Hour for daily digest (0-23)")
    shift_readiness_enabled: bool = True
    shift_readiness_lead_minutes: int = Field(60, ge=15, le=180, description="Minutes before shift to generate readiness report")
    shift_schedules: List[str] = Field(default_factory=lambda: ["morning", "afternoon", "day"])
    location_baseline_enabled: bool = True
    location_baseline_day_of_week: int = Field(0, ge=0, le=6, description="Day of week for baseline computation (0=Mon)")
    location_baseline_hour: int = Field(3, ge=0, le=23, description="Hour for baseline computation (0-23)")

    # Timestamps
    last_training_time: Optional[str] = None
    last_scoring_time: Optional[str] = None
    last_auto_retrain_time: Optional[str] = None
    last_daily_digest_time: Optional[str] = None
    last_shift_readiness_time: Optional[str] = None
    last_location_baseline_time: Optional[str] = None


class SchedulerConfigUpdate(BaseModel):
    """Scheduler configuration update request."""
    training_enabled: Optional[bool] = None
    training_interval: Optional[ScheduleInterval] = None
    training_hour: Optional[int] = Field(None, ge=0, le=23)
    training_day_of_week: Optional[int] = Field(None, ge=0, le=6)
    training_lookback_days: Optional[int] = Field(None, ge=7, le=365)
    training_validation_days: Optional[int] = Field(None, ge=1, le=30)
    scoring_enabled: Optional[bool] = None
    scoring_interval_minutes: Optional[int] = Field(None, ge=5, le=1440)
    auto_retrain_enabled: Optional[bool] = None
    auto_retrain_fp_threshold: Optional[float] = Field(None, ge=0.01, le=0.50)
    auto_retrain_min_feedback: Optional[int] = Field(None, ge=10, le=1000)
    auto_retrain_cooldown_hours: Optional[int] = Field(None, ge=1, le=168)
    alerting_enabled: Optional[bool] = None
    alert_on_high_anomaly_rate: Optional[bool] = None
    high_anomaly_rate_threshold: Optional[float] = Field(None, ge=0.01, le=0.50)
    # Insight generation settings
    insights_enabled: Optional[bool] = None
    daily_digest_hour: Optional[int] = Field(None, ge=0, le=23)
    shift_readiness_enabled: Optional[bool] = None
    shift_readiness_lead_minutes: Optional[int] = Field(None, ge=15, le=180)
    shift_schedules: Optional[List[str]] = None
    location_baseline_enabled: Optional[bool] = None
    location_baseline_day_of_week: Optional[int] = Field(None, ge=0, le=6)
    location_baseline_hour: Optional[int] = Field(None, ge=0, le=23)


class SchedulerStatusResponse(BaseModel):
    """Scheduler status response."""
    is_running: bool = False
    training_status: str = "idle"
    scoring_status: str = "idle"
    insights_status: str = "idle"
    last_training_result: Optional[Dict[str, Any]] = None
    last_scoring_result: Optional[Dict[str, Any]] = None
    last_insight_result: Optional[Dict[str, Any]] = None
    next_training_time: Optional[str] = None
    next_scoring_time: Optional[str] = None
    next_insight_time: Optional[str] = None
    total_anomalies_detected: int = 0
    total_insights_generated: int = 0
    false_positive_rate: float = 0.0
    uptime_seconds: int = 0
    errors: List[str] = Field(default_factory=list)


class AlertResponse(BaseModel):
    """Alert notification."""
    id: str
    timestamp: str
    message: str
    acknowledged: bool = False


class ManualJobRequest(BaseModel):
    """Request to trigger a manual job."""
    job_type: str = Field(..., description="Job type: 'training', 'scoring', 'daily_digest', 'shift_readiness', or 'location_baseline'")
    # Training-specific
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    validation_days: Optional[int] = 7
    # Shift readiness-specific
    shift_name: Optional[str] = Field(None, description="Shift name for shift_readiness job: 'morning', 'afternoon', 'night', or 'day'")


class ManualJobResponse(BaseModel):
    """Response from manual job trigger."""
    success: bool
    job_id: Optional[str] = None
    message: str


class ScoreRequest(BaseModel):
    """Request to score specific devices."""
    device_ids: Optional[List[int]] = None
    start_date: str
    end_date: str


class ScoreResponse(BaseModel):
    """Scoring response."""
    success: bool
    total_scored: int = 0
    anomalies_detected: int = 0
    anomaly_rate: float = 0.0
    results: Optional[List[Dict[str, Any]]] = None


class JobHistoryEntry(BaseModel):
    """Job history entry for automation history."""
    type: str
    timestamp: str
    triggered_by: str = "schedule"
    success: bool = True
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_alert_id(timestamp: str, message: str) -> str:
    """Generate a stable, unique alert ID based on timestamp and message content."""
    import hashlib
    content = f"{timestamp}:{message}"
    return f"alert_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

def get_redis_client() -> redis.Redis:
    """Get Redis client."""
    import os
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.from_url(redis_url, decode_responses=True)


def get_scheduler_config() -> Dict[str, Any]:
    """Get current scheduler configuration from Redis."""
    try:
        client = get_redis_client()
        data = client.get("scheduler:config")
        if data:
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to get scheduler config: {e}")
    return {}


def save_scheduler_config(config: Dict[str, Any]) -> None:
    """Save scheduler configuration to Redis."""
    try:
        client = get_redis_client()
        client.set("scheduler:config", json.dumps(config))
    except Exception as e:
        logger.error(f"Failed to save scheduler config: {e}")
        raise HTTPException(status_code=500, detail="Failed to save configuration")


def get_scheduler_status() -> Dict[str, Any]:
    """Get current scheduler status from Redis."""
    try:
        client = get_redis_client()
        data = client.get("scheduler:status")
        if data:
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to get scheduler status: {e}")
    return {"is_running": False}


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/config", response_model=SchedulerConfigResponse)
async def get_config():
    """
    Get current scheduler configuration.

    Returns all configurable settings for the automation scheduler.
    """
    config = get_scheduler_config()
    return SchedulerConfigResponse(**config) if config else SchedulerConfigResponse()


@router.put("/config", response_model=SchedulerConfigResponse)
async def update_config(update: SchedulerConfigUpdate):
    """
    Update scheduler configuration.

    Only provided fields will be updated. Omitted fields retain their current values.
    """
    current = get_scheduler_config()

    # Apply updates
    update_dict = update.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        if value is not None:
            if isinstance(value, Enum):
                current[key] = value.value
            else:
                current[key] = value

    save_scheduler_config(current)
    logger.info(f"Scheduler config updated: {list(update_dict.keys())}")

    return SchedulerConfigResponse(**current)


@router.get("/status", response_model=SchedulerStatusResponse)
async def get_status():
    """
    Get current scheduler status.

    Returns real-time information about scheduler state, running jobs,
    and recent results.
    """
    status = get_scheduler_status()
    return SchedulerStatusResponse(**status)


@router.post("/start", response_model=Dict[str, Any])
async def start_scheduler():
    """
    Start the automation scheduler.

    Note: In production, the scheduler runs as a separate service.
    This endpoint signals the scheduler to start processing.
    """
    try:
        client = get_redis_client()
        client.set("scheduler:command", "start")
        return {"success": True, "message": "Start signal sent to scheduler"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=Dict[str, Any])
async def stop_scheduler():
    """
    Stop the automation scheduler.

    Gracefully stops all scheduled jobs.
    """
    try:
        client = get_redis_client()
        client.set("scheduler:command", "stop")
        return {"success": True, "message": "Stop signal sent to scheduler"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger", response_model=ManualJobResponse)
async def trigger_manual_job(request: ManualJobRequest):
    """
    Manually trigger a training or scoring job.

    Useful for:
    - Testing the pipeline
    - Running an immediate training after configuration changes
    - On-demand scoring of specific date ranges
    """
    try:
        client = get_redis_client()

        if request.job_type == "training":
            # Create training job with format expected by ml-worker
            job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            submitted_at = datetime.utcnow().isoformat()

            # Use historical end date if configured, otherwise use today
            import os
            historical_date_str = os.getenv("HISTORICAL_DATA_END_DATE")
            if historical_date_str:
                try:
                    historical_end = datetime.strptime(historical_date_str, "%Y-%m-%d")
                except ValueError:
                    historical_end = None
            else:
                historical_end = None

            if historical_end:
                default_end = historical_end.strftime("%Y-%m-%d")
                # Start date: 90 days before end date for training
                default_start = (historical_end - timedelta(days=90)).strftime("%Y-%m-%d")
            else:
                default_end = datetime.utcnow().strftime("%Y-%m-%d")
                default_start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")

            job_data = {
                "job_id": job_id,
                "config": {
                    "start_date": request.start_date or default_start,
                    "end_date": request.end_date or default_end,
                    "validation_days": request.validation_days or 7,
                    "contamination": 0.03,
                    "n_estimators": 300,
                    "export_onnx": True,
                },
                "tenant_id": "default",
                "submitted_at": submitted_at,
                "status": "pending",
            }
            client.rpush("ml:training:queue", json.dumps(job_data))

            # Update status to show pending job
            client.set("ml:training:status", json.dumps({
                "run_id": job_id,
                "status": "pending",
                "progress": 0,
                "message": "Job queued, waiting for worker...",
                "submitted_at": submitted_at,
            }))

            return ManualJobResponse(
                success=True,
                job_id=job_id,
                message="Training job queued successfully",
            )

        elif request.job_type == "scoring":
            # Create scoring job
            job_data = {
                "type": "scoring",
                "start_date": request.start_date or datetime.utcnow().strftime("%Y-%m-%d"),
                "end_date": request.end_date or datetime.utcnow().strftime("%Y-%m-%d"),
                "triggered_by": "manual",
                "timestamp": datetime.utcnow().isoformat(),
            }
            client.rpush("scheduler:scoring:queue", json.dumps(job_data))
            return ManualJobResponse(
                success=True,
                job_id=f"score_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                message="Scoring job queued successfully",
            )

        elif request.job_type == "daily_digest":
            # Queue daily digest insight generation
            job_data = {
                "type": "daily_digest",
                "triggered_by": "manual",
                "timestamp": datetime.utcnow().isoformat(),
            }
            client.rpush("scheduler:insights:queue", json.dumps(job_data))
            return ManualJobResponse(
                success=True,
                job_id=f"digest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                message="Daily digest job queued successfully",
            )

        elif request.job_type == "shift_readiness":
            # Queue shift readiness analysis
            shift_name = request.shift_name or "morning"
            if shift_name not in ["morning", "afternoon", "night", "day"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid shift name: {shift_name}. Use 'morning', 'afternoon', 'night', or 'day'",
                )
            job_data = {
                "type": "shift_readiness",
                "shift_name": shift_name,
                "triggered_by": "manual",
                "timestamp": datetime.utcnow().isoformat(),
            }
            client.rpush("scheduler:insights:queue", json.dumps(job_data))
            return ManualJobResponse(
                success=True,
                job_id=f"shift_{shift_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                message=f"Shift readiness job for {shift_name} shift queued successfully",
            )

        elif request.job_type == "location_baseline":
            # Queue location baseline computation
            job_data = {
                "type": "location_baseline",
                "triggered_by": "manual",
                "timestamp": datetime.utcnow().isoformat(),
            }
            client.rpush("scheduler:insights:queue", json.dumps(job_data))
            return ManualJobResponse(
                success=True,
                job_id=f"baseline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                message="Location baseline job queued successfully",
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid job type: {request.job_type}. Use 'training', 'scoring', 'daily_digest', 'shift_readiness', or 'location_baseline'",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score", response_model=ScoreResponse)
async def score_data(request: ScoreRequest):
    """
    Score device data for anomalies.

    This endpoint allows on-demand scoring of specific devices or date ranges.
    Results are returned directly and also persisted to the database.
    """
    from device_anomaly.data_access.unified_loader import load_unified_device_dataset
    from device_anomaly.features.device_features import DeviceFeatureBuilder
    from device_anomaly.models.anomaly_detector import AnomalyDetector

    try:
        # Load data
        df = load_unified_device_dataset(
            start_date=request.start_date,
            end_date=request.end_date,
            device_ids=request.device_ids,
            row_limit=100_000,
        )

        if df.empty:
            return ScoreResponse(success=True, total_scored=0)

        # Build features
        builder = DeviceFeatureBuilder()
        df_features = builder.transform(df)

        # Load model and score
        detector = AnomalyDetector.load_latest()
        if detector is None:
            raise HTTPException(status_code=503, detail="No trained model available")

        scored_df = detector.score_dataframe(df_features)

        # Calculate stats
        anomaly_count = int((scored_df["anomaly_label"] == -1).sum())
        total_count = len(scored_df)
        anomaly_rate = anomaly_count / total_count if total_count > 0 else 0

        # Prepare results (top anomalies)
        top_anomalies = scored_df[scored_df["anomaly_label"] == -1].nsmallest(
            10, "anomaly_score"
        )
        results = []
        for _, row in top_anomalies.iterrows():
            results.append({
                "device_id": int(row.get("DeviceId", 0)),
                "timestamp": str(row.get("Timestamp", "")),
                "anomaly_score": float(row.get("anomaly_score", 0)),
            })

        return ScoreResponse(
            success=True,
            total_scored=total_count,
            anomalies_detected=anomaly_count,
            anomaly_rate=anomaly_rate,
            results=results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    limit: int = Query(20, ge=1, le=100),
    acknowledged: Optional[bool] = None,
):
    """
    Get recent alerts from the scheduler.

    Returns alerts about high anomaly rates, training failures, etc.
    """
    try:
        client = get_redis_client()
        alerts_data = client.lrange("scheduler:alerts", 0, limit - 1)

        alerts = []
        for data in alerts_data:
            alert = json.loads(data)
            # Generate stable ID based on content if not present
            if "id" not in alert:
                alert["id"] = generate_alert_id(
                    alert.get("timestamp", ""),
                    alert.get("message", "")
                )
            if acknowledged is None or alert.get("acknowledged", False) == acknowledged:
                alerts.append(AlertResponse(**alert))

        return alerts

    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return []


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """
    Acknowledge an alert by its ID.

    Marks the alert as seen/handled by the user.
    """
    try:
        client = get_redis_client()
        alerts_data = client.lrange("scheduler:alerts", 0, -1)

        # Find the alert by ID (using stable ID generation)
        for idx, data in enumerate(alerts_data):
            alert = json.loads(data)
            # Get stored ID or generate stable ID
            stored_id = alert.get("id")
            if not stored_id:
                stored_id = generate_alert_id(
                    alert.get("timestamp", ""),
                    alert.get("message", "")
                )

            if stored_id == alert_id:
                alert["acknowledged"] = True
                alert["id"] = stored_id  # Persist the stable ID
                client.lset("scheduler:alerts", idx, json.dumps(alert))
                return {"success": True, "message": "Alert acknowledged"}

        raise HTTPException(status_code=404, detail="Alert not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[JobHistoryEntry])
async def get_job_history(
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    limit: int = Query(20, ge=1, le=100),
):
    """
    Get history of completed jobs.

    Returns recent training and scoring job results.
    """
    try:
        client = get_redis_client()

        if job_type == "training":
            history_data = client.lrange("ml:training:history", 0, limit - 1)
            scoring_data = []
        elif job_type == "scoring":
            history_data = []
            scoring_data = client.lrange("scheduler:scoring:history", 0, limit - 1)
        else:
            # Get both
            history_data = client.lrange("ml:training:history", 0, limit // 2)
            scoring_data = client.lrange("scheduler:scoring:history", 0, limit // 2)

        history: List[JobHistoryEntry] = []

        # Process training history (from ml_worker format)
        for data in history_data:
            try:
                item = json.loads(data)
                # Training history from ml_worker has different format:
                # {run_id, status, completed_at, metrics, model_version, error}
                # We need to normalize to JobHistoryEntry format

                # Determine timestamp - try multiple fields
                timestamp = (
                    item.get("timestamp") or
                    item.get("completed_at") or
                    item.get("started_at") or
                    item.get("submitted_at") or
                    datetime.utcnow().isoformat()
                )

                # Determine success status
                status = item.get("status", "")
                success = status == "completed"

                # Extract details from metrics if available
                details = None
                metrics = item.get("metrics")
                if metrics:
                    details = {
                        "train_rows": metrics.get("train_rows"),
                        "validation_rows": metrics.get("validation_rows"),
                        "validation_auc": metrics.get("validation_auc"),
                        "model_version": item.get("model_version"),
                    }
                elif item.get("model_version"):
                    details = {"model_version": item.get("model_version")}

                entry = JobHistoryEntry(
                    type=item.get("type", "training"),  # Default to "training" for training history
                    timestamp=timestamp,
                    triggered_by=item.get("triggered_by", "schedule"),
                    success=success,
                    error=item.get("error") or item.get("message") if not success else None,
                    details=details,
                )
                history.append(entry)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse training history entry: {e}")
                continue

        # Process scoring history
        for data in scoring_data:
            try:
                item = json.loads(data)
                timestamp = (
                    item.get("timestamp") or
                    item.get("completed_at") or
                    datetime.utcnow().isoformat()
                )

                entry = JobHistoryEntry(
                    type=item.get("type", "scoring"),  # Default to "scoring" for scoring history
                    timestamp=timestamp,
                    triggered_by=item.get("triggered_by", "schedule"),
                    success=item.get("success", True),
                    error=item.get("error"),
                    details=item.get("details") or item.get("metrics"),
                )
                history.append(entry)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse scoring history entry: {e}")
                continue

        # Sort by timestamp (handle empty strings)
        def sort_key(x):
            if not x.timestamp:
                return ""
            return x.timestamp

        history.sort(key=sort_key, reverse=True)
        return history[:limit]

    except Exception as e:
        logger.error(f"Failed to get job history: {e}")
        return []


@router.get("/health")
async def health_check():
    """
    Health check for the automation system.

    Returns the status of all automation components.
    """
    status = get_scheduler_status()
    config = get_scheduler_config()

    return {
        "status": "healthy" if status.get("is_running") else "stopped",
        "scheduler_running": status.get("is_running", False),
        "training_enabled": config.get("training_enabled", True),
        "scoring_enabled": config.get("scoring_enabled", True),
        "auto_retrain_enabled": config.get("auto_retrain_enabled", True),
        "insights_enabled": config.get("insights_enabled", True),
        "shift_readiness_enabled": config.get("shift_readiness_enabled", True),
        "last_training": config.get("last_training_time"),
        "last_scoring": config.get("last_scoring_time"),
        "last_daily_digest": config.get("last_daily_digest_time"),
        "last_shift_readiness": config.get("last_shift_readiness_time"),
        "last_location_baseline": config.get("last_location_baseline_time"),
        "total_insights_generated": status.get("total_insights_generated", 0),
        "uptime_seconds": status.get("uptime_seconds", 0),
        "error_count": len(status.get("errors", [])),
    }
