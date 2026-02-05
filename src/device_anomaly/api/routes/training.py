"""API routes for ML model training and monitoring."""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode, get_tenant_id, require_role
from device_anomaly.api.request_context import set_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "")
USE_REDIS = bool(REDIS_URL)
_app_env = os.getenv("APP_ENV", "local")
_allow_in_process_training = os.getenv("ALLOW_IN_PROCESS_TRAINING", "false").lower() == "true"

# Redis keys
QUEUE_KEY = "ml:training:queue"
STATUS_KEY = "ml:training:status"
HISTORY_KEY = "ml:training:history"


# ============================================================================
# Pydantic Models
# ============================================================================


class TrainingConfigRequest(BaseModel):
    """Request to start a training job."""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    validation_days: int = Field(default=7, ge=1, le=30)
    contamination: float = Field(default=0.03, ge=0.001, le=0.1)
    n_estimators: int = Field(default=300, ge=50, le=1000)
    export_onnx: bool = Field(default=True)


class TrainingMetricsResponse(BaseModel):
    """Training metrics from a completed run."""

    train_rows: int
    validation_rows: int
    feature_count: int
    anomaly_rate_train: float
    anomaly_rate_validation: float
    validation_auc: float | None = None
    precision_at_recall_80: float | None = None
    feature_importance: dict[str, float] = Field(default_factory=dict)


class TrainingArtifactsResponse(BaseModel):
    """Paths to training artifacts."""

    model_path: str | None = None
    onnx_path: str | None = None
    baselines_path: str | None = None
    cohort_stats_path: str | None = None
    metadata_path: str | None = None


class TrainingStageInfo(BaseModel):
    """Information about a training stage."""

    name: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    started_at: str | None = None
    completed_at: str | None = None
    message: str | None = None


class TrainingRunResponse(BaseModel):
    """Response for a training run."""

    run_id: str
    status: str  # 'idle', 'pending', 'running', 'completed', 'failed'
    progress: float = 0.0
    message: str | None = None
    stage: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    estimated_completion: str | None = None
    config: dict[str, Any] | None = None
    metrics: TrainingMetricsResponse | None = None
    artifacts: TrainingArtifactsResponse | None = None
    model_version: str | None = None
    error: str | None = None
    stages: list[TrainingStageInfo] | None = None


class TrainingHistoryResponse(BaseModel):
    """List of past training runs."""

    runs: list[TrainingRunResponse]
    total: int


class TrainingQueueStatus(BaseModel):
    """Status of the training queue."""

    queue_length: int
    worker_available: bool
    last_job_completed_at: str | None = None
    next_scheduled: str | None = None


# ============================================================================
# Redis Client (lazy initialization)
# ============================================================================


_redis_client = None


def get_redis():
    """Get or create Redis client."""
    global _redis_client
    if _redis_client is None and USE_REDIS:
        try:
            import redis
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            _redis_client.ping()  # Test connection
            logger.info("Connected to Redis for training job queue")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory fallback: {e}")
            return None
    return _redis_client


# ============================================================================
# In-memory training state (fallback when Redis not available)
# ============================================================================

_training_state_lock = threading.Lock()
_training_state: dict[str, Any] = {
    "status": "idle",
    "progress": 0.0,
    "message": None,
    "stage": None,
    "started_at": None,
    "completed_at": None,
    "current_run": None,
    "history": [],
}


def _update_training_state(updates: dict[str, Any]) -> None:
    """Thread-safe update of training state."""
    with _training_state_lock:
        _training_state.update(updates)


def _get_training_state_copy() -> dict[str, Any]:
    """Thread-safe copy of training state."""
    with _training_state_lock:
        return dict(_training_state)


# Stage names in order of execution
TRAINING_STAGE_NAMES = [
    "Initialize",
    "Load Data",
    "Feature Engineering",
    "Training",
    "Validation",
    "Export",
]

# Map stage identifiers to display names
STAGE_NAME_MAP = {
    "initialize": "Initialize",
    "load_data": "Load Data",
    "features": "Feature Engineering",
    "split": "Feature Engineering",
    "cohort_stats": "Feature Engineering",
    "baselines": "Feature Engineering",
    "training": "Training",
    "validation": "Validation",
    "export": "Export",
}


def _build_stages_from_status(status: dict[str, Any]) -> list[TrainingStageInfo] | None:
    """Build stages list from current training status.

    Creates a stages representation based on current progress and stage information.
    Returns None for idle status.
    """
    job_status = status.get("status", "idle")
    if job_status == "idle":
        return None

    current_stage = status.get("stage", "")
    status.get("progress", 0.0)
    started_at = status.get("started_at")
    status.get("completed_at")

    # Map current stage to display name
    current_stage_name = STAGE_NAME_MAP.get(current_stage, current_stage.replace("_", " ").title() if current_stage else "")

    stages = []
    found_current = False

    for stage_name in TRAINING_STAGE_NAMES:
        if job_status == "completed":
            # All stages completed
            stages.append(TrainingStageInfo(
                name=stage_name,
                status="completed",
            ))
        elif job_status == "failed":
            # Mark stages up to current as completed, current as failed
            if stage_name == current_stage_name:
                stages.append(TrainingStageInfo(
                    name=stage_name,
                    status="failed",
                    message=status.get("error"),
                ))
                found_current = True
            elif not found_current:
                stages.append(TrainingStageInfo(
                    name=stage_name,
                    status="completed",
                ))
            else:
                stages.append(TrainingStageInfo(
                    name=stage_name,
                    status="pending",
                ))
        else:  # running or pending
            if stage_name == current_stage_name:
                stages.append(TrainingStageInfo(
                    name=stage_name,
                    status="running",
                    started_at=started_at,
                    message=status.get("message"),
                ))
                found_current = True
            elif not found_current:
                stages.append(TrainingStageInfo(
                    name=stage_name,
                    status="completed",
                ))
            else:
                stages.append(TrainingStageInfo(
                    name=stage_name,
                    status="pending",
                ))

    return stages


# ============================================================================
# Mock Data
# ============================================================================


def get_mock_training_stages() -> list[TrainingStageInfo]:
    """Generate mock training stages."""
    now = datetime.now(UTC)
    return [
        TrainingStageInfo(
            name="Initialize",
            status="completed",
            started_at=(now - timedelta(hours=2)).isoformat(),
            completed_at=(now - timedelta(hours=2, minutes=-2)).isoformat(),
        ),
        TrainingStageInfo(
            name="Load Data",
            status="completed",
            started_at=(now - timedelta(hours=1, minutes=58)).isoformat(),
            completed_at=(now - timedelta(hours=1, minutes=45)).isoformat(),
            message="Loaded 850,000 rows",
        ),
        TrainingStageInfo(
            name="Feature Engineering",
            status="completed",
            started_at=(now - timedelta(hours=1, minutes=45)).isoformat(),
            completed_at=(now - timedelta(hours=1, minutes=30)).isoformat(),
            message="Created 24 features",
        ),
        TrainingStageInfo(
            name="Training",
            status="completed",
            started_at=(now - timedelta(hours=1, minutes=30)).isoformat(),
            completed_at=(now - timedelta(hours=1, minutes=10)).isoformat(),
        ),
        TrainingStageInfo(
            name="Validation",
            status="completed",
            started_at=(now - timedelta(hours=1, minutes=10)).isoformat(),
            completed_at=(now - timedelta(hours=1, minutes=5)).isoformat(),
            message="AUC: 0.89",
        ),
        TrainingStageInfo(
            name="Export",
            status="completed",
            started_at=(now - timedelta(hours=1, minutes=5)).isoformat(),
            completed_at=(now - timedelta(hours=1)).isoformat(),
        ),
    ]


def get_mock_training_run() -> TrainingRunResponse:
    """Generate mock training run for development."""
    return TrainingRunResponse(
        run_id="mock-abc123",
        status="completed",
        progress=100.0,
        message="Training completed successfully",
        stage="complete",
        started_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        completed_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        config={
            "start_date": "2024-01-01",
            "end_date": "2024-12-01",
            "validation_days": 7,
            "contamination": 0.03,
            "n_estimators": 300,
        },
        metrics=TrainingMetricsResponse(
            train_rows=850000,
            validation_rows=150000,
            feature_count=24,
            anomaly_rate_train=0.028,
            anomaly_rate_validation=0.032,
            validation_auc=0.89,
            precision_at_recall_80=0.72,
            feature_importance={
                "TotalBatteryLevelDrop": 0.18,
                "TotalFreeStorageKb": 0.15,
                "Download": 0.12,
                "Upload": 0.10,
                "OfflineTime": 0.09,
                "AvgSignalStrength": 0.08,
                "BatteryTemp_std": 0.07,
                "CpuUsage_mean": 0.06,
                "MemoryUsage_max": 0.05,
                "NetworkLatency_p95": 0.04,
            },
        ),
        artifacts=TrainingArtifactsResponse(
            model_path="models/production/isolation_forest.pkl",
            onnx_path="models/production/isolation_forest.onnx",
            baselines_path="models/production/baselines.json",
            metadata_path="models/production/training_metadata.json",
        ),
        model_version="v20241228_143052",
        stages=get_mock_training_stages(),
    )


def get_mock_running_training() -> TrainingRunResponse:
    """Generate mock running training for demonstration."""
    now = datetime.now(UTC)
    stages = [
        TrainingStageInfo(
            name="Initialize",
            status="completed",
            started_at=(now - timedelta(minutes=15)).isoformat(),
            completed_at=(now - timedelta(minutes=14)).isoformat(),
        ),
        TrainingStageInfo(
            name="Load Data",
            status="completed",
            started_at=(now - timedelta(minutes=14)).isoformat(),
            completed_at=(now - timedelta(minutes=10)).isoformat(),
            message="Loaded 920,000 rows",
        ),
        TrainingStageInfo(
            name="Feature Engineering",
            status="completed",
            started_at=(now - timedelta(minutes=10)).isoformat(),
            completed_at=(now - timedelta(minutes=6)).isoformat(),
            message="Created 26 features",
        ),
        TrainingStageInfo(
            name="Training",
            status="running",
            started_at=(now - timedelta(minutes=6)).isoformat(),
            message="Training IsolationForest model...",
        ),
        TrainingStageInfo(name="Validation", status="pending"),
        TrainingStageInfo(name="Export", status="pending"),
    ]

    return TrainingRunResponse(
        run_id="mock-running-001",
        status="running",
        progress=65.0,
        message="Training IsolationForest model...",
        stage="training",
        started_at=(now - timedelta(minutes=15)).isoformat(),
        estimated_completion=(now + timedelta(minutes=10)).isoformat(),
        config={
            "start_date": "2024-06-01",
            "end_date": "2024-12-28",
            "validation_days": 7,
            "contamination": 0.03,
            "n_estimators": 300,
        },
        stages=stages,
    )


def get_mock_training_history() -> list[TrainingRunResponse]:
    """Generate mock training history."""
    now = datetime.now(UTC)
    return [
        get_mock_training_run(),
        TrainingRunResponse(
            run_id="mock-xyz789",
            status="completed",
            progress=100.0,
            message="Training completed",
            started_at=(now - timedelta(days=7)).isoformat(),
            completed_at=(now - timedelta(days=7, hours=-1)).isoformat(),
            model_version="v20241221_091523",
            metrics=TrainingMetricsResponse(
                train_rows=780000,
                validation_rows=140000,
                feature_count=22,
                anomaly_rate_train=0.031,
                anomaly_rate_validation=0.029,
                validation_auc=0.87,
                precision_at_recall_80=0.68,
                feature_importance={
                    "TotalBatteryLevelDrop": 0.20,
                    "Download": 0.14,
                    "OfflineTime": 0.11,
                },
            ),
        ),
        TrainingRunResponse(
            run_id="mock-def456",
            status="failed",
            progress=35.0,
            message="Insufficient training data",
            started_at=(now - timedelta(days=14)).isoformat(),
            completed_at=(now - timedelta(days=14, hours=-0.5)).isoformat(),
            error="ValueError: training_set requires at least 1000 rows; got 523.",
        ),
        TrainingRunResponse(
            run_id="mock-ghi321",
            status="completed",
            progress=100.0,
            started_at=(now - timedelta(days=21)).isoformat(),
            completed_at=(now - timedelta(days=21, hours=-1.5)).isoformat(),
            model_version="v20241207_154230",
            metrics=TrainingMetricsResponse(
                train_rows=720000,
                validation_rows=130000,
                feature_count=20,
                anomaly_rate_train=0.029,
                anomaly_rate_validation=0.031,
                validation_auc=0.85,
            ),
        ),
    ]


# ============================================================================
# Background Training Task (fallback when Redis not available)
# ============================================================================


def _run_training_job_background(config: TrainingConfigRequest, tenant_id: str):
    """Background task to run model training (in-process fallback)."""
    set_tenant_id(tenant_id)

    try:
        _update_training_state({
            "status": "running",
            "progress": 5.0,
            "stage": "initialize",
            "message": "Initializing training pipeline...",
        })

        from device_anomaly.pipeline.training import RealDataTrainingPipeline, TrainingConfig

        training_config = TrainingConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            validation_days=config.validation_days,
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            export_onnx=config.export_onnx,
        )

        pipeline = RealDataTrainingPipeline(training_config)

        # Stage: Load data
        _update_training_state({
            "progress": 10.0,
            "stage": "load_data",
            "message": "Loading training data...",
        })

        # Run the pipeline
        result = pipeline.run(output_dir="models/production")

        # Update state with results (thread-safe)
        with _training_state_lock:
            _training_state["status"] = result.status
            _training_state["progress"] = 100.0
            _training_state["stage"] = "complete"
            _training_state["message"] = "Training completed" if result.status == "completed" else result.error
            _training_state["completed_at"] = result.completed_at
            _training_state["current_run"] = result.to_dict()
            # Add to history
            _training_state["history"].insert(0, result.to_dict())
            _training_state["history"] = _training_state["history"][:50]

    except Exception as e:
        logger.error(f"Training job failed: {e}", exc_info=True)
        _update_training_state({
            "status": "failed",
            "stage": "error",
            "message": "Training failed. Check server logs for details.",
            "completed_at": datetime.now(UTC).isoformat(),
        })
    finally:
        set_tenant_id(None)


# ============================================================================
# Helper Functions
# ============================================================================


def _get_status_from_redis() -> dict[str, Any] | None:
    """Get training status from Redis."""
    redis_client = get_redis()
    if redis_client:
        try:
            data = redis_client.get(STATUS_KEY)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Failed to get status from Redis: {e}")
    return None


def _get_history_from_redis(limit: int = 10) -> list[dict[str, Any]]:
    """Get training history from Redis."""
    redis_client = get_redis()
    if redis_client:
        try:
            history = redis_client.lrange(HISTORY_KEY, 0, limit - 1)
            return [json.loads(h) for h in history]
        except Exception as e:
            logger.warning(f"Failed to get history from Redis: {e}")
    return []


def _submit_job_to_redis(config: TrainingConfigRequest, tenant_id: str) -> str:
    """Submit training job to Redis queue."""
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    job_id = f"job_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    job = {
        "job_id": job_id,
        "config": config.model_dump(),
        "tenant_id": tenant_id,
        "submitted_at": datetime.now(UTC).isoformat(),
        "status": "pending",
    }

    redis_client.rpush(QUEUE_KEY, json.dumps(job))

    # Update status
    redis_client.set(
        STATUS_KEY,
        json.dumps({
            "run_id": job_id,
            "status": "pending",
            "progress": 0,
            "message": "Job queued, waiting for ML worker...",
            "submitted_at": job["submitted_at"],
        }),
    )

    return job_id


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/start", response_model=TrainingRunResponse)
def start_training(
    config: TrainingConfigRequest,
    background_tasks: BackgroundTasks,
    mock_mode: bool = Depends(get_mock_mode),
    _: None = Depends(require_role(["admin"])),
):
    """
    Start a new model training job.

    In production (with Redis), the job is queued for the ML worker.
    In development (without Redis), training runs in the API process.
    Use /status to monitor progress.
    """
    global _training_state

    if mock_mode:
        return get_mock_running_training()

    tenant_id = get_tenant_id()
    if not USE_REDIS and _app_env == "production" and not _allow_in_process_training:
        raise HTTPException(
            status_code=503,
            detail="Training is disabled without a worker queue in production",
        )
    # Check if training is already running
    if USE_REDIS:
        status = _get_status_from_redis()
        if status and status.get("status") == "running":
            raise HTTPException(
                status_code=409,
                detail="A training job is already running. Wait for it to complete."
            )

        # Submit to Redis queue
        try:
            job_id = _submit_job_to_redis(config, tenant_id)
            return TrainingRunResponse(
                run_id=job_id,
                status="pending",
                progress=0.0,
                message="Training job queued, waiting for ML worker...",
                started_at=datetime.now(UTC).isoformat(),
                config=config.model_dump(),
            )
        except Exception as e:
            logger.warning(f"Failed to submit to Redis, falling back to background task: {e}")

    # Fallback: run in background task (thread-safe check)
    current_state = _get_training_state_copy()
    if current_state.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="A training job is already running. Wait for it to complete."
        )

    started_at = datetime.now(UTC).isoformat()
    with _training_state_lock:
        _training_state.update({
            "status": "running",
            "progress": 0.0,
            "message": "Starting training job...",
            "stage": "initialize",
            "started_at": started_at,
            "completed_at": None,
            "current_run": None,
        })
        # Preserve history

    background_tasks.add_task(_run_training_job_background, config, tenant_id)

    return TrainingRunResponse(
        run_id="local-run",
        status="running",
        progress=0.0,
        message="Training job started (running in API process)",
        stage="initialize",
        started_at=started_at,
        config=config.model_dump(),
    )


@router.get("/status", response_model=TrainingRunResponse)
def get_training_status(
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get status of the current or last training job."""
    if mock_mode:
        # Randomly return running or completed for demo
        import random
        if random.random() < 0.3:
            return get_mock_running_training()
        return get_mock_training_run()

    # Try Redis first
    if USE_REDIS:
        status = _get_status_from_redis()
        if status:
            # Build stages from current status
            stages = _build_stages_from_status(status)
            return TrainingRunResponse(
                run_id=status.get("run_id", "unknown"),
                status=status.get("status", "idle"),
                progress=status.get("progress", 0.0),
                message=status.get("message"),
                stage=status.get("stage"),
                started_at=status.get("started_at") or status.get("submitted_at"),
                completed_at=status.get("completed_at"),
                config=status.get("config"),
                metrics=TrainingMetricsResponse(**status["metrics"]) if status.get("metrics") else None,
                artifacts=TrainingArtifactsResponse(**status["artifacts"]) if status.get("artifacts") else None,
                model_version=status.get("model_version"),
                error=status.get("error"),
                stages=stages,
            )

    # Fallback to in-memory state (thread-safe access)
    state = _get_training_state_copy()
    if state.get("current_run"):
        run = state["current_run"]
        stages = _build_stages_from_status(state)
        return TrainingRunResponse(
            run_id=run.get("run_id", "unknown"),
            status=run.get("status", state.get("status", "idle")),
            progress=state.get("progress", 0.0),
            message=state.get("message"),
            stage=state.get("stage"),
            started_at=run.get("started_at", state.get("started_at")),
            completed_at=run.get("completed_at", state.get("completed_at")),
            config=run.get("config"),
            metrics=TrainingMetricsResponse(**run["metrics"]) if run.get("metrics") else None,
            artifacts=TrainingArtifactsResponse(**run["artifacts"]) if run.get("artifacts") else None,
            model_version=run.get("model_version"),
            error=run.get("error"),
            stages=stages,
        )

    return TrainingRunResponse(
        run_id="none",
        status=state.get("status", "idle"),
        progress=state.get("progress", 0.0),
        message=state.get("message", "No training job has been run"),
        stage=state.get("stage"),
        started_at=state.get("started_at"),
        completed_at=state.get("completed_at"),
        stages=None,  # No stages for idle state
    )


@router.get("/history", response_model=TrainingHistoryResponse)
def get_training_history(
    limit: int = Query(default=10, ge=1, le=50),
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get history of past training runs."""
    if mock_mode:
        mock_runs = get_mock_training_history()
        return TrainingHistoryResponse(
            runs=mock_runs[:limit],
            total=len(mock_runs),
        )

    # Try Redis first
    if USE_REDIS:
        history = _get_history_from_redis(limit)
        if history:
            runs = []
            for run in history:
                # Parse metrics if present
                metrics = None
                metrics_data = run.get("metrics")
                if metrics_data and isinstance(metrics_data, dict):
                    # Ensure required fields have defaults
                    metrics = TrainingMetricsResponse(
                        train_rows=metrics_data.get("train_rows", 0),
                        validation_rows=metrics_data.get("validation_rows", 0),
                        feature_count=metrics_data.get("feature_count", 0),
                        anomaly_rate_train=metrics_data.get("anomaly_rate_train", 0.0),
                        anomaly_rate_validation=metrics_data.get("anomaly_rate_validation", 0.0),
                        validation_auc=metrics_data.get("validation_auc"),
                        precision_at_recall_80=metrics_data.get("precision_at_recall_80"),
                        feature_importance=metrics_data.get("feature_importance", {}),
                    )

                runs.append(TrainingRunResponse(
                    run_id=run.get("run_id", "unknown"),
                    status=run.get("status", "unknown"),
                    progress=100.0 if run.get("status") == "completed" else 0.0,
                    message=run.get("message"),
                    started_at=run.get("started_at") or run.get("submitted_at"),
                    completed_at=run.get("completed_at"),
                    model_version=run.get("model_version"),
                    metrics=metrics,
                    error=run.get("error"),
                ))
            return TrainingHistoryResponse(runs=runs, total=len(history))

    # Fallback to in-memory history (thread-safe access)
    state = _get_training_state_copy()
    history = state.get("history", [])
    runs = []
    for run in history[:limit]:
        runs.append(TrainingRunResponse(
            run_id=run.get("run_id", "unknown"),
            status=run.get("status", "unknown"),
            progress=100.0 if run.get("status") == "completed" else 0.0,
            started_at=run.get("started_at"),
            completed_at=run.get("completed_at"),
            model_version=run.get("model_version"),
            metrics=TrainingMetricsResponse(**run["metrics"]) if run.get("metrics") else None,
            error=run.get("error"),
        ))

    return TrainingHistoryResponse(
        runs=runs,
        total=len(history),
    )


@router.get("/queue", response_model=TrainingQueueStatus)
def get_queue_status(
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get status of the training job queue."""
    if mock_mode:
        return TrainingQueueStatus(
            queue_length=0,
            worker_available=True,
            last_job_completed_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            next_scheduled=(datetime.now(UTC) + timedelta(hours=6)).isoformat(),
        )

    redis_client = get_redis()
    if redis_client:
        try:
            queue_length = redis_client.llen(QUEUE_KEY)

            # Get next scheduled training time from scheduler status
            next_scheduled = None
            last_completed = None
            try:
                scheduler_status_raw = redis_client.get("scheduler:status")
                if scheduler_status_raw:
                    scheduler_status = json.loads(scheduler_status_raw)
                    next_scheduled = scheduler_status.get("next_training_time")
                    # Get last completed from training history
                    if scheduler_status.get("last_training_result"):
                        last_completed = scheduler_status["last_training_result"].get("completed_at")
            except Exception as e:
                logger.debug(f"Could not get scheduler status: {e}")

            return TrainingQueueStatus(
                queue_length=queue_length,
                worker_available=True,  # TODO: Add worker health check
                last_job_completed_at=last_completed,
                next_scheduled=next_scheduled,
            )
        except Exception as e:
            logger.warning(f"Failed to get queue status: {e}")

    return TrainingQueueStatus(
        queue_length=0,
        worker_available=False,
    )


@router.get("/{run_id}/metrics", response_model=TrainingMetricsResponse)
def get_run_metrics(
    run_id: str,
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get detailed metrics for a specific training run."""
    if mock_mode:
        mock_run = get_mock_training_run()
        if mock_run.metrics:
            return mock_run.metrics
        raise HTTPException(status_code=404, detail="Run not found")

    # Try Redis history first
    if USE_REDIS:
        history = _get_history_from_redis(50)
        for run in history:
            if run.get("run_id") == run_id and run.get("metrics"):
                return TrainingMetricsResponse(**run["metrics"])

    # Fallback to in-memory (thread-safe access)
    state = _get_training_state_copy()
    for run in state.get("history", []):
        if run.get("run_id") == run_id and run.get("metrics"):
            return TrainingMetricsResponse(**run["metrics"])

    current = state.get("current_run")
    if current and current.get("run_id") == run_id and current.get("metrics"):
        return TrainingMetricsResponse(**current["metrics"])

    raise HTTPException(status_code=404, detail=f"Training run '{run_id}' not found")


@router.get("/{run_id}/artifacts", response_model=TrainingArtifactsResponse)
def get_run_artifacts(
    run_id: str,
    mock_mode: bool = Depends(get_mock_mode),
):
    """Get artifact paths for a specific training run."""
    if mock_mode:
        mock_run = get_mock_training_run()
        if mock_run.artifacts:
            return mock_run.artifacts
        raise HTTPException(status_code=404, detail="Run not found")

    # Try Redis history first
    if USE_REDIS:
        history = _get_history_from_redis(50)
        for run in history:
            if run.get("run_id") == run_id and run.get("artifacts"):
                return TrainingArtifactsResponse(**run["artifacts"])

    # Fallback to in-memory (thread-safe access)
    state = _get_training_state_copy()
    for run in state.get("history", []):
        if run.get("run_id") == run_id and run.get("artifacts"):
            return TrainingArtifactsResponse(**run["artifacts"])

    current = state.get("current_run")
    if current and current.get("run_id") == run_id and current.get("artifacts"):
        return TrainingArtifactsResponse(**current["artifacts"])

    raise HTTPException(status_code=404, detail=f"Training run '{run_id}' not found")
