"""
ML Worker - Background service for training jobs.

This worker processes training jobs from a Redis queue, allowing for:
- Proper progress tracking with real-time updates
- Job persistence across restarts
- Separate resource allocation from the API
- Reliable long-running ML training jobs
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis

from device_anomaly.api.request_context import set_tenant_id
from device_anomaly.pipeline.training import (
    RealDataTrainingPipeline,
    TrainingConfig,
    TrainingResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Redis keys
QUEUE_KEY = "ml:training:queue"
STATUS_KEY = "ml:training:status"
HISTORY_KEY = "ml:training:history"
CURRENT_JOB_KEY = "ml:training:current"


class TrainingJobProcessor:
    """
    Processes training jobs from Redis queue with progress updates.

    Job lifecycle:
    1. Job submitted to queue via API
    2. Worker picks up job from queue
    3. Worker updates status as training progresses
    4. On completion, job moves to history
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.running = True
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Handle graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown handler."""
        logger.info("Received shutdown signal, finishing current job...")
        self.running = False

    def update_progress(
        self,
        run_id: str,
        progress: float,
        message: str,
        stage: str = "",
    ):
        """Update training progress in Redis for real-time UI updates."""
        status = {
            "run_id": run_id,
            "status": "running",
            "progress": progress,
            "message": message,
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.set(STATUS_KEY, json.dumps(status))
        self.redis.set(CURRENT_JOB_KEY, json.dumps(status))
        logger.info(f"[{run_id}] {progress:.0f}% - {message}")

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get current training status."""
        data = self.redis.get(STATUS_KEY)
        if data:
            return json.loads(data)
        return None

    def get_history(self, limit: int = 10) -> list:
        """Get training history."""
        history = self.redis.lrange(HISTORY_KEY, 0, limit - 1)
        return [json.loads(h) for h in history]

    def submit_job(self, config: Dict[str, Any], tenant_id: str = "default") -> str:
        """Submit a new training job to the queue."""
        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        job = {
            "job_id": job_id,
            "config": config,
            "tenant_id": tenant_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        self.redis.rpush(QUEUE_KEY, json.dumps(job))

        # Update status to show pending job
        self.redis.set(
            STATUS_KEY,
            json.dumps({
                "run_id": job_id,
                "status": "pending",
                "progress": 0,
                "message": "Job queued, waiting for worker...",
                "submitted_at": job["submitted_at"],
            }),
        )

        logger.info(f"Job {job_id} submitted to queue")
        return job_id

    def process_job(self, job_data: Dict[str, Any]) -> TrainingResult:
        """Process a single training job with progress updates."""
        job_id = job_data["job_id"]
        config_dict = job_data["config"]
        tenant_id = job_data.get("tenant_id") or "default"

        logger.info(f"Starting training job: {job_id}")

        # Create training config
        config = TrainingConfig(
            start_date=config_dict["start_date"],
            end_date=config_dict["end_date"],
            validation_days=config_dict.get("validation_days", 7),
            contamination=config_dict.get("contamination", 0.03),
            n_estimators=config_dict.get("n_estimators", 300),
            export_onnx=config_dict.get("export_onnx", True),
        )

        # Initialize pipeline
        set_tenant_id(tenant_id)
        pipeline = RealDataTrainingPipeline(config)
        run_id = pipeline.run_id

        try:
            # Stage 1: Initialize (0-5%)
            self.update_progress(run_id, 5, "Initializing training pipeline...", "initialize")

            # Stage 2: Load data (5-20%)
            self.update_progress(run_id, 10, "Loading training data from database...", "load_data")
            df = pipeline.load_training_data()
            self.update_progress(run_id, 20, f"Loaded {len(df):,} rows", "load_data")

            # Stage 3: Feature engineering (20-40%)
            self.update_progress(run_id, 25, "Applying feature engineering...", "features")
            df_features = pipeline.prepare_features(df)
            self.update_progress(run_id, 40, f"Created {len(df_features.columns)} features", "features")

            # Stage 4: Compute baselines (40-50%)
            self.update_progress(run_id, 45, "Computing data-driven baselines...", "baselines")
            pipeline.compute_baselines(df_features)
            self.update_progress(run_id, 50, "Baselines computed", "baselines")

            # Stage 5: Train/validation split (50-55%)
            self.update_progress(run_id, 52, "Splitting data for validation...", "split")
            df_train, df_val = pipeline.train_validation_split(df_features)
            self.update_progress(
                run_id, 55,
                f"Train: {len(df_train):,} rows, Val: {len(df_val):,} rows",
                "split",
            )

            # Stage 6: Training (55-80%)
            self.update_progress(run_id, 60, "Training IsolationForest model...", "training")
            detector = pipeline.train_model(df_train)
            self.update_progress(
                run_id, 80,
                f"Model trained with {len(detector.feature_cols)} features",
                "training",
            )

            # Stage 7: Validation (80-90%)
            self.update_progress(run_id, 85, "Evaluating model on validation set...", "validation")
            metrics = pipeline.evaluate_model(detector, df_val)
            self.update_progress(
                run_id, 90,
                f"Validation complete - AUC: {metrics.validation_auc or 'N/A'}",
                "validation",
            )

            # Stage 8: Export artifacts (90-100%)
            self.update_progress(run_id, 95, "Exporting model artifacts...", "export")
            # Sanitize run_id to prevent path traversal attacks
            safe_run_id = re.sub(r'[^a-zA-Z0-9_-]', '', run_id)
            output_dir = f"models/production/{safe_run_id}"
            artifacts = pipeline.export_artifacts(detector, metrics, output_dir)

            # Create result
            completed_at = datetime.now(timezone.utc).isoformat()
            version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

            result = TrainingResult(
                run_id=run_id,
                model_version=version,
                config=config,
                metrics=metrics,
                artifacts=artifacts,
                started_at=job_data["submitted_at"],
                completed_at=completed_at,
                status="completed",
            )

            # Update final status - ensure status is "completed" not "running"
            result_dict = result.to_dict()
            final_status = {
                "run_id": run_id,
                "status": "completed",
                "progress": 100,
                "message": "Training completed successfully",
                "stage": "complete",
                "model_version": result.model_version,
                "completed_at": completed_at,
                "metrics": asdict(metrics) if metrics else None,
            }
            self.redis.set(STATUS_KEY, json.dumps(final_status))
            self.redis.set(CURRENT_JOB_KEY, json.dumps(final_status))

            # Add to history
            self.redis.lpush(HISTORY_KEY, json.dumps(result_dict))
            self.redis.ltrim(HISTORY_KEY, 0, 49)  # Keep last 50 runs

            logger.info(f"[{run_id}] 100% - Training completed successfully!")
            logger.info(f"Training job {job_id} completed successfully")

            return result

        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}", exc_info=True)

            # Update status with error
            error_status = {
                "run_id": run_id,
                "status": "failed",
                "progress": 0,
                "message": str(e),
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.redis.set(STATUS_KEY, json.dumps(error_status))

            # Add failed job to history
            self.redis.lpush(HISTORY_KEY, json.dumps(error_status))
            self.redis.ltrim(HISTORY_KEY, 0, 49)

            raise
        finally:
            set_tenant_id(None)

    def run(self):
        """Main worker loop - process jobs from queue."""
        logger.info("ML Worker started, waiting for jobs...")

        while self.running:
            try:
                # Block wait for job with 5 second timeout
                job = self.redis.blpop(QUEUE_KEY, timeout=5)

                if job:
                    _, job_data = job
                    job_dict = json.loads(job_data)

                    try:
                        self.process_job(job_dict)
                    except Exception as e:
                        logger.error(f"Job processing failed: {e}")
                        # Job failure is already logged and recorded

            except redis.ConnectionError:
                logger.warning("Redis connection lost, retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                time.sleep(1)

        logger.info("ML Worker shutting down...")


def main():
    """Entry point for the ML worker."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    logger.info(f"Connecting to Redis at {redis_url}")

    processor = TrainingJobProcessor(redis_url)
    processor.run()


if __name__ == "__main__":
    main()
