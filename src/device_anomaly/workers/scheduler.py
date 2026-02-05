"""
Automated Scheduler Service for Anomaly Detection.

This module provides autonomous scheduling for:
- Periodic model retraining (configurable interval)
- Continuous data scoring (real-time anomaly detection)
- Auto-retrain on feedback threshold
- Health monitoring and alerting

The scheduler runs as a background service and can be configured
via API or environment variables.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import redis

logger = logging.getLogger(__name__)


def is_mock_mode() -> bool:
    """Check if mock mode is enabled via environment variable."""
    return os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")


def get_historical_end_date() -> datetime | None:
    """
    Get the historical data end date for demo/testing with backup data.

    When set, the scheduler will use this date instead of today's date
    for scoring queries. This allows testing with historical database backups.

    Set HISTORICAL_DATA_END_DATE=2025-12-26 (or any date in your backup)
    """
    date_str = os.getenv("HISTORICAL_DATA_END_DATE")
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid HISTORICAL_DATA_END_DATE format: {date_str}. Use YYYY-MM-DD.")
    return None


class ScheduleInterval(StrEnum):
    """Supported scheduling intervals."""
    HOURLY = "hourly"
    EVERY_6_HOURS = "every_6_hours"
    EVERY_12_HOURS = "every_12_hours"
    DAILY = "daily"
    WEEKLY = "weekly"
    MANUAL = "manual"  # Disabled, only manual triggers


class ShiftSchedule(StrEnum):
    """Pre-defined shift schedules for shift readiness analysis."""
    MORNING = "morning"  # 06:00 - 14:00
    AFTERNOON = "afternoon"  # 14:00 - 22:00
    NIGHT = "night"  # 22:00 - 06:00
    DAY = "day"  # 08:00 - 17:00 (standard day shift)


@dataclass
class SchedulerConfig:
    """Configuration for the automation scheduler."""

    # Training schedule
    training_enabled: bool = True
    training_interval: ScheduleInterval = ScheduleInterval.DAILY
    training_hour: int = 2  # Hour of day for daily/weekly (0-23)
    training_day_of_week: int = 0  # 0=Monday for weekly
    training_lookback_days: int = 90  # Days of data to use for training
    training_validation_days: int = 7

    # Scoring schedule
    scoring_enabled: bool = True
    scoring_interval_minutes: int = 15  # How often to score new data

    # Auto-retrain settings
    auto_retrain_enabled: bool = True
    auto_retrain_fp_threshold: float = 0.15  # False positive rate threshold
    auto_retrain_min_feedback: int = 50  # Minimum feedback items before considering
    auto_retrain_cooldown_hours: int = 24  # Hours between auto-retrains

    # Alerting
    alerting_enabled: bool = True
    alert_on_high_anomaly_rate: bool = True
    high_anomaly_rate_threshold: float = 0.10  # Alert if >10% anomalies

    # Insight generation
    insights_enabled: bool = True
    daily_digest_hour: int = 5  # Hour to generate daily digest (5 AM)
    shift_readiness_enabled: bool = True
    shift_readiness_lead_minutes: int = 60  # Generate readiness report N minutes before shift
    shift_schedules: list[str] = field(default_factory=lambda: ["morning", "afternoon", "day"])
    location_baseline_enabled: bool = True
    location_baseline_day_of_week: int = 0  # Monday for weekly baseline computation
    location_baseline_hour: int = 3  # 3 AM for baseline computation

    # Device metadata sync settings
    device_metadata_sync_enabled: bool = True
    device_metadata_sync_interval_minutes: int = 30  # Every 30 minutes
    device_metadata_sync_since_days: int = 30  # Only sync devices active in last N days

    # General
    timezone: str = "UTC"
    last_training_time: str | None = None
    last_scoring_time: str | None = None
    last_auto_retrain_time: str | None = None
    last_daily_digest_time: str | None = None
    last_shift_readiness_time: str | None = None
    last_location_baseline_time: str | None = None
    last_device_metadata_sync_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        result = asdict(self)
        result["training_interval"] = self.training_interval.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchedulerConfig:
        """Create from dictionary."""
        if "training_interval" in data and isinstance(data["training_interval"], str):
            data["training_interval"] = ScheduleInterval(data["training_interval"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SchedulerStatus:
    """Current status of the scheduler."""
    is_running: bool = False
    training_status: str = "idle"  # idle, running, completed, failed
    scoring_status: str = "idle"
    insights_status: str = "idle"  # idle, running, completed, failed
    last_training_result: dict | None = None
    last_scoring_result: dict | None = None
    last_insight_result: dict | None = None
    next_training_time: str | None = None
    next_scoring_time: str | None = None
    next_insight_time: str | None = None
    total_anomalies_detected: int = 0
    total_insights_generated: int = 0
    false_positive_rate: float = 0.0
    uptime_seconds: int = 0
    errors: list[str] = field(default_factory=list)


class AutomationScheduler:
    """
    Main scheduler service for autonomous anomaly detection.

    Features:
    - Configurable training schedule (hourly to weekly)
    - Real-time scoring of new data
    - Auto-retrain when feedback threshold exceeded
    - Health monitoring and status reporting
    """

    REDIS_CONFIG_KEY = "scheduler:config"
    REDIS_STATUS_KEY = "scheduler:status"
    REDIS_LOCK_KEY = "scheduler:lock"

    def __init__(self, redis_url: str | None = None):
        """Initialize the scheduler."""
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis: redis.Redis | None = None
        self._redis_available = False
        self._redis_max_retries = 3
        self._redis_retry_delay = 2  # seconds, doubles each retry
        self._running = False
        self._start_time: datetime | None = None
        self._config: SchedulerConfig | None = None
        self._config_last_loaded: datetime | None = None
        self._config_cache_ttl = 10  # Reload config from Redis every 10 seconds max
        self._status = SchedulerStatus()
        self._tasks: list[asyncio.Task] = []

    def _connect_redis(self) -> redis.Redis | None:
        """Attempt to connect to Redis with retry and backoff.

        Returns the Redis client if successful, None otherwise.
        """
        delay = self._redis_retry_delay
        for attempt in range(self._redis_max_retries):
            try:
                client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                client.ping()
                self._redis_available = True
                logger.info("Redis connection established")
                return client
            except redis.ConnectionError as e:
                logger.warning(
                    f"Redis connection attempt {attempt + 1}/{self._redis_max_retries} failed: {e}"
                )
                if attempt < self._redis_max_retries - 1:
                    import time
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected Redis error: {e}")
                break

        logger.error("Redis unavailable after all retries. Scheduler will run with limited functionality.")
        self._redis_available = False
        return None

    @property
    def redis(self) -> redis.Redis | None:
        """Get Redis connection with lazy initialization and retry.

        Returns None if Redis is unavailable, allowing graceful degradation.
        """
        if self._redis is None:
            self._redis = self._connect_redis()
        return self._redis

    def _config_from_env(self) -> SchedulerConfig:
        """Create configuration from environment variables."""
        interval_map = {
            "hourly": ScheduleInterval.HOURLY,
            "every_6_hours": ScheduleInterval.EVERY_6_HOURS,
            "every_12_hours": ScheduleInterval.EVERY_12_HOURS,
            "daily": ScheduleInterval.DAILY,
            "weekly": ScheduleInterval.WEEKLY,
            "manual": ScheduleInterval.MANUAL,
        }
        interval_str = os.getenv("SCHEDULER_TRAINING_INTERVAL", "daily").lower()

        # Parse shift schedules from comma-separated string with validation
        valid_shifts = {"morning", "afternoon", "night", "day"}
        shift_schedules_str = os.getenv("SCHEDULER_SHIFT_SCHEDULES", "morning,afternoon,day")
        shift_schedules = [
            s.strip().lower()
            for s in shift_schedules_str.split(",")
            if s.strip().lower() in valid_shifts
        ]
        # Fallback to default if no valid shifts
        if not shift_schedules:
            shift_schedules = ["morning", "afternoon", "day"]

        return SchedulerConfig(
            training_enabled=os.getenv("SCHEDULER_TRAINING_ENABLED", "true").lower() == "true",
            training_interval=interval_map.get(interval_str, ScheduleInterval.DAILY),
            training_hour=int(os.getenv("SCHEDULER_TRAINING_HOUR", "2")),
            training_lookback_days=int(os.getenv("SCHEDULER_TRAINING_LOOKBACK_DAYS", "90")),
            scoring_enabled=os.getenv("SCHEDULER_SCORING_ENABLED", "true").lower() == "true",
            scoring_interval_minutes=int(os.getenv("SCHEDULER_SCORING_INTERVAL_MINUTES", "15")),
            auto_retrain_enabled=os.getenv("SCHEDULER_AUTO_RETRAIN_ENABLED", "true").lower() == "true",
            auto_retrain_fp_threshold=float(os.getenv("SCHEDULER_AUTO_RETRAIN_FP_THRESHOLD", "0.15")),
            auto_retrain_min_feedback=int(os.getenv("SCHEDULER_AUTO_RETRAIN_MIN_FEEDBACK", "50")),
            alerting_enabled=os.getenv("SCHEDULER_ALERTING_ENABLED", "true").lower() == "true",
            high_anomaly_rate_threshold=float(os.getenv("SCHEDULER_HIGH_ANOMALY_THRESHOLD", "0.10")),
            # Insight generation settings
            insights_enabled=os.getenv("SCHEDULER_INSIGHTS_ENABLED", "true").lower() == "true",
            daily_digest_hour=int(os.getenv("SCHEDULER_DAILY_DIGEST_HOUR", "5")),
            shift_readiness_enabled=os.getenv("SCHEDULER_SHIFT_READINESS_ENABLED", "true").lower() == "true",
            shift_readiness_lead_minutes=int(os.getenv("SCHEDULER_SHIFT_READINESS_LEAD_MINUTES", "60")),
            shift_schedules=shift_schedules,
            location_baseline_enabled=os.getenv("SCHEDULER_LOCATION_BASELINE_ENABLED", "true").lower() == "true",
            location_baseline_day_of_week=int(os.getenv("SCHEDULER_LOCATION_BASELINE_DAY", "0")),
            location_baseline_hour=int(os.getenv("SCHEDULER_LOCATION_BASELINE_HOUR", "3")),
            # Device metadata sync settings
            device_metadata_sync_enabled=os.getenv("DEVICE_METADATA_SYNC_ENABLED", "true").lower() == "true",
            device_metadata_sync_interval_minutes=int(os.getenv("DEVICE_METADATA_SYNC_INTERVAL_MINUTES", "30")),
            device_metadata_sync_since_days=int(os.getenv("DEVICE_METADATA_SYNC_SINCE_DAYS", "30")),
        )

    def load_config(self, force: bool = False) -> SchedulerConfig:
        """
        Load configuration from Redis, falling back to environment variables.

        Uses a short cache TTL to avoid hammering Redis while still picking up
        config changes quickly.
        """
        now = datetime.utcnow()

        # Return cached config if still fresh (unless force reload)
        if (
            not force
            and self._config is not None
            and self._config_last_loaded is not None
            and (now - self._config_last_loaded).total_seconds() < self._config_cache_ttl
        ):
            return self._config

        # Try to load from Redis if available
        if self.redis is not None:
            try:
                data = self.redis.get(self.REDIS_CONFIG_KEY)
                if data:
                    self._config = SchedulerConfig.from_dict(json.loads(data))
                    self._config_last_loaded = now
                    return self._config
            except Exception as e:
                logger.warning(f"Failed to load config from Redis: {e}")

        # Use environment-based defaults
        self._config = self._config_from_env()
        self._config_last_loaded = now
        return self._config

    def save_config(self, config: SchedulerConfig) -> None:
        """Save configuration to Redis."""
        self._config = config  # Always update local cache
        if self.redis is None:
            logger.warning("Redis unavailable, config saved locally only")
            return
        try:
            self.redis.set(self.REDIS_CONFIG_KEY, json.dumps(config.to_dict()))
            logger.info("Scheduler configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config to Redis: {e}")

    def update_status(self, **kwargs) -> None:
        """Update scheduler status."""
        for key, value in kwargs.items():
            if hasattr(self._status, key):
                setattr(self._status, key, value)

        # Update uptime
        if self._start_time:
            self._status.uptime_seconds = int(
                (datetime.utcnow() - self._start_time).total_seconds()
            )

        # Persist to Redis if available
        if self.redis is None:
            return  # Status updated locally only
        try:
            status_dict = asdict(self._status)
            self.redis.set(self.REDIS_STATUS_KEY, json.dumps(status_dict))
        except Exception as e:
            logger.warning(f"Failed to update status in Redis: {e}")

    def get_status(self) -> SchedulerStatus:
        """Get current scheduler status."""
        if self.redis is None:
            return self._status  # Return local status if Redis unavailable

        try:
            data = self.redis.get(self.REDIS_STATUS_KEY)
            if data:
                status_data = json.loads(data)
                # Reconstruct SchedulerStatus
                self._status = SchedulerStatus(**{
                    k: v for k, v in status_data.items()
                    if k in SchedulerStatus.__dataclass_fields__
                })
        except Exception as e:
            logger.warning(f"Failed to get status from Redis: {e}")

        return self._status

    def calculate_next_training_time(self) -> datetime | None:
        """Calculate next scheduled training time."""
        if not self._config or not self._config.training_enabled:
            return None

        if self._config.training_interval == ScheduleInterval.MANUAL:
            return None

        now = datetime.utcnow()
        interval = self._config.training_interval

        if interval == ScheduleInterval.HOURLY:
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        elif interval == ScheduleInterval.EVERY_6_HOURS:
            hour = (now.hour // 6 + 1) * 6
            if hour >= 24:
                next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)

        elif interval == ScheduleInterval.EVERY_12_HOURS:
            hour = (now.hour // 12 + 1) * 12
            if hour >= 24:
                next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)

        elif interval == ScheduleInterval.DAILY:
            target_hour = self._config.training_hour
            next_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            if next_time <= now:
                next_time += timedelta(days=1)

        elif interval == ScheduleInterval.WEEKLY:
            target_hour = self._config.training_hour
            target_day = self._config.training_day_of_week
            days_ahead = target_day - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and now.hour >= target_hour):
                days_ahead += 7
            next_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            next_time += timedelta(days=days_ahead)

        else:
            return None

        return next_time

    async def run_training_job(self) -> dict[str, Any]:
        """Execute a training job."""
        logger.info("Starting scheduled training job...")
        self.update_status(training_status="running")

        # In mock mode, skip real training and return mock results
        if is_mock_mode():
            logger.info("Mock mode enabled - skipping real training, returning mock results")
            self._config.last_training_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                training_status="completed",
                last_training_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_path": "mock_model.pkl",
                    "training_samples": 1000,
                    "validation_samples": 100,
                    "mock_mode": True,
                },
            )
            logger.info("Mock training completed")
            return {"success": True, "model_path": "mock_model.pkl", "mock_mode": True}

        from device_anomaly.pipeline.training import RealDataTrainingPipeline, TrainingConfig

        try:
            # Use historical end date if configured (for demo with backup data)
            historical_end = get_historical_end_date()
            if historical_end:
                reference_time = historical_end
                logger.info(f"Using historical data end date for training: {historical_end.strftime('%Y-%m-%d')}")
            else:
                reference_time = datetime.utcnow()

            # Calculate date range
            end_date = reference_time.strftime("%Y-%m-%d")
            start_date = (
                reference_time - timedelta(days=self._config.training_lookback_days)
            ).strftime("%Y-%m-%d")

            config = TrainingConfig(
                start_date=start_date,
                end_date=end_date,
                validation_days=self._config.training_validation_days,
                contamination=0.03,
                n_estimators=300,
                export_onnx=True,
            )

            pipeline = RealDataTrainingPipeline(config)
            result = pipeline.run()

            # Update config with last training time
            self._config.last_training_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                training_status="completed",
                last_training_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": result.metrics if hasattr(result, "metrics") else {},
                },
            )

            logger.info("Scheduled training completed successfully")
            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Training job failed: {e}")
            try:
                self.update_status(
                    training_status="failed",
                    last_training_result={
                        "success": False,
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": str(e),
                    },
                    errors=self._status.errors[-9:] + [f"Training failed: {e}"],
                )
            except Exception as status_err:
                # Ensure we always record the failure, even if Redis is down
                logger.error(f"Failed to update status after training failure: {status_err}")
                self._status.training_status = "failed"
            return {"success": False, "error": str(e)}

    async def run_scoring_job(self) -> dict[str, Any]:
        """Execute a scoring job on recent data."""
        logger.info("Starting scheduled scoring job...")
        self.update_status(scoring_status="running")

        # In mock mode, skip real database operations and return mock results
        if is_mock_mode():
            logger.info("Mock mode enabled - skipping real scoring, returning mock results")
            import random
            mock_scored = random.randint(20, 100)
            mock_anomalies = random.randint(0, min(5, mock_scored))
            mock_rate = mock_anomalies / mock_scored if mock_scored > 0 else 0

            self._config.last_scoring_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                scoring_status="completed",
                last_scoring_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_scored": mock_scored,
                    "anomalies_detected": mock_anomalies,
                    "anomaly_rate": mock_rate,
                    "mock_mode": True,
                },
                total_anomalies_detected=self._status.total_anomalies_detected + mock_anomalies,
            )
            logger.info(f"Mock scoring completed: {mock_anomalies}/{mock_scored} anomalies")
            return {"success": True, "scored": mock_scored, "anomalies": mock_anomalies, "mock_mode": True}

        from device_anomaly.data_access.anomaly_persistence import persist_anomaly_results
        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from device_anomaly.data_access.watermark_store import get_watermark_store
        from device_anomaly.features.cohort_stats import (
            apply_cohort_stats,
            load_latest_cohort_stats,
        )
        from device_anomaly.features.device_features import (
            build_feature_builder_from_metadata,
            load_feature_metadata,
        )
        from device_anomaly.models.anomaly_detector import AnomalyDetector

        try:
            # Check if there's new source data to score
            # This prevents anomaly count inflation on static/backup databases
            watermark_store = get_watermark_store()
            has_new_data, source_timestamp, freshness_reason = watermark_store.check_source_data_freshness()

            if not has_new_data:
                logger.info(f"Skipping scoring: {freshness_reason}")
                self.update_status(
                    scoring_status="idle",
                    last_scoring_result={
                        "success": True,
                        "timestamp": datetime.utcnow().isoformat(),
                        "total_scored": 0,
                        "anomalies_detected": 0,
                        "skipped": True,
                        "reason": freshness_reason,
                    },
                )
                return {"success": True, "scored": 0, "skipped": True, "reason": freshness_reason}

            logger.info(f"Proceeding with scoring: {freshness_reason}")

            # Use historical end date if configured (for demo with backup data)
            # Otherwise use current time
            historical_end = get_historical_end_date()
            if historical_end:
                reference_time = historical_end
                logger.info(f"Using historical data end date: {historical_end.strftime('%Y-%m-%d')}")
            else:
                reference_time = datetime.utcnow()

            # Load enough data to compute trend features (7-day trends require 7+ days)
            # We load 14 days to ensure sufficient history for trend computation
            end_date = reference_time.strftime("%Y-%m-%d")
            start_date = (
                reference_time - timedelta(days=14)  # Look back 14 days for trend features
            ).strftime("%Y-%m-%d")

            # Load data
            df = load_unified_device_dataset(
                start_date=start_date,
                end_date=end_date,
                row_limit=100_000,
            )

            if df.empty:
                logger.info("No new data to score")
                self.update_status(scoring_status="idle")
                return {"success": True, "scored": 0}

            # Build features
            metadata = load_feature_metadata()
            builder = build_feature_builder_from_metadata(metadata, compute_cohort=False)
            df_features = builder.transform(df)
            cohort_stats = load_latest_cohort_stats()
            df_features = apply_cohort_stats(df_features, cohort_stats)

            # Load model and score
            detector = AnomalyDetector.load_latest()
            if detector is None:
                logger.warning("No trained model available for scoring")
                self.update_status(scoring_status="idle")
                return {"success": False, "error": "No model available"}

            scored_df = detector.score_dataframe(df_features)

            # Count anomalies
            anomaly_count = int((scored_df["anomaly_label"] == -1).sum())
            total_count = len(scored_df)
            anomaly_rate = anomaly_count / total_count if total_count > 0 else 0

            # Persist results (sync function, run in thread pool)
            await asyncio.to_thread(persist_anomaly_results, scored_df)

            # Record scoring watermark to prevent re-scoring same data
            # This is critical for preventing anomaly inflation on static databases
            scoring_timestamp = source_timestamp or datetime.utcnow()
            watermark_store.set_last_scoring_watermark(scoring_timestamp)
            logger.info(f"Recorded scoring watermark: {scoring_timestamp.isoformat()}")

            # Update status
            self._config.last_scoring_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                scoring_status="completed",
                last_scoring_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_scored": total_count,
                    "anomalies_detected": anomaly_count,
                    "anomaly_rate": anomaly_rate,
                },
                total_anomalies_detected=self._status.total_anomalies_detected + anomaly_count,
            )

            # Check for high anomaly rate alert
            if (
                self._config.alerting_enabled
                and self._config.alert_on_high_anomaly_rate
                and anomaly_rate > self._config.high_anomaly_rate_threshold
            ):
                await self._send_alert(
                    f"High anomaly rate detected: {anomaly_rate:.1%} ({anomaly_count}/{total_count})"
                )

            logger.info(f"Scoring completed: {anomaly_count}/{total_count} anomalies")
            return {"success": True, "scored": total_count, "anomalies": anomaly_count}

        except Exception as e:
            logger.error(f"Scoring job failed: {e}")
            try:
                self.update_status(
                    scoring_status="failed",
                    errors=self._status.errors[-9:] + [f"Scoring failed: {e}"],
                )
            except Exception as status_err:
                # Ensure we always record the failure, even if Redis is down
                logger.error(f"Failed to update status after scoring failure: {status_err}")
                self._status.scoring_status = "failed"
            return {"success": False, "error": str(e)}

    async def check_auto_retrain(self) -> bool:
        """Check if auto-retrain should be triggered based on feedback."""
        if not self._config.auto_retrain_enabled:
            return False

        # Check cooldown
        if self._config.last_auto_retrain_time:
            last_retrain = datetime.fromisoformat(self._config.last_auto_retrain_time)
            cooldown = timedelta(hours=self._config.auto_retrain_cooldown_hours)
            if datetime.utcnow() - last_retrain < cooldown:
                return False

        try:
            from device_anomaly.data_access.anomaly_persistence import get_feedback_stats

            stats = await asyncio.to_thread(get_feedback_stats)
            total_feedback = stats.get("total_feedback", 0)
            false_positives = stats.get("false_positives", 0)

            if total_feedback < self._config.auto_retrain_min_feedback:
                return False

            fp_rate = false_positives / total_feedback if total_feedback > 0 else 0
            self.update_status(false_positive_rate=fp_rate)

            if fp_rate > self._config.auto_retrain_fp_threshold:
                logger.info(
                    f"Auto-retrain triggered: FP rate {fp_rate:.1%} > {self._config.auto_retrain_fp_threshold:.1%}"
                )
                self._config.last_auto_retrain_time = datetime.utcnow().isoformat()
                self.save_config(self._config)
                return True

        except Exception as e:
            logger.warning(f"Failed to check auto-retrain condition: {e}")

        return False

    async def run_daily_digest_job(self) -> dict[str, Any]:
        """Generate daily insight digest for all locations."""
        logger.info("Starting daily insight digest generation...")
        self.update_status(insights_status="running")

        # In mock mode, skip real insight generation
        if is_mock_mode():
            logger.info("Mock mode enabled - skipping real insight generation")
            self._config.last_daily_digest_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                insights_status="completed",
                last_insight_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "daily_digest",
                    "insights_generated": 10,
                    "mock_mode": True,
                },
                total_insights_generated=self._status.total_insights_generated + 10,
            )
            logger.info("Mock daily digest completed")
            return {"success": True, "insights_generated": 10, "mock_mode": True}

        try:
            from device_anomaly.database.connection import get_results_db_session
            from device_anomaly.insights.generator import InsightGenerator

            # Use historical end date if configured
            historical_end = get_historical_end_date()
            if historical_end:
                insight_date = historical_end.date()
                logger.info(f"Using historical date for insights: {insight_date}")
            else:
                insight_date = datetime.utcnow().date()

            # Get tenant_id from environment or use default
            tenant_id = os.getenv("TENANT_ID", "default")

            db_session = get_results_db_session()
            try:
                generator = InsightGenerator(db_session, tenant_id)

                # Generate daily digest
                digest = generator.generate_daily_insights(insight_date=insight_date)

                # Save insights to database
                saved_count = generator.save_insights_to_db(digest.all_insights)
            finally:
                db_session.close()

            # Update config
            self._config.last_daily_digest_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                insights_status="completed",
                last_insight_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "daily_digest",
                    "insights_generated": saved_count,
                    "critical_count": digest.critical_count,
                    "high_count": digest.high_count,
                },
                total_insights_generated=self._status.total_insights_generated + saved_count,
            )

            logger.info(f"Daily digest completed: {saved_count} insights generated")
            return {"success": True, "insights_generated": saved_count}

        except Exception as e:
            logger.error(f"Daily digest generation failed: {e}")
            try:
                self.update_status(
                    insights_status="failed",
                    last_insight_result={
                        "success": False,
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "daily_digest",
                        "error": str(e),
                    },
                    errors=self._status.errors[-9:] + [f"Daily digest failed: {e}"],
                )
            except Exception as status_err:
                logger.error(f"Failed to update status after insights failure: {status_err}")
                self._status.insights_status = "failed"
            return {"success": False, "error": str(e)}

    async def run_shift_readiness_job(self, shift_name: str) -> dict[str, Any]:
        """Generate shift readiness reports for all locations."""
        logger.info(f"Starting shift readiness analysis for {shift_name} shift...")
        self.update_status(insights_status="running")

        # In mock mode, skip real analysis
        if is_mock_mode():
            logger.info("Mock mode enabled - skipping real shift readiness analysis")
            self._config.last_shift_readiness_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                insights_status="completed",
                last_insight_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "shift_readiness",
                    "shift_name": shift_name,
                    "locations_analyzed": 5,
                    "devices_at_risk": 3,
                    "mock_mode": True,
                },
            )
            logger.info(f"Mock shift readiness for {shift_name} completed")
            return {"success": True, "shift_name": shift_name, "mock_mode": True}

        try:
            from device_anomaly.database.connection import get_results_db_session
            from device_anomaly.database.schema import LocationMetadata
            from device_anomaly.insights.battery_shift import BatteryShiftAnalyzer

            # Use historical end date if configured
            historical_end = get_historical_end_date()
            shift_date = historical_end.date() if historical_end else datetime.utcnow().date()

            # Get tenant_id from environment or use default
            tenant_id = os.getenv("TENANT_ID", "default")

            # Get all configured locations and analyze shift readiness
            session = get_results_db_session()
            try:
                analyzer = BatteryShiftAnalyzer(session, tenant_id)

                locations = session.query(LocationMetadata).filter(
                    LocationMetadata.is_active
                ).all()

                total_at_risk = 0
                reports_generated = 0

                for location in locations:
                    try:
                        report = analyzer.analyze_shift_readiness(
                            location_id=location.location_id,
                            shift_date=shift_date,
                            shift_name=shift_name,
                        )
                        if report:
                            analyzer.save_shift_performance(report)
                            total_at_risk += report.devices_at_risk
                            reports_generated += 1
                    except Exception as loc_error:
                        logger.warning(f"Shift readiness failed for location {location.location_id}: {loc_error}")
            finally:
                session.close()

            self._config.last_shift_readiness_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                insights_status="completed",
                last_insight_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "shift_readiness",
                    "shift_name": shift_name,
                    "locations_analyzed": reports_generated,
                    "devices_at_risk": total_at_risk,
                },
            )

            # Alert if many devices at risk
            if total_at_risk > 10:
                await self._send_alert(
                    f"Shift readiness warning: {total_at_risk} devices may not last {shift_name} shift"
                )

            logger.info(f"Shift readiness completed: {reports_generated} locations, {total_at_risk} devices at risk")
            return {"success": True, "locations_analyzed": reports_generated, "devices_at_risk": total_at_risk}

        except Exception as e:
            logger.error(f"Shift readiness analysis failed: {e}")
            self.update_status(
                insights_status="failed",
                last_insight_result={
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "shift_readiness",
                    "error": str(e),
                },
                errors=self._status.errors[-9:] + [f"Shift readiness failed: {e}"],
            )
            return {"success": False, "error": str(e)}

    async def run_location_baseline_job(self) -> dict[str, Any]:
        """Compute weekly location baselines for comparison."""
        logger.info("Starting weekly location baseline computation...")
        self.update_status(insights_status="running")

        # In mock mode, skip real computation
        if is_mock_mode():
            logger.info("Mock mode enabled - skipping real baseline computation")
            self._config.last_location_baseline_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                insights_status="completed",
                last_insight_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "location_baseline",
                    "locations_updated": 5,
                    "mock_mode": True,
                },
            )
            logger.info("Mock location baseline completed")
            return {"success": True, "locations_updated": 5, "mock_mode": True}

        try:
            from device_anomaly.database.connection import get_results_db_session
            from device_anomaly.database.schema import LocationMetadata
            from device_anomaly.insights.entities import EntityAggregator

            # Calculate baselines over the past 4 weeks
            baseline_period_days = 28

            # Get tenant_id from environment or use default
            tenant_id = os.getenv("TENANT_ID", "default")

            session = get_results_db_session()
            try:
                aggregator = EntityAggregator(session, tenant_id)

                locations = session.query(LocationMetadata).filter(
                    LocationMetadata.is_active
                ).all()

                updated_count = 0

                for location in locations:
                    try:
                        # Get location aggregates
                        location_data = aggregator.aggregate_by_location(
                            anomalies=None,  # Will fetch from DB
                            period_days=baseline_period_days,
                        )

                        if location.location_id in location_data:
                            agg = location_data[location.location_id]

                            # Update location baselines
                            location.baseline_battery_drain_per_hour = agg.get("avg_battery_drain_per_hour")
                            location.baseline_disconnect_rate = agg.get("avg_disconnect_rate")
                            location.baseline_drop_rate = agg.get("avg_drop_rate")
                            location.baseline_computed_at = datetime.utcnow()

                            updated_count += 1

                    except Exception as loc_error:
                        logger.warning(f"Baseline computation failed for location {location.location_id}: {loc_error}")

                session.commit()
            finally:
                session.close()

            self._config.last_location_baseline_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            self.update_status(
                insights_status="completed",
                last_insight_result={
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "location_baseline",
                    "locations_updated": updated_count,
                },
            )

            logger.info(f"Location baseline computation completed: {updated_count} locations updated")
            return {"success": True, "locations_updated": updated_count}

        except Exception as e:
            logger.error(f"Location baseline computation failed: {e}")
            self.update_status(
                insights_status="failed",
                last_insight_result={
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "location_baseline",
                    "error": str(e),
                },
                errors=self._status.errors[-9:] + [f"Location baseline failed: {e}"],
            )
            return {"success": False, "error": str(e)}

    async def run_device_metadata_sync_job(self) -> dict[str, Any]:
        """Execute device metadata sync from MobiControl to PostgreSQL."""
        logger.info("Starting device metadata sync job...")

        # In mock mode, skip real sync
        if is_mock_mode():
            logger.info("Mock mode enabled - skipping real device metadata sync")
            self._config.last_device_metadata_sync_time = datetime.utcnow().isoformat()
            self.save_config(self._config)
            return {"success": True, "synced_count": 0, "mock_mode": True}

        try:
            from device_anomaly.services.device_metadata_sync import sync_device_metadata

            result = await asyncio.to_thread(
                sync_device_metadata,
                since_days=self._config.device_metadata_sync_since_days,
            )

            self._config.last_device_metadata_sync_time = datetime.utcnow().isoformat()
            self.save_config(self._config)

            logger.info(f"Device metadata sync completed: {result.get('synced_count', 0)} devices")
            return result

        except Exception as e:
            logger.error(f"Device metadata sync failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_shift_times(self, shift_name: str) -> tuple[int, int]:
        """Get start hour and end hour for a shift."""
        shift_times = {
            "morning": (6, 14),
            "afternoon": (14, 22),
            "night": (22, 6),
            "day": (8, 17),
        }
        return shift_times.get(shift_name, (8, 17))

    def calculate_next_shift_readiness_time(self) -> tuple[datetime, str] | None:
        """Calculate next time shift readiness should run and which shift."""
        if not self._config or not self._config.shift_readiness_enabled:
            return None

        now = datetime.utcnow()
        lead_minutes = self._config.shift_readiness_lead_minutes
        next_times = []

        for shift_name in self._config.shift_schedules:
            start_hour, _ = self._get_shift_times(shift_name)

            # Calculate when to run (lead_minutes before shift start)
            run_time = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            run_time -= timedelta(minutes=lead_minutes)

            # If run time has passed today, schedule for tomorrow
            if run_time <= now:
                run_time += timedelta(days=1)

            next_times.append((run_time, shift_name))

        if not next_times:
            return None

        # Return the soonest shift readiness time
        return min(next_times, key=lambda x: x[0])

    async def _send_alert(self, message: str) -> None:
        """Send an alert notification."""
        logger.warning(f"ALERT: {message}")
        # TODO: Implement actual alerting (email, Slack, webhook)
        try:
            timestamp = datetime.utcnow().isoformat()
            # Generate a unique alert ID based on timestamp
            alert_id = f"alert_{timestamp[:19].replace(':', '').replace('-', '').replace('T', '_')}"
            self.redis.lpush("scheduler:alerts", json.dumps({
                "id": alert_id,
                "timestamp": timestamp,
                "message": message,
                "acknowledged": False,
            }))
            self.redis.ltrim("scheduler:alerts", 0, 99)  # Keep last 100 alerts
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")

    async def _training_loop(self) -> None:
        """Background loop for scheduled training."""
        loop_iteration = 0
        while self._running:
            loop_iteration += 1
            try:
                # Reload config from Redis to pick up any changes from API
                self._config = self.load_config()
                now = datetime.utcnow()

                # Log loop status periodically (every 10 iterations = ~5 minutes at 30s sleep)
                if loop_iteration % 10 == 1:
                    logger.info(
                        f"[Training Loop] iteration={loop_iteration}, "
                        f"enabled={self._config.training_enabled}, "
                        f"interval={self._config.training_interval.value}, "
                        f"last_training={self._config.last_training_time or 'never'}"
                    )

                # Check for manual training jobs in the queue first (if Redis available)
                if self.redis is not None:
                    try:
                        manual_job = self.redis.lpop("ml:training:queue")
                        if manual_job:
                            try:
                                job_data = json.loads(manual_job)
                                logger.info(f"Processing manual training job: {job_data.get('job_id', 'unknown')}")
                                await self.run_training_job()
                                continue  # Check for more queued jobs immediately
                            except json.JSONDecodeError:
                                logger.error(f"Invalid training job data in queue: {manual_job}")
                    except Exception as e:
                        logger.warning(f"Failed to check training queue: {e}")

                if not self._config.training_enabled:
                    logger.info("[Training Loop] Training is DISABLED - skipping")
                    self.update_status(next_training_time=None)
                    await asyncio.sleep(60)
                    continue

                next_time = self.calculate_next_training_time()
                if next_time:
                    self.update_status(next_training_time=next_time.isoformat())

                    # Wait until next training time
                    wait_seconds = (next_time - now).total_seconds()

                    # Log next training time on first iteration and periodically
                    if loop_iteration % 10 == 1 or loop_iteration == 1:
                        hours_until = wait_seconds / 3600
                        logger.info(
                            f"[Training Loop] Next training: {next_time.isoformat()} "
                            f"({hours_until:.1f} hours from now)"
                        )

                    if wait_seconds > 0:
                        # Sleep in shorter intervals to allow config reload and quick response
                        # Use smaller sleep to ensure we don't miss the training window
                        sleep_duration = min(wait_seconds, 30)
                        logger.debug(f"Training scheduled in {wait_seconds:.0f}s, sleeping for {sleep_duration:.0f}s")
                        await asyncio.sleep(sleep_duration)
                        # Don't continue - fall through to check again if it's time
                        # This ensures we eventually reach the training job when wait_seconds <= 0
                        if (next_time - datetime.utcnow()).total_seconds() > 0:
                            continue

                    # Time to train!
                    logger.info("[Training Loop] Training time reached! Starting scheduled training job...")
                    await self.run_training_job()

                    # After training, sleep briefly to avoid immediate re-trigger
                    await asyncio.sleep(5)

                else:
                    logger.info(f"[Training Loop] No next training time (interval={self._config.training_interval.value})")
                    self.update_status(next_training_time=None)
                    await asyncio.sleep(60)

            except asyncio.CancelledError:
                logger.info("[Training Loop] Cancelled, exiting loop")
                break
            except Exception as e:
                logger.error(f"[Training Loop] Error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _scoring_loop(self) -> None:
        """Background loop for continuous scoring."""
        while self._running:
            try:
                # Reload config from Redis to pick up any changes from API
                self._config = self.load_config()

                # Check for manual scoring jobs in the queue first
                manual_job = self.redis.lpop("scheduler:scoring:queue")
                if manual_job:
                    try:
                        job_data = json.loads(manual_job)
                        logger.info(f"Processing manual scoring job: {job_data}")
                        await self.run_scoring_job()
                        continue  # Check for more queued jobs immediately
                    except json.JSONDecodeError:
                        logger.error(f"Invalid job data in queue: {manual_job}")

                if not self._config.scoring_enabled:
                    self.update_status(next_scoring_time=None)
                    await asyncio.sleep(60)
                    continue

                # Calculate next scoring time based on current config
                interval = self._config.scoring_interval_minutes * 60
                next_time = datetime.utcnow() + timedelta(seconds=interval)
                self.update_status(next_scoring_time=next_time.isoformat())

                await self.run_scoring_job()

                # Wait for next interval, but check queue every 5 seconds
                elapsed = 0
                while elapsed < interval and self._running:
                    # Check every 5 seconds for manual jobs or config changes
                    sleep_time = min(5, interval - elapsed)
                    await asyncio.sleep(sleep_time)
                    elapsed += sleep_time

                    # Check for manual scoring jobs in queue
                    if self.redis.llen("scheduler:scoring:queue") > 0:
                        break  # Exit wait loop to process queued job

                    # Check for config changes every 30 seconds
                    if elapsed % 30 < 5:
                        new_config = self.load_config()
                        if new_config.scoring_interval_minutes != self._config.scoring_interval_minutes:
                            self._config = new_config
                            break  # Restart loop with new interval

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scoring loop error: {e}")
                await asyncio.sleep(60)

    async def _auto_retrain_loop(self) -> None:
        """Background loop for checking auto-retrain conditions."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Reload config from Redis
                self._config = self.load_config()

                if await self.check_auto_retrain():
                    logger.info("Triggering auto-retrain based on feedback threshold")
                    await self.run_training_job()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-retrain loop error: {e}")

    async def _insights_loop(self) -> None:
        """Background loop for insight generation (daily digest, shift readiness, baselines)."""
        # Short delay on startup to let other loops initialize
        await asyncio.sleep(10)

        while self._running:
            try:
                # Reload config from Redis
                self._config = self.load_config()

                # Check for manually queued insight jobs first
                manual_job = self.redis.lpop("scheduler:insights:queue")
                if manual_job:
                    try:
                        job_data = json.loads(manual_job)
                        job_type = job_data.get("type")
                        logger.info(f"Processing manual insight job: {job_type}")

                        if job_type == "daily_digest":
                            await self.run_daily_digest_job()
                        elif job_type == "shift_readiness":
                            shift_name = job_data.get("shift_name", "morning")
                            await self.run_shift_readiness_job(shift_name)
                        elif job_type == "location_baseline":
                            await self.run_location_baseline_job()
                        else:
                            logger.warning(f"Unknown insight job type: {job_type}")

                        continue  # Check for more queued jobs immediately
                    except json.JSONDecodeError:
                        logger.error(f"Invalid insight job data in queue: {manual_job}")

                if not self._config.insights_enabled:
                    self.update_status(next_insight_time=None)
                    await asyncio.sleep(60)
                    continue

                now = datetime.utcnow()
                next_jobs = []

                # 1. Daily digest scheduling
                target_digest_time = now.replace(
                    hour=self._config.daily_digest_hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                if target_digest_time <= now:
                    target_digest_time += timedelta(days=1)

                # Check if we should run daily digest now
                if self._config.last_daily_digest_time:
                    last_digest = datetime.fromisoformat(self._config.last_daily_digest_time)
                    # Run if we're within the target hour and haven't run today
                    if (
                        now.hour == self._config.daily_digest_hour
                        and last_digest.date() < now.date()
                    ):
                        logger.info("Running scheduled daily digest...")
                        await self.run_daily_digest_job()
                        continue
                else:
                    # Never run before - run if we're in the target hour
                    if now.hour == self._config.daily_digest_hour:
                        logger.info("Running initial daily digest...")
                        await self.run_daily_digest_job()
                        continue

                next_jobs.append(("daily_digest", target_digest_time))

                # 2. Shift readiness scheduling
                if self._config.shift_readiness_enabled:
                    next_shift = self.calculate_next_shift_readiness_time()
                    if next_shift:
                        shift_time, shift_name = next_shift
                        next_jobs.append(("shift_readiness", shift_time, shift_name))

                        # Check if we should run shift readiness now
                        if shift_time <= now + timedelta(seconds=60):
                            logger.info(f"Running scheduled shift readiness for {shift_name}...")
                            await self.run_shift_readiness_job(shift_name)
                            continue

                # 3. Weekly location baseline scheduling
                if self._config.location_baseline_enabled:
                    target_baseline_time = now.replace(
                        hour=self._config.location_baseline_hour,
                        minute=0,
                        second=0,
                        microsecond=0,
                    )

                    # Calculate next baseline day (specific day of week)
                    days_until_baseline = (
                        self._config.location_baseline_day_of_week - now.weekday()
                    ) % 7
                    if days_until_baseline == 0 and target_baseline_time <= now:
                        days_until_baseline = 7
                    target_baseline_time += timedelta(days=days_until_baseline)

                    next_jobs.append(("location_baseline", target_baseline_time))

                    # Check if we should run baseline now
                    if self._config.last_location_baseline_time:
                        last_baseline = datetime.fromisoformat(
                            self._config.last_location_baseline_time
                        )
                        # Run if we're on the right day/hour and haven't run this week
                        if (
                            now.weekday() == self._config.location_baseline_day_of_week
                            and now.hour == self._config.location_baseline_hour
                            and (now - last_baseline).days >= 6
                        ):
                            logger.info("Running scheduled weekly baseline computation...")
                            await self.run_location_baseline_job()
                            continue
                    else:
                        # Never run before - run if we're on the right day/hour
                        if (
                            now.weekday() == self._config.location_baseline_day_of_week
                            and now.hour == self._config.location_baseline_hour
                        ):
                            logger.info("Running initial location baseline computation...")
                            await self.run_location_baseline_job()
                            continue

                # Update status with next scheduled insight job
                if next_jobs:
                    next_job = min(next_jobs, key=lambda x: x[1])
                    self.update_status(next_insight_time=next_job[1].isoformat())

                # Sleep for a minute before checking again
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Insights loop error: {e}")
                await asyncio.sleep(60)

    async def _device_metadata_sync_loop(self) -> None:
        """Background loop for periodic device metadata sync from MobiControl."""
        # Run initial sync shortly after startup
        await asyncio.sleep(15)

        while self._running:
            try:
                # Reload config from Redis
                self._config = self.load_config()

                # Check for manually queued sync jobs
                if self.redis is not None:
                    try:
                        manual_job = self.redis.lpop("scheduler:device_sync:queue")
                        if manual_job:
                            logger.info("Processing manual device metadata sync job")
                            await self.run_device_metadata_sync_job()
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to check device sync queue: {e}")

                if not self._config.device_metadata_sync_enabled:
                    await asyncio.sleep(60)
                    continue

                # Run sync job
                logger.info("Running scheduled device metadata sync...")
                await self.run_device_metadata_sync_job()

                # Wait for configured interval
                interval_seconds = self._config.device_metadata_sync_interval_minutes * 60
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Device metadata sync loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 min on error

    async def start(self) -> None:
        """Start the scheduler service."""
        logger.info("=" * 60)
        logger.info("AUTOMATION SCHEDULER STARTING")
        logger.info("=" * 60)

        self._running = True
        self._start_time = datetime.utcnow()
        self._config = self.load_config(force=True)

        # Detailed startup banner
        logger.info(f"Start time: {self._start_time.isoformat()}")
        logger.info(f"Mock mode: {is_mock_mode()}")
        logger.info(f"Redis URL: {self.redis_url}")
        logger.info(f"Redis connected: {self._redis_available}")

        # Training config
        logger.info("-" * 40)
        logger.info("TRAINING CONFIG:")
        logger.info(f"  enabled: {self._config.training_enabled}")
        logger.info(f"  interval: {self._config.training_interval.value}")
        logger.info(f"  training_hour: {self._config.training_hour} (for daily/weekly)")
        logger.info(f"  lookback_days: {self._config.training_lookback_days}")
        logger.info(f"  last_training_time: {self._config.last_training_time or 'never'}")

        # Calculate and show next training time
        next_training = self.calculate_next_training_time()
        if next_training:
            wait_hours = (next_training - datetime.utcnow()).total_seconds() / 3600
            logger.info(f"  NEXT TRAINING: {next_training.isoformat()} ({wait_hours:.1f}h from now)")
        else:
            logger.info("  NEXT TRAINING: None (disabled or manual)")

        # Scoring config
        logger.info("-" * 40)
        logger.info("SCORING CONFIG:")
        logger.info(f"  enabled: {self._config.scoring_enabled}")
        logger.info(f"  interval_minutes: {self._config.scoring_interval_minutes}")

        # Insights config
        logger.info("-" * 40)
        logger.info("INSIGHTS CONFIG:")
        logger.info(f"  enabled: {self._config.insights_enabled}")
        logger.info(f"  shift_readiness_enabled: {self._config.shift_readiness_enabled}")
        logger.info("=" * 60)

        # Update initial status
        self.update_status(
            is_running=True,
            training_status="idle",
            scoring_status="idle",
            insights_status="idle",
        )

        # Start background loops
        self._tasks = [
            asyncio.create_task(self._training_loop()),
            asyncio.create_task(self._scoring_loop()),
            asyncio.create_task(self._auto_retrain_loop()),
            asyncio.create_task(self._insights_loop()),
            asyncio.create_task(self._device_metadata_sync_loop()),
        ]

        logger.info("Scheduler started successfully with insight generation and device metadata sync enabled")

    async def stop(self) -> None:
        """Stop the scheduler service."""
        logger.info("Stopping Automation Scheduler...")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

        self.update_status(is_running=False)
        logger.info("Scheduler stopped")

    def _check_redis_command(self) -> str | None:
        """Check for and consume any pending command from Redis.

        Returns the command if one was found, or None.
        """
        if self.redis is None:
            return None  # Can't check commands without Redis

        try:
            command = self.redis.get("scheduler:command")
            if command:
                # Clear the command after reading
                self.redis.delete("scheduler:command")
                return command.lower()
        except Exception as e:
            logger.warning(f"Failed to check Redis command: {e}")
        return None

    async def run_forever(self) -> None:
        """Run the scheduler until interrupted."""
        await self.start()

        # Setup signal handlers
        loop = asyncio.get_running_loop()

        def signal_handler():
            asyncio.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        # Wait until stopped, checking for Redis commands
        while self._running:
            # Check for stop command from API
            command = self._check_redis_command()
            if command == "stop":
                logger.info("Received stop command from API")
                await self.stop()
                break
            # Note: "start" command is ignored while running; scheduler must be restarted externally

            await asyncio.sleep(1)


async def main():
    """Entry point for the scheduler service."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    scheduler = AutomationScheduler()
    await scheduler.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
