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
from enum import Enum
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


def is_mock_mode() -> bool:
    """Check if mock mode is enabled via environment variable."""
    return os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")


def get_historical_end_date() -> Optional[datetime]:
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


class ScheduleInterval(str, Enum):
    """Supported scheduling intervals."""
    HOURLY = "hourly"
    EVERY_6_HOURS = "every_6_hours"
    EVERY_12_HOURS = "every_12_hours"
    DAILY = "daily"
    WEEKLY = "weekly"
    MANUAL = "manual"  # Disabled, only manual triggers


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

    # General
    timezone: str = "UTC"
    last_training_time: Optional[str] = None
    last_scoring_time: Optional[str] = None
    last_auto_retrain_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        result = asdict(self)
        result["training_interval"] = self.training_interval.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulerConfig":
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
    last_training_result: Optional[Dict] = None
    last_scoring_result: Optional[Dict] = None
    next_training_time: Optional[str] = None
    next_scoring_time: Optional[str] = None
    total_anomalies_detected: int = 0
    false_positive_rate: float = 0.0
    uptime_seconds: int = 0
    errors: List[str] = field(default_factory=list)


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

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize the scheduler."""
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis: Optional[redis.Redis] = None
        self._running = False
        self._start_time: Optional[datetime] = None
        self._config: Optional[SchedulerConfig] = None
        self._config_last_loaded: Optional[datetime] = None
        self._config_cache_ttl = 10  # Reload config from Redis every 10 seconds max
        self._status = SchedulerStatus()
        self._tasks: List[asyncio.Task] = []

    @property
    def redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
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
        try:
            self.redis.set(self.REDIS_CONFIG_KEY, json.dumps(config.to_dict()))
            self._config = config
            logger.info("Scheduler configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

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

        # Persist to Redis
        try:
            status_dict = asdict(self._status)
            self.redis.set(self.REDIS_STATUS_KEY, json.dumps(status_dict))
        except Exception as e:
            logger.warning(f"Failed to update status in Redis: {e}")

    def get_status(self) -> SchedulerStatus:
        """Get current scheduler status."""
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

    def calculate_next_training_time(self) -> Optional[datetime]:
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

    async def run_training_job(self) -> Dict[str, Any]:
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
            self.update_status(
                training_status="failed",
                last_training_result={
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                },
                errors=self._status.errors[-9:] + [f"Training failed: {e}"],
            )
            return {"success": False, "error": str(e)}

    async def run_scoring_job(self) -> Dict[str, Any]:
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

        from device_anomaly.data_access.unified_loader import load_unified_device_dataset
        from device_anomaly.features.device_features import DeviceFeatureBuilder
        from device_anomaly.models.anomaly_detector import AnomalyDetector
        from device_anomaly.data_access.anomaly_persistence import persist_anomaly_results

        try:
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
            builder = DeviceFeatureBuilder()
            df_features = builder.transform(df)

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
            self.update_status(
                scoring_status="failed",
                errors=self._status.errors[-9:] + [f"Scoring failed: {e}"],
            )
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
        while self._running:
            try:
                # Reload config from Redis to pick up any changes from API
                self._config = self.load_config()

                if not self._config.training_enabled:
                    self.update_status(next_training_time=None)
                    await asyncio.sleep(60)
                    continue

                next_time = self.calculate_next_training_time()
                if next_time:
                    self.update_status(next_training_time=next_time.isoformat())

                    # Wait until next training time
                    wait_seconds = (next_time - datetime.utcnow()).total_seconds()
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
                    logger.info(f"Training time reached, starting scheduled training job...")
                    await self.run_training_job()

                    # After training, sleep briefly to avoid immediate re-trigger
                    await asyncio.sleep(5)

                else:
                    self.update_status(next_training_time=None)
                    await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training loop error: {e}")
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

    async def start(self) -> None:
        """Start the scheduler service."""
        logger.info("Starting Automation Scheduler...")
        self._running = True
        self._start_time = datetime.utcnow()
        self._config = self.load_config(force=True)

        # Log configuration
        logger.info(
            f"Scheduler config: training_enabled={self._config.training_enabled}, "
            f"training_interval={self._config.training_interval.value}, "
            f"scoring_enabled={self._config.scoring_enabled}, "
            f"scoring_interval_minutes={self._config.scoring_interval_minutes}"
        )

        # Update initial status
        self.update_status(
            is_running=True,
            training_status="idle",
            scoring_status="idle",
        )

        # Start background loops
        self._tasks = [
            asyncio.create_task(self._training_loop()),
            asyncio.create_task(self._scoring_loop()),
            asyncio.create_task(self._auto_retrain_loop()),
        ]

        logger.info("Scheduler started successfully")

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

    async def run_forever(self) -> None:
        """Run the scheduler until interrupted."""
        await self.start()

        # Setup signal handlers
        loop = asyncio.get_running_loop()

        def signal_handler():
            asyncio.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        # Wait until stopped
        while self._running:
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
