"""
Ingestion Orchestrator with Weight-Based Parallelism Throttling.

Controls concurrent table loads to prevent overwhelming SQL Server.
Uses a weighted semaphore where total concurrent weight never exceeds MAX_INGEST_WEIGHT.

Table Weight Assignment:
- XSight hourly huge tables (cs_DataUsageByHour, cs_PresetApps): weight 5
- Other XSight extended tables: weight 2
- MobiControl time-series tables: weight 2
- Small tables (Alert, Events, Devices): weight 1
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from device_anomaly.config.settings import get_settings

logger = logging.getLogger(__name__)


# Default maximum concurrent weight
DEFAULT_MAX_INGEST_WEIGHT = 5

# Table weight assignments
XSIGHT_HOURLY_HUGE_TABLES: Set[str] = {
    "cs_DataUsageByHour",
    "cs_DataUsageByHourApp",
    "cs_PresetApps",
    "cs_WiFiLocation",  # 6M+ rows
}

XSIGHT_EXTENDED_TABLES: Set[str] = {
    "cs_BatteryDrain",
    "cs_CrashLogs",
    "cs_WiFiByHour",
    "cs_CellTowerByHour",
    "cs_RoamingByHour",
}

MC_TIMESERIES_TABLES: Set[str] = {
    "DeviceStatInt",
    "DeviceStatString",
    "DeviceStatLocation",
    "DeviceStatNetTraffic",
    "MainLog",
    "DeviceInstalledApp",
}

SMALL_TABLES: Set[str] = {
    "Alert",
    "Events",
    "Devices",
    "Device",
    "cs_Devices",
}


class TableCategory(Enum):
    """Table category for weight assignment."""
    XSIGHT_HOURLY_HUGE = "xsight_hourly_huge"
    XSIGHT_EXTENDED = "xsight_extended"
    MC_TIMESERIES = "mc_timeseries"
    SMALL = "small"
    DEFAULT = "default"


def get_table_weight(table_name: str) -> int:
    """
    Get the weight for a table based on its category.

    Weight reflects query cost - higher weight = more resource intensive.
    """
    if table_name in XSIGHT_HOURLY_HUGE_TABLES:
        return 5  # Full capacity - only one at a time
    elif table_name in XSIGHT_EXTENDED_TABLES:
        return 2  # Can run 2 in parallel
    elif table_name in MC_TIMESERIES_TABLES:
        return 2  # Can run 2 in parallel
    elif table_name in SMALL_TABLES:
        return 1  # Can run 5 in parallel
    else:
        return 1  # Default to lightweight


def get_table_category(table_name: str) -> TableCategory:
    """Get the category for a table."""
    if table_name in XSIGHT_HOURLY_HUGE_TABLES:
        return TableCategory.XSIGHT_HOURLY_HUGE
    elif table_name in XSIGHT_EXTENDED_TABLES:
        return TableCategory.XSIGHT_EXTENDED
    elif table_name in MC_TIMESERIES_TABLES:
        return TableCategory.MC_TIMESERIES
    elif table_name in SMALL_TABLES:
        return TableCategory.SMALL
    else:
        return TableCategory.DEFAULT


class WeightedSemaphore:
    """
    A semaphore that tracks weight instead of count.

    Allows fine-grained control over concurrent operations where
    different operations have different resource costs.

    Example:
        sem = WeightedSemaphore(max_weight=5)
        async with sem.acquire(weight=2):
            # This operation takes weight 2
            await heavy_operation()
    """

    def __init__(self, max_weight: int = DEFAULT_MAX_INGEST_WEIGHT):
        self.max_weight = max_weight
        self._current_weight = 0
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._waiters: List[int] = []  # Track waiting weights for fairness

    @property
    def current_weight(self) -> int:
        """Current total weight of running operations."""
        return self._current_weight

    @property
    def available_weight(self) -> int:
        """Weight available for new operations."""
        return self.max_weight - self._current_weight

    async def acquire(self, weight: int = 1) -> "WeightedSemaphoreContext":
        """
        Acquire the semaphore with the given weight.

        Blocks until enough weight is available.
        Returns a context manager for automatic release.
        """
        if weight > self.max_weight:
            logger.warning(
                f"Requested weight {weight} exceeds max {self.max_weight}, "
                f"clamping to max"
            )
            weight = self.max_weight

        async with self._condition:
            # Add to waiters queue for fairness tracking
            self._waiters.append(weight)

            try:
                # Wait until we have enough capacity AND we're first in queue
                # (or our weight is small enough to squeeze in)
                while True:
                    if (self._current_weight + weight <= self.max_weight and
                            (not self._waiters or self._waiters[0] == weight)):
                        break
                    await self._condition.wait()

                # Remove from waiters and acquire
                self._waiters.remove(weight)
                self._current_weight += weight
                logger.debug(
                    f"Acquired weight {weight}, current total: {self._current_weight}/{self.max_weight}"
                )
            except Exception:
                # Clean up waiter on error
                if weight in self._waiters:
                    self._waiters.remove(weight)
                raise

        return WeightedSemaphoreContext(self, weight)

    async def release(self, weight: int = 1) -> None:
        """Release the given weight back to the semaphore."""
        async with self._condition:
            self._current_weight -= weight
            if self._current_weight < 0:
                logger.error(f"Semaphore weight went negative: {self._current_weight}")
                self._current_weight = 0
            logger.debug(
                f"Released weight {weight}, current total: {self._current_weight}/{self.max_weight}"
            )
            self._condition.notify_all()


class WeightedSemaphoreContext:
    """Context manager for weighted semaphore acquisition."""

    def __init__(self, semaphore: WeightedSemaphore, weight: int):
        self._semaphore = semaphore
        self._weight = weight

    async def __aenter__(self) -> "WeightedSemaphoreContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._semaphore.release(self._weight)


@dataclass
class IngestionTask:
    """Represents a single table ingestion task."""
    table_name: str
    source_db: str  # "xsight" or "mobicontrol"
    weight: int = field(default=1)
    category: TableCategory = field(default=TableCategory.DEFAULT)

    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rows_fetched: int = 0
    rows_inserted: int = 0
    rows_deduped: int = 0
    error: Optional[str] = None

    def __post_init__(self):
        if self.weight == 1:  # Only auto-assign if not explicitly set
            self.weight = get_table_weight(self.table_name)
        if self.category == TableCategory.DEFAULT:
            self.category = get_table_category(self.table_name)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of the task in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def success(self) -> bool:
        """Whether the task completed successfully."""
        return self.completed_at is not None and self.error is None


@dataclass
class IngestionBatchResult:
    """Result of an ingestion batch run."""
    started_at: datetime
    completed_at: datetime
    tasks: List[IngestionTask]
    total_rows_fetched: int = 0
    total_rows_inserted: int = 0
    total_rows_deduped: int = 0

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def success_count(self) -> int:
        return sum(1 for t in self.tasks if t.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for t in self.tasks if not t.success)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "total_tables": len(self.tasks),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_rows_fetched": self.total_rows_fetched,
            "total_rows_inserted": self.total_rows_inserted,
            "total_rows_deduped": self.total_rows_deduped,
            "tasks": [
                {
                    "table_name": t.table_name,
                    "source_db": t.source_db,
                    "weight": t.weight,
                    "category": t.category.value,
                    "rows_fetched": t.rows_fetched,
                    "rows_inserted": t.rows_inserted,
                    "rows_deduped": t.rows_deduped,
                    "duration_seconds": t.duration_seconds,
                    "success": t.success,
                    "error": t.error,
                }
                for t in self.tasks
            ],
        }


class IngestionOrchestrator:
    """
    Orchestrates parallel table ingestion with weight-based throttling.

    Usage:
        orchestrator = IngestionOrchestrator(max_weight=5)
        result = await orchestrator.run_batch(tables, loader_func)
    """

    def __init__(
        self,
        max_weight: Optional[int] = None,
    ):
        settings = get_settings()
        self.max_weight = max_weight or settings.ingest_max_tables_parallel
        self._semaphore = WeightedSemaphore(max_weight=self.max_weight)
        self._active_tasks: Dict[str, IngestionTask] = {}

    async def _run_task(
        self,
        task: IngestionTask,
        loader_func: Callable[[str, str], Dict[str, Any]],
    ) -> IngestionTask:
        """
        Run a single ingestion task with weight-based throttling.

        Args:
            task: The ingestion task to run
            loader_func: Function that takes (table_name, source_db) and returns
                        dict with keys: rows_fetched, rows_inserted, rows_deduped
        """
        weight = task.weight
        table_key = f"{task.source_db}.{task.table_name}"

        logger.info(
            f"Queuing {table_key} (weight={weight}, "
            f"current={self._semaphore.current_weight}/{self.max_weight})"
        )

        ctx = await self._semaphore.acquire(weight)
        async with ctx:
            task.started_at = datetime.now(timezone.utc)
            self._active_tasks[table_key] = task

            logger.info(
                f"Starting {table_key} (weight={weight}, "
                f"current={self._semaphore.current_weight}/{self.max_weight})"
            )

            try:
                # Run the loader in a thread pool to not block the event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    loader_func,
                    task.table_name,
                    task.source_db,
                )

                task.rows_fetched = result.get("rows_fetched", 0)
                task.rows_inserted = result.get("rows_inserted", 0)
                task.rows_deduped = result.get("rows_deduped", 0)
                task.completed_at = datetime.now(timezone.utc)

                logger.info(
                    f"Completed {table_key}: "
                    f"fetched={task.rows_fetched}, "
                    f"inserted={task.rows_inserted}, "
                    f"deduped={task.rows_deduped}, "
                    f"duration={task.duration_seconds:.2f}s"
                )

            except Exception as e:
                task.error = str(e)
                task.completed_at = datetime.now(timezone.utc)
                logger.error(f"Failed {table_key}: {e}")

            finally:
                del self._active_tasks[table_key]

        return task

    async def run_batch(
        self,
        tables: List[Dict[str, str]],
        loader_func: Callable[[str, str], Dict[str, Any]],
    ) -> IngestionBatchResult:
        """
        Run ingestion for a batch of tables with weight-based parallelism.

        Args:
            tables: List of dicts with keys: table_name, source_db
            loader_func: Function that takes (table_name, source_db) and returns
                        dict with keys: rows_fetched, rows_inserted, rows_deduped

        Returns:
            IngestionBatchResult with all task results
        """
        started_at = datetime.now(timezone.utc)

        # Create tasks with weights
        tasks = [
            IngestionTask(
                table_name=t["table_name"],
                source_db=t["source_db"],
            )
            for t in tables
        ]

        # Sort by weight descending - run heavy tables first while light ones fill gaps
        tasks.sort(key=lambda t: t.weight, reverse=True)

        logger.info(
            f"Starting batch ingestion: {len(tasks)} tables, "
            f"max_weight={self.max_weight}, "
            f"total_weight={sum(t.weight for t in tasks)}"
        )

        # Run all tasks concurrently (semaphore handles throttling)
        completed_tasks = await asyncio.gather(
            *[self._run_task(task, loader_func) for task in tasks],
            return_exceptions=False,
        )

        completed_at = datetime.now(timezone.utc)

        result = IngestionBatchResult(
            started_at=started_at,
            completed_at=completed_at,
            tasks=list(completed_tasks),
            total_rows_fetched=sum(t.rows_fetched for t in completed_tasks),
            total_rows_inserted=sum(t.rows_inserted for t in completed_tasks),
            total_rows_deduped=sum(t.rows_deduped for t in completed_tasks),
        )

        logger.info(
            f"Batch complete: {result.success_count}/{len(tasks)} succeeded, "
            f"duration={result.duration_seconds:.2f}s, "
            f"rows_fetched={result.total_rows_fetched}"
        )

        return result

    def get_active_tasks(self) -> Dict[str, IngestionTask]:
        """Get currently running tasks."""
        return dict(self._active_tasks)

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            "max_weight": self.max_weight,
            "current_weight": self._semaphore.current_weight,
            "available_weight": self._semaphore.available_weight,
            "active_tasks": [
                {
                    "table": k,
                    "weight": v.weight,
                    "started_at": v.started_at.isoformat() if v.started_at else None,
                }
                for k, v in self._active_tasks.items()
            ],
        }


def create_table_list_for_ingestion(
    xsight_tables: Optional[List[str]] = None,
    mc_tables: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Create a list of tables for ingestion with source database info.

    Args:
        xsight_tables: List of XSight table names
        mc_tables: List of MobiControl table names

    Returns:
        List of dicts with table_name and source_db keys
    """
    tables = []

    if xsight_tables:
        tables.extend([
            {"table_name": t, "source_db": "xsight"}
            for t in xsight_tables
        ])

    if mc_tables:
        tables.extend([
            {"table_name": t, "source_db": "mobicontrol"}
            for t in mc_tables
        ])

    return tables


# Synchronous wrapper for non-async contexts
def run_batch_sync(
    tables: List[Dict[str, str]],
    loader_func: Callable[[str, str], Dict[str, Any]],
    max_weight: Optional[int] = None,
) -> IngestionBatchResult:
    """
    Synchronous wrapper for batch ingestion.

    Creates a new event loop if needed.
    """
    orchestrator = IngestionOrchestrator(max_weight=max_weight)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, need to use run_in_executor
            raise RuntimeError(
                "Cannot use run_batch_sync from within an async context. "
                "Use orchestrator.run_batch() directly instead."
            )
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(orchestrator.run_batch(tables, loader_func))
    finally:
        loop.close()
