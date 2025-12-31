"""Repository pattern implementations for database access.

This module provides the repository layer for data access,
implementing the Repository pattern for clean separation of
database operations from business logic.

Example:
    from device_anomaly.db import DatabaseSession
    from device_anomaly.db.repositories import AnomalyRepository

    db = DatabaseSession()
    with db.session() as session:
        repo = AnomalyRepository(session)
        anomalies = repo.get_by_tenant('tenant-1', severity='high')
"""
from device_anomaly.db.repositories.anomaly_repo import AnomalyRepository, MAX_PAGE_SIZE
from device_anomaly.db.repositories.base import BaseRepository
from device_anomaly.db.repositories.baseline_repo import BaselineRepository
from device_anomaly.db.repositories.device_repo import DeviceRepository

__all__ = [
    "AnomalyRepository",
    "BaseRepository",
    "BaselineRepository",
    "DeviceRepository",
    "MAX_PAGE_SIZE",
]
