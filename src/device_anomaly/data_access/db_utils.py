"""Shared database utilities for data access layer.

This module provides common utilities for all data loaders:
- table_exists: Check if a table exists
- get_valid_columns: Validate which columns exist in a table
- BaseLoader: Abstract base class for SQL Server loaders
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

T = TypeVar("T")


def table_exists(engine: Engine, table_name: str, base_table_only: bool = False) -> bool:
    """Check if a table exists in the database.

    Args:
        engine: SQLAlchemy engine to use for the query.
        table_name: Name of the table to check.
        base_table_only: If True, only match BASE TABLE (excludes views).
                        Default is False for backwards compatibility.

    Returns:
        True if the table exists, False otherwise.
    """
    if base_table_only:
        query = text("""
            SELECT 1 FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = :table_name AND TABLE_TYPE = 'BASE TABLE'
        """)
    else:
        query = text("""
            SELECT 1 FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = :table_name
        """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"table_name": table_name}).fetchone()
            return result is not None
    except Exception as e:
        logger.debug("Error checking if table %s exists: %s", table_name, e)
        return False


def get_valid_columns(engine: Engine, table_name: str, requested_columns: list[str]) -> list[str]:
    """Validate which columns exist in the table.

    Args:
        engine: SQLAlchemy engine to use for the query.
        table_name: Name of the table to check.
        requested_columns: List of column names to validate.

    Returns:
        List of column names that exist in the table.
    """
    query = text("""
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = :table_name
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"table_name": table_name})
            existing = {row[0] for row in result.fetchall()}
            valid = [c for c in requested_columns if c in existing]
            missing = set(requested_columns) - existing
            if missing:
                logger.debug("Columns not found in %s: %s", table_name, missing)
            return valid
    except Exception as e:
        logger.warning("Failed to validate columns for %s: %s", table_name, e)
        return requested_columns  # Return all, let query fail if invalid


@dataclass
class LoaderConfig:
    """Configuration for a data loader."""

    table_name: str
    timestamp_col: str
    device_col: str | None = None
    columns: list[str] | None = None
    base_table_only: bool = False


class BaseLoader(ABC, Generic[T]):
    """Abstract base class for SQL Server data loaders.

    Provides common functionality:
    - Engine management
    - Table existence checking
    - Column validation
    - Standard error handling

    Subclasses must implement:
    - _create_engine(): Create the SQLAlchemy engine
    - _load_data(): Execute the actual data loading logic
    """

    def __init__(self, config: LoaderConfig):
        self.config = config
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        """Lazily create and cache the engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @abstractmethod
    def _create_engine(self) -> Engine:
        """Create the SQLAlchemy engine for this loader."""
        pass

    @abstractmethod
    def _load_data(self, **kwargs) -> T:
        """Execute the data loading logic."""
        pass

    def table_exists(self) -> bool:
        """Check if the configured table exists."""
        return table_exists(
            self.engine, self.config.table_name, base_table_only=self.config.base_table_only
        )

    def get_valid_columns(self) -> list[str]:
        """Get the list of valid columns for the configured table."""
        if not self.config.columns:
            return []
        return get_valid_columns(self.engine, self.config.table_name, self.config.columns)

    def load(self, **kwargs) -> T | None:
        """Load data with standard error handling.

        Returns None if the table doesn't exist or an error occurs.
        """
        if not self.table_exists():
            logger.warning("Table %s not found", self.config.table_name)
            return None

        try:
            return self._load_data(**kwargs)
        except Exception as e:
            logger.error("Error loading data from %s: %s", self.config.table_name, e)
            return None
