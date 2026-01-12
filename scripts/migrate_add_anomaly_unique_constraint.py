#!/usr/bin/env python3
"""
Migration script: Add unique constraint to anomaly_results table.

This prevents duplicate anomalies from being created when re-scoring
the same data, which caused anomaly count inflation on static/backup databases.

Run with: python scripts/migrate_add_anomaly_unique_constraint.py

The script will:
1. Check if the unique constraint already exists
2. Handle duplicate records by keeping the most recent one
3. Add the unique constraint

This is safe to run multiple times (idempotent).
"""
from __future__ import annotations

import logging
import os
import sys

from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get the backend database URL from environment or settings."""
    # Try environment variable first
    db_url = os.getenv("BACKEND_DATABASE_URL")
    if db_url:
        return db_url

    # Try loading from settings
    try:
        from device_anomaly.config.settings import get_settings
        settings = get_settings()
        return settings.backend_db.url
    except Exception as e:
        logger.warning(f"Could not load settings: {e}")

    # Default for local development
    return "postgresql://postgres:postgres@localhost:5432/anomaly_detection"


def check_constraint_exists(conn, constraint_name: str) -> bool:
    """Check if a constraint already exists."""
    result = conn.execute(
        text("""
            SELECT 1 FROM pg_indexes
            WHERE indexname = :constraint_name
        """),
        {"constraint_name": constraint_name}
    ).fetchone()
    return result is not None


def count_duplicates(conn) -> int:
    """Count the number of duplicate anomaly records."""
    result = conn.execute(
        text("""
            SELECT COUNT(*) as dup_count
            FROM (
                SELECT tenant_id, device_id, timestamp, COUNT(*) as cnt
                FROM anomaly_results
                GROUP BY tenant_id, device_id, timestamp
                HAVING COUNT(*) > 1
            ) duplicates
        """)
    ).fetchone()
    return result[0] if result else 0


def remove_duplicates(conn) -> int:
    """Remove duplicate anomaly records, keeping the most recent one (by updated_at)."""
    # First, identify duplicates and which ones to keep
    # We keep the record with the highest id (most recent insertion) or updated_at
    result = conn.execute(
        text("""
            DELETE FROM anomaly_results
            WHERE id IN (
                SELECT id FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY tenant_id, device_id, timestamp
                               ORDER BY updated_at DESC, id DESC
                           ) as rn
                    FROM anomaly_results
                ) ranked
                WHERE rn > 1
            )
        """)
    )
    return result.rowcount


def add_unique_constraint(conn, constraint_name: str) -> bool:
    """Add the unique constraint to anomaly_results."""
    try:
        conn.execute(
            text(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {constraint_name}
                ON anomaly_results (tenant_id, device_id, timestamp)
            """)
        )
        return True
    except Exception as e:
        logger.error(f"Failed to create constraint: {e}")
        return False


def migrate():
    """Run the migration."""
    constraint_name = "idx_anomaly_unique_device_timestamp"

    logger.info("=" * 60)
    logger.info("Migration: Add unique constraint to anomaly_results")
    logger.info("=" * 60)

    db_url = get_database_url()
    # Mask password in log output
    safe_url = db_url.replace(db_url.split("@")[0].split(":")[-1], "***") if "@" in db_url else db_url
    logger.info(f"Database: {safe_url}")

    engine = create_engine(db_url)

    with engine.connect() as conn:
        # Check if table exists
        result = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'anomaly_results'
                )
            """)
        ).fetchone()

        if not result or not result[0]:
            logger.info("Table 'anomaly_results' does not exist yet. Skipping migration.")
            logger.info("The constraint will be created automatically when the table is created.")
            return

        # Check if constraint already exists
        if check_constraint_exists(conn, constraint_name):
            logger.info(f"Constraint '{constraint_name}' already exists. Nothing to do.")
            return

        # Check for duplicates
        dup_count = count_duplicates(conn)
        logger.info(f"Found {dup_count} groups of duplicate records.")

        if dup_count > 0:
            logger.info("Removing duplicates (keeping most recent record per device/timestamp)...")
            removed = remove_duplicates(conn)
            logger.info(f"Removed {removed} duplicate records.")
            conn.commit()

        # Add the unique constraint
        logger.info(f"Adding unique constraint: {constraint_name}")
        if add_unique_constraint(conn, constraint_name):
            conn.commit()
            logger.info("Constraint added successfully!")
        else:
            logger.error("Failed to add constraint.")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Migration completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    migrate()
