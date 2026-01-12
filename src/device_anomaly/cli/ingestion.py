"""
CLI entrypoint for batch ingestion.

Supports dry-run mode with mock data for safe verification.
"""
from __future__ import annotations

import argparse
import logging

from device_anomaly.config.logging_config import setup_logging
from device_anomaly.services.ingestion_pipeline import run_ingestion_batch

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run batch ingestion for telemetry tables.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock data and skip database writes.",
    )
    parser.add_argument(
        "--xsight-table",
        action="append",
        dest="xsight_tables",
        help="XSight table name to ingest (repeatable).",
    )
    parser.add_argument(
        "--mc-table",
        action="append",
        dest="mc_tables",
        help="MobiControl table name to ingest (repeatable).",
    )
    parser.add_argument(
        "--max-weight",
        type=int,
        default=None,
        help="Override max ingestion weight (default: config).",
    )
    args = parser.parse_args()

    result = run_ingestion_batch(
        xsight_tables=args.xsight_tables,
        mc_tables=args.mc_tables,
        dry_run=args.dry_run,
        max_weight=args.max_weight,
    )

    logger.info(
        "Ingestion run complete: %s tables, fetched=%s, inserted=%s, deduped=%s",
        len(result.tasks),
        result.total_rows_fetched,
        result.total_rows_inserted,
        result.total_rows_deduped,
    )


if __name__ == "__main__":
    main()
