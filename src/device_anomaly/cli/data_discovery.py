"""
CLI command for data discovery and profiling.

Analyzes SQL Server telemetry tables to understand data distributions,
patterns, and baselines for ML model training.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from device_anomaly.config.logging_config import setup_logging
from device_anomaly.data_access.data_profiler import (
    DW_TELEMETRY_TABLES,
    analyze_temporal_patterns,
    generate_profile_report,
    get_available_metrics,
    profile_dw_tables,
    profile_mc_tables,
)
from device_anomaly.data_access.unified_loader import load_unified_device_dataset

logger = logging.getLogger(__name__)


def run_data_discovery(
    start_date: str,
    end_date: str,
    output_dir: str = "artifacts/data_discovery",
    include_mc: bool = True,
    sample_limit: int = 100_000,
    analyze_patterns: bool = True,
    load_sample_data: bool = True,
) -> dict:
    """
    Run comprehensive data discovery on SQL Server databases.

    Args:
        start_date: Start date for data range (YYYY-MM-DD)
        end_date: End date for data range (YYYY-MM-DD)
        output_dir: Directory to save reports
        include_mc: Include MobiControl database in profiling
        sample_limit: Max rows to sample per table
        analyze_patterns: Whether to analyze temporal patterns
        load_sample_data: Whether to load actual data for pattern analysis

    Returns:
        Dictionary with discovery results summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "discovery_started": datetime.utcnow().isoformat(),
        "date_range": {"start": start_date, "end": end_date},
        "dw_profiles": {},
        "mc_profiles": {},
        "temporal_patterns": {},
        "available_metrics": [],
    }

    # 1. Profile DW tables
    logger.info("=== Profiling XSight Database Tables ===")
    try:
        dw_profiles = profile_dw_tables(
            start_date=start_date,
            end_date=end_date,
            sample_limit=sample_limit,
        )
        results["dw_profiles"] = {name: p.to_dict() for name, p in dw_profiles.items()}

        # Generate DW report
        dw_report_path = output_path / "dw_profile_report.json"
        generate_profile_report(dw_profiles, dw_report_path)
        logger.info(f"DW profile report saved to {dw_report_path}")

        # Extract available metrics
        results["available_metrics"] = get_available_metrics(dw_profiles)
        logger.info(
            f"Found {len(results['available_metrics'])} numeric metrics across {len(dw_profiles)} tables"
        )

    except Exception as e:
        logger.error(f"Failed to profile DW tables: {e}")
        results["dw_error"] = str(e)

    # 2. Profile MC tables (optional)
    if include_mc:
        logger.info("=== Profiling MobiControl Tables ===")
        try:
            mc_profiles = profile_mc_tables(sample_limit=sample_limit)
            results["mc_profiles"] = {name: p.to_dict() for name, p in mc_profiles.items()}

            mc_report_path = output_path / "mc_profile_report.json"
            generate_profile_report(mc_profiles, mc_report_path)
            logger.info(f"MC profile report saved to {mc_report_path}")

        except Exception as e:
            logger.warning(f"Failed to profile MC tables (optional): {e}")
            results["mc_error"] = str(e)

    # 3. Analyze temporal patterns (requires loading sample data)
    if analyze_patterns and load_sample_data:
        logger.info("=== Analyzing Temporal Patterns ===")
        try:
            # Load a sample of unified data for pattern analysis
            df = load_unified_device_dataset(
                start_date=start_date,
                end_date=end_date,
                row_limit=sample_limit,
                include_mc_labels=False,
            )

            if not df.empty:
                # Identify numeric metric columns
                metric_cols = [
                    col
                    for col in df.columns
                    if df[col].dtype in ["float64", "int64", "float32", "int32"]
                    and col not in ["DeviceId", "ModelId", "ManufacturerId", "OsVersionId"]
                ]

                patterns = analyze_temporal_patterns(df, metric_cols)
                results["temporal_patterns"] = {name: p.to_dict() for name, p in patterns.items()}

                # Save patterns separately
                patterns_path = output_path / "temporal_patterns.json"
                with open(patterns_path, "w") as f:
                    json.dump(results["temporal_patterns"], f, indent=2)
                logger.info(f"Temporal patterns saved to {patterns_path}")

        except Exception as e:
            logger.warning(f"Failed to analyze temporal patterns: {e}")
            results["patterns_error"] = str(e)

    # 4. Generate summary
    results["discovery_completed"] = datetime.utcnow().isoformat()

    summary = {
        "total_tables_profiled": len(results.get("dw_profiles", {}))
        + len(results.get("mc_profiles", {})),
        "total_rows": sum(p.get("row_count", 0) for p in results.get("dw_profiles", {}).values()),
        "total_devices": max(
            (p.get("device_count", 0) for p in results.get("dw_profiles", {}).values()), default=0
        ),
        "metrics_discovered": len(results.get("available_metrics", [])),
        "patterns_analyzed": len(results.get("temporal_patterns", {})),
    }
    results["summary"] = summary

    # Save complete results
    results_path = output_path / "discovery_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Complete discovery results saved to {results_path}")

    # Print summary
    logger.info("=== Discovery Summary ===")
    logger.info(f"Tables profiled: {summary['total_tables_profiled']}")
    logger.info(f"Total rows: {summary['total_rows']:,}")
    logger.info(f"Unique devices: {summary['total_devices']:,}")
    logger.info(f"Metrics discovered: {summary['metrics_discovered']}")
    logger.info(f"Patterns analyzed: {summary['patterns_analyzed']}")

    return results


def main() -> None:
    """CLI entry point for data discovery."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run data discovery on SQL Server telemetry databases."
    )
    parser.add_argument(
        "-s",
        "--start-date",
        help="Start date (YYYY-MM-DD). Defaults to 30 days ago.",
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "-e",
        "--end-date",
        help="End date (YYYY-MM-DD). Defaults to today.",
        default=datetime.now().strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for reports",
        default="artifacts/data_discovery",
    )
    parser.add_argument(
        "--no-mc",
        action="store_true",
        help="Skip MobiControl database profiling",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100_000,
        help="Max rows to sample per table for statistics",
    )
    parser.add_argument(
        "--no-patterns",
        action="store_true",
        help="Skip temporal pattern analysis",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help=f"Specific tables to profile (default: {DW_TELEMETRY_TABLES})",
    )

    args = parser.parse_args()

    logger.info("Starting data discovery...")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")

    run_data_discovery(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        include_mc=not args.no_mc,
        sample_limit=args.sample_limit,
        analyze_patterns=not args.no_patterns,
    )

    logger.info("Data discovery complete!")


if __name__ == "__main__":
    main()
