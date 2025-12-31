"""
Unified Device Dataset Loader - Combines DW Telemetry + MC Inventory.

This module provides a unified interface for loading comprehensive device data
by merging XSight Data Warehouse telemetry with MobiControl device inventory.

The result is a rich dataset with:
- Daily telemetry metrics (battery, app usage, data, RF/signal)
- Device metadata (hardware, OS, firmware, security status)
- Derived connectivity features
- Labels pivoted into columns
- Security composite scores
- Temporal context features

For multi-tenant training, use `load_multi_source_training_data()` to aggregate
data from multiple customer databases while keeping runtime isolated.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import pandas as pd

from device_anomaly.config.settings import get_settings, TrainingDataSource
from device_anomaly.data_access.dw_loader import (
    load_device_daily_telemetry,
    load_device_telemetry_with_fallback,
)
from device_anomaly.data_access.mc_loader import (
    load_mc_device_inventory_snapshot,
    load_mc_inventory_with_fallback,
)

LOGGER = logging.getLogger(__name__)


def _latest_mc_snapshot(mc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the MC snapshot to one row per DeviceId, keeping the latest
    LastCheckInTime to avoid row multiplication when joining to DW facts.
    """
    if mc_df.empty:
        return mc_df

    mc_df = mc_df.sort_values("LastCheckInTime")
    deduped = mc_df.drop_duplicates(subset=["DeviceId"], keep="last").copy()

    if {"LabelName", "LabelValue"} <= set(mc_df.columns):
        label_pivot = (
            mc_df.dropna(subset=["LabelName", "LabelValue"])
            .pivot_table(
                index="DeviceId",
                columns="LabelName",
                values="LabelValue",
                aggfunc="last",
            )
            .reset_index()
        )
        deduped = deduped.merge(label_pivot, on="DeviceId", how="left")

    # Normalize a few semantic columns that downstream code expects
    if "OEMVersion" in deduped.columns:
        deduped["FirmwareVersion"] = deduped["OEMVersion"]

    if "OSVersion" in deduped.columns:
        deduped["OsVersionName"] = deduped["OSVersion"]

    if "Model" in deduped.columns:
        deduped["ModelName"] = deduped["Model"]

    for col, code_col in [
        ("FirmwareVersion", "FirmwareVersionCode"),
        ("OsVersionName", "OsVersionCode"),
        ("ModelName", "ModelCode"),
        ("Manufacturer", "ManufacturerCode"),
        ("Carrier", "CarrierCode"),
    ]:
        if col in deduped.columns:
            deduped[code_col] = pd.factorize(deduped[col].astype(str))[0]

    return deduped


def _add_disconnect_flags(mc_df: pd.DataFrame, end_dt: Optional[str]) -> pd.DataFrame:
    """Add disconnect recency indicators based on LastDisconnTime."""
    if mc_df.empty or "LastDisconnTime" not in mc_df.columns or end_dt is None:
        return mc_df

    mc_df = mc_df.copy()
    try:
        end_ts = pd.to_datetime(end_dt)
    except Exception:
        end_ts = datetime.now(timezone.utc)

    mc_df["LastDisconnTime"] = pd.to_datetime(mc_df["LastDisconnTime"])
    mc_df["DisconnectWithinWindow"] = (
        (end_ts - mc_df["LastDisconnTime"]).dt.total_seconds() <= 7 * 24 * 3600
    )
    mc_df["DisconnectRecencyHours"] = (
        (end_ts - mc_df["LastDisconnTime"]).dt.total_seconds() / 3600.0
    )
    return mc_df


def _add_security_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite security features from MC security indicators.
    """
    if df.empty:
        return df

    df = df.copy()

    # Security indicators with their weights
    security_cols = [
        ("HasPasscode", 1.0),
        ("IsEncrypted", 1.0),
        ("IsAndroidSafetynetAttestationPassed", 1.0),
        ("IsSupervised", 0.5),  # iOS
        ("FileVaultEnabled", 1.0),  # Mac
        ("IsSystemIntegrityProtectionEnabled", 1.0),  # Mac
        ("IsRooted", -1.0),  # Negative indicator
        ("IsJailbroken", -1.0),
        ("IsDeveloperModeEnabled", -0.5),
        ("IsUSBDebuggingEnabled", -0.5),
        ("KnoxAttestationStatus", 0.5),  # Samsung
        ("TrustStatus", 0.5),
        ("CompromisedStatus", -1.0),
    ]

    score = pd.Series(0.0, index=df.index)
    max_score = 0.0

    for col, weight in security_cols:
        if col in df.columns:
            val = df[col].fillna(0)
            if val.dtype == "object":
                val = val.map(lambda x: 1 if str(x).lower() in ("true", "1", "yes", "passed") else 0)
            else:
                val = val.astype(float)
            score += val * weight
            max_score += abs(weight)

    if max_score > 0:
        df["CompositeSecurityScore"] = ((score + max_score) / (2 * max_score)).clip(0, 1)

    # Binary compliance indicator
    if "ComplianceState" in df.columns:
        df["IsCompliant"] = df["ComplianceState"].fillna("").str.lower().isin(
            ["compliant", "true", "1", "yes"]
        ).astype(int)

    return df


def _add_storage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute storage utilization features from MC inventory.
    """
    if df.empty:
        return df

    df = df.copy()

    # Storage utilization
    if all(c in df.columns for c in ["TotalStorage", "AvailableStorage"]):
        total = df["TotalStorage"].fillna(0) + 1
        avail = df["AvailableStorage"].fillna(0)
        df["StorageUtilizationPct"] = ((total - avail) / total * 100).clip(0, 100)

    # RAM utilization
    if all(c in df.columns for c in ["TotalRAM", "AvailableRAM"]):
        total = df["TotalRAM"].fillna(0) + 1
        avail = df["AvailableRAM"].fillna(0)
        df["RAMUtilizationPct"] = ((total - avail) / total * 100).clip(0, 100)

    # Internal storage utilization
    if all(c in df.columns for c in ["TotalInternalStorage", "AvailableInternalStorage"]):
        total = df["TotalInternalStorage"].fillna(0) + 1
        avail = df["AvailableInternalStorage"].fillna(0)
        df["InternalStorageUtilizationPct"] = ((total - avail) / total * 100).clip(0, 100)

    return df


def _add_temporal_features(df: pd.DataFrame, end_date: str) -> pd.DataFrame:
    """
    Add temporal context features based on device timestamps.
    """
    if df.empty:
        return df

    df = df.copy()

    try:
        end_ts = pd.to_datetime(end_date)
    except Exception:
        end_ts = datetime.now(timezone.utc)

    # Device age (days since enrollment)
    if "EnrollmentTime" in df.columns:
        enroll = pd.to_datetime(df["EnrollmentTime"], errors="coerce")
        df["DeviceAgeDays"] = (end_ts - enroll).dt.days.clip(lower=0)

    # Days since last check-in
    if "LastCheckInTime" in df.columns:
        last_checkin = pd.to_datetime(df["LastCheckInTime"], errors="coerce")
        df["DaysSinceLastCheckin"] = (end_ts - last_checkin).dt.days.abs()

    # Days since last policy update
    if "LastPolicyUpdateTime" in df.columns:
        last_policy = pd.to_datetime(df["LastPolicyUpdateTime"], errors="coerce")
        df["DaysSinceLastPolicyUpdate"] = (end_ts - last_policy).dt.days.abs()

    return df


def _normalize_download_upload(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Download/Upload column names for consistency.
    """
    if df.empty:
        return df

    df = df.copy()

    # Create standard aliases
    if "TotalDownload" in df.columns and "Download" not in df.columns:
        df["Download"] = df["TotalDownload"]

    if "TotalUpload" in df.columns and "Upload" not in df.columns:
        df["Upload"] = df["TotalUpload"]

    return df


def load_unified_device_dataset(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
    row_limit: int | None = 1_000_000,
    include_mc_labels: bool = True,
    use_fallback: bool = True,
    enrich_features: bool = True,
) -> pd.DataFrame:
    """
    Pull DW daily telemetry and enrich with MobiControl inventory metadata so the
    model can correlate anomalies with firmware/OS/model/site information.

    This is the primary entry point for loading training data. It:
    1. Loads comprehensive telemetry from XSight DW
    2. Loads device metadata from MobiControl
    3. Joins on DeviceId with MC snapshot closest to DW date
    4. Pivots labels into columns
    5. Adds derived connectivity, security, and temporal features
    6. Normalizes columns for ML compatibility

    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        device_ids: Optional list of device IDs to filter
        row_limit: Maximum rows to load (applied to DW query)
        include_mc_labels: Whether to include MC labels (pivoted)
        use_fallback: Whether to use fallback queries if comprehensive ones fail
        enrich_features: Whether to compute additional derived features

    Returns a dataframe keyed by (DeviceId, Timestamp) with added columns:
      - FirmwareVersion / OEMVersion
      - OSVersion (name) and ModelName/Manufacturer
      - Agent/Mode/Online flags for context
      - Disconnect recency indicators
      - Security composite scores
      - Storage/RAM utilization
      - Temporal context (device age, days since check-in)
      - Label_* columns if MC labels are enabled (e.g., StoreId, SiteId)
    """
    LOGGER.info(f"Loading unified dataset: {start_date} to {end_date}")

    # ==========================================================================
    # Step 1: Load DW telemetry
    # ==========================================================================
    LOGGER.info("Loading XSight DW telemetry...")

    if use_fallback:
        dw_df = load_device_telemetry_with_fallback(
            start_date=start_date,
            end_date=end_date,
            device_ids=device_ids,
            limit=row_limit,
        )
    else:
        dw_df = load_device_daily_telemetry(
            start_date=start_date,
            end_date=end_date,
            device_ids=device_ids,
            limit=row_limit,
        )

    LOGGER.info(f"DW telemetry: {len(dw_df):,} rows, {len(dw_df.columns)} columns")

    if dw_df.empty:
        LOGGER.warning("DW loader returned no rows for %s - %s", start_date, end_date)
        return dw_df

    # Normalize Download/Upload
    dw_df = _normalize_download_upload(dw_df)

    # ==========================================================================
    # Step 2: Load MC inventory
    # ==========================================================================
    settings = get_settings()
    mc_settings = settings.mc
    if not (mc_settings.host and mc_settings.user and mc_settings.password):
        LOGGER.info("MC settings missing; skipping MC enrichment.")
        return dw_df

    LOGGER.info("Loading MobiControl inventory...")

    # Expand date range for MC to capture device snapshots
    mc_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    mc_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        if use_fallback:
            mc_df = load_mc_inventory_with_fallback(
                start_dt=mc_start,
                end_dt=mc_end,
                device_ids=device_ids,
                include_labels=include_mc_labels,
                limit=None,
            )
        else:
            mc_df = load_mc_device_inventory_snapshot(
                start_dt=mc_start,
                end_dt=mc_end,
                device_ids=device_ids,
                include_labels=include_mc_labels,
                limit=None,
            )
    except Exception as exc:
        LOGGER.warning("Failed to load MC inventory data; skipping enrichment. Error: %s", exc)
        return dw_df

    LOGGER.info(f"MC inventory: {len(mc_df):,} rows, {len(mc_df.columns)} columns")

    # ==========================================================================
    # Step 3: Process MC data
    # ==========================================================================
    mc_df = _add_disconnect_flags(mc_df, end_date)
    mc_latest = _latest_mc_snapshot(mc_df)

    if mc_latest.empty:
        LOGGER.info("No MC metadata found; returning DW telemetry only.")
        return dw_df

    LOGGER.info(f"MC after processing: {len(mc_latest):,} unique devices")

    # ==========================================================================
    # Step 4: Merge DW + MC
    # ==========================================================================
    # Select MC columns to merge (avoid duplicating DW columns)
    mc_merge_cols = ["DeviceId"]
    for col in mc_latest.columns:
        if col == "DeviceId":
            continue
        # Skip columns that already exist in DW (except some we want to override)
        if col in dw_df.columns and col not in (
            "FirmwareVersion", "OsVersionName", "ModelName",
            "Manufacturer", "FirmwareVersionCode", "OsVersionCode", "ModelCode"
        ):
            continue
        mc_merge_cols.append(col)

    enriched = dw_df.merge(
        mc_latest[mc_merge_cols],
        on="DeviceId",
        how="left",
        suffixes=("", "_mc"),
    )

    LOGGER.info(f"Merged dataset: {len(enriched):,} rows, {len(enriched.columns)} columns")

    # ==========================================================================
    # Step 5: Add derived features
    # ==========================================================================
    # Derived disconnect metrics
    if "TotalDropCnt" in enriched.columns:
        enriched["DisconnectCount"] = enriched["TotalDropCnt"]
    if "DisconnectWithinWindow" in enriched.columns:
        enriched["DisconnectFlag"] = enriched["DisconnectWithinWindow"].fillna(False).astype(int)

    # Normalize RSSI naming for downstream usage
    if "AvgSignalStrength" in enriched.columns and "Rssi" not in enriched.columns:
        enriched["Rssi"] = enriched["AvgSignalStrength"]

    # Factorize string metadata after join to align encodings with telemetry rows
    for col, code_col in [
        ("FirmwareVersion", "FirmwareVersionCode"),
        ("OsVersionName", "OsVersionCode"),
        ("ModelName", "ModelCode"),
    ]:
        if col in enriched.columns and code_col not in enriched.columns:
            enriched[code_col] = pd.factorize(enriched[col].astype(str))[0]

    # Additional enrichment
    if enrich_features:
        enriched = _add_security_features(enriched)
        enriched = _add_storage_features(enriched)
        enriched = _add_temporal_features(enriched, end_date)

    LOGGER.info(f"Final unified dataset: {len(enriched):,} rows, {len(enriched.columns)} columns")

    return enriched


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_training_data(
    start_date: str,
    end_date: str,
    row_limit: int = 500_000,
) -> pd.DataFrame:
    """
    Convenience function for loading training data.

    Wraps load_unified_device_dataset with common defaults.
    """
    return load_unified_device_dataset(
        start_date=start_date,
        end_date=end_date,
        include_mc_labels=True,
        row_limit=row_limit,
        use_fallback=True,
        enrich_features=True,
    )


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of the unified dataset.
    """
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "unique_devices": df["DeviceId"].nunique() if "DeviceId" in df.columns else 0,
    }

    if "Timestamp" in df.columns:
        summary["date_range"] = {
            "start": str(df["Timestamp"].min()),
            "end": str(df["Timestamp"].max()),
        }

    # Column categories
    summary["columns_by_type"] = {
        "numeric": len(df.select_dtypes(include=["number"]).columns),
        "categorical": len(df.select_dtypes(include=["object", "category"]).columns),
        "datetime": len(df.select_dtypes(include=["datetime"]).columns),
        "boolean": len(df.select_dtypes(include=["bool"]).columns),
    }

    # Missing value summary
    missing = df.isnull().sum()
    summary["columns_with_missing"] = int((missing > 0).sum())
    summary["total_missing_values"] = int(missing.sum())

    return summary


# =============================================================================
# MULTI-SOURCE TRAINING DATA LOADER
# =============================================================================

def _load_from_source(
    source: TrainingDataSource,
    start_date: str,
    end_date: str,
    row_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load unified dataset from a specific training data source.

    This temporarily overrides the database settings to load from the
    specified customer database, then restores the original settings.

    Args:
        source: Training data source configuration
        start_date: Start date for data
        end_date: End date for data
        row_limit: Maximum rows to load per source

    Returns:
        DataFrame with unified device data from the source
    """
    import os
    from device_anomaly.config.settings import reset_settings

    # Save original environment
    original_env = {
        "DW_DB_NAME": os.environ.get("DW_DB_NAME"),
        "MC_DB_NAME": os.environ.get("MC_DB_NAME"),
        "DW_DB_HOST": os.environ.get("DW_DB_HOST"),
        "MC_DB_HOST": os.environ.get("MC_DB_HOST"),
        "DW_DB_USER": os.environ.get("DW_DB_USER"),
        "MC_DB_USER": os.environ.get("MC_DB_USER"),
        "DW_DB_PASS": os.environ.get("DW_DB_PASS"),
        "MC_DB_PASS": os.environ.get("MC_DB_PASS"),
        "DW_DB_PORT": os.environ.get("DW_DB_PORT"),
        "MC_DB_PORT": os.environ.get("MC_DB_PORT"),
    }

    try:
        # Override environment for this source
        os.environ["DW_DB_NAME"] = source.xsight_db
        os.environ["MC_DB_NAME"] = source.mc_db

        if source.host:
            os.environ["DW_DB_HOST"] = source.host
            os.environ["MC_DB_HOST"] = source.host
        if source.port:
            os.environ["DW_DB_PORT"] = str(source.port)
            os.environ["MC_DB_PORT"] = str(source.port)
        if source.user:
            os.environ["DW_DB_USER"] = source.user
            os.environ["MC_DB_USER"] = source.user
        if source.password:
            os.environ["DW_DB_PASS"] = source.password
            os.environ["MC_DB_PASS"] = source.password

        # Reset cached settings to pick up new environment
        reset_settings()

        LOGGER.info(f"Loading data from source: {source.name}")
        LOGGER.info(f"  XSight DB: {source.xsight_db}")
        LOGGER.info(f"  MobiControl DB: {source.mc_db}")

        # Load unified dataset from this source
        df = load_unified_device_dataset(
            start_date=start_date,
            end_date=end_date,
            row_limit=row_limit,
            include_mc_labels=True,
            use_fallback=True,
            enrich_features=True,
        )

        # Add source identifier column
        if not df.empty:
            df["_source"] = source.name

        LOGGER.info(f"Loaded {len(df):,} rows from {source.name}")
        return df

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        # Reset settings again to restore original configuration
        reset_settings()


def load_multi_source_training_data(
    start_date: str,
    end_date: str,
    sources: Optional[List[TrainingDataSource]] = None,
    row_limit_per_source: Optional[int] = 500_000,
) -> pd.DataFrame:
    """
    Load and combine training data from multiple customer databases.

    This function is the key to multi-tenant training: it aggregates data from
    all configured training sources while keeping the runtime application
    isolated to a single customer's data.

    The model learns patterns from diverse customer environments, making it
    more robust and generalizable.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        sources: List of training data sources. If None, uses TRAINING_DATA_SOURCES
                 from settings/environment.
        row_limit_per_source: Maximum rows to load from each source.
                              Total rows = sources * row_limit_per_source

    Returns:
        Combined DataFrame with data from all sources.
        Includes '_source' column identifying the origin of each row.

    Example:
        # Load from all configured sources
        df = load_multi_source_training_data("2024-01-01", "2024-12-31")

        # Load from specific sources
        sources = [
            TrainingDataSource(name="BENELUX", xsight_db="XSight_BENELUX", mc_db="MobiControl_BENELUX"),
            TrainingDataSource(name="PIBLIC", xsight_db="XSight_PIBLIC", mc_db="MobiControl_PIBLIC"),
        ]
        df = load_multi_source_training_data("2024-01-01", "2024-12-31", sources=sources)
    """
    if sources is None:
        settings = get_settings()
        sources = settings.training_data_sources

    if not sources:
        LOGGER.warning(
            "No training data sources configured. "
            "Set TRAINING_DATA_SOURCES environment variable or pass sources parameter. "
            "Falling back to single-source loading from current DW/MC settings."
        )
        return load_unified_device_dataset(
            start_date=start_date,
            end_date=end_date,
            row_limit=row_limit_per_source,
            include_mc_labels=True,
            use_fallback=True,
            enrich_features=True,
        )

    LOGGER.info(f"=== Multi-Source Training Data Load ===")
    LOGGER.info(f"Date range: {start_date} to {end_date}")
    LOGGER.info(f"Sources: {[s.name for s in sources]}")

    all_dfs = []
    for source in sources:
        try:
            df = _load_from_source(
                source=source,
                start_date=start_date,
                end_date=end_date,
                row_limit=row_limit_per_source,
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            LOGGER.error(f"Failed to load from source {source.name}: {e}")
            # Continue with other sources

    if not all_dfs:
        LOGGER.warning("No data loaded from any source!")
        return pd.DataFrame()

    # Combine all sources
    combined = pd.concat(all_dfs, ignore_index=True)

    LOGGER.info(f"=== Multi-Source Load Complete ===")
    LOGGER.info(f"Total rows: {len(combined):,}")
    LOGGER.info(f"Unique devices: {combined['DeviceId'].nunique():,}")
    LOGGER.info(f"Sources represented: {combined['_source'].nunique()}")

    return combined


def get_multi_source_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for multi-source training data.

    Args:
        df: DataFrame with '_source' column from load_multi_source_training_data

    Returns:
        Dictionary with per-source and aggregate statistics
    """
    if "_source" not in df.columns:
        return get_dataset_summary(df)

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "unique_devices": df["DeviceId"].nunique() if "DeviceId" in df.columns else 0,
        "sources": {},
    }

    # Per-source statistics
    for source_name in df["_source"].unique():
        source_df = df[df["_source"] == source_name]
        summary["sources"][source_name] = {
            "rows": len(source_df),
            "unique_devices": source_df["DeviceId"].nunique() if "DeviceId" in source_df.columns else 0,
            "percentage": len(source_df) / len(df) * 100 if len(df) > 0 else 0,
        }
        if "Timestamp" in source_df.columns:
            summary["sources"][source_name]["date_range"] = {
                "start": str(source_df["Timestamp"].min()),
                "end": str(source_df["Timestamp"].max()),
            }

    if "Timestamp" in df.columns:
        summary["date_range"] = {
            "start": str(df["Timestamp"].min()),
            "end": str(df["Timestamp"].max()),
        }

    return summary
