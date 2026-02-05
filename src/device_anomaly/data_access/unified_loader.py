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
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

import pandas as pd

from device_anomaly.config.settings import TrainingDataSource, get_settings
from device_anomaly.data_access.dw_loader import (
    load_device_daily_telemetry,
    load_device_telemetry_with_fallback,
)
from device_anomaly.data_access.mc_loader import (
    load_mc_device_inventory_snapshot,
    load_mc_inventory_with_fallback,
)

LOGGER = logging.getLogger(__name__)


def _temporal_aware_mc_join(
    dw_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    max_staleness_hours: int = 24,
) -> pd.DataFrame:
    """
    Join DW telemetry with temporally-closest MC snapshot.

    For each DW row (DeviceId, Timestamp), finds the MC snapshot
    closest in time (within max_staleness_hours) and joins it.

    This prevents joining stale MC data with fresh telemetry, which can
    cause misleading correlations. Rows with stale MC data are flagged
    with _mc_data_stale=True.

    Args:
        dw_df: XSight telemetry with Timestamp column
        mc_df: MobiControl inventory with LastCheckInTime column
        max_staleness_hours: Maximum hours between DW and MC timestamps

    Returns:
        Joined DataFrame with _mc_staleness_hours and _mc_data_stale columns
    """
    if dw_df.empty or mc_df.empty:
        return dw_df

    # Ensure datetime types
    dw_df = dw_df.copy()
    mc_df = mc_df.copy()

    # Find timestamp columns
    dw_time_col = "Timestamp" if "Timestamp" in dw_df.columns else "CollectedDate"
    mc_time_col = "LastCheckInTime"

    if dw_time_col not in dw_df.columns or mc_time_col not in mc_df.columns:
        LOGGER.warning(
            "Missing timestamp columns for temporal-aware join, falling back to static join"
        )
        return dw_df

    dw_df[dw_time_col] = pd.to_datetime(dw_df[dw_time_col], errors="coerce")
    mc_df[mc_time_col] = pd.to_datetime(mc_df[mc_time_col], errors="coerce")

    # Get unique devices
    devices = dw_df["DeviceId"].unique()

    # Build merged result
    merged_rows = []

    for device_id in devices:
        device_dw = dw_df[dw_df["DeviceId"] == device_id]
        device_mc = mc_df[mc_df["DeviceId"] == device_id]

        if device_mc.empty:
            # No MC data - mark as missing
            for _, dw_row in device_dw.iterrows():
                row_dict = dw_row.to_dict()
                row_dict["_mc_staleness_hours"] = float("inf")
                row_dict["_mc_data_stale"] = True
                row_dict["_mc_data_missing"] = True
                merged_rows.append(row_dict)
            continue

        # Sort MC by time
        device_mc = device_mc.sort_values(mc_time_col)

        for _, dw_row in device_dw.iterrows():
            dw_time = dw_row[dw_time_col]
            if pd.isna(dw_time):
                row_dict = dw_row.to_dict()
                merged_rows.append(row_dict)
                continue

            # Find closest MC snapshot before this timestamp
            mc_before = device_mc[device_mc[mc_time_col] <= dw_time]

            closest_mc = device_mc.iloc[0] if mc_before.empty else mc_before.iloc[-1]

            # Compute staleness
            mc_time = closest_mc[mc_time_col]
            if pd.isna(mc_time):
                time_gap = float("inf")
            else:
                time_gap = abs((dw_time - mc_time).total_seconds() / 3600)

            # Merge row data
            row_dict = dw_row.to_dict()
            for col in closest_mc.index:
                if col not in row_dict and col != "DeviceId":
                    row_dict[col] = closest_mc[col]

            row_dict["_mc_staleness_hours"] = time_gap
            row_dict["_mc_data_stale"] = time_gap > max_staleness_hours
            row_dict["_mc_data_missing"] = False

            merged_rows.append(row_dict)

    if not merged_rows:
        return dw_df

    result = pd.DataFrame(merged_rows)

    # Log staleness statistics
    stale_count = result["_mc_data_stale"].sum() if "_mc_data_stale" in result.columns else 0
    total_count = len(result)
    LOGGER.info(
        f"Temporal-aware join complete: {stale_count}/{total_count} "
        f"({100 * stale_count / max(1, total_count):.1f}%) rows have stale MC data"
    )

    return result


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


def _add_disconnect_flags(mc_df: pd.DataFrame, end_dt: str | None) -> pd.DataFrame:
    """Add disconnect recency indicators based on LastDisconnTime.

    IMPORTANT: Uses the row's LastCheckInTime as reference when available,
    to prevent feature drift when re-scoring static data. The disconnect
    recency should reflect the state AT THE TIME of the record, not at
    the time of scoring.
    """
    if mc_df.empty or "LastDisconnTime" not in mc_df.columns or end_dt is None:
        return mc_df

    mc_df = mc_df.copy()

    # Use LastCheckInTime as reference when available (more stable for static data)
    # This ensures features don't drift when re-scoring the same data
    if "LastCheckInTime" in mc_df.columns:
        reference_ts = pd.to_datetime(mc_df["LastCheckInTime"], errors="coerce")
        try:
            fallback_ts = pd.to_datetime(end_dt)
        except Exception:
            fallback_ts = datetime.now(UTC)
        reference_ts = reference_ts.fillna(fallback_ts)
    else:
        try:
            reference_ts = pd.to_datetime(end_dt)
        except Exception:
            reference_ts = datetime.now(UTC)

    mc_df["LastDisconnTime"] = pd.to_datetime(mc_df["LastDisconnTime"], errors="coerce")
    mc_df["DisconnectWithinWindow"] = (
        reference_ts - mc_df["LastDisconnTime"]
    ).dt.total_seconds() <= 7 * 24 * 3600
    mc_df["DisconnectRecencyHours"] = (
        reference_ts - mc_df["LastDisconnTime"]
    ).dt.total_seconds() / 3600.0
    # Cap to prevent unbounded values on very old data
    mc_df["DisconnectRecencyHours"] = mc_df["DisconnectRecencyHours"].clip(upper=8760)  # 1 year max
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
                val = val.map(
                    lambda x: 1 if str(x).lower() in ("true", "1", "yes", "passed") else 0
                )
            else:
                val = val.astype(float)
            score += val * weight
            max_score += abs(weight)

    if max_score > 0:
        df["CompositeSecurityScore"] = ((score + max_score) / (2 * max_score)).clip(0, 1)

    # Binary compliance indicator
    if "ComplianceState" in df.columns:
        df["IsCompliant"] = (
            df["ComplianceState"]
            .fillna("")
            .str.lower()
            .isin(["compliant", "true", "1", "yes"])
            .astype(int)
        )

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

    IMPORTANT: For scoring stability on static/backup databases, temporal features
    are computed relative to the ROW's own Timestamp (when available), not the
    scoring run time. This prevents feature drift when re-scoring the same data.

    For example, DaysSinceLastCheckin should reflect how stale the device was
    AT THE TIME of the telemetry record, not at the time of scoring. This ensures
    that re-running scoring on the same data produces the same features.
    """
    if df.empty:
        return df

    df = df.copy()

    # Use per-row reference time (Timestamp column) when available
    # This prevents features from drifting when re-scoring static data
    if "Timestamp" in df.columns:
        reference_ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        # Fall back to end_date for rows with missing timestamps
        try:
            fallback_ts = pd.to_datetime(end_date)
        except Exception:
            fallback_ts = datetime.now(UTC)
        reference_ts = reference_ts.fillna(fallback_ts)
    else:
        # No per-row timestamp - use end_date for all rows
        try:
            reference_ts = pd.to_datetime(end_date)
        except Exception:
            reference_ts = datetime.now(UTC)

    # Device age (days since enrollment)
    if "EnrollmentTime" in df.columns:
        enroll = pd.to_datetime(df["EnrollmentTime"], errors="coerce")
        df["DeviceAgeDays"] = (reference_ts - enroll).dt.days.clip(lower=0)
        # Cap extreme values to prevent unbounded growth
        df["DeviceAgeDays"] = df["DeviceAgeDays"].clip(upper=3650)  # 10 years max

    # Days since last check-in (relative to row's timestamp, not current time)
    if "LastCheckInTime" in df.columns:
        last_checkin = pd.to_datetime(df["LastCheckInTime"], errors="coerce")
        df["DaysSinceLastCheckin"] = (reference_ts - last_checkin).dt.days.abs()
        # Cap to prevent unbounded drift on very old/static data
        df["DaysSinceLastCheckin"] = df["DaysSinceLastCheckin"].clip(upper=365)

    # Days since last policy update
    if "LastPolicyUpdateTime" in df.columns:
        last_policy = pd.to_datetime(df["LastPolicyUpdateTime"], errors="coerce")
        df["DaysSinceLastPolicyUpdate"] = (reference_ts - last_policy).dt.days.abs()
        # Cap to prevent unbounded drift
        df["DaysSinceLastPolicyUpdate"] = df["DaysSinceLastPolicyUpdate"].clip(upper=365)

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

    # Detailed diagnostic logging for MC configuration
    LOGGER.info(
        "MC configuration: host=%s, database=%s, user=%s, password=%s",
        mc_settings.host or "(not set)",
        mc_settings.database or "(not set)",
        mc_settings.user or "(not set)",
        "***" if mc_settings.password else "(not set)",
    )

    if not (mc_settings.host and mc_settings.user and mc_settings.password):
        LOGGER.warning(
            "MC settings incomplete - skipping MC enrichment. "
            "Set MC_DB_HOST, MC_DB_USER, MC_DB_PASS environment variables."
        )
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
        LOGGER.error(
            "Failed to load MC inventory data - skipping enrichment. Error type: %s, Message: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return dw_df

    LOGGER.info(f"MC inventory: {len(mc_df):,} rows, {len(mc_df.columns)} columns")
    if mc_df.empty:
        LOGGER.warning(
            "MC loader returned 0 rows. This may indicate: "
            "1) No devices in MobiControlDB with LastCheckInTime in range [%s, %s], "
            "2) DevInfo table is empty, or "
            "3) Query returned no matching results.",
            mc_start,
            mc_end,
        )

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
            "FirmwareVersion",
            "OsVersionName",
            "ModelName",
            "Manufacturer",
            "FirmwareVersionCode",
            "OsVersionCode",
            "ModelCode",
        ):
            continue
        mc_merge_cols.append(col)

    enriched = dw_df.merge(
        mc_latest[mc_merge_cols],
        on="DeviceId",
        how="left",
        suffixes=("", "_mc"),
    )

    # Log which MC columns were added
    mc_added_cols = [c for c in mc_merge_cols if c != "DeviceId"]
    LOGGER.info(
        "Merged dataset: %d rows, %d columns. MC columns added: %s",
        len(enriched),
        len(enriched.columns),
        mc_added_cols[:10] if len(mc_added_cols) > 10 else mc_added_cols,
    )
    if len(mc_added_cols) > 10:
        LOGGER.debug("Full MC columns list: %s", mc_added_cols)

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
    row_limit: int | None = None,
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
    sources: list[TrainingDataSource] | None = None,
    row_limit_per_source: int | None = 500_000,
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

    LOGGER.info("=== Multi-Source Training Data Load ===")
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

    LOGGER.info("=== Multi-Source Load Complete ===")
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
            "unique_devices": source_df["DeviceId"].nunique()
            if "DeviceId" in source_df.columns
            else 0,
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


# =============================================================================
# EXTENDED DATA LOADING (MC Timeseries + XSight Extended)
# =============================================================================


def _load_mc_location_data(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Load location data from MobiControl DeviceStatLocation.

    Returns GPS coordinates with timestamps for movement analysis.
    """
    try:
        from device_anomaly.data_access.db_connection import create_mc_engine
        from device_anomaly.data_access.mc_timeseries_loader import (
            load_mc_timeseries_incremental,
        )

        engine = create_mc_engine()
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)

        df, _ = load_mc_timeseries_incremental(
            table_name="DeviceStatLocation",
            start_time=start_dt,
            end_time=end_dt,
            device_ids=list(device_ids) if device_ids else None,
            use_watermark=False,
            engine=engine,
        )

        if not df.empty:
            LOGGER.info(f"Loaded {len(df):,} MC location records")
        return df

    except Exception as e:
        LOGGER.warning(f"Failed to load MC location data: {e}")
        return pd.DataFrame()


def _load_mc_system_health_data(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Load system health metrics from MobiControl DeviceStatInt.

    Returns CPU, RAM, storage metrics from time-series data.
    """
    try:
        from device_anomaly.data_access.db_connection import create_mc_engine
        from device_anomaly.data_access.mc_timeseries_loader import (
            load_mc_timeseries_incremental,
        )

        engine = create_mc_engine()
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)

        df, _ = load_mc_timeseries_incremental(
            table_name="DeviceStatInt",
            start_time=start_dt,
            end_time=end_dt,
            device_ids=list(device_ids) if device_ids else None,
            use_watermark=False,
            engine=engine,
        )

        if not df.empty:
            LOGGER.info(f"Loaded {len(df):,} MC system health records")
        return df

    except Exception as e:
        LOGGER.warning(f"Failed to load MC system health data: {e}")
        return pd.DataFrame()


def _load_mc_events_data(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Load event/log data from MobiControl MainLog.

    Returns device events for crash/error pattern analysis.
    """
    try:
        from device_anomaly.data_access.db_connection import create_mc_engine
        from device_anomaly.data_access.mc_timeseries_loader import (
            load_mc_timeseries_incremental,
        )

        engine = create_mc_engine()
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)

        df, _ = load_mc_timeseries_incremental(
            table_name="MainLog",
            start_time=start_dt,
            end_time=end_dt,
            device_ids=list(device_ids) if device_ids else None,
            use_watermark=False,
            engine=engine,
        )

        if not df.empty:
            LOGGER.info(f"Loaded {len(df):,} MC event records")
        return df

    except Exception as e:
        LOGGER.warning(f"Failed to load MC events data: {e}")
        return pd.DataFrame()


def _load_xsight_location_data(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Load WiFi location data from XSight cs_WiFiLocation and cs_LastKnown.

    Returns GPS + WiFi signal strength for location-based anomaly detection.
    """
    try:
        from device_anomaly.data_access.db_connection import create_dw_engine
        from device_anomaly.data_access.xsight_loader_extended import (
            load_xsight_table_incremental,
        )

        engine = create_dw_engine()
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)

        dfs = []

        # Load WiFiLocation
        df_wifi, _, _ = load_xsight_table_incremental(
            table_name="cs_WiFiLocation",
            start_date=start_dt,
            end_date=end_dt,
            device_ids=list(device_ids) if device_ids else None,
            use_watermark=False,
            engine=engine,
        )
        if not df_wifi.empty:
            # Normalize column names
            if "Deviceid" in df_wifi.columns:
                df_wifi = df_wifi.rename(columns={"Deviceid": "DeviceId"})
            dfs.append(df_wifi)

        # Load LastKnown
        df_last, _, _ = load_xsight_table_incremental(
            table_name="cs_LastKnown",
            start_date=start_dt,
            end_date=end_dt,
            device_ids=list(device_ids) if device_ids else None,
            use_watermark=False,
            engine=engine,
        )
        if not df_last.empty:
            dfs.append(df_last)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            LOGGER.info(f"Loaded {len(combined):,} XSight location records")
            return combined
        return pd.DataFrame()

    except Exception as e:
        LOGGER.warning(f"Failed to load XSight location data: {e}")
        return pd.DataFrame()


def _load_xsight_wifi_data(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Load WiFi connectivity patterns from XSight cs_WifiHour.

    Returns WiFi signal strength, connection time, and disconnect counts.
    """
    try:
        from device_anomaly.data_access.db_connection import create_dw_engine
        from device_anomaly.data_access.xsight_loader_extended import (
            load_xsight_table_incremental,
        )

        engine = create_dw_engine()
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)

        df, _, _ = load_xsight_table_incremental(
            table_name="cs_WifiHour",
            start_date=start_dt,
            end_date=end_dt,
            device_ids=list(device_ids) if device_ids else None,
            use_watermark=False,
            engine=engine,
        )

        if not df.empty:
            # Normalize column names
            if "Deviceid" in df.columns:
                df = df.rename(columns={"Deviceid": "DeviceId"})
            LOGGER.info(f"Loaded {len(df):,} XSight WiFi records")
        return df

    except Exception as e:
        LOGGER.warning(f"Failed to load XSight WiFi data: {e}")
        return pd.DataFrame()


def _aggregate_location_features(
    df_location: pd.DataFrame,
    device_col: str = "DeviceId",
) -> pd.DataFrame:
    """
    Aggregate raw location data to device-level features.

    Computes: avg latitude/longitude, location count, avg signal strength.
    """
    if df_location.empty:
        return pd.DataFrame()

    agg_dict = {}

    if "Latitude" in df_location.columns:
        agg_dict["Latitude"] = ["mean", "std", "count"]
    if "Longitude" in df_location.columns:
        agg_dict["Longitude"] = ["mean", "std"]
    if "WiFiSignalStrength" in df_location.columns:
        agg_dict["WiFiSignalStrength"] = ["mean", "min", "max"]
    if "SignalStrength" in df_location.columns:
        agg_dict["SignalStrength"] = ["mean", "min", "max"]

    if not agg_dict or device_col not in df_location.columns:
        return pd.DataFrame()

    aggregated = df_location.groupby(device_col).agg(agg_dict)
    aggregated.columns = ["_".join(col).strip() for col in aggregated.columns.values]
    aggregated = aggregated.reset_index()

    # Rename for clarity
    rename_map = {
        "Latitude_mean": "LocationLatitudeMean",
        "Latitude_std": "LocationLatitudeStd",
        "Latitude_count": "LocationReadingCount",
        "Longitude_mean": "LocationLongitudeMean",
        "Longitude_std": "LocationLongitudeStd",
        "WiFiSignalStrength_mean": "WiFiSignalMean",
        "WiFiSignalStrength_min": "WiFiSignalMin",
        "WiFiSignalStrength_max": "WiFiSignalMax",
        "SignalStrength_mean": "CellSignalMean",
        "SignalStrength_min": "CellSignalMin",
        "SignalStrength_max": "CellSignalMax",
    }
    aggregated = aggregated.rename(
        columns={k: v for k, v in rename_map.items() if k in aggregated.columns}
    )

    return aggregated


def _aggregate_event_features(
    df_events: pd.DataFrame,
    device_col: str = "DeviceId",
) -> pd.DataFrame:
    """
    Aggregate raw event data to device-level features.

    Computes: event counts by severity, error ratios.
    """
    if df_events.empty or device_col not in df_events.columns:
        return pd.DataFrame()

    # Total event count per device
    event_counts = df_events.groupby(device_col).size().reset_index(name="TotalEventCount")

    # Count by severity if available
    if "Severity" in df_events.columns:
        severity_pivot = df_events.pivot_table(
            index=device_col,
            columns="Severity",
            values="DateTime" if "DateTime" in df_events.columns else df_events.columns[0],
            aggfunc="count",
            fill_value=0,
        ).reset_index()
        severity_pivot.columns = [
            f"EventSeverity_{c}" if c != device_col else c for c in severity_pivot.columns
        ]
        event_counts = event_counts.merge(severity_pivot, on=device_col, how="left")

    # Count by event class if available
    if "EventClass" in df_events.columns:
        class_pivot = df_events.pivot_table(
            index=device_col,
            columns="EventClass",
            values="DateTime" if "DateTime" in df_events.columns else df_events.columns[0],
            aggfunc="count",
            fill_value=0,
        ).reset_index()
        class_pivot.columns = [
            f"EventClass_{c}" if c != device_col else c for c in class_pivot.columns
        ]
        event_counts = event_counts.merge(class_pivot, on=device_col, how="left")

    return event_counts


def _aggregate_system_health_features(
    df_health: pd.DataFrame,
    device_col: str = "DeviceId",
) -> pd.DataFrame:
    """
    Aggregate system health time-series to device-level features.

    Uses StatType to identify metric types (CPU, RAM, etc).
    """
    if df_health.empty or device_col not in df_health.columns:
        return pd.DataFrame()

    # If StatType is available, pivot by stat type
    if "StatType" in df_health.columns and "IntValue" in df_health.columns:
        pivot = df_health.pivot_table(
            index=device_col,
            columns="StatType",
            values="IntValue",
            aggfunc=["mean", "max", "min"],
        )
        pivot.columns = [f"Health_{col[1]}_{col[0]}" for col in pivot.columns]
        return pivot.reset_index()

    # Fallback: aggregate IntValue directly
    if "IntValue" in df_health.columns:
        return (
            df_health.groupby(device_col)
            .agg({"IntValue": ["mean", "max", "min", "std"]})
            .reset_index()
        )

    return pd.DataFrame()


def load_extended_device_dataset(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
    row_limit: int | None = 1_000_000,
    include_location: bool = True,
    include_events: bool = True,
    include_system_health: bool = True,
    include_wifi: bool = True,
) -> pd.DataFrame:
    """
    Load comprehensive device dataset with all available data sources.

    Extends load_unified_device_dataset with:
    - MC DeviceStatLocation (GPS coordinates)
    - MC DeviceStatInt (CPU, RAM, storage metrics)
    - MC MainLog (events/alerts)
    - XSight cs_WiFiLocation (WiFi + GPS)
    - XSight cs_WifiHour (WiFi patterns)
    - XSight cs_LastKnown (last known location)

    All additional data sources are aggregated to device-level and merged
    with the base telemetry data.

    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        device_ids: Optional list of device IDs to filter
        row_limit: Maximum rows for base dataset
        include_location: Load location data from MC + XSight
        include_events: Load event/log data from MC
        include_system_health: Load system health metrics from MC
        include_wifi: Load WiFi pattern data from XSight

    Returns:
        DataFrame with comprehensive device telemetry + enriched features
    """
    LOGGER.info(f"Loading extended dataset: {start_date} to {end_date}")

    # Start with base unified dataset
    df = load_unified_device_dataset(
        start_date=start_date,
        end_date=end_date,
        device_ids=device_ids,
        row_limit=row_limit,
        include_mc_labels=True,
        use_fallback=True,
        enrich_features=True,
    )

    if df.empty:
        return df

    settings = get_settings()
    device_id_list = list(device_ids) if device_ids else None

    # Load and merge location data
    if include_location and settings.enable_mc_timeseries:
        df_mc_loc = _load_mc_location_data(start_date, end_date, device_id_list)
        df_mc_loc_agg = _aggregate_location_features(df_mc_loc)
        if not df_mc_loc_agg.empty:
            df = df.merge(df_mc_loc_agg, on="DeviceId", how="left", suffixes=("", "_mc_loc"))
            LOGGER.info(f"Merged MC location features: {len(df_mc_loc_agg.columns) - 1} columns")

    if include_location and settings.enable_xsight_extended:
        df_xs_loc = _load_xsight_location_data(start_date, end_date, device_id_list)
        df_xs_loc_agg = _aggregate_location_features(df_xs_loc)
        if not df_xs_loc_agg.empty:
            df = df.merge(df_xs_loc_agg, on="DeviceId", how="left", suffixes=("", "_xs_loc"))
            LOGGER.info(
                f"Merged XSight location features: {len(df_xs_loc_agg.columns) - 1} columns"
            )

    # Load and merge event data
    if include_events and settings.enable_mc_timeseries:
        df_events = _load_mc_events_data(start_date, end_date, device_id_list)
        df_events_agg = _aggregate_event_features(df_events)
        if not df_events_agg.empty:
            df = df.merge(df_events_agg, on="DeviceId", how="left", suffixes=("", "_events"))
            LOGGER.info(f"Merged event features: {len(df_events_agg.columns) - 1} columns")

    # Load and merge system health data
    if include_system_health and settings.enable_mc_timeseries:
        df_health = _load_mc_system_health_data(start_date, end_date, device_id_list)
        df_health_agg = _aggregate_system_health_features(df_health)
        if not df_health_agg.empty:
            df = df.merge(df_health_agg, on="DeviceId", how="left", suffixes=("", "_health"))
            LOGGER.info(f"Merged system health features: {len(df_health_agg.columns) - 1} columns")

    # Load and merge WiFi data
    if include_wifi and settings.enable_xsight_extended:
        df_wifi = _load_xsight_wifi_data(start_date, end_date, device_id_list)
        if not df_wifi.empty:
            # Aggregate WiFi data to device level
            wifi_agg = (
                df_wifi.groupby("DeviceId")
                .agg(
                    {
                        col: ["mean", "sum"]
                        if col in ["ConnectionTime", "DisconnectCount"]
                        else "mean"
                        for col in df_wifi.columns
                        if col != "DeviceId" and df_wifi[col].dtype in ["float64", "int64"]
                    }
                )
                .reset_index()
            )
            if len(wifi_agg.columns) > 1:
                wifi_agg.columns = ["DeviceId"] + [
                    f"WiFi_{col[0]}_{col[1]}" if isinstance(col, tuple) else col
                    for col in wifi_agg.columns[1:]
                ]
                df = df.merge(wifi_agg, on="DeviceId", how="left", suffixes=("", "_wifi"))
                LOGGER.info(f"Merged WiFi features: {len(wifi_agg.columns) - 1} columns")

    LOGGER.info(f"Extended dataset complete: {len(df):,} rows, {len(df.columns)} columns")
    return df


def get_extended_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Get summary of extended dataset including data source contributions.
    """
    summary = get_dataset_summary(df)

    # Identify columns by source
    location_cols = [
        c
        for c in df.columns
        if any(x in c.lower() for x in ["location", "latitude", "longitude", "gps"])
    ]
    event_cols = [
        c for c in df.columns if any(x in c.lower() for x in ["event", "severity", "eventclass"])
    ]
    health_cols = [
        c for c in df.columns if any(x in c.lower() for x in ["health_", "cpu", "ram", "storage"])
    ]
    wifi_cols = [c for c in df.columns if any(x in c.lower() for x in ["wifi", "signal"])]

    summary["extended_features"] = {
        "location_features": len(location_cols),
        "event_features": len(event_cols),
        "system_health_features": len(health_cols),
        "wifi_features": len(wifi_cols),
    }

    return summary


# =============================================================================
# HOURLY DATA LOADING
# =============================================================================


def load_hourly_device_dataset(
    start_date: str,
    end_date: str,
    device_ids: Iterable[int] | None = None,
    tables: list[str] | None = None,
    aggregation_level: str = "device_day",
    max_rows_per_table: int = 1_000_000,
    max_days: int = 7,
) -> pd.DataFrame:
    """
    Load hourly granularity data from XSight hourly tables.

    Enables finer-grained anomaly detection by loading hour-level metrics
    from tables like cs_DataUsageByHour (104M+ rows), cs_BatteryLevelDrop,
    and cs_WifiHour.

    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        device_ids: Optional list of device IDs to filter
        tables: List of hourly tables to load. Defaults to:
                ["cs_DataUsageByHour", "cs_BatteryLevelDrop", "cs_WifiHour"]
        aggregation_level: How to aggregate hourly data:
            - "hourly": Keep hour-level granularity (returns many rows per device per day)
            - "device_day": Aggregate to device-day level (matches daily training data)
            - "device_hour": Group by device and hour-of-day (temporal patterns)
        max_rows_per_table: Maximum rows to load per table (memory safety)
        max_days: Maximum days of hourly data to load (performance/memory)

    Returns:
        DataFrame with hourly telemetry aggregated per aggregation_level
    """
    if tables is None:
        tables = ["cs_DataUsageByHour", "cs_BatteryLevelDrop", "cs_WifiHour"]

    LOGGER.info(
        f"Loading hourly data: {tables}, aggregation={aggregation_level}, max_days={max_days}"
    )

    # Limit date range for hourly tables (high volume)
    from datetime import datetime as dt

    end_dt = dt.strptime(end_date, "%Y-%m-%d")
    start_dt = dt.strptime(start_date, "%Y-%m-%d")
    if (end_dt - start_dt).days > max_days:
        start_dt = end_dt - timedelta(days=max_days)
        start_date = start_dt.strftime("%Y-%m-%d")
        LOGGER.info(f"Limiting hourly data to last {max_days} days: {start_date} to {end_date}")

    try:
        from device_anomaly.data_access.db_connection import create_dw_engine
        from device_anomaly.data_access.xsight_loader_extended import (
            load_xsight_table_incremental,
        )

        engine = create_dw_engine()
        start_datetime = dt.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        end_datetime = dt.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)

        all_dfs = []

        for table_name in tables:
            try:
                LOGGER.info(f"Loading hourly table: {table_name}")
                df_table, _, _ = load_xsight_table_incremental(
                    table_name=table_name,
                    start_date=start_datetime,
                    end_date=end_datetime,
                    device_ids=list(device_ids) if device_ids else None,
                    use_watermark=False,
                    engine=engine,
                    batch_size=max_rows_per_table,
                )

                if df_table.empty:
                    LOGGER.warning(f"No data from {table_name}")
                    continue

                # Normalize column names
                if "Deviceid" in df_table.columns:
                    df_table = df_table.rename(columns={"Deviceid": "DeviceId"})

                # Add source table identifier
                df_table["_hourly_source"] = table_name

                LOGGER.info(f"Loaded {len(df_table):,} rows from {table_name}")
                all_dfs.append(df_table)

            except Exception as e:
                LOGGER.warning(f"Failed to load {table_name}: {e}")
                continue

        if not all_dfs:
            LOGGER.warning("No hourly data loaded from any table")
            return pd.DataFrame()

        # Combine all hourly tables
        df_hourly = pd.concat(all_dfs, ignore_index=True)
        LOGGER.info(f"Combined hourly data: {len(df_hourly):,} rows")

        # Aggregate based on level
        df_hourly = _aggregate_hourly_data(df_hourly, aggregation_level)

        LOGGER.info(
            f"Final hourly dataset: {len(df_hourly):,} rows, {len(df_hourly.columns)} columns"
        )
        return df_hourly

    except Exception as e:
        LOGGER.error(f"Failed to load hourly data: {e}")
        return pd.DataFrame()


def _aggregate_hourly_data(df: pd.DataFrame, aggregation_level: str) -> pd.DataFrame:
    """
    Aggregate hourly data to specified level.

    Args:
        df: Raw hourly DataFrame
        aggregation_level: "hourly", "device_day", or "device_hour"

    Returns:
        Aggregated DataFrame
    """
    if df.empty or "DeviceId" not in df.columns:
        return df

    # Identify timestamp column
    ts_col = None
    for col in ["CollectedDate", "CollectedTime", "Timestamp", "DateTime"]:
        if col in df.columns:
            ts_col = col
            break

    if ts_col is None:
        LOGGER.warning("No timestamp column found in hourly data")
        return df

    # Ensure datetime type
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    # Identify numeric columns for aggregation
    exclude_cols = {"DeviceId", ts_col, "_hourly_source", "Hour", "CollectedDate"}
    numeric_cols = [
        col
        for col in df.columns
        if df[col].dtype in ["float64", "int64", "float32", "int32"] and col not in exclude_cols
    ]

    if aggregation_level == "hourly":
        # Keep hour-level granularity
        return df

    elif aggregation_level == "device_day":
        # Aggregate to device-day
        df["_date"] = df[ts_col].dt.date

        agg_dict = {
            col: ["mean", "sum", "max", "min"] for col in numeric_cols[:20]
        }  # Limit columns
        grouped = df.groupby(["DeviceId", "_date"]).agg(agg_dict)
        grouped.columns = [f"{col}_{agg}_hourly" for col, agg in grouped.columns]
        grouped = grouped.reset_index()

        # Rename _date to CollectedDate for merge compatibility
        grouped = grouped.rename(columns={"_date": "CollectedDate"})

        return grouped

    elif aggregation_level == "device_hour":
        # Aggregate by device and hour-of-day (temporal patterns)
        df["_hour"] = df[ts_col].dt.hour

        agg_dict = dict.fromkeys(numeric_cols[:20], "mean")
        grouped = df.groupby(["DeviceId", "_hour"]).agg(agg_dict)
        grouped.columns = [f"{col}_hour_pattern" for col in grouped.columns]
        grouped = grouped.reset_index()

        # Pivot hour dimension into columns for ML
        pivoted = grouped.pivot(index="DeviceId", columns="_hour")
        pivoted.columns = [f"{col}_h{hour}" for col, hour in pivoted.columns]
        pivoted = pivoted.reset_index()

        return pivoted

    else:
        LOGGER.warning(f"Unknown aggregation level: {aggregation_level}, returning raw data")
        return df
