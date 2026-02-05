"""
Device Metadata Sync Service.

Syncs device metadata from XSight Device table and MobiControlDB DevInfo
to PostgreSQL device_metadata table for:
- Fleet Overview UI display
- Real-time streaming enrichment (cohort identification)

Data Sources:
- XSight Device table: DeviceName, ModelId (matches telemetry DeviceId)
- MobiControlDB DevInfo: Firmware, OS, Manufacturer, Security indicators
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text

from device_anomaly.config.settings import get_settings
from device_anomaly.data_access.anomaly_persistence import upsert_device_metadata
from device_anomaly.data_access.db_connection import create_dw_engine, create_mc_engine
from device_anomaly.database.connection import get_results_db_session

logger = logging.getLogger(__name__)


def load_xsight_device_metadata(since_days: int = 30, limit: int = 100_000) -> pd.DataFrame:
    """
    Load essential device metadata from XSight Device table.

    This query fetches DeviceName and Model from XSight's Device table
    joined with Model table for the full model name.

    Args:
        since_days: Only include devices active in the last N days
        limit: Maximum number of devices to sync

    Returns:
        DataFrame with device metadata
    """
    engine = create_dw_engine()
    since_date = (datetime.utcnow() - timedelta(days=since_days)).strftime("%Y-%m-%d")

    sql = text(f"""
        SELECT TOP ({int(limit)})
            d.DeviceId,
            d.DeviceName AS DevName,
            d.SerialNumber,
            m.ModelName AS Model,
            d.LastCheckinTime AS LastCheckInTime,
            d.IsActive
        FROM dbo.Device d
        LEFT JOIN dbo.Model m ON d.ModelId = m.ModelId
        WHERE d.LastCheckinTime >= :since_date
           OR d.IsActive = 1
        ORDER BY d.LastCheckinTime DESC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"since_date": since_date})
            logger.info(f"Loaded {len(df)} devices from XSight (active since {since_date})")
            return df
    except Exception as e:
        logger.error(f"Failed to load device metadata from XSight: {e}")
        raise


def _derive_device_name(row: pd.Series) -> str:
    """
    Derive display name with fallback chain: DevName -> SerialNumber -> DeviceId

    Args:
        row: DataFrame row with device fields

    Returns:
        Device display name
    """
    # Priority 1: DevName (user-configured name in MobiControl)
    devname = row.get("DevName")
    if (
        devname
        and str(devname).strip()
        and str(devname).strip().lower() not in ("null", "none", "")
    ):
        return str(devname).strip()

    # Priority 2: SerialNumber
    serial = row.get("SerialNumber")
    if serial and str(serial).strip() and str(serial).strip().lower() not in ("null", "none", ""):
        return str(serial).strip()

    # Priority 3: DeviceId as string
    device_id = row.get("DeviceId")
    if device_id is not None:
        return f"Device-{device_id}"

    return "Unknown"


def _derive_device_model(row: pd.Series) -> str:
    """
    Get model display string from XSight ModelName.

    XSight's Model table already has full model names like "Samsung SM-G390F"
    so we just use that directly.

    Args:
        row: DataFrame row with device fields

    Returns:
        Model name string
    """
    model = str(row.get("Model", "") or "").strip()

    # Filter out placeholder values
    placeholder_values = ("null", "unknown", "", "none", "n/a")
    if model.lower() in placeholder_values:
        return "Unknown Device"

    return model


def transform_device_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw XSight Device data into format expected by upsert_device_metadata.

    Applies:
    - NAME fallback logic (DevName -> SerialNumber -> DeviceId)
    - MODEL: uses ModelName from XSight (already joined)

    Args:
        df: Raw DataFrame from load_xsight_device_metadata

    Returns:
        Transformed DataFrame ready for upsert
    """
    if df.empty:
        return df

    df = df.copy()

    # Apply transformation for each row
    df["device_name"] = df.apply(_derive_device_name, axis=1)
    df["device_model"] = df.apply(_derive_device_model, axis=1)

    # Rename columns to match expected format
    rename_map = {
        "LastCheckInTime": "last_seen",
    }
    df = df.rename(columns=rename_map)

    # Derive status from IsActive
    def derive_status(row):
        is_active = row.get("IsActive")
        if is_active is True or is_active == 1:
            return "online"
        elif is_active is False or is_active == 0:
            return "offline"
        return "unknown"

    df["status"] = df.apply(derive_status, axis=1)

    logger.info(f"Transformed {len(df)} device records")
    return df


def sync_device_metadata(
    tenant_id: str = None,
    since_days: int = 30,
    limit: int = 100_000,
    raise_on_error: bool = False,
) -> dict:
    """
    Sync device metadata from XSight to PostgreSQL.

    This function:
    1. Loads device metadata from XSight Device table (with Model join)
    2. Transforms names (fallback logic) and models
    3. Upserts to PostgreSQL device_metadata table

    Args:
        tenant_id: Optional tenant ID (uses default if not provided)
        since_days: Only sync devices active in last N days
        limit: Maximum devices to sync
        raise_on_error: If True, raise exceptions instead of returning error dict

    Returns:
        Dict with sync results:
        {
            "success": bool,
            "synced_count": int,
            "duration_seconds": float,
            "errors": list[str],
        }
    """
    start_time = time.time()
    errors = []

    try:
        # 1. Load from XSight Device table
        logger.info(f"Starting device metadata sync (since_days={since_days}, limit={limit})")
        df = load_xsight_device_metadata(since_days=since_days, limit=limit)

        if df.empty:
            logger.warning("No device metadata returned from XSight")
            return {
                "success": True,
                "synced_count": 0,
                "duration_seconds": time.time() - start_time,
                "errors": ["No devices found in XSight for the specified time range"],
            }

        # 2. Transform
        df = transform_device_metadata(df)

        # 3. Upsert to PostgreSQL
        db = get_results_db_session()
        try:
            synced_count = upsert_device_metadata(df, db=db, tenant_id=tenant_id)
        finally:
            db.close()

        duration = time.time() - start_time
        logger.info(f"Device metadata sync completed: {synced_count} devices in {duration:.2f}s")

        return {
            "success": True,
            "synced_count": synced_count,
            "duration_seconds": round(duration, 2),
            "errors": errors,
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Device metadata sync failed: {e}", exc_info=True)

        if raise_on_error:
            raise

        return {
            "success": False,
            "synced_count": 0,
            "duration_seconds": round(duration, 2),
            "errors": [str(e)],
        }


def get_sync_stats() -> dict:
    """
    Get statistics about the current device_metadata table state.

    Returns:
        Dict with stats about populated/empty name/model fields
    """
    db = get_results_db_session()
    try:
        from device_anomaly.database.schema import DeviceMetadata

        total = db.query(DeviceMetadata).count()
        with_name = db.query(DeviceMetadata).filter(DeviceMetadata.device_name.isnot(None)).count()
        with_model = (
            db.query(DeviceMetadata).filter(DeviceMetadata.device_model.isnot(None)).count()
        )

        return {
            "total_devices": total,
            "devices_with_name": with_name,
            "devices_with_model": with_model,
            "name_coverage_pct": round(with_name / total * 100, 1) if total > 0 else 0,
            "model_coverage_pct": round(with_model / total * 100, 1) if total > 0 else 0,
        }
    finally:
        db.close()


# =============================================================================
# MobiControl Device Metadata Cache for Streaming Enrichment
# =============================================================================

# In-memory cache for device metadata used by streaming enrichment
_MC_DEVICE_CACHE: dict[int, dict] = {}
_MC_CACHE_LOADED_AT: datetime | None = None


def load_mc_device_metadata_cache(since_days: int = 30, limit: int = 200_000) -> int:
    """
    Load MobiControl device metadata into in-memory cache for streaming enrichment.

    This populates the cache with device cohort information (model, manufacturer,
    OS version, firmware) that the streaming path needs for cohort-based anomaly detection.

    Args:
        since_days: Only include devices checked in within N days
        limit: Maximum devices to cache

    Returns:
        Number of devices loaded into cache
    """
    global _MC_DEVICE_CACHE, _MC_CACHE_LOADED_AT

    settings = get_settings()
    mc_settings = settings.mc

    if not (mc_settings.host and mc_settings.user and mc_settings.password):
        logger.warning(
            "MC credentials not configured - streaming enrichment will not have device metadata. "
            "Set MC_DB_HOST, MC_DB_USER, MC_DB_PASS environment variables."
        )
        return 0

    try:
        engine = create_mc_engine()
        since_date = (datetime.utcnow() - timedelta(days=since_days)).strftime("%Y-%m-%d")

        # Query essential device metadata for streaming enrichment
        # MobiControlDB DevInfo has Manufacturer/Model/OSVersion as strings
        sql = text(f"""
            SELECT TOP ({int(limit)})
                DeviceId,
                Manufacturer,
                Model,
                OSVersion,
                OEMVersion
            FROM dbo.DevInfo
            WHERE LastCheckInTime >= :since_date
            ORDER BY LastCheckInTime DESC
        """)

        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"since_date": since_date})

        if df.empty:
            logger.warning(
                "MobiControl returned 0 devices for cache - streaming enrichment disabled"
            )
            _MC_DEVICE_CACHE = {}
            _MC_CACHE_LOADED_AT = datetime.utcnow()
            return 0

        # Build cache with stable integer IDs derived from string values
        # Use deterministic hash (not Python's hash() which varies by session)
        import hashlib

        def stable_hash(s: str) -> int:
            """Convert string to stable positive integer for cohort identification."""
            if not s or str(s).lower() in ("none", "null", ""):
                return 0
            # Use MD5 for deterministic hashing (fast, consistent across runs)
            h = hashlib.md5(str(s).encode()).hexdigest()
            return int(h[:8], 16)  # Use first 8 hex chars -> max ~4 billion

        new_cache = {}
        for _, row in df.iterrows():
            device_id = int(row["DeviceId"])
            manufacturer = row.get("Manufacturer")
            model = row.get("Model")
            os_version = row.get("OSVersion")
            oem_version = row.get("OEMVersion")

            new_cache[device_id] = {
                # Integer IDs for cohort grouping (expected by TelemetryEvent)
                "manufacturer_id": stable_hash(manufacturer),
                "model_id": stable_hash(model),
                "os_version_id": stable_hash(os_version),
                "firmware_version": str(oem_version) if oem_version else None,
                # Also store string values for display/debugging
                "manufacturer": manufacturer,
                "model": model,
                "os_version": os_version,
            }

        _MC_DEVICE_CACHE = new_cache
        _MC_CACHE_LOADED_AT = datetime.utcnow()

        logger.info(
            "Loaded %d devices into MC metadata cache for streaming enrichment",
            len(_MC_DEVICE_CACHE),
        )
        return len(_MC_DEVICE_CACHE)

    except Exception as e:
        logger.error("Failed to load MC device metadata cache: %s", e, exc_info=True)
        return 0


def get_mc_device_metadata(device_id: int) -> dict | None:
    """
    Get cached device metadata for streaming enrichment.

    Args:
        device_id: The device ID to look up

    Returns:
        Dict with device metadata or None if not in cache
    """
    return _MC_DEVICE_CACHE.get(device_id)


def get_mc_cache_stats() -> dict:
    """
    Get statistics about the MC device metadata cache.

    Returns:
        Dict with cache stats
    """
    return {
        "cache_size": len(_MC_DEVICE_CACHE),
        "loaded_at": _MC_CACHE_LOADED_AT.isoformat() if _MC_CACHE_LOADED_AT else None,
        "is_populated": len(_MC_DEVICE_CACHE) > 0,
    }


def refresh_mc_cache_if_stale(max_age_minutes: int = 30) -> bool:
    """
    Refresh the MC device metadata cache if it's stale.

    Args:
        max_age_minutes: Consider cache stale after this many minutes

    Returns:
        True if cache was refreshed, False otherwise
    """
    global _MC_CACHE_LOADED_AT

    if _MC_CACHE_LOADED_AT is None:
        load_mc_device_metadata_cache()
        return True

    age = datetime.utcnow() - _MC_CACHE_LOADED_AT
    if age > timedelta(minutes=max_age_minutes):
        logger.info("MC device cache is stale (%s), refreshing...", age)
        load_mc_device_metadata_cache()
        return True

    return False
