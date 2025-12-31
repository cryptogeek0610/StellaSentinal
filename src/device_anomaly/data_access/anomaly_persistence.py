"""Module for persisting anomaly detection results to database."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from device_anomaly.api.dependencies import get_tenant_id
from device_anomaly.database.connection import get_results_db_session
from device_anomaly.database.schema import AnomalyResult, AnomalyStatus, DeviceMetadata

logger = logging.getLogger(__name__)


def _get_str_value(row: pd.Series, *col_names: str) -> Optional[str]:
    """Get string value from row, trying multiple column names."""
    for col in col_names:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                return str(val)
    return None


def _get_datetime_value(row: pd.Series, *col_names: str) -> Optional[datetime]:
    """Get datetime value from row, trying multiple column names."""
    for col in col_names:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                try:
                    return pd.to_datetime(val).to_pydatetime()
                except Exception:
                    pass
    return None


def _derive_status(row: pd.Series) -> str:
    """Derive device status from available columns."""
    # Check Online column first
    if "Online" in row.index and pd.notna(row["Online"]):
        online_val = row["Online"]
        if isinstance(online_val, bool):
            return "online" if online_val else "offline"
        if str(online_val).lower() in ("true", "1", "yes"):
            return "online"
        if str(online_val).lower() in ("false", "0", "no"):
            return "offline"

    # Check Mode column
    if "Mode" in row.index and pd.notna(row["Mode"]):
        mode = str(row["Mode"]).lower()
        if "online" in mode:
            return "online"
        if "offline" in mode:
            return "offline"

    return "unknown"


def upsert_device_metadata(df: pd.DataFrame, db=None, tenant_id: str = None) -> int:
    """
    Upsert device metadata from DataFrame.

    Extracts device information from the scored DataFrame and updates
    the DeviceMetadata table. This ensures device details are populated
    when anomalies are detected.

    Args:
        df: DataFrame containing device data (from unified loader)
        db: Optional database session (will create one if not provided)
        tenant_id: Optional tenant ID (will get from context if not provided)

    Returns:
        Number of devices updated
    """
    if df.empty or "DeviceId" not in df.columns:
        return 0

    close_db = False
    if db is None:
        db = get_results_db_session()
        close_db = True

    if tenant_id is None:
        tenant_id = get_tenant_id()

    try:
        # Get unique devices with their latest data
        device_ids = df["DeviceId"].unique()
        devices_updated = 0

        for device_id in device_ids:
            device_id = int(device_id)

            # Get the latest row for this device (by Timestamp if available)
            device_rows = df[df["DeviceId"] == device_id]
            if "Timestamp" in device_rows.columns:
                device_rows = device_rows.sort_values("Timestamp", ascending=False)
            row = device_rows.iloc[0]

            # Extract metadata from row
            device_name = _get_str_value(row, "DevName", "DeviceName", "device_name")
            device_model = _get_str_value(row, "Model", "ModelName", "device_model")
            location = _get_str_value(row, "StoreName", "SiteId", "StoreId", "Location", "location")
            os_version = _get_str_value(row, "OSVersion", "OsVersionName", "os_version")
            agent_version = _get_str_value(row, "AgentVersion", "agent_version")
            last_seen = _get_datetime_value(row, "LastCheckInTime", "LastConnTime", "Timestamp", "last_seen")
            status = _derive_status(row)

            # Try to get existing device metadata
            existing = (
                db.query(DeviceMetadata)
                .filter(DeviceMetadata.device_id == device_id)
                .filter(DeviceMetadata.tenant_id == tenant_id)
                .first()
            )

            if existing:
                # Update existing record (only update non-null values)
                if device_name:
                    existing.device_name = device_name
                if device_model:
                    existing.device_model = device_model
                if location:
                    existing.location = location
                if os_version:
                    existing.os_version = os_version
                if agent_version:
                    existing.agent_version = agent_version
                if last_seen:
                    existing.last_seen = last_seen
                if status != "unknown":
                    existing.status = status
            else:
                # Create new record
                new_device = DeviceMetadata(
                    device_id=device_id,
                    tenant_id=tenant_id,
                    device_name=device_name,
                    device_model=device_model,
                    location=location,
                    os_version=os_version,
                    agent_version=agent_version,
                    last_seen=last_seen,
                    status=status or "unknown",
                )
                db.add(new_device)

            devices_updated += 1

        db.commit()
        logger.info(f"Upserted metadata for {devices_updated} devices")
        return devices_updated

    except Exception as e:
        db.rollback()
        logger.error(f"Error upserting device metadata: {e}", exc_info=True)
        raise
    finally:
        if close_db:
            db.close()


def persist_anomaly_results(
    df_scored: pd.DataFrame,
    batch_size: int = 1000,
    only_anomalies: bool = True,
    update_device_metadata: bool = True,
) -> int:
    """
    Persist anomaly detection results to the database.

    Args:
        df_scored: DataFrame with anomaly scores and labels (from detector.score_dataframe())
        batch_size: Number of records to insert per batch
        only_anomalies: If True, only persist records with anomaly_label == -1
        update_device_metadata: If True, also upsert device metadata from the DataFrame

    Returns:
        Number of records persisted
    """
    db = get_results_db_session()
    tenant_id = get_tenant_id()
    try:
        # Required columns
        required_cols = ["DeviceId", "Timestamp", "anomaly_score", "anomaly_label"]
        missing_cols = [c for c in required_cols if c not in df_scored.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Update device metadata first (before filtering to anomalies only)
        # This ensures all devices in the scored data get their metadata updated
        if update_device_metadata:
            try:
                upsert_device_metadata(df_scored, db=db, tenant_id=tenant_id)
            except Exception as e:
                logger.warning(f"Failed to update device metadata: {e}")
                # Continue with anomaly persistence even if metadata update fails

        # Filter to only anomalies if requested
        if only_anomalies:
            df_to_save = df_scored[df_scored["anomaly_label"] == -1].copy()
        else:
            df_to_save = df_scored.copy()

        if df_to_save.empty:
            logger.info("No anomalies to persist.")
            return 0

        # Metric columns (optional)
        metric_cols = [
            "TotalBatteryLevelDrop",
            "TotalFreeStorageKb",
            "Download",
            "Upload",
            "OfflineTime",
            "DisconnectCount",
            "WiFiSignalStrength",
            "ConnectionTime",
        ]

        # Feature columns (to store as JSON)
        feature_cols = [c for c in df_to_save.columns if any(token in c for token in ("_mean_", "_std_", "_delta"))]

        records_persisted = 0

        # Process in batches
        for i in range(0, len(df_to_save), batch_size):
            batch = df_to_save.iloc[i : i + batch_size]
            anomalies_to_insert = []

            for _, row in batch.iterrows():
                # Extract feature values as dict
                feature_values = {}
                if feature_cols:
                    for col in feature_cols:
                        if col in row:
                            feature_values[col] = float(row[col]) if pd.notna(row[col]) else None

                anomaly = AnomalyResult(
                    tenant_id=tenant_id,
                    device_id=int(row["DeviceId"]),
                    timestamp=pd.to_datetime(row["Timestamp"]),
                    anomaly_score=float(row["anomaly_score"]),
                    anomaly_label=int(row["anomaly_label"]),
                    total_battery_level_drop=float(row.get("TotalBatteryLevelDrop", 0)) if pd.notna(row.get("TotalBatteryLevelDrop")) else None,
                    total_free_storage_kb=float(row.get("TotalFreeStorageKb", 0)) if pd.notna(row.get("TotalFreeStorageKb")) else None,
                    download=float(row.get("Download", 0)) if pd.notna(row.get("Download")) else None,
                    upload=float(row.get("Upload", 0)) if pd.notna(row.get("Upload")) else None,
                    offline_time=float(row.get("OfflineTime", 0)) if pd.notna(row.get("OfflineTime")) else None,
                    disconnect_count=float(row.get("DisconnectCount", 0)) if pd.notna(row.get("DisconnectCount")) else None,
                    wifi_signal_strength=float(row.get("WiFiSignalStrength", 0)) if pd.notna(row.get("WiFiSignalStrength")) else None,
                    connection_time=float(row.get("ConnectionTime", 0)) if pd.notna(row.get("ConnectionTime")) else None,
                    feature_values_json=json.dumps(feature_values) if feature_values else None,
                    status=AnomalyStatus.OPEN.value,
                )
                anomalies_to_insert.append(anomaly)

            # Bulk insert
            db.add_all(anomalies_to_insert)
            db.commit()
            records_persisted += len(anomalies_to_insert)
            logger.info(f"Persisted batch: {len(anomalies_to_insert)} anomalies (total: {records_persisted})")

        logger.info(f"Successfully persisted {records_persisted} anomaly records to database.")
        return records_persisted

    except Exception as e:
        db.rollback()
        logger.error(f"Error persisting anomaly results: {e}", exc_info=True)
        raise
    finally:
        db.close()


def get_feedback_stats() -> Dict[str, int]:
    """
    Get feedback statistics for auto-retrain decisions.

    Uses the status field to determine feedback:
    - Total feedback: count of anomalies with status != 'open'
    - False positives: count of anomalies with status = 'false_positive'

    Returns:
        Dict with 'total_feedback' and 'false_positives' counts
    """
    from sqlalchemy import func

    db = get_results_db_session()
    tenant_id = get_tenant_id()
    try:
        # Count reviewed entries (status changed from 'open')
        total_feedback = (
            db.query(func.count(AnomalyResult.id))
            .filter(
                AnomalyResult.tenant_id == tenant_id,
                AnomalyResult.status != AnomalyStatus.OPEN.value,
            )
            .scalar()
            or 0
        )

        # Count false positives
        false_positives = (
            db.query(func.count(AnomalyResult.id))
            .filter(
                AnomalyResult.tenant_id == tenant_id,
                AnomalyResult.status == AnomalyStatus.FALSE_POSITIVE.value,
            )
            .scalar()
            or 0
        )

        return {
            "total_feedback": total_feedback,
            "false_positives": false_positives,
        }

    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}", exc_info=True)
        return {"total_feedback": 0, "false_positives": 0}
    finally:
        db.close()
