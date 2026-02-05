"""
Location Sync Service.

Syncs location metadata from MobiControl labels/groups to PostgreSQL
location_metadata table for "Warehouse A vs Warehouse B" comparisons.

This enables Carl's requirement: "Relate any anomalies to location"
"""
import logging
import time
from datetime import datetime

import pandas as pd
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from device_anomaly.data_access.db_connection import create_mc_engine
from device_anomaly.database.connection import get_results_db_session
from device_anomaly.database.schema import DeviceMetadata, LocationMappingType, LocationMetadata

logger = logging.getLogger(__name__)

# Default label types that represent locations
DEFAULT_LOCATION_LABEL_TYPES = [
    "Store",
    "Warehouse",
    "Site",
    "Location",
    "Building",
    "Branch",
    "Office",
    "Facility",
    "Region",
]


def load_location_labels_from_mc(
    label_types: list[str] | None = None,
    limit: int = 10_000,
) -> pd.DataFrame:
    """
    Load distinct location values from MobiControl LabelDevice table.

    Args:
        label_types: Label type names to treat as locations.
                    Defaults to DEFAULT_LOCATION_LABEL_TYPES.
        limit: Maximum number of locations to sync.

    Returns:
        DataFrame with columns: LabelTypeName, LabelValue, DeviceCount
    """
    if label_types is None:
        label_types = DEFAULT_LOCATION_LABEL_TYPES

    engine = create_mc_engine()

    # Build IN clause with proper escaping
    label_types_escaped = ", ".join(f"'{lt}'" for lt in label_types)

    sql = text(f"""
        SELECT TOP ({int(limit)})
            lt.Name AS LabelTypeName,
            CONVERT(nvarchar(4000), ld.Value) AS LabelValue,
            COUNT(DISTINCT ld.DeviceId) AS DeviceCount
        FROM dbo.LabelDevice ld
        JOIN dbo.LabelType lt ON lt.Id = ld.LabelTypeId
        WHERE lt.Name IN ({label_types_escaped})
          AND ld.Value IS NOT NULL
          AND CONVERT(nvarchar(4000), ld.Value) != ''
        GROUP BY lt.Name, CONVERT(nvarchar(4000), ld.Value)
        ORDER BY COUNT(DISTINCT ld.DeviceId) DESC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
            logger.info(f"Loaded {len(df)} distinct location labels from MC")
            return df
    except Exception as e:
        logger.error(f"Failed to load location labels from MC: {e}")
        raise


def load_device_group_hierarchy_from_mc(limit: int = 1_000) -> pd.DataFrame:
    """
    Load device group hierarchy from MobiControl for location mapping.

    Device groups can represent organizational hierarchy (Region > Site > Building).

    Returns:
        DataFrame with columns: DeviceGroupId, GroupName, ParentGroupId, DeviceCount
    """
    engine = create_mc_engine()

    sql = text(f"""
        SELECT TOP ({int(limit)})
            dg.DeviceGroupId,
            dg.Name AS GroupName,
            dg.ParentDeviceGroupId,
            (SELECT COUNT(*) FROM dbo.DevInfo d WHERE d.DeviceGroupId = dg.DeviceGroupId) AS DeviceCount
        FROM dbo.DeviceGroup dg
        WHERE dg.Name IS NOT NULL
          AND dg.Name != ''
        ORDER BY DeviceCount DESC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
            logger.info(f"Loaded {len(df)} device groups from MC")
            return df
    except Exception as e:
        logger.warning(f"Could not load device groups from MC: {e}")
        return pd.DataFrame(columns=["DeviceGroupId", "GroupName", "ParentDeviceGroupId", "DeviceCount"])


def load_device_to_location_mapping(
    label_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load mapping of devices to their location labels.

    Args:
        label_types: Label type names to treat as locations.

    Returns:
        DataFrame with columns: DeviceId, LocationLabelType, LocationValue
    """
    if label_types is None:
        label_types = DEFAULT_LOCATION_LABEL_TYPES

    engine = create_mc_engine()
    label_types_escaped = ", ".join(f"'{lt}'" for lt in label_types)

    sql = text(f"""
        SELECT
            ld.DeviceId,
            lt.Name AS LocationLabelType,
            CONVERT(nvarchar(4000), ld.Value) AS LocationValue
        FROM dbo.LabelDevice ld
        JOIN dbo.LabelType lt ON lt.Id = ld.LabelTypeId
        WHERE lt.Name IN ({label_types_escaped})
          AND ld.Value IS NOT NULL
          AND CONVERT(nvarchar(4000), ld.Value) != ''
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
            logger.info(f"Loaded {len(df)} device-to-location mappings")
            return df
    except Exception as e:
        logger.error(f"Failed to load device-to-location mappings: {e}")
        raise


def upsert_location_metadata(
    locations_df: pd.DataFrame,
    tenant_id: str,
    mapping_type: str = LocationMappingType.LABEL.value,
    db: Session | None = None,
) -> int:
    """
    Upsert location metadata records to PostgreSQL.

    Args:
        locations_df: DataFrame with LabelTypeName, LabelValue, DeviceCount columns.
        tenant_id: Tenant identifier.
        mapping_type: How devices map to this location (label, device_group, etc.).
        db: Optional database session.

    Returns:
        Number of locations upserted.
    """
    close_db = False
    if db is None:
        db = get_results_db_session()
        close_db = True

    try:
        count = 0
        for _, row in locations_df.iterrows():
            label_type = row.get("LabelTypeName", "Location")
            label_value = row.get("LabelValue", "")

            if not label_value:
                continue

            # Generate location_id from label type and value
            location_id = f"{label_type.lower()}-{label_value.lower().replace(' ', '-')}"

            # Check if exists
            existing = db.query(LocationMetadata).filter(
                LocationMetadata.tenant_id == tenant_id,
                LocationMetadata.location_id == location_id,
            ).first()

            if existing:
                # Update existing
                existing.location_name = label_value
                existing.mapping_type = mapping_type
                existing.mapping_attribute = label_type
                existing.mapping_value = label_value
                existing.updated_at = datetime.utcnow()
            else:
                # Create new
                location = LocationMetadata(
                    tenant_id=tenant_id,
                    location_id=location_id,
                    location_name=label_value,
                    mapping_type=mapping_type,
                    mapping_attribute=label_type,
                    mapping_value=label_value,
                    is_active=True,
                )
                db.add(location)

            count += 1

        db.commit()
        logger.info(f"Upserted {count} location metadata records for tenant {tenant_id}")
        return count

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to upsert location metadata: {e}")
        raise
    finally:
        if close_db:
            db.close()


def upsert_device_group_locations(
    groups_df: pd.DataFrame,
    tenant_id: str,
    db: Session | None = None,
) -> int:
    """
    Upsert device groups as location metadata.

    Args:
        groups_df: DataFrame with DeviceGroupId, GroupName columns.
        tenant_id: Tenant identifier.
        db: Optional database session.

    Returns:
        Number of locations upserted.
    """
    close_db = False
    if db is None:
        db = get_results_db_session()
        close_db = True

    try:
        count = 0
        for _, row in groups_df.iterrows():
            group_id = row.get("DeviceGroupId")
            group_name = row.get("GroupName", "")

            if not group_name or group_id is None:
                continue

            location_id = f"group-{group_id}"

            existing = db.query(LocationMetadata).filter(
                LocationMetadata.tenant_id == tenant_id,
                LocationMetadata.location_id == location_id,
            ).first()

            if existing:
                existing.location_name = group_name
                existing.device_group_id = int(group_id)
                existing.updated_at = datetime.utcnow()
            else:
                location = LocationMetadata(
                    tenant_id=tenant_id,
                    location_id=location_id,
                    location_name=group_name,
                    mapping_type=LocationMappingType.DEVICE_GROUP.value,
                    device_group_id=int(group_id),
                    is_active=True,
                )
                db.add(location)

            count += 1

        db.commit()
        logger.info(f"Upserted {count} device group locations for tenant {tenant_id}")
        return count

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to upsert device group locations: {e}")
        raise
    finally:
        if close_db:
            db.close()


def sync_locations_from_labels(
    tenant_id: str,
    label_types: list[str] | None = None,
    limit: int = 10_000,
    raise_on_error: bool = False,
) -> dict:
    """
    Sync locations from MobiControl labels to PostgreSQL.

    This is the primary sync function that:
    1. Loads distinct location label values from MC
    2. Creates/updates location_metadata records in PostgreSQL

    Args:
        tenant_id: Tenant identifier.
        label_types: Label type names to treat as locations.
        limit: Maximum locations to sync.
        raise_on_error: If True, raise exceptions instead of returning error dict.

    Returns:
        Dict with sync results.
    """
    start_time = time.time()
    errors = []

    try:
        logger.info(f"Starting location sync from labels (tenant={tenant_id})")

        # 1. Load from MC
        locations_df = load_location_labels_from_mc(label_types=label_types, limit=limit)

        if locations_df.empty:
            logger.warning("No location labels found in MobiControl")
            return {
                "success": True,
                "synced_count": 0,
                "duration_seconds": time.time() - start_time,
                "errors": ["No location labels found in MobiControl"],
                "label_types_searched": label_types or DEFAULT_LOCATION_LABEL_TYPES,
            }

        # 2. Upsert to PostgreSQL
        synced_count = upsert_location_metadata(
            locations_df=locations_df,
            tenant_id=tenant_id,
            mapping_type=LocationMappingType.LABEL.value,
        )

        duration = time.time() - start_time
        logger.info(f"Location sync completed: {synced_count} locations in {duration:.2f}s")

        return {
            "success": True,
            "synced_count": synced_count,
            "duration_seconds": round(duration, 2),
            "errors": errors,
            "label_types_searched": label_types or DEFAULT_LOCATION_LABEL_TYPES,
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Location sync failed: {e}", exc_info=True)

        if raise_on_error:
            raise

        return {
            "success": False,
            "synced_count": 0,
            "duration_seconds": round(duration, 2),
            "errors": [str(e)],
        }


def sync_locations_from_device_groups(
    tenant_id: str,
    limit: int = 1_000,
    raise_on_error: bool = False,
) -> dict:
    """
    Sync locations from MobiControl device groups to PostgreSQL.

    Args:
        tenant_id: Tenant identifier.
        limit: Maximum groups to sync.
        raise_on_error: If True, raise exceptions.

    Returns:
        Dict with sync results.
    """
    start_time = time.time()

    try:
        logger.info(f"Starting location sync from device groups (tenant={tenant_id})")

        # 1. Load from MC
        groups_df = load_device_group_hierarchy_from_mc(limit=limit)

        if groups_df.empty:
            return {
                "success": True,
                "synced_count": 0,
                "duration_seconds": time.time() - start_time,
                "errors": ["No device groups found in MobiControl"],
            }

        # 2. Upsert to PostgreSQL
        synced_count = upsert_device_group_locations(
            groups_df=groups_df,
            tenant_id=tenant_id,
        )

        duration = time.time() - start_time
        return {
            "success": True,
            "synced_count": synced_count,
            "duration_seconds": round(duration, 2),
            "errors": [],
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Device group sync failed: {e}", exc_info=True)

        if raise_on_error:
            raise

        return {
            "success": False,
            "synced_count": 0,
            "duration_seconds": round(duration, 2),
            "errors": [str(e)],
        }


def sync_all_locations(
    tenant_id: str,
    label_types: list[str] | None = None,
    include_device_groups: bool = True,
    raise_on_error: bool = False,
) -> dict:
    """
    Sync all location sources (labels + device groups) to PostgreSQL.

    Args:
        tenant_id: Tenant identifier.
        label_types: Label types to treat as locations.
        include_device_groups: Also sync device groups as locations.
        raise_on_error: If True, raise exceptions.

    Returns:
        Combined sync results.
    """
    start_time = time.time()
    total_synced = 0
    all_errors = []

    # Sync labels
    labels_result = sync_locations_from_labels(
        tenant_id=tenant_id,
        label_types=label_types,
        raise_on_error=raise_on_error,
    )
    total_synced += labels_result.get("synced_count", 0)
    all_errors.extend(labels_result.get("errors", []))

    # Sync device groups
    groups_result = {"synced_count": 0, "errors": []}
    if include_device_groups:
        groups_result = sync_locations_from_device_groups(
            tenant_id=tenant_id,
            raise_on_error=raise_on_error,
        )
        total_synced += groups_result.get("synced_count", 0)
        all_errors.extend(groups_result.get("errors", []))

    duration = time.time() - start_time

    return {
        "success": len([e for e in all_errors if not e.startswith("No ")]) == 0,
        "synced_count": total_synced,
        "duration_seconds": round(duration, 2),
        "labels_synced": labels_result.get("synced_count", 0),
        "groups_synced": groups_result.get("synced_count", 0),
        "errors": all_errors,
    }


def get_location_sync_stats(tenant_id: str | None = None) -> dict:
    """
    Get statistics about the current location_metadata table state.

    Args:
        tenant_id: Optional tenant filter.

    Returns:
        Dict with location statistics.
    """
    db = get_results_db_session()
    try:
        query = db.query(LocationMetadata)
        if tenant_id:
            query = query.filter(LocationMetadata.tenant_id == tenant_id)

        total = query.count()
        active = query.filter(LocationMetadata.is_active == True).count()  # noqa: E712

        # Count by mapping type
        label_count = query.filter(
            LocationMetadata.mapping_type == LocationMappingType.LABEL.value
        ).count()
        group_count = query.filter(
            LocationMetadata.mapping_type == LocationMappingType.DEVICE_GROUP.value
        ).count()

        last_sync = (
            query.with_entities(func.max(LocationMetadata.updated_at)).scalar()
        )

        device_query = db.query(DeviceMetadata)
        if tenant_id:
            device_query = device_query.filter(DeviceMetadata.tenant_id == tenant_id)
        devices_mapped = device_query.filter(DeviceMetadata.location.isnot(None)).count()

        return {
            "total_locations": total,
            "active_locations": active,
            "locations_from_labels": label_count,
            "locations_from_groups": group_count,
            "last_sync": last_sync.isoformat() if last_sync else None,
            "locations_count": total,
            "devices_mapped": devices_mapped,
            "pending_updates": max(total - active, 0),
        }
    finally:
        db.close()


def get_device_location(
    device_id: int,
    tenant_id: str,
    label_types: list[str] | None = None,
) -> str | None:
    """
    Get the location name for a device by checking its labels.

    Args:
        device_id: Device identifier.
        tenant_id: Tenant identifier.
        label_types: Label types to check.

    Returns:
        Location name or None if not found.
    """
    if label_types is None:
        label_types = DEFAULT_LOCATION_LABEL_TYPES

    db = get_results_db_session()
    try:
        # First, try to find via direct label lookup
        mapping_df = load_device_to_location_mapping(label_types=label_types)
        device_locations = mapping_df[mapping_df["DeviceId"] == device_id]

        if device_locations.empty:
            return None

        # Return first matching location
        first_match = device_locations.iloc[0]
        label_type = first_match["LocationLabelType"]
        label_value = first_match["LocationValue"]
        location_id = f"{label_type.lower()}-{label_value.lower().replace(' ', '-')}"

        # Look up the location name
        location = db.query(LocationMetadata).filter(
            LocationMetadata.tenant_id == tenant_id,
            LocationMetadata.location_id == location_id,
        ).first()

        if location:
            return location.location_name

        return label_value  # Fallback to raw label value

    finally:
        db.close()


def enrich_dataframe_with_locations(
    df: pd.DataFrame,
    tenant_id: str,
    device_id_column: str = "DeviceId",
    label_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add location_id and location_name columns to a DataFrame.

    Args:
        df: DataFrame with device IDs.
        tenant_id: Tenant identifier.
        device_id_column: Name of the device ID column.
        label_types: Label types to check.

    Returns:
        DataFrame with added location columns.
    """
    if df.empty or device_id_column not in df.columns:
        return df

    df = df.copy()

    # Load device-to-location mapping
    mapping_df = load_device_to_location_mapping(label_types=label_types)

    if mapping_df.empty:
        df["location_id"] = None
        df["location_name"] = None
        return df

    # Create location_id from mapping
    mapping_df["location_id"] = (
        mapping_df["LocationLabelType"].str.lower() + "-" +
        mapping_df["LocationValue"].str.lower().str.replace(" ", "-")
    )

    # Merge with main dataframe (take first location per device)
    device_locations = mapping_df.groupby("DeviceId").first().reset_index()
    device_locations = device_locations[["DeviceId", "location_id", "LocationValue"]]
    device_locations = device_locations.rename(columns={"LocationValue": "location_name"})

    df = df.merge(
        device_locations,
        left_on=device_id_column,
        right_on="DeviceId",
        how="left",
    )

    # Clean up duplicate DeviceId column if created
    if "DeviceId_y" in df.columns:
        df = df.drop(columns=["DeviceId_y"])
    if "DeviceId_x" in df.columns:
        df = df.rename(columns={"DeviceId_x": device_id_column})

    return df
