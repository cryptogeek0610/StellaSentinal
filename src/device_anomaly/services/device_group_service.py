"""
DeviceGroup Hierarchy Service for Network Intelligence.

Provides functions to load DeviceGroup hierarchy from MobiControl
and compute full PATH strings for navigation.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import text

from device_anomaly.data_access.db_connection import create_mc_engine

logger = logging.getLogger(__name__)


def load_device_groups_with_path(limit: int = 10_000) -> list[dict[str, Any]]:
    """
    Load device groups from MobiControl and compute full PATH for each.

    Returns a flat list of groups with their full hierarchical path.

    Returns:
        List of dicts with keys:
        - device_group_id: int
        - group_name: str
        - parent_device_group_id: int | None
        - device_count: int
        - full_path: str (e.g., "Root > Region > Site")
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
            logger.info(f"Loaded {len(df)} device groups from MobiControl")
    except Exception as e:
        logger.warning(f"Could not load device groups from MC: {e}")
        return []

    if df.empty:
        return []

    # Build lookup maps for path computation
    parent_map: dict[int, int | None] = {}
    name_map: dict[int, str] = {}

    for _, row in df.iterrows():
        gid = int(row["DeviceGroupId"])
        parent_map[gid] = (
            int(row["ParentDeviceGroupId"]) if pd.notna(row["ParentDeviceGroupId"]) else None
        )
        name_map[gid] = str(row["GroupName"])

    def build_path(group_id: int) -> str:
        """Recursively build full path from parent chain."""
        path_parts: list[str] = []
        current: int | None = group_id
        visited: set[int] = set()

        while current is not None and current not in visited:
            visited.add(current)
            if current in name_map:
                path_parts.insert(0, name_map[current])
            current = parent_map.get(current)

        return " > ".join(path_parts)

    # Build result with full_path
    result: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        gid = int(row["DeviceGroupId"])
        result.append(
            {
                "device_group_id": gid,
                "group_name": str(row["GroupName"]),
                "parent_device_group_id": (
                    int(row["ParentDeviceGroupId"])
                    if pd.notna(row["ParentDeviceGroupId"])
                    else None
                ),
                "device_count": int(row["DeviceCount"]),
                "full_path": build_path(gid),
            }
        )

    return result


def build_hierarchy_tree(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert flat list of groups to nested tree structure.

    Args:
        groups: Flat list from load_device_groups_with_path()

    Returns:
        List of root nodes, each with nested 'children' arrays
    """
    if not groups:
        return []

    # Build lookup by ID, adding empty children list
    by_id: dict[int, dict[str, Any]] = {}
    for g in groups:
        by_id[g["device_group_id"]] = {**g, "children": []}

    roots: list[dict[str, Any]] = []

    for g in groups:
        node = by_id[g["device_group_id"]]
        parent_id = g["parent_device_group_id"]

        if parent_id is None or parent_id not in by_id:
            roots.append(node)
        else:
            by_id[parent_id]["children"].append(node)

    # Sort children by device_count descending at each level
    def sort_children(node: dict[str, Any]) -> None:
        node["children"].sort(key=lambda x: x["device_count"], reverse=True)
        for child in node["children"]:
            sort_children(child)

    for root in roots:
        sort_children(root)

    # Sort roots by device_count descending
    roots.sort(key=lambda x: x["device_count"], reverse=True)

    return roots


def get_device_ids_in_group(device_group_id: int, include_descendants: bool = True) -> list[int]:
    """
    Get all device IDs belonging to a device group.

    Args:
        device_group_id: The group to query
        include_descendants: If True, include devices in child groups (recursive)

    Returns:
        List of device IDs
    """
    engine = create_mc_engine()

    if include_descendants:
        # Recursive CTE to get all descendant groups
        sql = text("""
            WITH GroupHierarchy AS (
                SELECT DeviceGroupId
                FROM dbo.DeviceGroup
                WHERE DeviceGroupId = :group_id

                UNION ALL

                SELECT dg.DeviceGroupId
                FROM dbo.DeviceGroup dg
                INNER JOIN GroupHierarchy gh ON dg.ParentDeviceGroupId = gh.DeviceGroupId
            )
            SELECT DISTINCT d.DeviceId
            FROM dbo.DevInfo d
            WHERE d.DeviceGroupId IN (SELECT DeviceGroupId FROM GroupHierarchy)
        """)
    else:
        sql = text("""
            SELECT DISTINCT d.DeviceId
            FROM dbo.DevInfo d
            WHERE d.DeviceGroupId = :group_id
        """)

    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"group_id": device_group_id})
            device_ids = [int(row[0]) for row in result]
            logger.info(
                f"Found {len(device_ids)} devices in group {device_group_id} "
                f"(include_descendants={include_descendants})"
            )
            return device_ids
    except Exception as e:
        logger.error(f"Failed to get device IDs for group {device_group_id}: {e}")
        return []


def get_group_by_id(device_group_id: int) -> dict[str, Any] | None:
    """
    Get a single device group by ID with its full path.

    Returns:
        Dict with group info or None if not found
    """
    engine = create_mc_engine()

    sql = text("""
        SELECT
            dg.DeviceGroupId,
            dg.Name AS GroupName,
            dg.ParentDeviceGroupId,
            (SELECT COUNT(*) FROM dbo.DevInfo d WHERE d.DeviceGroupId = dg.DeviceGroupId) AS DeviceCount
        FROM dbo.DeviceGroup dg
        WHERE dg.DeviceGroupId = :group_id
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"group_id": device_group_id})
            row = result.fetchone()
            if not row:
                return None

            # Get full path by loading hierarchy
            groups = load_device_groups_with_path(limit=10000)
            group_lookup = {g["device_group_id"]: g for g in groups}

            if device_group_id in group_lookup:
                return group_lookup[device_group_id]

            # Fallback if not in hierarchy
            return {
                "device_group_id": int(row[0]),
                "group_name": str(row[1]),
                "parent_device_group_id": (int(row[2]) if row[2] is not None else None),
                "device_count": int(row[3]),
                "full_path": str(row[1]),
            }
    except Exception as e:
        logger.error(f"Failed to get group {device_group_id}: {e}")
        return None
