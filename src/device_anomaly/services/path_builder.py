"""
PATH Builder Service.

Builds hierarchical PATH strings from MobiControl DeviceGroup relationships.
Used for intelligent device grouping in the Security Posture dashboard.

Example PATH: "North America / East Region / Store-NYC-001"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from device_anomaly.data_access.db_connection import create_mc_engine

logger = logging.getLogger(__name__)

PATH_SEPARATOR = " / "


@dataclass
class PathNode:
    """Represents a node in the device group hierarchy."""

    group_id: int
    name: str
    parent_id: Optional[int] = None
    device_count: int = 0
    children: List[PathNode] = field(default_factory=list)
    full_path: str = ""
    depth: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "path_id": f"group-{self.group_id}",
            "path_name": self.name,
            "full_path": self.full_path,
            "parent_path_id": f"group-{self.parent_id}" if self.parent_id else None,
            "depth": self.depth,
            "device_count": self.device_count,
            "children": [c.to_dict() for c in self.children],
        }


class PathBuilder:
    """
    Builds and caches PATH hierarchy from MobiControl DeviceGroup table.

    Usage:
        builder = PathBuilder()
        builder.load_hierarchy()
        path = builder.get_device_path(device_group_id=123)
        # Returns: "North America / East Region / Store-NYC-001"
    """

    def __init__(self):
        self._nodes: Dict[int, PathNode] = {}
        self._roots: List[PathNode] = []
        self._loaded = False

    def load_hierarchy(self, limit: int = 5000) -> None:
        """Load device group hierarchy from MobiControl database."""
        try:
            engine = create_mc_engine()

            sql = f"""
                SELECT TOP ({int(limit)})
                    dg.DeviceGroupId,
                    dg.Name AS GroupName,
                    dg.ParentDeviceGroupId,
                    (SELECT COUNT(*) FROM dbo.DevInfo d
                     WHERE d.DeviceGroupId = dg.DeviceGroupId) AS DeviceCount
                FROM dbo.DeviceGroup dg
                WHERE dg.Name IS NOT NULL
                  AND dg.Name != ''
                ORDER BY dg.ParentDeviceGroupId, dg.Name
            """

            with engine.connect() as conn:
                df = pd.read_sql(sql, conn)

            self._build_tree(df)
            self._loaded = True
            logger.info(f"Loaded {len(self._nodes)} device groups into PATH hierarchy")

        except Exception as e:
            logger.warning(f"Could not load device group hierarchy: {e}")
            self._loaded = False

    def _build_tree(self, df: pd.DataFrame) -> None:
        """Build tree structure from flat DataFrame."""
        self._nodes.clear()
        self._roots.clear()

        # First pass: create all nodes
        for _, row in df.iterrows():
            group_id = int(row["DeviceGroupId"])
            parent_id = row["ParentDeviceGroupId"]
            if pd.notna(parent_id):
                parent_id = int(parent_id)
            else:
                parent_id = None

            self._nodes[group_id] = PathNode(
                group_id=group_id,
                name=str(row["GroupName"]),
                parent_id=parent_id,
                device_count=int(row["DeviceCount"]) if pd.notna(row["DeviceCount"]) else 0,
            )

        # Second pass: build parent-child relationships
        for node in self._nodes.values():
            if node.parent_id is None:
                self._roots.append(node)
            elif node.parent_id in self._nodes:
                self._nodes[node.parent_id].children.append(node)
            else:
                # Parent not found, treat as root
                self._roots.append(node)

        # Third pass: compute full paths and depths
        self._compute_paths(self._roots, "", 0)

    def _compute_paths(self, nodes: List[PathNode], parent_path: str, depth: int) -> None:
        """Recursively compute full paths for all nodes."""
        for node in nodes:
            if parent_path:
                node.full_path = f"{parent_path}{PATH_SEPARATOR}{node.name}"
            else:
                node.full_path = node.name
            node.depth = depth
            self._compute_paths(node.children, node.full_path, depth + 1)

    def get_device_path(self, device_group_id: int) -> Optional[str]:
        """
        Get full PATH string for a device group ID.

        Args:
            device_group_id: The DeviceGroupId from DevInfo table.

        Returns:
            Full path string like "North America / East Region / Store-NYC-001"
            or None if not found.
        """
        if not self._loaded:
            self.load_hierarchy()

        node = self._nodes.get(device_group_id)
        if node:
            return node.full_path
        return None

    def get_hierarchy_tree(self) -> List[dict]:
        """
        Get the full hierarchy as a nested dictionary structure.

        Returns:
            List of root nodes with nested children.
        """
        if not self._loaded:
            self.load_hierarchy()

        return [root.to_dict() for root in self._roots]

    def get_all_paths(self) -> Dict[int, str]:
        """
        Get mapping of all group IDs to their full paths.

        Returns:
            Dict mapping DeviceGroupId to full path string.
        """
        if not self._loaded:
            self.load_hierarchy()

        return {gid: node.full_path for gid, node in self._nodes.items()}

    def get_nodes_at_depth(self, depth: int) -> List[PathNode]:
        """
        Get all nodes at a specific depth level.

        Args:
            depth: 0 = root, 1 = first level children, etc.

        Returns:
            List of PathNode objects at that depth.
        """
        if not self._loaded:
            self.load_hierarchy()

        return [node for node in self._nodes.values() if node.depth == depth]

    def get_children(self, group_id: int) -> List[PathNode]:
        """
        Get direct children of a group.

        Args:
            group_id: Parent DeviceGroupId.

        Returns:
            List of child PathNode objects.
        """
        if not self._loaded:
            self.load_hierarchy()

        node = self._nodes.get(group_id)
        if node:
            return node.children
        return []

    def get_descendants_count(self, group_id: int) -> int:
        """
        Get total device count including all descendants.

        Args:
            group_id: DeviceGroupId to count from.

        Returns:
            Total device count across this group and all descendants.
        """
        if not self._loaded:
            self.load_hierarchy()

        node = self._nodes.get(group_id)
        if not node:
            return 0

        return self._count_recursive(node)

    def _count_recursive(self, node: PathNode) -> int:
        """Recursively count devices in node and all children."""
        total = node.device_count
        for child in node.children:
            total += self._count_recursive(child)
        return total


# Singleton instance for caching
_path_builder: Optional[PathBuilder] = None


def get_path_builder() -> PathBuilder:
    """Get or create the singleton PathBuilder instance."""
    global _path_builder
    if _path_builder is None:
        _path_builder = PathBuilder()
    return _path_builder


def get_device_path(device_group_id: int) -> Optional[str]:
    """
    Convenience function to get PATH for a device group.

    Args:
        device_group_id: The DeviceGroupId from DevInfo.

    Returns:
        Full path string or None.
    """
    return get_path_builder().get_device_path(device_group_id)


def get_path_hierarchy() -> List[dict]:
    """
    Convenience function to get the full hierarchy tree.

    Returns:
        Nested hierarchy structure for API response.
    """
    return get_path_builder().get_hierarchy_tree()
