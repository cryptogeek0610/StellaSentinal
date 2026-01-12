"""
Model Governance - Versioning, A/B Testing, and Rollback.

This module manages the model lifecycle including:
- Model versioning with status tracking
- Shadow mode for A/B testing
- One-click rollback capability
- Metrics tracking per version
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""

    TRAINING = "training"
    VALIDATION = "validation"
    SHADOW = "shadow"        # Running alongside production for A/B testing
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Represents a versioned model."""

    version_id: str
    model_path: Path
    status: ModelStatus
    created_at: datetime
    promoted_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    training_reason: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version_id": self.version_id,
            "model_path": str(self.model_path),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "metrics": self.metrics,
            "config": self.config,
            "parent_version": self.parent_version,
            "training_reason": self.training_reason,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            model_path=Path(data["model_path"]),
            status=ModelStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            promoted_at=datetime.fromisoformat(data["promoted_at"]) if data.get("promoted_at") else None,
            deprecated_at=datetime.fromisoformat(data["deprecated_at"]) if data.get("deprecated_at") else None,
            metrics=data.get("metrics", {}),
            config=data.get("config", {}),
            parent_version=data.get("parent_version"),
            training_reason=data.get("training_reason", ""),
            notes=data.get("notes", ""),
        )


class ModelGovernanceRegistry:
    """
    Manages model lifecycle: versioning, promotion, and rollback.

    Tracks all model versions and their statuses, enabling:
    - Safe model deployment with shadow testing
    - Quick rollback on issues
    - Audit trail of model changes
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self._versions: dict[str, ModelVersion] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                    for version_data in data.get("versions", []):
                        version = ModelVersion.from_dict(version_data)
                        self._versions[version.version_id] = version
                logger.info(f"Loaded {len(self._versions)} model versions from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self._versions = {}
        else:
            logger.info("No existing registry found, starting fresh")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "versions": [v.to_dict() for v in self._versions.values()],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Registry saved")

    def register_model(
        self,
        model_path: Path,
        metrics: dict[str, Any],
        config: dict[str, Any],
        training_reason: str = "",
        version_id: Optional[str] = None,
    ) -> ModelVersion:
        """
        Register a new trained model.

        Args:
            model_path: Path to the saved model
            metrics: Training metrics (anomaly_rate, accuracy, etc.)
            config: Training configuration
            training_reason: Why this model was trained
            version_id: Optional custom version ID

        Returns:
            The registered ModelVersion
        """
        if version_id is None:
            version_id = f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            status=ModelStatus.VALIDATION,
            created_at=datetime.now(timezone.utc),
            metrics=metrics,
            config=config,
            parent_version=self.get_production_version_id(),
            training_reason=training_reason,
        )

        self._versions[version_id] = version
        self._save_registry()

        logger.info(f"Registered model version {version_id}")
        return version

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._versions.get(version_id)

    def get_production_version(self) -> Optional[ModelVersion]:
        """Get the current production model."""
        for v in self._versions.values():
            if v.status == ModelStatus.PRODUCTION:
                return v
        return None

    def get_production_version_id(self) -> Optional[str]:
        """Get the current production model version ID."""
        prod = self.get_production_version()
        return prod.version_id if prod else None

    def get_shadow_version(self) -> Optional[ModelVersion]:
        """Get the current shadow model (if any)."""
        for v in self._versions.values():
            if v.status == ModelStatus.SHADOW:
                return v
        return None

    def promote_to_shadow(self, version_id: str) -> ModelVersion:
        """
        Start A/B testing by promoting model to shadow mode.

        In shadow mode, the model runs alongside production but
        results are logged without affecting the main output.

        Args:
            version_id: Version to promote to shadow

        Returns:
            Updated ModelVersion
        """
        version = self._versions.get(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")

        if version.status not in (ModelStatus.VALIDATION, ModelStatus.DEPRECATED):
            raise ValueError(f"Cannot promote to shadow from status: {version.status}")

        # Demote any existing shadow model
        for v in self._versions.values():
            if v.status == ModelStatus.SHADOW:
                v.status = ModelStatus.VALIDATION
                logger.info(f"Demoted {v.version_id} from shadow")

        version.status = ModelStatus.SHADOW
        self._save_registry()

        logger.info(f"Promoted {version_id} to shadow mode")
        return version

    def promote_to_production(
        self,
        version_id: str,
        notes: str = "",
    ) -> ModelVersion:
        """
        Promote model to production.

        Demotes the current production model and activates the new one.

        Args:
            version_id: Version to promote
            notes: Notes about the promotion

        Returns:
            Updated ModelVersion
        """
        version = self._versions.get(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")

        if version.status not in (ModelStatus.SHADOW, ModelStatus.VALIDATION):
            raise ValueError(f"Cannot promote to production from status: {version.status}")

        # Demote current production
        current_prod = self.get_production_version()
        if current_prod:
            current_prod.status = ModelStatus.DEPRECATED
            current_prod.deprecated_at = datetime.now(timezone.utc)
            logger.info(f"Deprecated previous production model {current_prod.version_id}")

        # Promote new version
        version.status = ModelStatus.PRODUCTION
        version.promoted_at = datetime.now(timezone.utc)
        version.notes = notes
        self._save_registry()

        logger.info(f"Promoted {version_id} to production")
        return version

    def rollback(
        self,
        to_version_id: Optional[str] = None,
        reason: str = "",
    ) -> Optional[ModelVersion]:
        """
        Rollback to a previous model version.

        Args:
            to_version_id: Specific version to rollback to (defaults to parent)
            reason: Reason for rollback

        Returns:
            The restored ModelVersion, or None if rollback failed
        """
        current_prod = self.get_production_version()
        if not current_prod:
            logger.warning("No production model to rollback from")
            return None

        # Mark current as rolled back
        current_prod.status = ModelStatus.ROLLED_BACK
        current_prod.notes = f"Rolled back: {reason}"
        current_prod.deprecated_at = datetime.now(timezone.utc)

        # Find version to restore
        if to_version_id:
            target = self._versions.get(to_version_id)
        else:
            # Rollback to parent version
            target = self._versions.get(current_prod.parent_version) if current_prod.parent_version else None

        if not target:
            logger.error("No rollback target available")
            self._save_registry()
            return None

        if target.status in (ModelStatus.FAILED, ModelStatus.TRAINING):
            logger.error(f"Cannot rollback to version in status: {target.status}")
            self._save_registry()
            return None

        target.status = ModelStatus.PRODUCTION
        target.promoted_at = datetime.now(timezone.utc)
        target.notes = f"Restored via rollback from {current_prod.version_id}: {reason}"
        self._save_registry()

        logger.info(f"Rolled back to {target.version_id}")
        return target

    def mark_failed(
        self,
        version_id: str,
        reason: str = "",
    ) -> Optional[ModelVersion]:
        """Mark a model version as failed."""
        version = self._versions.get(version_id)
        if not version:
            return None

        version.status = ModelStatus.FAILED
        version.notes = f"Failed: {reason}"
        self._save_registry()

        logger.info(f"Marked {version_id} as failed: {reason}")
        return version

    def get_version_history(
        self,
        limit: int = 20,
    ) -> list[ModelVersion]:
        """Get model version history, most recent first."""
        versions = sorted(
            self._versions.values(),
            key=lambda v: v.created_at,
            reverse=True
        )
        return versions[:limit]

    def get_metrics_comparison(
        self,
        version_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Compare metrics across model versions."""
        comparison = {}
        for version_id in version_ids:
            version = self._versions.get(version_id)
            if version:
                comparison[version_id] = {
                    "status": version.status.value,
                    "created_at": version.created_at.isoformat(),
                    "metrics": version.metrics,
                }
        return comparison

    def cleanup_old_versions(
        self,
        keep_count: int = 10,
        keep_days: int = 90,
    ) -> int:
        """
        Clean up old deprecated/failed versions.

        Keeps production, shadow, and recent versions.

        Args:
            keep_count: Minimum versions to keep
            keep_days: Versions newer than this are kept

        Returns:
            Number of versions removed
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=keep_days)

        # Identify versions to remove
        removable_statuses = {ModelStatus.DEPRECATED, ModelStatus.FAILED, ModelStatus.ROLLED_BACK}
        candidates = [
            v for v in self._versions.values()
            if v.status in removable_statuses and v.created_at < cutoff
        ]

        # Sort by age, keep the most recent
        candidates.sort(key=lambda v: v.created_at, reverse=True)

        # Remove older ones beyond keep_count
        total_versions = len(self._versions)
        to_remove = []

        if total_versions > keep_count:
            excess = total_versions - keep_count
            to_remove = candidates[:min(excess, len(candidates))]

        for version in to_remove:
            del self._versions[version.version_id]
            logger.info(f"Cleaned up old version {version.version_id}")

        if to_remove:
            self._save_registry()

        return len(to_remove)


def create_model_registry(
    registry_path: Optional[Path] = None,
) -> ModelGovernanceRegistry:
    """
    Create a model governance registry.

    Args:
        registry_path: Path to registry file (default: models/registry.json)

    Returns:
        Configured ModelGovernanceRegistry
    """
    if registry_path is None:
        registry_path = Path("models/registry.json")

    return ModelGovernanceRegistry(registry_path)
