"""Base repository pattern for database access.

This module provides a generic repository pattern that can be extended
for specific entities. It implements common CRUD operations with
tenant isolation support.
"""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy.orm import Session

from device_anomaly.db.models import Base

# Type variable for generic repository
T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """Generic repository for database operations.

    Provides common CRUD operations for any SQLAlchemy model.
    Subclasses can override or extend these methods for specific behavior.

    Example:
        class AnomalyRepository(BaseRepository[Anomaly]):
            def __init__(self, session: Session):
                super().__init__(session, Anomaly)

            def get_by_severity(self, severity: str) -> List[Anomaly]:
                return self.session.query(self.model).filter_by(severity=severity).all()
    """

    def __init__(self, session: Session, model: type[T]):
        """Initialize the repository.

        Args:
            session: SQLAlchemy session for database operations
            model: The SQLAlchemy model class this repository manages
        """
        self.session = session
        self.model = model

    def get(self, id: Any) -> T | None:
        """Get a single entity by primary key.

        Args:
            id: The primary key value

        Returns:
            The entity if found, None otherwise
        """
        return self.session.query(self.model).get(id)

    def get_by_id(self, id: Any, tenant_id: str | None = None) -> T | None:
        """Get a single entity by ID with optional tenant filtering.

        Args:
            id: The entity ID
            tenant_id: Optional tenant ID for filtering

        Returns:
            The entity if found, None otherwise
        """
        query = self.session.query(self.model)

        # Get the primary key column name
        pk_columns = [c.name for c in self.model.__table__.primary_key.columns]
        if len(pk_columns) == 1:
            query = query.filter(getattr(self.model, pk_columns[0]) == id)

        # Add tenant filter if model has tenant_id and tenant_id is provided
        if tenant_id and hasattr(self.model, "tenant_id"):
            query = query.filter(self.model.tenant_id == tenant_id)

        return query.first()

    def get_all(
        self,
        tenant_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[T]:
        """Get all entities with optional pagination.

        Args:
            tenant_id: Optional tenant ID for filtering
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of entities
        """
        query = self.session.query(self.model)

        if tenant_id and hasattr(self.model, "tenant_id"):
            query = query.filter(self.model.tenant_id == tenant_id)

        query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        return query.all()

    def create(self, entity: T) -> T:
        """Create a new entity.

        Args:
            entity: The entity to create

        Returns:
            The created entity with generated ID
        """
        self.session.add(entity)
        self.session.flush()  # Flush to get generated ID
        return entity

    def create_many(self, entities: list[T]) -> list[T]:
        """Create multiple entities.

        Args:
            entities: List of entities to create

        Returns:
            List of created entities
        """
        self.session.add_all(entities)
        self.session.flush()
        return entities

    def update(self, entity: T) -> T:
        """Update an existing entity.

        Args:
            entity: The entity with updated values

        Returns:
            The updated entity
        """
        self.session.merge(entity)
        self.session.flush()
        return entity

    def delete(self, entity: T) -> None:
        """Delete an entity.

        Args:
            entity: The entity to delete
        """
        self.session.delete(entity)
        self.session.flush()

    def delete_by_id(self, id: Any, tenant_id: str | None = None) -> bool:
        """Delete an entity by ID.

        Args:
            id: The entity ID
            tenant_id: Optional tenant ID for validation

        Returns:
            True if deleted, False if not found
        """
        entity = self.get_by_id(id, tenant_id)
        if entity:
            self.delete(entity)
            return True
        return False

    def count(self, tenant_id: str | None = None) -> int:
        """Count all entities.

        Args:
            tenant_id: Optional tenant ID for filtering

        Returns:
            Number of entities
        """
        query = self.session.query(self.model)

        if tenant_id and hasattr(self.model, "tenant_id"):
            query = query.filter(self.model.tenant_id == tenant_id)

        return query.count()

    def exists(self, id: Any, tenant_id: str | None = None) -> bool:
        """Check if an entity exists.

        Args:
            id: The entity ID
            tenant_id: Optional tenant ID for filtering

        Returns:
            True if exists, False otherwise
        """
        return self.get_by_id(id, tenant_id) is not None

    def filter_by(
        self,
        tenant_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        **kwargs: Any,
    ) -> list[T]:
        """Filter entities by column values.

        Args:
            tenant_id: Optional tenant ID for filtering
            limit: Maximum number of records
            offset: Number of records to skip
            **kwargs: Column=value pairs to filter by

        Returns:
            List of matching entities
        """
        query = self.session.query(self.model)

        if tenant_id and hasattr(self.model, "tenant_id"):
            query = query.filter(self.model.tenant_id == tenant_id)

        for key, value in kwargs.items():
            if hasattr(self.model, key):
                query = query.filter(getattr(self.model, key) == value)

        query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        return query.all()
