"""FastAPI dependencies for database sessions and repositories."""
from __future__ import annotations

import logging
import os
from typing import Generator, Optional, Sequence

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from device_anomaly.api.request_context import (
    RequestUser,
    get_tenant_id_value,
    get_user_context,
    set_tenant_id,
)
from device_anomaly.database.connection import get_results_db_session

logger = logging.getLogger(__name__)

# Environment-based mock mode control
# In production, set DISABLE_MOCK_MODE=true to completely disable mock mode
_mock_mode_disabled = os.getenv("DISABLE_MOCK_MODE", "false").lower() == "true"
_app_env = os.getenv("APP_ENV", "local")
_require_auth = os.getenv("REQUIRE_API_KEY", "false").lower() == "true" or _app_env == "production"


def get_mock_mode(request: Request) -> bool:
    """Check if Mock Mode is enabled via X-Mock-Mode header.

    This allows the frontend to toggle mock mode by sending a header,
    keeping the backend stateless.

    Security: Mock mode is automatically disabled in production environments
    unless explicitly allowed. Set DISABLE_MOCK_MODE=true to fully disable.
    """
    # Always disable mock mode if explicitly disabled via environment
    if _mock_mode_disabled:
        return False

    # In production, don't allow mock mode unless explicitly enabled
    if _app_env == "production":
        logger.warning(
            "Mock mode header received in production environment. "
            "Mock mode is disabled in production. Set APP_ENV to 'local' or 'development' to enable."
        )
        return False

    return request.headers.get("X-Mock-Mode", "false").lower() == "true"


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting results database session.

    Use this for the existing anomaly results storage.
    """
    db = get_results_db_session()
    try:
        yield db
    finally:
        db.close()


def get_backend_db() -> Generator[Session, None, None]:
    """Dependency for getting backend database session (SQL Server).

    Use this for the new enterprise backend database with:
    - Multi-tenant support
    - Devices from multiple sources
    - Baselines and explanations
    """
    from device_anomaly.db.session import DatabaseSession

    db = DatabaseSession()
    session = None
    try:
        session = db.get_session()
        yield session
        session.commit()
    except Exception:
        if session:
            session.rollback()
        raise
    finally:
        if session:
            session.close()


# Repository factory dependencies
def get_anomaly_repository(session: Session):
    """Create an AnomalyRepository for the given session."""
    from device_anomaly.db.repositories import AnomalyRepository
    return AnomalyRepository(session)


def get_baseline_repository(session: Session):
    """Create a BaselineRepository for the given session."""
    from device_anomaly.db.repositories import BaselineRepository
    return BaselineRepository(session)


def get_device_repository(session: Session):
    """Create a DeviceRepository for the given session."""
    from device_anomaly.db.repositories import DeviceRepository
    return DeviceRepository(session)


# Tenant context (for future multi-tenant support)
class TenantContext:
    """Thread-safe tenant context using contextvars.

    Safe for use in async ASGI servers like uvicorn where
    multiple requests may be processed concurrently.
    """

    @classmethod
    def set(cls, tenant_id: str) -> None:
        """Set the current tenant context."""
        set_tenant_id(tenant_id)

    @classmethod
    def get(cls) -> Optional[str]:
        """Get the current tenant ID."""
        return get_tenant_id_value()

    @classmethod
    def clear(cls) -> None:
        """Clear the tenant context."""
        set_tenant_id(None)


def get_tenant_id(allow_default: bool = True) -> str:
    """Get the current tenant ID from context.

    In a real implementation, this would extract the tenant from:
    - JWT token claims
    - Request headers
    - Session data

    Args:
        allow_default: If True, returns "default" when no tenant is set.
                      If False, raises ValueError. Set to False in production.

    Returns:
        The current tenant ID.

    Raises:
        ValueError: If no tenant context and allow_default is False.
    """
    tenant_id = TenantContext.get()
    if tenant_id:
        return tenant_id

    if not allow_default:
        raise ValueError("Tenant context not established. Authentication required.")

    # Default tenant for development only
    return "default"


def get_current_user() -> RequestUser:
    """Return the current request user context."""
    return get_user_context()


def require_role(allowed_roles: Sequence[str]):
    """Dependency factory enforcing role-based access control."""
    def _dependency() -> RequestUser:
        user = get_user_context()
        if user.user_id is None and _require_auth:
            raise HTTPException(status_code=401, detail="Authentication required")
        if user.user_id is None and not _require_auth:
            return user
        if user.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user

    return _dependency
