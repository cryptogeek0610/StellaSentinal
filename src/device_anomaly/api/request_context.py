from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_tenant_id_ctx: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)
_user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_user_role_ctx: ContextVar[Optional[str]] = ContextVar("user_role", default=None)


@dataclass(frozen=True)
class RequestUser:
    user_id: Optional[str]
    role: str
    tenant_id: Optional[str]


@dataclass(frozen=True)
class RequestContext:
    request_id: Optional[str]
    tenant_id: Optional[str]
    user_id: Optional[str]
    role: str


def set_request_context(
    *,
    request_id: Optional[str],
    tenant_id: Optional[str],
    user_id: Optional[str],
    user_role: Optional[str],
) -> None:
    _request_id_ctx.set(request_id)
    _tenant_id_ctx.set(tenant_id)
    _user_id_ctx.set(user_id)
    _user_role_ctx.set(user_role)


def set_request_id(request_id: Optional[str]) -> None:
    _request_id_ctx.set(request_id)


def set_tenant_id(tenant_id: Optional[str]) -> None:
    _tenant_id_ctx.set(tenant_id)


def set_user_context(user_id: Optional[str], role: Optional[str]) -> None:
    _user_id_ctx.set(user_id)
    _user_role_ctx.set(role)


def clear_request_context() -> None:
    _request_id_ctx.set(None)
    _tenant_id_ctx.set(None)
    _user_id_ctx.set(None)
    _user_role_ctx.set(None)


def get_request_id() -> Optional[str]:
    return _request_id_ctx.get()


def get_tenant_id_value() -> Optional[str]:
    return _tenant_id_ctx.get()


def get_user_context() -> RequestUser:
    role = _user_role_ctx.get() or "viewer"
    return RequestUser(
        user_id=_user_id_ctx.get(),
        role=role,
        tenant_id=_tenant_id_ctx.get(),
    )


def get_request_context() -> RequestContext:
    role = _user_role_ctx.get() or "viewer"
    return RequestContext(
        request_id=_request_id_ctx.get(),
        tenant_id=_tenant_id_ctx.get(),
        user_id=_user_id_ctx.get(),
        role=role,
    )
