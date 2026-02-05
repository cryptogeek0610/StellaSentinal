from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
_tenant_id_ctx: ContextVar[str | None] = ContextVar("tenant_id", default=None)
_user_id_ctx: ContextVar[str | None] = ContextVar("user_id", default=None)
_user_role_ctx: ContextVar[str | None] = ContextVar("user_role", default=None)


@dataclass(frozen=True)
class RequestUser:
    user_id: str | None
    role: str
    tenant_id: str | None


@dataclass(frozen=True)
class RequestContext:
    request_id: str | None
    tenant_id: str | None
    user_id: str | None
    role: str


def set_request_context(
    *,
    request_id: str | None,
    tenant_id: str | None,
    user_id: str | None,
    user_role: str | None,
) -> None:
    _request_id_ctx.set(request_id)
    _tenant_id_ctx.set(tenant_id)
    _user_id_ctx.set(user_id)
    _user_role_ctx.set(user_role)


def set_request_id(request_id: str | None) -> None:
    _request_id_ctx.set(request_id)


def set_tenant_id(tenant_id: str | None) -> None:
    _tenant_id_ctx.set(tenant_id)


def set_user_context(user_id: str | None, role: str | None) -> None:
    _user_id_ctx.set(user_id)
    _user_role_ctx.set(role)


def clear_request_context() -> None:
    _request_id_ctx.set(None)
    _tenant_id_ctx.set(None)
    _user_id_ctx.set(None)
    _user_role_ctx.set(None)


def get_request_id() -> str | None:
    return _request_id_ctx.get()


def get_tenant_id_value() -> str | None:
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
