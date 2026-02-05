"""Structured error responses for the Stella Sentinel API.

Provides a standard error envelope so all error responses share the
same shape regardless of which route raised them.

Response shape:
    {
        "error": {
            "code": "not_found",
            "message": "Anomaly not found",
            "request_id": "abc-123"
        }
    }
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from device_anomaly.api.request_context import get_request_context

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    """Standard error detail envelope."""

    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Top-level error response wrapper."""

    error: ErrorDetail


# ---- Helpers ----

_STATUS_CODE_DEFAULTS: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


def _request_id() -> str | None:
    """Best-effort retrieval of current request id from context."""
    try:
        ctx = get_request_context()
        return ctx.request_id if ctx else None
    except Exception:
        return None


def _build_error_response(status_code: int, message: str, code: str | None = None) -> dict:
    """Build a standard error dict."""
    return ErrorResponse(
        error=ErrorDetail(
            code=code or _STATUS_CODE_DEFAULTS.get(status_code, "error"),
            message=message,
            request_id=_request_id(),
        )
    ).model_dump()


# ---- Exception handlers (register in main.py) ----


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert FastAPI HTTPException into structured error response."""
    # If detail is already a dict (e.g. from correlations structured errors), pass through
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    message = str(exc.detail) if exc.detail else "An error occurred"
    return JSONResponse(
        status_code=exc.status_code,
        content=_build_error_response(exc.status_code, message),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions. Prevents internal details from leaking."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=_build_error_response(
            500,
            "An internal error occurred. Please try again later.",
            code="internal_error",
        ),
    )
