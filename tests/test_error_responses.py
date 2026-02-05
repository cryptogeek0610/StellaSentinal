"""Tests for structured error responses."""

from device_anomaly.api.errors import (
    ErrorDetail,
    ErrorResponse,
    _build_error_response,
)
from device_anomaly.api.request_context import clear_request_context, set_request_context


class TestBuildErrorResponse:
    def setup_method(self):
        clear_request_context()

    def teardown_method(self):
        clear_request_context()

    def test_includes_request_id_from_context(self):
        set_request_context(
            request_id="req-abc",
            tenant_id="t",
            user_id=None,
            user_role=None,
        )
        body = _build_error_response(404, "Not found")
        assert body["error"]["request_id"] == "req-abc"

    def test_default_code_from_status(self):
        body = _build_error_response(400, "bad")
        assert body["error"]["code"] == "bad_request"

        body = _build_error_response(401, "unauth")
        assert body["error"]["code"] == "unauthorized"

        body = _build_error_response(403, "nope")
        assert body["error"]["code"] == "forbidden"

        body = _build_error_response(404, "gone")
        assert body["error"]["code"] == "not_found"

        body = _build_error_response(429, "slow down")
        assert body["error"]["code"] == "rate_limited"

        body = _build_error_response(500, "oops")
        assert body["error"]["code"] == "internal_error"

    def test_custom_code_overrides_default(self):
        body = _build_error_response(500, "db down", code="database_error")
        assert body["error"]["code"] == "database_error"

    def test_message_preserved(self):
        body = _build_error_response(400, "Missing field X")
        assert body["error"]["message"] == "Missing field X"

    def test_unknown_status_code(self):
        body = _build_error_response(418, "I'm a teapot")
        assert body["error"]["code"] == "error"

    def test_request_id_none_when_no_context(self):
        clear_request_context()
        body = _build_error_response(500, "fail")
        assert body["error"]["request_id"] is None


class TestErrorModels:
    def test_error_detail_serialization(self):
        detail = ErrorDetail(code="not_found", message="Not found", request_id="r-1")
        d = detail.model_dump()
        assert d == {"code": "not_found", "message": "Not found", "request_id": "r-1"}

    def test_error_response_envelope(self):
        resp = ErrorResponse(error=ErrorDetail(code="bad_request", message="Oops"))
        d = resp.model_dump()
        assert "error" in d
        assert d["error"]["code"] == "bad_request"
        assert d["error"]["request_id"] is None
