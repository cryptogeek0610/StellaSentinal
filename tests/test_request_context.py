"""Tests for the request context module."""

from device_anomaly.api.request_context import (
    RequestContext,
    RequestUser,
    clear_request_context,
    get_request_context,
    get_request_id,
    get_tenant_id_value,
    get_user_context,
    set_request_context,
    set_request_id,
    set_tenant_id,
    set_user_context,
)


class TestSetAndGetContext:
    def test_set_and_get_full_context(self):
        set_request_context(
            request_id="req-123",
            tenant_id="tenant-a",
            user_id="user-1",
            user_role="admin",
        )
        ctx = get_request_context()
        assert ctx == RequestContext(
            request_id="req-123",
            tenant_id="tenant-a",
            user_id="user-1",
            role="admin",
        )
        clear_request_context()

    def test_clear_resets_all_fields(self):
        set_request_context(
            request_id="req-999",
            tenant_id="t",
            user_id="u",
            user_role="admin",
        )
        clear_request_context()
        ctx = get_request_context()
        assert ctx.request_id is None
        assert ctx.tenant_id is None
        assert ctx.user_id is None
        assert ctx.role == "viewer"  # default

    def test_default_role_is_viewer(self):
        set_request_context(
            request_id=None,
            tenant_id=None,
            user_id=None,
            user_role=None,
        )
        ctx = get_request_context()
        assert ctx.role == "viewer"
        clear_request_context()


class TestIndividualSetters:
    def test_set_request_id(self):
        set_request_id("abc")
        assert get_request_id() == "abc"
        clear_request_context()

    def test_set_tenant_id(self):
        set_tenant_id("tenant-x")
        assert get_tenant_id_value() == "tenant-x"
        clear_request_context()

    def test_set_user_context(self):
        set_user_context("u-42", "analyst")
        user = get_user_context()
        assert user == RequestUser(user_id="u-42", role="analyst", tenant_id=None)
        clear_request_context()


class TestDataclasses:
    def test_request_context_is_frozen(self):
        import pytest

        ctx = RequestContext(request_id="r", tenant_id="t", user_id="u", role="admin")
        with pytest.raises(AttributeError):
            ctx.role = "viewer"  # type: ignore[misc]

    def test_request_user_is_frozen(self):
        import pytest

        user = RequestUser(user_id="u", role="admin", tenant_id="t")
        with pytest.raises(AttributeError):
            user.role = "viewer"  # type: ignore[misc]
