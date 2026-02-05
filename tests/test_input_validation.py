"""Tests for input validation helpers."""

import pytest
from fastapi import HTTPException

from device_anomaly.api.routes.devices import validate_attribute_name


class TestValidateAttributeName:
    def test_accepts_simple_names(self):
        assert validate_attribute_name("Store") == "Store"
        assert validate_attribute_name("Warehouse") == "Warehouse"
        assert validate_attribute_name("Location") == "Location"

    def test_accepts_underscores_and_hyphens(self):
        assert validate_attribute_name("store_id") == "store_id"
        assert validate_attribute_name("my-attribute") == "my-attribute"

    def test_accepts_dots_and_spaces(self):
        assert validate_attribute_name("Store Name") == "Store Name"
        assert validate_attribute_name("config.key") == "config.key"

    def test_accepts_numbers(self):
        assert validate_attribute_name("attr123") == "attr123"
        assert validate_attribute_name("123") == "123"

    def test_rejects_empty_string(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_attribute_name("")
        assert exc_info.value.status_code == 400

    def test_rejects_special_characters(self):
        for dangerous in [
            "'; DROP TABLE--",
            "attr; rm -rf /",
            "name<script>",
            "col$(whoami)",
            "name\x00null",
            "path/../../../etc/passwd",
            "key=value&other=x",
        ]:
            with pytest.raises(HTTPException) as exc_info:
                validate_attribute_name(dangerous)
            assert exc_info.value.status_code == 400, f"Should reject: {dangerous!r}"

    def test_rejects_too_long(self):
        with pytest.raises(HTTPException):
            validate_attribute_name("a" * 129)

    def test_accepts_max_length(self):
        name = "a" * 128
        assert validate_attribute_name(name) == name

    def test_custom_param_in_error_message(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_attribute_name("bad<>name", param="group_by")
        assert "group_by" in str(exc_info.value.detail)
