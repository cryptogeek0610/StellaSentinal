"""Unit tests for NextBestActionEngine."""
import pytest

from projects.SOTI_Advanced_Analytics_Plus.ai_service.nba_engine import NextBestActionEngine


@pytest.fixture
def engine():
    return NextBestActionEngine()


class TestRecommendConnectivity:
    def test_profile_rollback(self, engine):
        anomaly = {
            "type": "connectivity",
            "root_cause_hint": "Profile Misconfiguration",
            "summary": "Devices losing connectivity after profile push",
            "affected_cohort": "Site-A",
        }
        result = engine.recommend(anomaly)
        assert result["action"] == "Rollback Profile"
        assert result["target"] == "Site-A"
        assert result["risk"] == "Low"

    def test_connectivity_without_profile_hint(self, engine):
        anomaly = {
            "type": "connectivity",
            "root_cause_hint": "DNS Failure",
            "summary": "Intermittent connectivity loss",
            "affected_cohort": "Site-B",
        }
        result = engine.recommend(anomaly)
        assert result["action"] == "Manual Investigation Required"


class TestRecommendHardware:
    def test_thermal_throttle(self, engine):
        anomaly = {
            "type": "hardware",
            "root_cause_hint": "Battery Degradation",
            "summary": "Thermal Warning on 14 devices",
            "affected_cohort": "Warehouse-Fleet",
        }
        result = engine.recommend(anomaly)
        assert result["action"] == "Throttle Background Apps"
        assert result["target"] == "Warehouse-Fleet"
        assert result["risk"] == "Zero"

    def test_hardware_without_thermal(self, engine):
        anomaly = {
            "type": "hardware",
            "root_cause_hint": "Disk Failure",
            "summary": "Storage degradation detected",
            "affected_cohort": "Office-Fleet",
        }
        result = engine.recommend(anomaly)
        assert result["action"] == "Manual Investigation Required"


class TestRecommendFallback:
    def test_unknown_type(self, engine):
        anomaly = {
            "type": "unknown",
            "root_cause_hint": "N/A",
            "summary": "Unclassified issue",
            "affected_cohort": "All",
        }
        result = engine.recommend(anomaly)
        assert result["action"] == "Manual Investigation Required"
        assert result["target"] == "Admin"


class TestRecommendBadInput:
    def test_missing_type_key(self, engine):
        result = engine.recommend({"summary": "something"})
        assert result["action"] == "Manual Investigation Required"

    def test_missing_all_keys(self, engine):
        result = engine.recommend({})
        assert result["action"] == "Manual Investigation Required"

    def test_none_input(self, engine):
        result = engine.recommend(None)
        assert result["action"] == "Manual Investigation Required"

    def test_string_input(self, engine):
        result = engine.recommend("not a dict")
        assert result["action"] == "Manual Investigation Required"

    def test_empty_string_values(self, engine):
        anomaly = {
            "type": "",
            "root_cause_hint": "",
            "summary": "",
            "affected_cohort": "",
        }
        result = engine.recommend(anomaly)
        assert result["action"] == "Manual Investigation Required"
