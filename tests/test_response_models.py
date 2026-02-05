"""Tests for Pydantic response models."""

from device_anomaly.api.models import (
    FeedbackResponse,
    LearnRemediationResponse,
    RemediationOutcomeResponse,
    SuccessResponse,
)


class TestSuccessResponse:
    def test_default_success_is_true(self):
        resp = SuccessResponse(message="ok")
        assert resp.success is True
        assert resp.message == "ok"

    def test_serialization(self):
        d = SuccessResponse(message="done").model_dump()
        assert d == {"success": True, "message": "done"}


class TestFeedbackResponse:
    def test_inherits_success(self):
        resp = FeedbackResponse(message="Feedback recorded")
        assert resp.success is True
        assert resp.message == "Feedback recorded"


class TestRemediationOutcomeResponse:
    def test_includes_outcome_id(self):
        resp = RemediationOutcomeResponse(message="Recorded", outcome_id=42)
        d = resp.model_dump()
        assert d["outcome_id"] == 42
        assert d["success"] is True


class TestLearnRemediationResponse:
    def test_with_current_confidence(self):
        resp = LearnRemediationResponse(
            message="Updated",
            learned_remediation_id=7,
            current_confidence=0.85,
        )
        d = resp.model_dump()
        assert d["learned_remediation_id"] == 7
        assert d["current_confidence"] == 0.85
        assert d["initial_confidence"] is None

    def test_with_initial_confidence(self):
        resp = LearnRemediationResponse(
            message="New pattern",
            learned_remediation_id=1,
            initial_confidence=0.6,
        )
        d = resp.model_dump()
        assert d["initial_confidence"] == 0.6
        assert d["current_confidence"] is None
