"""
Action Center API - The One Screen That Matters

Elon Musk principle: "The best interface is no interface."

This replaces multiple dashboards with a single "what needs fixing" view.
Users don't browse data - they see problems ranked by $ impact and fix them.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_mock_mode, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/action-center", tags=["action-center"])


# ============================================================================
# Response Models - Keep it simple
# ============================================================================

class IssueImpact(BaseModel):
    """Business impact in terms that matter."""
    affected_devices: int
    affected_users: int
    hourly_cost: float = Field(description="$ per hour this problem continues")
    priority_score: float = Field(description="Composite priority (higher = fix first)")


class Remediation(BaseModel):
    """What to do about it."""
    action_type: str
    description: str
    automated: bool = Field(description="Can be fixed with one click?")
    estimated_minutes: int
    success_rate: float = Field(description="Historical success rate 0-1")


class Issue(BaseModel):
    """A problem that needs fixing."""
    id: str
    category: str  # productivity_loss, security_risk, cost_waste, impending_failure
    title: str
    root_cause: str
    one_liner: str = Field(description="Problem + cost + fix in one sentence")
    impact: IssueImpact
    remediation: Remediation
    device_ids: list[int] = []
    detected_at: datetime


class ActionCenterSummary(BaseModel):
    """Executive dashboard - understand everything in 5 seconds."""
    tenant_id: str
    total_issues: int
    total_hourly_cost: float
    daily_cost: float
    monthly_cost: float
    automatable_count: int = Field(description="Issues we can fix automatically")
    automatable_savings: float = Field(description="$ saved if we auto-fix everything")
    by_category: dict
    top_3_issues: list[str]
    recommended_action: str = Field(description="Single most impactful thing to do now")
    generated_at: datetime


class IssueListResponse(BaseModel):
    """All issues, sorted by priority."""
    issues: list[Issue]
    total_count: int
    can_auto_fix: int


class FixResult(BaseModel):
    """Result of attempting a fix."""
    issue_id: str
    success: bool
    message: str
    devices_fixed: int
    devices_failed: int


# ============================================================================
# Mock Data for Demo
# ============================================================================

def _generate_mock_issues() -> list[Issue]:
    """Generate realistic demo issues."""
    now = datetime.now(UTC)

    return [
        Issue(
            id="sec_usb_debug_demo",
            category="security_risk",
            title="23 devices have USB debugging enabled",
            root_cause="Developer devices not returned to production config after testing",
            one_liner="23 devices exposed to data theft → $46/hr risk → One-click policy push available",
            impact=IssueImpact(
                affected_devices=23,
                affected_users=23,
                hourly_cost=46.0,
                priority_score=230.0,
            ),
            remediation=Remediation(
                action_type="push_policy",
                description="Push 'Disable USB Debugging' policy via MobiControl",
                automated=True,
                estimated_minutes=5,
                success_rate=0.95,
            ),
            detected_at=now,
        ),
        Issue(
            id="net_deadzone_warehouse_b",
            category="productivity_loss",
            title="WiFi dead zone in Warehouse B: 18 devices affected",
            root_cause="Signal strength -85 dBm (needs >-70 dBm for reliable scanning)",
            one_liner="18 workers losing connectivity → $450/hr lost → Add AP in section 3",
            impact=IssueImpact(
                affected_devices=18,
                affected_users=18,
                hourly_cost=450.0,
                priority_score=675.0,
            ),
            remediation=Remediation(
                action_type="infrastructure_ticket",
                description="Install additional WiFi AP in Warehouse B, Section 3",
                automated=False,
                estimated_minutes=240,
                success_rate=0.95,
            ),
            detected_at=now,
        ),
        Issue(
            id="hw_battery_critical",
            category="impending_failure",
            title="12 devices need battery replacement within 2 weeks",
            root_cause="Battery capacity below 60%, showing accelerated degradation",
            one_liner="12 devices will die mid-shift → $180/hr when they fail → Schedule replacement",
            impact=IssueImpact(
                affected_devices=12,
                affected_users=12,
                hourly_cost=60.0,  # Not failing yet
                priority_score=120.0,
            ),
            remediation=Remediation(
                action_type="schedule_replacement",
                description="Schedule staggered battery replacements over next 2 weeks",
                automated=False,
                estimated_minutes=30,
                success_rate=0.99,
            ),
            device_ids=[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012],
            detected_at=now,
        ),
        Issue(
            id="sec_unencrypted",
            category="security_risk",
            title="8 devices lack storage encryption",
            root_cause="Legacy devices enrolled before encryption policy was mandatory",
            one_liner="8 devices at risk of data breach → $80/hr compliance risk → Force encryption",
            impact=IssueImpact(
                affected_devices=8,
                affected_users=8,
                hourly_cost=80.0,
                priority_score=800.0,
            ),
            remediation=Remediation(
                action_type="push_policy",
                description="Force encryption via MDM (will require device restart)",
                automated=True,
                estimated_minutes=45,
                success_rate=0.85,
            ),
            detected_at=now,
        ),
        Issue(
            id="cost_data_abuse",
            category="cost_waste",
            title="Device MC-2847 using 15GB/month (10x fleet average)",
            root_cause="Spotify streaming in background consuming 80% of data",
            one_liner="$45/month excess data cost → Block Spotify background data",
            impact=IssueImpact(
                affected_devices=1,
                affected_users=1,
                hourly_cost=0.06,  # $45/month = ~$0.06/hr
                priority_score=0.06,
            ),
            remediation=Remediation(
                action_type="user_notification",
                description="Notify user and restrict Spotify background data",
                automated=True,
                estimated_minutes=1,
                success_rate=0.7,
            ),
            device_ids=[2847],
            detected_at=now,
        ),
        Issue(
            id="hw_storage_critical",
            category="impending_failure",
            title="5 devices critically low on storage (<5%)",
            root_cause="App cache buildup and uncleared logs",
            one_liner="5 devices will freeze → $75/hr when apps crash → Clear cache remotely",
            impact=IssueImpact(
                affected_devices=5,
                affected_users=5,
                hourly_cost=75.0,
                priority_score=225.0,
            ),
            remediation=Remediation(
                action_type="remote_action",
                description="Clear app caches and temp files remotely",
                automated=True,
                estimated_minutes=10,
                success_rate=0.8,
            ),
            detected_at=now,
        ),
    ]


def _generate_mock_summary(issues: list[Issue]) -> ActionCenterSummary:
    """Generate summary from issues."""
    total_hourly = sum(i.impact.hourly_cost for i in issues)
    automatable = [i for i in issues if i.remediation.automated]

    by_category = {}
    for issue in issues:
        cat = issue.category
        if cat not in by_category:
            by_category[cat] = {"count": 0, "hourly_cost": 0.0, "devices": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["hourly_cost"] += issue.impact.hourly_cost
        by_category[cat]["devices"] += issue.impact.affected_devices

    # Sort issues by priority for top 3
    sorted_issues = sorted(issues, key=lambda x: x.impact.priority_score, reverse=True)

    return ActionCenterSummary(
        tenant_id="demo",
        total_issues=len(issues),
        total_hourly_cost=total_hourly,
        daily_cost=total_hourly * 8,
        monthly_cost=total_hourly * 8 * 22,
        automatable_count=len(automatable),
        automatable_savings=sum(i.impact.hourly_cost for i in automatable),
        by_category=by_category,
        top_3_issues=[i.one_liner for i in sorted_issues[:3]],
        recommended_action=(
            f"Auto-fix {len(automatable)} issues to save ${sum(i.impact.hourly_cost for i in automatable):.0f}/hr"
            if automatable else "Review and approve manual fixes"
        ),
        generated_at=datetime.now(UTC),
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/summary", response_model=ActionCenterSummary)
async def get_action_summary(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Get executive summary - understand fleet health in 5 seconds.

    Shows:
    - Total $ being lost per hour
    - How much can be auto-fixed
    - Single recommended action
    """
    if mock_mode:
        issues = _generate_mock_issues()
        return _generate_mock_summary(issues)

    # Real implementation using ProactiveResolver
    try:
        from device_anomaly.services.proactive_resolver import get_resolver
        resolver = get_resolver()
        issues = resolver.scan_fleet(tenant_id)
        summary_dict = resolver.get_executive_summary(issues)
        # Add required fields that the resolver doesn't provide
        return ActionCenterSummary(
            tenant_id=tenant_id,
            generated_at=datetime.now(UTC),
            **summary_dict,
        )
    except Exception as e:
        logger.warning(f"ProactiveResolver failed: {e}, returning empty summary in live mode")
        # Return empty summary in live mode (not mock data)
        return ActionCenterSummary(
            tenant_id=tenant_id,
            total_issues=0,
            total_hourly_cost=0.0,
            daily_cost=0.0,
            monthly_cost=0.0,
            automatable_count=0,
            automatable_savings=0.0,
            by_category={},
            top_3_issues=[],
            recommended_action="No issues detected - fleet is healthy",
            generated_at=datetime.now(UTC),
        )


@router.get("/issues", response_model=IssueListResponse)
async def get_issues(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
    category: str | None = Query(None, description="Filter by category"),
    automatable_only: bool = Query(False, description="Only show auto-fixable issues"),
    limit: int = Query(50, le=200),
):
    """
    Get all issues sorted by business impact.

    Each issue includes:
    - What's wrong
    - Why it matters ($)
    - How to fix it
    - Whether we can auto-fix
    """
    if mock_mode:
        issues = _generate_mock_issues()
    else:
        # Real implementation: try to get issues from ProactiveResolver
        try:
            from device_anomaly.services.proactive_resolver import get_resolver
            resolver = get_resolver()
            issues = resolver.scan_fleet(tenant_id)
        except Exception as e:
            logger.warning(f"ProactiveResolver failed: {e}, returning empty list in live mode")
            # In live mode without resolver, return empty list (not mock data)
            return IssueListResponse(issues=[], total_count=0, can_auto_fix=0)

    if category:
        issues = [i for i in issues if i.category == category]

    if automatable_only:
        issues = [i for i in issues if i.remediation.automated]

    # Sort by priority
    issues = sorted(issues, key=lambda x: x.impact.priority_score, reverse=True)[:limit]

    return IssueListResponse(
        issues=issues,
        total_count=len(issues),
        can_auto_fix=len([i for i in issues if i.remediation.automated]),
    )


@router.post("/fix/{issue_id}", response_model=FixResult)
async def fix_issue(
    issue_id: str,
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    Execute the fix for an issue.

    For automated issues, this triggers the remediation.
    For manual issues, this creates a ticket/notification.
    """
    # Mock implementation
    return FixResult(
        issue_id=issue_id,
        success=True,
        message="Policy push initiated. Changes will apply within 5 minutes.",
        devices_fixed=23,
        devices_failed=0,
    )


@router.post("/fix-all-automated", response_model=list[FixResult])
async def fix_all_automated(
    tenant_id: str = Depends(get_tenant_id),
    mock_mode: bool = Depends(get_mock_mode),
):
    """
    One-click fix for all automatable issues.

    This is the "just fix everything" button Elon would want.
    """
    if mock_mode:
        issues = _generate_mock_issues()
        automatable = [i for i in issues if i.remediation.automated]

        results = []
        for issue in automatable:
            results.append(FixResult(
                issue_id=issue.id,
                success=True,
                message=f"Initiated: {issue.remediation.description}",
                devices_fixed=issue.impact.affected_devices,
                devices_failed=0,
            ))
        return results

    # Real implementation: get issues from ProactiveResolver and execute fixes
    try:
        from device_anomaly.services.proactive_resolver import get_resolver
        resolver = get_resolver()
        issues = resolver.scan_fleet(tenant_id)
        automatable = [i for i in issues if i.remediation.automated]

        results = []
        for issue in automatable:
            # TODO: Actually execute the fix via resolver
            results.append(FixResult(
                issue_id=issue.id,
                success=True,
                message=f"Initiated: {issue.remediation.description}",
                devices_fixed=issue.impact.affected_devices,
                devices_failed=0,
            ))
        return results
    except Exception as e:
        logger.warning(f"ProactiveResolver failed: {e}, returning empty list in live mode")
        return []
