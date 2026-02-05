"""
Proactive Issue Resolver - Elon Musk First Principles Approach

Instead of dashboards showing data, this service:
1. Detects problems automatically
2. Calculates business impact
3. Suggests/executes fixes
4. Learns from outcomes

"The best interface is no interface" - delete the need for users to hunt for problems.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class IssueCategory(StrEnum):
    """Categories that matter to the business, not technical silos."""
    PRODUCTIVITY_LOSS = "productivity_loss"  # Devices not working = people not working
    SECURITY_RISK = "security_risk"          # Exposure to breach/compliance failure
    COST_WASTE = "cost_waste"                # Money being burned unnecessarily
    IMPENDING_FAILURE = "impending_failure"  # About to break, fix before it does


class RemediationStatus(StrEnum):
    SUGGESTED = "suggested"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BusinessImpact:
    """Quantify the actual cost of not fixing this."""
    affected_devices: int
    affected_users: int
    hourly_cost: float  # $ per hour this continues
    risk_multiplier: float  # 1.0 = normal, 10.0 = could cause major incident

    @property
    def priority_score(self) -> float:
        """Single number for sorting. Higher = fix first."""
        return self.hourly_cost * self.risk_multiplier * (1 + self.affected_users / 100)


@dataclass
class Remediation:
    """What to do about it."""
    action_type: str  # "push_policy", "send_alert", "schedule_replacement", etc.
    description: str
    automated: bool  # Can we just do it?
    estimated_fix_time_minutes: int
    success_probability: float  # 0-1 based on historical data
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedIssue:
    """A problem that needs fixing."""
    id: str
    category: IssueCategory
    title: str  # "12 devices have USB debugging enabled"
    root_cause: str  # "Developer devices not returned to secure config"
    impact: BusinessImpact
    remediation: Remediation
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    device_ids: list[int] = field(default_factory=list)

    @property
    def one_liner(self) -> str:
        """What Elon would want to see: problem + cost + fix in one line."""
        return (
            f"{self.title} → ${self.impact.hourly_cost:.0f}/hr lost → "
            f"{self.remediation.description}"
        )


class ProactiveResolver:
    """
    The brain that finds and fixes problems.

    Philosophy:
    - Don't wait for users to find problems
    - Quantify everything in $ impact
    - Automate what we can, simplify what we can't
    - Learn from every fix
    """

    def __init__(self):
        self.issue_detectors: list[Callable] = []
        self.remediation_executors: dict[str, Callable] = {}
        self._register_default_detectors()

    def _register_default_detectors(self):
        """Built-in issue detection rules."""
        self.issue_detectors = [
            self._detect_security_exposures,
            self._detect_network_dead_zones,
            self._detect_battery_failures,
            self._detect_storage_exhaustion,
            self._detect_connectivity_issues,
            self._detect_cost_anomalies,
        ]

    def scan_fleet(self, tenant_id: str) -> list[DetectedIssue]:
        """
        Scan everything, return prioritized list of issues to fix.

        This replaces the need for users to check multiple dashboards.
        """
        all_issues: list[DetectedIssue] = []

        for detector in self.issue_detectors:
            try:
                issues = detector(tenant_id)
                all_issues.extend(issues)
            except Exception as e:
                logger.warning(f"Detector {detector.__name__} failed: {e}")

        # Sort by business impact - highest cost first
        all_issues.sort(key=lambda x: x.impact.priority_score, reverse=True)

        return all_issues

    def get_executive_summary(self, issues: list[DetectedIssue]) -> dict[str, Any]:
        """
        One-page summary for executives.

        Elon's rule: If it takes more than 5 seconds to understand, it's wrong.
        """
        total_hourly_cost = sum(i.impact.hourly_cost for i in issues)

        by_category = {}
        for issue in issues:
            cat = issue.category.value
            if cat not in by_category:
                by_category[cat] = {"count": 0, "hourly_cost": 0, "devices": 0}
            by_category[cat]["count"] += 1
            by_category[cat]["hourly_cost"] += issue.impact.hourly_cost
            by_category[cat]["devices"] += issue.impact.affected_devices

        automatable = [i for i in issues if i.remediation.automated]

        return {
            "total_issues": len(issues),
            "total_hourly_cost": total_hourly_cost,
            "daily_cost": total_hourly_cost * 8,  # Assuming 8 work hours
            "monthly_cost": total_hourly_cost * 8 * 22,  # 22 work days
            "by_category": by_category,
            "automatable_count": len(automatable),
            "automatable_savings": sum(i.impact.hourly_cost for i in automatable),
            "top_3_issues": [i.one_liner for i in issues[:3]],
            "recommended_action": self._get_recommended_action(issues),
        }

    def _get_recommended_action(self, issues: list[DetectedIssue]) -> str:
        """Single most impactful thing to do right now."""
        if not issues:
            return "Fleet is healthy. No action needed."

        top = issues[0]
        if top.remediation.automated:
            return f"Auto-fix available: {top.remediation.description} (saves ${top.impact.hourly_cost:.0f}/hr)"
        else:
            return f"Requires attention: {top.title}"

    # =========================================================================
    # Issue Detectors - Each returns list of issues
    # =========================================================================

    def _detect_security_exposures(self, tenant_id: str) -> list[DetectedIssue]:
        """Find security issues that create real business risk."""
        issues = []

        # This would query real data - showing structure
        # In production, this calls security_features and DevInfo tables

        # Example: USB debugging enabled
        usb_debug_count = self._query_usb_debugging_enabled(tenant_id)
        if usb_debug_count > 0:
            issues.append(DetectedIssue(
                id=f"sec_usb_debug_{tenant_id}",
                category=IssueCategory.SECURITY_RISK,
                title=f"{usb_debug_count} devices have USB debugging enabled",
                root_cause="Developer/test devices not returned to production config",
                impact=BusinessImpact(
                    affected_devices=usb_debug_count,
                    affected_users=usb_debug_count,
                    hourly_cost=usb_debug_count * 2.0,  # $2/device/hr risk cost
                    risk_multiplier=5.0,  # High risk - data exfiltration possible
                ),
                remediation=Remediation(
                    action_type="push_policy",
                    description="Push 'Disable USB Debugging' policy via MobiControl",
                    automated=True,
                    estimated_fix_time_minutes=5,
                    success_probability=0.95,
                    parameters={"policy_id": "disable_usb_debug"},
                ),
            ))

        # Unencrypted devices
        unencrypted = self._query_unencrypted_devices(tenant_id)
        if unencrypted > 0:
            issues.append(DetectedIssue(
                id=f"sec_unencrypted_{tenant_id}",
                category=IssueCategory.SECURITY_RISK,
                title=f"{unencrypted} devices lack encryption",
                root_cause="Devices enrolled before encryption policy, or policy not enforced",
                impact=BusinessImpact(
                    affected_devices=unencrypted,
                    affected_users=unencrypted,
                    hourly_cost=unencrypted * 10.0,  # High compliance risk
                    risk_multiplier=10.0,  # Critical - data breach risk
                ),
                remediation=Remediation(
                    action_type="push_policy",
                    description="Force encryption via MDM policy",
                    automated=True,
                    estimated_fix_time_minutes=30,  # Encryption takes time
                    success_probability=0.85,
                    parameters={"policy_id": "enforce_encryption"},
                ),
            ))

        return issues

    def _detect_network_dead_zones(self, tenant_id: str) -> list[DetectedIssue]:
        """Find locations where devices can't connect reliably."""
        issues = []

        dead_zones = self._query_network_dead_zones(tenant_id)
        for zone in dead_zones:
            issues.append(DetectedIssue(
                id=f"net_deadzone_{zone['location_id']}",
                category=IssueCategory.PRODUCTIVITY_LOSS,
                title=f"Dead zone at {zone['location_name']}: {zone['device_count']} devices affected",
                root_cause=f"WiFi signal strength avg {zone['avg_signal']} dBm (needs >-70 dBm)",
                impact=BusinessImpact(
                    affected_devices=zone['device_count'],
                    affected_users=zone['device_count'],
                    hourly_cost=zone['device_count'] * 25.0,  # Lost productivity
                    risk_multiplier=1.5,
                ),
                remediation=Remediation(
                    action_type="infrastructure_ticket",
                    description=f"Add WiFi AP at {zone['location_name']}",
                    automated=False,  # Requires physical install
                    estimated_fix_time_minutes=240,
                    success_probability=0.9,
                    parameters={"location": zone['location_name']},
                ),
                device_ids=zone.get('device_ids', []),
            ))

        return issues

    def _detect_battery_failures(self, tenant_id: str) -> list[DetectedIssue]:
        """Find devices with degraded batteries that will fail soon."""
        issues = []

        failing_batteries = self._query_degraded_batteries(tenant_id)
        if failing_batteries:
            issues.append(DetectedIssue(
                id=f"hw_battery_{tenant_id}",
                category=IssueCategory.IMPENDING_FAILURE,
                title=f"{len(failing_batteries)} devices need battery replacement",
                root_cause="Battery capacity <60% of original, avg age 18+ months",
                impact=BusinessImpact(
                    affected_devices=len(failing_batteries),
                    affected_users=len(failing_batteries),
                    hourly_cost=len(failing_batteries) * 5.0,
                    risk_multiplier=2.0,  # Will fail during shift
                ),
                remediation=Remediation(
                    action_type="schedule_replacement",
                    description="Schedule battery replacements before failure",
                    automated=False,
                    estimated_fix_time_minutes=30,  # Per device
                    success_probability=0.99,
                    parameters={"device_ids": failing_batteries},
                ),
                device_ids=failing_batteries,
            ))

        return issues

    def _detect_storage_exhaustion(self, tenant_id: str) -> list[DetectedIssue]:
        """Find devices about to run out of storage."""
        issues = []

        low_storage = self._query_low_storage_devices(tenant_id)
        if low_storage:
            issues.append(DetectedIssue(
                id=f"hw_storage_{tenant_id}",
                category=IssueCategory.IMPENDING_FAILURE,
                title=f"{len(low_storage)} devices critically low on storage (<10%)",
                root_cause="Cache buildup, app data growth, or user files",
                impact=BusinessImpact(
                    affected_devices=len(low_storage),
                    affected_users=len(low_storage),
                    hourly_cost=len(low_storage) * 15.0,  # Apps will crash
                    risk_multiplier=3.0,
                ),
                remediation=Remediation(
                    action_type="remote_action",
                    description="Clear app cache and temp files remotely",
                    automated=True,
                    estimated_fix_time_minutes=10,
                    success_probability=0.8,
                    parameters={"action": "clear_cache"},
                ),
                device_ids=low_storage,
            ))

        return issues

    def _detect_connectivity_issues(self, tenant_id: str) -> list[DetectedIssue]:
        """Find devices with frequent disconnections."""
        # Similar pattern - query data, create issues
        return []

    def _detect_cost_anomalies(self, tenant_id: str) -> list[DetectedIssue]:
        """Find unusual data usage or other cost drivers."""
        issues = []

        data_abusers = self._query_data_usage_anomalies(tenant_id)
        for device in data_abusers:
            issues.append(DetectedIssue(
                id=f"cost_data_{device['device_id']}",
                category=IssueCategory.COST_WASTE,
                title=f"Device {device['device_name']} using {device['data_gb']:.1f}GB/month (10x fleet avg)",
                root_cause=f"App '{device['top_app']}' consuming {device['top_app_pct']:.0f}% of data",
                impact=BusinessImpact(
                    affected_devices=1,
                    affected_users=1,
                    hourly_cost=device['excess_cost'] / 720,  # Monthly to hourly
                    risk_multiplier=1.0,
                ),
                remediation=Remediation(
                    action_type="user_notification",
                    description=f"Notify user and restrict background data for {device['top_app']}",
                    automated=True,
                    estimated_fix_time_minutes=1,
                    success_probability=0.7,
                    parameters={"device_id": device['device_id'], "app": device['top_app']},
                ),
                device_ids=[device['device_id']],
            ))

        return issues

    # =========================================================================
    # Data Query Stubs - These would hit real databases
    # =========================================================================

    def _query_usb_debugging_enabled(self, tenant_id: str) -> int:
        """Query DevInfo for USB debugging status."""
        # TODO: Real implementation
        return 0

    def _query_unencrypted_devices(self, tenant_id: str) -> int:
        """Query DevInfo for encryption status."""
        return 0

    def _query_network_dead_zones(self, tenant_id: str) -> list[dict]:
        """Query WiFi data for dead zones."""
        return []

    def _query_degraded_batteries(self, tenant_id: str) -> list[int]:
        """Query battery health predictions."""
        return []

    def _query_low_storage_devices(self, tenant_id: str) -> list[int]:
        """Query storage levels."""
        return []

    def _query_data_usage_anomalies(self, tenant_id: str) -> list[dict]:
        """Query data usage for anomalies."""
        return []


# Singleton instance
_resolver: ProactiveResolver | None = None


def get_resolver() -> ProactiveResolver:
    """Get the singleton resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = ProactiveResolver()
    return _resolver
