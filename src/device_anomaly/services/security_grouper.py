"""
Security Grouper Service.

Provides intelligent grouping of devices for security analysis:
- Risk clusters (group by violation type)
- Path-based comparison
- Temporal correlation (devices with issues appearing at similar times)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DeviceSecurityStatus:
    """Security status for a single device."""

    device_id: int
    device_name: str
    device_group_id: int | None = None
    full_path: str | None = None
    security_score: float = 100.0
    is_encrypted: bool = True
    is_rooted: bool = False
    has_passcode: bool = True
    patch_age_days: int = 0
    usb_debugging: bool = False
    developer_mode: bool = False
    risk_level: str = "low"
    violations: list[str] = field(default_factory=list)
    last_check_in: datetime | None = None
    security_issue_detected_at: datetime | None = None


@dataclass
class RiskCluster:
    """A group of devices with similar security violations."""

    cluster_id: str
    cluster_name: str
    violation_type: str
    severity: str  # critical, high, medium, low
    device_count: int
    avg_security_score: float
    affected_paths: list[str]
    first_detected: datetime | None
    sample_devices: list[DeviceSecurityStatus]
    recommendation: str


@dataclass
class PathSecuritySummary:
    """Security summary for a specific PATH."""

    path: str
    path_name: str
    device_count: int
    security_score: float
    compliant_count: int
    at_risk_count: int
    critical_count: int
    rooted_count: int
    unencrypted_count: int
    no_passcode_count: int
    outdated_patch_count: int
    usb_debugging_count: int
    developer_mode_count: int


@dataclass
class PathComparison:
    """Comparison result between paths."""

    path: str
    path_name: str
    security_score: float
    device_count: int
    compliant_pct: float
    rooted_count: int
    unencrypted_count: int
    outdated_patch_count: int
    delta_from_fleet: float


@dataclass
class TemporalCluster:
    """Devices that developed similar security issues around the same time."""

    cluster_id: str
    cluster_name: str
    issue_appeared_at: datetime
    device_count: int
    common_violations: list[str]
    affected_paths: list[str]
    correlation_insight: str
    sample_devices: list[DeviceSecurityStatus]


class SecurityGrouper:
    """
    Intelligent grouping of devices for security analysis.

    Provides:
    - Risk clusters by violation type
    - Path-based aggregation and comparison
    - Temporal correlation analysis
    """

    # Violation type definitions with severity
    VIOLATION_TYPES = {
        "rooted": {
            "name": "Rooted/Jailbroken",
            "severity": "critical",
            "check": lambda d: d.is_rooted,
            "recommendation": "Investigate for potential compromise and consider device wipe",
        },
        "unencrypted": {
            "name": "Storage Not Encrypted",
            "severity": "critical",
            "check": lambda d: not d.is_encrypted,
            "recommendation": "Enable encryption policy and verify compliance",
        },
        "no_passcode": {
            "name": "No Passcode Set",
            "severity": "high",
            "check": lambda d: not d.has_passcode,
            "recommendation": "Enforce passcode policy immediately",
        },
        "outdated_patch": {
            "name": "Outdated Security Patches",
            "severity": "high",
            "check": lambda d: d.patch_age_days > 60,
            "recommendation": "Schedule security patch deployment",
        },
        "usb_debugging": {
            "name": "USB Debugging Enabled",
            "severity": "medium",
            "check": lambda d: d.usb_debugging,
            "recommendation": "Push policy to disable USB debugging",
        },
        "developer_mode": {
            "name": "Developer Mode Enabled",
            "severity": "medium",
            "check": lambda d: d.developer_mode,
            "recommendation": "Disable developer mode on production devices",
        },
    }

    SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def __init__(self):
        self._devices: list[DeviceSecurityStatus] = []

    def set_devices(self, devices: list[DeviceSecurityStatus]) -> None:
        """Set the devices to analyze."""
        self._devices = devices

    def group_by_violation_type(
        self,
        devices: list[DeviceSecurityStatus] | None = None,
        min_cluster_size: int = 1,
    ) -> list[RiskCluster]:
        """
        Group devices by violation type into risk clusters.

        Args:
            devices: Devices to analyze (uses internal list if None).
            min_cluster_size: Minimum devices for a cluster to be included.

        Returns:
            List of RiskCluster objects sorted by severity.
        """
        if devices is None:
            devices = self._devices

        clusters = []

        for vtype, vconfig in self.VIOLATION_TYPES.items():
            affected = [d for d in devices if vconfig["check"](d)]

            if len(affected) < min_cluster_size:
                continue

            # Get unique paths affected
            paths = list({d.full_path for d in affected if d.full_path})
            paths.sort()

            # Calculate average score
            avg_score = (
                sum(d.security_score for d in affected) / len(affected) if affected else 0
            )

            # Find earliest detection
            dates = [d.security_issue_detected_at for d in affected if d.security_issue_detected_at]
            first_detected = min(dates) if dates else None

            # Sample devices (up to 5)
            sample = sorted(affected, key=lambda d: d.security_score)[:5]

            cluster = RiskCluster(
                cluster_id=f"cluster-{vtype}",
                cluster_name=f"{vconfig['name']} ({len(affected)} devices)",
                violation_type=vtype,
                severity=vconfig["severity"],
                device_count=len(affected),
                avg_security_score=round(avg_score, 1),
                affected_paths=paths[:10],  # Limit to 10 paths
                first_detected=first_detected,
                sample_devices=sample,
                recommendation=vconfig["recommendation"],
            )
            clusters.append(cluster)

        # Sort by severity (critical first)
        clusters.sort(key=lambda c: self.SEVERITY_ORDER.get(c.severity, 99))

        return clusters

    def group_by_path(
        self,
        devices: list[DeviceSecurityStatus] | None = None,
    ) -> dict[str, PathSecuritySummary]:
        """
        Group devices by PATH and compute security summaries.

        Args:
            devices: Devices to analyze.

        Returns:
            Dict mapping path to PathSecuritySummary.
        """
        if devices is None:
            devices = self._devices

        by_path: dict[str, list[DeviceSecurityStatus]] = defaultdict(list)

        for device in devices:
            path = device.full_path or "Unknown"
            by_path[path].append(device)

        summaries = {}

        for path, path_devices in by_path.items():
            count = len(path_devices)
            avg_score = sum(d.security_score for d in path_devices) / count if count else 0

            # Count violations
            rooted = sum(1 for d in path_devices if d.is_rooted)
            unencrypted = sum(1 for d in path_devices if not d.is_encrypted)
            no_passcode = sum(1 for d in path_devices if not d.has_passcode)
            outdated = sum(1 for d in path_devices if d.patch_age_days > 60)
            usb = sum(1 for d in path_devices if d.usb_debugging)
            dev_mode = sum(1 for d in path_devices if d.developer_mode)

            # Count by risk level
            at_risk = sum(1 for d in path_devices if d.risk_level in ("high", "critical"))
            critical = sum(1 for d in path_devices if d.risk_level == "critical")
            compliant = count - at_risk

            # Extract just the last segment as the name
            path_name = path.split(" / ")[-1] if path else "Unknown"

            summaries[path] = PathSecuritySummary(
                path=path,
                path_name=path_name,
                device_count=count,
                security_score=round(avg_score, 1),
                compliant_count=compliant,
                at_risk_count=at_risk,
                critical_count=critical,
                rooted_count=rooted,
                unencrypted_count=unencrypted,
                no_passcode_count=no_passcode,
                outdated_patch_count=outdated,
                usb_debugging_count=usb,
                developer_mode_count=dev_mode,
            )

        return summaries

    def compare_paths(
        self,
        paths: list[str],
        devices: list[DeviceSecurityStatus] | None = None,
    ) -> tuple[list[PathComparison], float, list[str]]:
        """
        Compare security posture across multiple paths.

        Args:
            paths: List of paths to compare.
            devices: Devices to analyze.

        Returns:
            Tuple of (comparisons, fleet_average, insights).
        """
        if devices is None:
            devices = self._devices

        # Get summaries for all paths
        all_summaries = self.group_by_path(devices)

        # Calculate fleet average
        total_devices = len(devices)
        fleet_avg = (
            sum(d.security_score for d in devices) / total_devices
            if total_devices > 0
            else 0
        )

        comparisons = []
        for path in paths:
            summary = all_summaries.get(path)
            if not summary:
                continue

            compliant_pct = (
                (summary.compliant_count / summary.device_count * 100)
                if summary.device_count > 0
                else 0
            )

            comp = PathComparison(
                path=path,
                path_name=summary.path_name,
                security_score=summary.security_score,
                device_count=summary.device_count,
                compliant_pct=round(compliant_pct, 1),
                rooted_count=summary.rooted_count,
                unencrypted_count=summary.unencrypted_count,
                outdated_patch_count=summary.outdated_patch_count,
                delta_from_fleet=round(summary.security_score - fleet_avg, 1),
            )
            comparisons.append(comp)

        # Sort by score descending
        comparisons.sort(key=lambda c: c.security_score, reverse=True)

        # Generate insights
        insights = self._generate_comparison_insights(comparisons, fleet_avg)

        return comparisons, round(fleet_avg, 1), insights

    def _generate_comparison_insights(
        self,
        comparisons: list[PathComparison],
        fleet_avg: float,
    ) -> list[str]:
        """Generate human-readable insights from path comparison."""
        insights = []

        if not comparisons:
            return insights

        # Best and worst performers
        best = comparisons[0]
        worst = comparisons[-1]

        if best.delta_from_fleet > 5:
            insights.append(
                f"{best.path_name} leads with {best.security_score} score "
                f"(+{best.delta_from_fleet} above fleet average)"
            )

        if worst.delta_from_fleet < -5:
            insights.append(
                f"{worst.path_name} needs attention with {worst.security_score} score "
                f"({worst.delta_from_fleet} below fleet average)"
            )

        # Specific issues
        high_rooted = [c for c in comparisons if c.rooted_count > 0]
        if high_rooted:
            paths_with_rooted = ", ".join(c.path_name for c in high_rooted[:3])
            total_rooted = sum(c.rooted_count for c in high_rooted)
            insights.append(
                f"{total_rooted} rooted devices detected across: {paths_with_rooted}"
            )

        high_unencrypted = [c for c in comparisons if c.unencrypted_count > 2]
        if high_unencrypted:
            paths = ", ".join(c.path_name for c in high_unencrypted[:3])
            insights.append(f"Encryption gaps found at: {paths}")

        return insights

    def find_temporal_correlations(
        self,
        devices: list[DeviceSecurityStatus] | None = None,
        window_hours: int = 72,
        min_cluster_size: int = 3,
    ) -> list[TemporalCluster]:
        """
        Find devices that developed security issues around the same time.

        This can indicate:
        - Coordinated attacks
        - Batch policy failures
        - Software update issues

        Args:
            devices: Devices to analyze.
            window_hours: Time window for correlation.
            min_cluster_size: Minimum devices for a cluster.

        Returns:
            List of TemporalCluster objects.
        """
        if devices is None:
            devices = self._devices

        # Filter to devices with known issue detection times
        with_times = [
            d for d in devices
            if d.security_issue_detected_at and d.violations
        ]

        if len(with_times) < min_cluster_size:
            return []

        # Sort by detection time
        with_times.sort(key=lambda d: d.security_issue_detected_at)

        clusters = []
        window = timedelta(hours=window_hours)
        used = set()

        for i, device in enumerate(with_times):
            if device.device_id in used:
                continue

            # Find all devices within window
            cluster_devices = [device]
            base_time = device.security_issue_detected_at

            for other in with_times[i + 1:]:
                if other.device_id in used:
                    continue
                time_diff = other.security_issue_detected_at - base_time
                if time_diff <= window:
                    cluster_devices.append(other)
                else:
                    break

            if len(cluster_devices) >= min_cluster_size:
                # Mark as used
                for d in cluster_devices:
                    used.add(d.device_id)

                # Find common violations
                all_violations = []
                for d in cluster_devices:
                    all_violations.extend(d.violations)
                violation_counts = defaultdict(int)
                for v in all_violations:
                    violation_counts[v] += 1
                common = [v for v, c in violation_counts.items() if c >= len(cluster_devices) // 2]

                # Get affected paths
                paths = list({d.full_path for d in cluster_devices if d.full_path})

                # Generate insight
                insight = self._generate_temporal_insight(cluster_devices, common, paths)

                cluster = TemporalCluster(
                    cluster_id=f"temporal-{len(clusters) + 1}",
                    cluster_name=f"Correlated Issues ({len(cluster_devices)} devices)",
                    issue_appeared_at=base_time,
                    device_count=len(cluster_devices),
                    common_violations=common[:5],
                    affected_paths=paths[:5],
                    correlation_insight=insight,
                    sample_devices=cluster_devices[:5],
                )
                clusters.append(cluster)

        return clusters

    def _generate_temporal_insight(
        self,
        devices: list[DeviceSecurityStatus],
        violations: list[str],
        paths: list[str],
    ) -> str:
        """Generate insight about temporal correlation."""
        if len(paths) == 1:
            return f"All {len(devices)} devices at {paths[0]} developed issues simultaneously - check for local policy or network issue"

        if "Device is rooted" in violations:
            return f"{len(devices)} devices were rooted within a short time window - possible coordinated attack"

        if any("patch" in v.lower() for v in violations):
            return f"Security patch issues appeared across {len(paths)} locations - verify patch deployment process"

        return f"{len(devices)} devices across {len(paths)} locations developed similar issues - investigate common cause"


def compute_security_score(
    is_rooted: bool = False,
    is_encrypted: bool = True,
    has_passcode: bool = True,
    usb_debugging: bool = False,
    developer_mode: bool = False,
    patch_age_days: int = 0,
) -> tuple[float, str, list[str]]:
    """
    Compute security score, risk level, and violations for a device.

    Returns:
        Tuple of (score, risk_level, violations).
    """
    score = 100.0
    violations = []

    if is_rooted:
        score -= 40
        violations.append("Device is rooted")

    if not is_encrypted:
        score -= 25
        violations.append("Storage not encrypted")

    if not has_passcode:
        score -= 15
        violations.append("No passcode set")

    if usb_debugging:
        score -= 10
        violations.append("USB debugging enabled")

    if developer_mode:
        score -= 5
        violations.append("Developer mode enabled")

    if patch_age_days > 60:
        penalty = min(20, patch_age_days / 3)
        score -= penalty
        violations.append(f"Security patch {patch_age_days} days old")

    score = max(0, score)

    if score < 50:
        risk_level = "critical"
    elif score < 70:
        risk_level = "high"
    elif score < 85:
        risk_level = "medium"
    else:
        risk_level = "low"

    return round(score, 1), risk_level, violations
