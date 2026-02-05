"""Network pattern analysis for WiFi and cellular insights.

Carl's key requirements:
- "AP hopping/stickiness"
- "Tower hopping/stickiness"
- "Patterns by carrier, network type (5G, 4G)"
- "Throughput metrics (upload, download, ping)"
- "Server disconnection patterns"
- "Devices/locations with hidden patterns (taken home, etc.)"

This analyzer provides network-aware insights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

import pandas as pd
from sqlalchemy.orm import Session

from device_anomaly.database.schema import DeviceFeature, LocationMetadata
from device_anomaly.insights.categories import InsightSeverity

logger = logging.getLogger(__name__)


class NetworkIssueType(StrEnum):
    """Types of network issues."""

    AP_HOPPING = "ap_hopping"
    AP_STICKINESS = "ap_stickiness"
    WIFI_DEAD_ZONE = "wifi_dead_zone"
    TOWER_HOPPING = "tower_hopping"
    CARRIER_DEGRADATION = "carrier_degradation"
    TECH_FALLBACK = "tech_fallback"  # 5G -> 4G -> 3G
    DISCONNECT_PATTERN = "disconnect_pattern"
    HIDDEN_DEVICE = "hidden_device"
    THROUGHPUT_ISSUE = "throughput_issue"


@dataclass
class WifiRoamingIssue:
    """A WiFi roaming issue for a device."""

    device_id: int
    device_name: str | None
    issue_type: NetworkIssueType  # AP_HOPPING or AP_STICKINESS
    unique_aps_connected: int
    ap_changes_per_hour: float
    avg_signal_strength: float
    weak_signal_percentage: float  # % time on weak signal
    description: str
    severity: InsightSeverity
    affected_location: str | None
    recommendation: str


@dataclass
class WifiRoamingReport:
    """WiFi roaming analysis for a location or tenant."""

    entity_id: str  # location_id or tenant_id
    entity_name: str
    entity_type: str  # "location" or "tenant"
    analysis_period_days: int

    # Summary
    total_devices: int
    devices_with_roaming_issues: int
    devices_with_stickiness: int

    # Fleet metrics
    avg_aps_per_device: float
    avg_ap_changes_per_hour: float
    avg_signal_strength: float

    # Problematic APs (if available)
    worst_aps: list[tuple[str, float]]  # (BSSID/SSID, issue_rate)

    # Issues
    issues: list[WifiRoamingIssue]

    # Dead zones detected
    potential_dead_zones: list[str]

    recommendations: list[str]


@dataclass
class CellularIssue:
    """A cellular connectivity issue."""

    device_id: int
    device_name: str | None
    issue_type: NetworkIssueType
    carrier: str | None
    primary_network_type: str  # "5G", "LTE", "3G"
    tower_changes_per_hour: float
    tech_fallback_rate: float  # % time on lower tech than expected
    description: str
    severity: InsightSeverity
    recommendation: str


@dataclass
class CellularPatternReport:
    """Cellular pattern analysis for a tenant."""

    tenant_id: str
    analysis_period_days: int

    # Summary by carrier
    carrier_stats: dict[str, dict[str, float]]  # carrier -> {devices, avg_signal, fallback_rate}

    # Summary by network type
    network_type_distribution: dict[str, float]  # "5G" -> % time

    # Issues
    total_devices: int
    devices_with_tower_hopping: int
    devices_with_tech_fallback: int
    issues: list[CellularIssue]

    # Carrier comparison
    best_carrier: str | None
    worst_carrier: str | None

    recommendations: list[str]


@dataclass
class DisconnectEvent:
    """A network disconnect event."""

    device_id: int
    timestamp: datetime
    duration_minutes: float
    network_type_before: str  # "wifi", "cellular", "none"
    network_type_after: str
    location_id: str | None
    is_predictable: bool  # Part of a pattern


@dataclass
class DisconnectPatternReport:
    """Disconnect pattern analysis."""

    entity_id: str
    entity_name: str
    entity_type: str  # "device", "location", "tenant"
    analysis_period_days: int

    # Summary
    total_disconnects: int
    avg_disconnects_per_device: float
    avg_disconnect_duration_minutes: float
    total_offline_hours: float

    # Pattern detection
    has_predictable_pattern: bool
    pattern_description: str | None  # e.g., "Daily at 14:00"
    pattern_confidence: float

    # Breakdown
    disconnects_by_hour: dict[int, int]  # hour -> count
    disconnects_by_day: dict[int, int]  # day_of_week -> count
    disconnects_by_location: dict[str, int]  # location -> count

    # Top offenders
    devices_with_most_disconnects: list[tuple[int, int]]  # (device_id, count)

    recommendations: list[str]


@dataclass
class HiddenDeviceIndicator:
    """Indicator that a device may be hidden or taken off-site."""

    device_id: int
    device_name: str | None
    assigned_location: str | None

    # Evidence
    offline_hours_last_week: float
    unusual_offline_periods: int  # Count of off-hours offline periods
    last_seen_location: str | None
    never_seen_at_assigned: bool

    # Scoring
    hidden_score: float  # 0-1, higher = more likely hidden
    confidence: float

    indicators: list[str]  # Human-readable evidence


@dataclass
class HiddenDeviceReport:
    """Analysis of potentially hidden/off-site devices."""

    tenant_id: str
    analysis_period_days: int

    # Summary
    total_devices_analyzed: int
    devices_flagged: int
    devices_highly_suspicious: int  # score > 0.8

    # Flagged devices
    flagged_devices: list[HiddenDeviceIndicator]

    # By location
    locations_missing_devices: dict[str, int]  # location -> count of missing devices

    recommendations: list[str]


class NetworkPatternAnalyzer:
    """Analyzer for network connectivity patterns.

    Addresses Carl's requirements:
    - AP hopping/stickiness detection
    - Cellular tower hopping
    - Carrier performance comparison
    - Disconnect pattern detection
    - Hidden device detection

    Usage:
        analyzer = NetworkPatternAnalyzer(db_session, tenant_id)
        wifi_report = analyzer.analyze_wifi_roaming("warehouse_1", period_days=7)
        hidden = analyzer.detect_hidden_devices()
    """

    # Thresholds
    AP_HOPPING_THRESHOLD = 5  # APs per hour = excessive hopping
    AP_CHANGES_HIGH = 10  # AP changes per hour = definitely hopping
    WEAK_SIGNAL_DBM = -75  # Below this is weak signal
    TOWER_HOPPING_THRESHOLD = 3  # Tower changes per hour
    HIDDEN_OFFLINE_HOURS = 40  # Offline hours/week to flag

    def __init__(
        self,
        db_session: Session,
        tenant_id: str,
    ):
        """Initialize the analyzer.

        Args:
            db_session: SQLAlchemy database session
            tenant_id: Tenant ID for multi-tenant filtering
        """
        self.db = db_session
        self.tenant_id = tenant_id

    def analyze_wifi_roaming(
        self,
        location_id: str | None = None,
        period_days: int = 7,
    ) -> WifiRoamingReport:
        """Analyze WiFi roaming patterns.

        Args:
            location_id: Optional location to analyze (None = whole tenant)
            period_days: Days of history to analyze

        Returns:
            WifiRoamingReport with roaming analysis
        """
        entity_type = "location" if location_id else "tenant"
        entity_id = location_id or self.tenant_id

        # Get entity name
        if location_id:
            loc = self.db.query(LocationMetadata).filter(
                LocationMetadata.tenant_id == self.tenant_id,
                LocationMetadata.location_id == location_id,
            ).first()
            entity_name = loc.location_name if loc else location_id
        else:
            entity_name = self.tenant_id

        # Get network data
        network_data = self._get_network_features(location_id, period_days)

        if network_data.empty:
            return WifiRoamingReport(
                entity_id=entity_id,
                entity_name=entity_name,
                entity_type=entity_type,
                analysis_period_days=period_days,
                total_devices=0,
                devices_with_roaming_issues=0,
                devices_with_stickiness=0,
                avg_aps_per_device=0,
                avg_ap_changes_per_hour=0,
                avg_signal_strength=0,
                worst_aps=[],
                issues=[],
                potential_dead_zones=[],
                recommendations=[],
            )

        # Analyze each device
        issues: list[WifiRoamingIssue] = []
        device_ids = network_data["device_id"].unique()

        hopping_count = 0
        stickiness_count = 0

        for device_id in device_ids:
            device_data = network_data[network_data["device_id"] == device_id]
            device_issues = self._analyze_device_wifi_roaming(device_id, device_data)

            for issue in device_issues:
                if issue.issue_type == NetworkIssueType.AP_HOPPING:
                    hopping_count += 1
                elif issue.issue_type == NetworkIssueType.AP_STICKINESS:
                    stickiness_count += 1

            issues.extend(device_issues)

        # Calculate fleet metrics
        avg_aps = network_data["unique_aps"].mean() if "unique_aps" in network_data.columns else 0
        avg_signal = network_data["wifi_signal"].mean() if "wifi_signal" in network_data.columns else 0

        # Detect potential dead zones
        dead_zones = self._detect_dead_zones(network_data)

        # Generate recommendations
        recommendations = self._generate_wifi_recommendations(
            issues, dead_zones, hopping_count, stickiness_count, len(device_ids)
        )

        return WifiRoamingReport(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_type=entity_type,
            analysis_period_days=period_days,
            total_devices=len(device_ids),
            devices_with_roaming_issues=hopping_count,
            devices_with_stickiness=stickiness_count,
            avg_aps_per_device=float(avg_aps),
            avg_ap_changes_per_hour=0,  # Would need time-series data
            avg_signal_strength=float(avg_signal),
            worst_aps=[],  # Would need BSSID-level data
            issues=issues,
            potential_dead_zones=dead_zones,
            recommendations=recommendations,
        )

    def analyze_cellular_patterns(
        self,
        period_days: int = 7,
    ) -> CellularPatternReport:
        """Analyze cellular connectivity patterns across the tenant.

        Args:
            period_days: Days of history to analyze

        Returns:
            CellularPatternReport with cellular analysis
        """
        network_data = self._get_network_features(None, period_days)

        if network_data.empty:
            return CellularPatternReport(
                tenant_id=self.tenant_id,
                analysis_period_days=period_days,
                carrier_stats={},
                network_type_distribution={},
                total_devices=0,
                devices_with_tower_hopping=0,
                devices_with_tech_fallback=0,
                issues=[],
                best_carrier=None,
                worst_carrier=None,
                recommendations=[],
            )

        # Analyze each device
        issues: list[CellularIssue] = []
        device_ids = network_data["device_id"].unique()

        tower_hopping_count = 0
        tech_fallback_count = 0

        for device_id in device_ids:
            device_data = network_data[network_data["device_id"] == device_id]
            device_issues = self._analyze_device_cellular(device_id, device_data)

            for issue in device_issues:
                if issue.issue_type == NetworkIssueType.TOWER_HOPPING:
                    tower_hopping_count += 1
                elif issue.issue_type == NetworkIssueType.TECH_FALLBACK:
                    tech_fallback_count += 1

            issues.extend(device_issues)

        # Calculate carrier stats
        carrier_stats = self._calculate_carrier_stats(network_data)

        # Find best/worst carrier
        best_carrier = None
        worst_carrier = None
        if carrier_stats:
            by_signal = sorted(carrier_stats.items(), key=lambda x: x[1].get("avg_signal", 0), reverse=True)
            best_carrier = by_signal[0][0] if by_signal else None
            worst_carrier = by_signal[-1][0] if len(by_signal) > 1 else None

        # Network type distribution
        network_type_dist = self._calculate_network_type_distribution(network_data)

        recommendations = self._generate_cellular_recommendations(
            issues, carrier_stats, tower_hopping_count, len(device_ids)
        )

        return CellularPatternReport(
            tenant_id=self.tenant_id,
            analysis_period_days=period_days,
            carrier_stats=carrier_stats,
            network_type_distribution=network_type_dist,
            total_devices=len(device_ids),
            devices_with_tower_hopping=tower_hopping_count,
            devices_with_tech_fallback=tech_fallback_count,
            issues=issues,
            best_carrier=best_carrier,
            worst_carrier=worst_carrier,
            recommendations=recommendations,
        )

    def analyze_disconnect_patterns(
        self,
        group_by: str = "tenant",  # "device", "location", "tenant"
        entity_id: str | None = None,
        period_days: int = 7,
    ) -> DisconnectPatternReport:
        """Analyze network disconnection patterns.

        Args:
            group_by: Entity type to group by
            entity_id: Optional specific entity ID
            period_days: Days of history to analyze

        Returns:
            DisconnectPatternReport with disconnect analysis
        """
        location_id = entity_id if group_by == "location" else None

        network_data = self._get_network_features(location_id, period_days)

        if network_data.empty:
            return DisconnectPatternReport(
                entity_id=entity_id or self.tenant_id,
                entity_name=entity_id or self.tenant_id,
                entity_type=group_by,
                analysis_period_days=period_days,
                total_disconnects=0,
                avg_disconnects_per_device=0,
                avg_disconnect_duration_minutes=0,
                total_offline_hours=0,
                has_predictable_pattern=False,
                pattern_description=None,
                pattern_confidence=0,
                disconnects_by_hour={},
                disconnects_by_day={},
                disconnects_by_location={},
                devices_with_most_disconnects=[],
                recommendations=[],
            )

        # Calculate disconnect metrics
        device_ids = network_data["device_id"].unique()

        # Get disconnect counts per device
        disconnect_col = "disconnect_count" if "disconnect_count" in network_data.columns else "wifi_disconnect_count"
        if disconnect_col not in network_data.columns:
            network_data[disconnect_col] = 0

        total_disconnects = int(network_data[disconnect_col].sum())
        avg_disconnects = network_data.groupby("device_id")[disconnect_col].mean().mean()

        # Get offline time
        offline_col = "time_no_network" if "time_no_network" in network_data.columns else None
        total_offline_hours = 0
        if offline_col and offline_col in network_data.columns:
            total_offline_hours = float(network_data[offline_col].sum() / 60)  # Assuming minutes

        # Detect patterns
        has_pattern, pattern_desc, confidence = self._detect_disconnect_pattern(network_data)

        # Top offenders
        device_disconnects = (
            network_data.groupby("device_id")[disconnect_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        top_offenders = [(int(d), int(c)) for d, c in device_disconnects.items()]

        recommendations = self._generate_disconnect_recommendations(
            total_disconnects, has_pattern, len(device_ids)
        )

        return DisconnectPatternReport(
            entity_id=entity_id or self.tenant_id,
            entity_name=entity_id or self.tenant_id,
            entity_type=group_by,
            analysis_period_days=period_days,
            total_disconnects=total_disconnects,
            avg_disconnects_per_device=float(avg_disconnects),
            avg_disconnect_duration_minutes=0,  # Would need event-level data
            total_offline_hours=total_offline_hours,
            has_predictable_pattern=has_pattern,
            pattern_description=pattern_desc,
            pattern_confidence=confidence,
            disconnects_by_hour={},  # Would need event-level data
            disconnects_by_day={},
            disconnects_by_location={},
            devices_with_most_disconnects=top_offenders,
            recommendations=recommendations,
        )

    def detect_hidden_devices(
        self,
        period_days: int = 7,
    ) -> HiddenDeviceReport:
        """Detect devices that may be hidden or taken off-site.

        Carl's requirement: "Device showing suspicious offline patterns (taken home, hidden)"

        Args:
            period_days: Days of history to analyze

        Returns:
            HiddenDeviceReport with flagged devices
        """
        network_data = self._get_network_features(None, period_days)

        if network_data.empty:
            return HiddenDeviceReport(
                tenant_id=self.tenant_id,
                analysis_period_days=period_days,
                total_devices_analyzed=0,
                devices_flagged=0,
                devices_highly_suspicious=0,
                flagged_devices=[],
                locations_missing_devices={},
                recommendations=[],
            )

        flagged: list[HiddenDeviceIndicator] = []
        device_ids = network_data["device_id"].unique()

        for device_id in device_ids:
            device_data = network_data[network_data["device_id"] == device_id]
            indicator = self._analyze_device_hidden_patterns(device_id, device_data, period_days)

            if indicator and indicator.hidden_score > 0.5:
                flagged.append(indicator)

        # Sort by hidden score
        flagged.sort(key=lambda x: x.hidden_score, reverse=True)

        highly_suspicious = sum(1 for d in flagged if d.hidden_score > 0.8)

        # Count missing devices by location
        locations_missing: dict[str, int] = {}
        for device in flagged:
            if device.assigned_location:
                locations_missing[device.assigned_location] = locations_missing.get(device.assigned_location, 0) + 1

        recommendations = self._generate_hidden_recommendations(flagged, highly_suspicious)

        return HiddenDeviceReport(
            tenant_id=self.tenant_id,
            analysis_period_days=period_days,
            total_devices_analyzed=len(device_ids),
            devices_flagged=len(flagged),
            devices_highly_suspicious=highly_suspicious,
            flagged_devices=flagged,
            locations_missing_devices=locations_missing,
            recommendations=recommendations,
        )

    # Private helper methods

    def _get_network_features(
        self,
        location_id: str | None,
        period_days: int,
    ) -> pd.DataFrame:
        """Get network features for analysis."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        query = (
            self.db.query(DeviceFeature)
            .filter(
                DeviceFeature.tenant_id == self.tenant_id,
                DeviceFeature.computed_at >= start_date,
                DeviceFeature.computed_at <= end_date,
            )
            .all()
        )

        if not query:
            return pd.DataFrame()

        import json

        records = []
        for feature in query:
            try:
                metadata = json.loads(feature.metadata_json) if feature.metadata_json else {}

                # Filter by location if specified
                if location_id and metadata.get("location_id") != location_id:
                    continue

                features = json.loads(feature.feature_values_json) if feature.feature_values_json else {}

                records.append({
                    "device_id": feature.device_id,
                    "computed_at": feature.computed_at,
                    "location_id": metadata.get("location_id"),
                    "carrier": metadata.get("carrier"),
                    # WiFi metrics
                    "unique_aps": features.get("UniqueAPsConnected", 0),
                    "unique_ssids": features.get("UniqueSSIDsConnected", 0),
                    "wifi_signal": features.get("WifiSignalStrength", 0),
                    "wifi_disconnect_count": features.get("WifiDisconnectCount", 0),
                    # Cellular metrics
                    "unique_cells": features.get("UniqueCellIds", 0),
                    "unique_lacs": features.get("UniqueLACs", 0),
                    "cell_signal": features.get("CellSignalStrength", 0),
                    "network_type_count": features.get("NetworkTypeCount", 0),
                    # General
                    "time_no_network": features.get("TimeOnNoNetwork", 0),
                    "time_on_wifi": features.get("TimeOnWifi", 0),
                    "time_on_5g": features.get("TimeOn5G", 0),
                    "time_on_4g": features.get("TimeOn4G", 0),
                    "time_on_3g": features.get("TimeOn3G", 0),
                })
            except (json.JSONDecodeError, TypeError):
                continue

        return pd.DataFrame(records)

    def _analyze_device_wifi_roaming(
        self,
        device_id: int,
        device_data: pd.DataFrame,
    ) -> list[WifiRoamingIssue]:
        """Analyze WiFi roaming for a single device."""
        issues: list[WifiRoamingIssue] = []

        if device_data.empty:
            return issues

        avg_aps = device_data["unique_aps"].mean() if "unique_aps" in device_data.columns else 0
        avg_signal = device_data["wifi_signal"].mean() if "wifi_signal" in device_data.columns else 0

        # Check for AP hopping
        if avg_aps > self.AP_HOPPING_THRESHOLD:
            severity = InsightSeverity.HIGH if avg_aps > self.AP_CHANGES_HIGH else InsightSeverity.MEDIUM
            issues.append(WifiRoamingIssue(
                device_id=device_id,
                device_name=None,
                issue_type=NetworkIssueType.AP_HOPPING,
                unique_aps_connected=int(avg_aps),
                ap_changes_per_hour=float(avg_aps),  # Simplified
                avg_signal_strength=float(avg_signal),
                weak_signal_percentage=0,
                description=f"Device connects to {avg_aps:.0f} APs on average (excessive roaming)",
                severity=severity,
                affected_location=None,
                recommendation="Check for overlapping AP coverage; consider adjusting roaming thresholds",
            ))

        # Check for stickiness (low APs + weak signal)
        elif avg_aps <= 1 and avg_signal < self.WEAK_SIGNAL_DBM:
            issues.append(WifiRoamingIssue(
                device_id=device_id,
                device_name=None,
                issue_type=NetworkIssueType.AP_STICKINESS,
                unique_aps_connected=int(avg_aps),
                ap_changes_per_hour=0,
                avg_signal_strength=float(avg_signal),
                weak_signal_percentage=100,  # Simplified
                description=f"Device stuck on weak AP (signal: {avg_signal:.0f} dBm)",
                severity=InsightSeverity.MEDIUM,
                affected_location=None,
                recommendation="Check roaming sensitivity; device may need to be configured to roam more aggressively",
            ))

        return issues

    def _analyze_device_cellular(
        self,
        device_id: int,
        device_data: pd.DataFrame,
    ) -> list[CellularIssue]:
        """Analyze cellular connectivity for a single device."""
        issues: list[CellularIssue] = []

        if device_data.empty:
            return issues

        avg_cells = device_data["unique_cells"].mean() if "unique_cells" in device_data.columns else 0
        device_data["cell_signal"].mean() if "cell_signal" in device_data.columns else 0

        # Check for tower hopping
        if avg_cells > self.TOWER_HOPPING_THRESHOLD:
            issues.append(CellularIssue(
                device_id=device_id,
                device_name=None,
                issue_type=NetworkIssueType.TOWER_HOPPING,
                carrier=device_data["carrier"].iloc[0] if "carrier" in device_data.columns else None,
                primary_network_type="LTE",  # Would need actual data
                tower_changes_per_hour=float(avg_cells),
                tech_fallback_rate=0,
                description=f"Device connects to {avg_cells:.0f} cell towers on average (excessive hopping)",
                severity=InsightSeverity.MEDIUM,
                recommendation="May indicate poor cellular coverage in work area",
            ))

        # Check for tech fallback
        time_5g = device_data["time_on_5g"].sum() if "time_on_5g" in device_data.columns else 0
        time_4g = device_data["time_on_4g"].sum() if "time_on_4g" in device_data.columns else 0
        time_3g = device_data["time_on_3g"].sum() if "time_on_3g" in device_data.columns else 0
        total_time = time_5g + time_4g + time_3g

        if total_time > 0:
            fallback_rate = (time_3g + time_4g * 0.5) / total_time  # Weighted fallback
            if fallback_rate > 0.5:  # More than 50% on lower tech
                issues.append(CellularIssue(
                    device_id=device_id,
                    device_name=None,
                    issue_type=NetworkIssueType.TECH_FALLBACK,
                    carrier=device_data["carrier"].iloc[0] if "carrier" in device_data.columns else None,
                    primary_network_type="3G/4G",
                    tower_changes_per_hour=0,
                    tech_fallback_rate=float(fallback_rate),
                    description=f"Device spends {fallback_rate*100:.0f}% of time on lower network technologies",
                    severity=InsightSeverity.LOW,
                    recommendation="Check carrier coverage; device may need 5G-capable SIM or plan",
                ))

        return issues

    def _detect_dead_zones(self, network_data: pd.DataFrame) -> list[str]:
        """Detect potential WiFi dead zones based on signal data."""
        dead_zones: list[str] = []

        if network_data.empty or "wifi_signal" not in network_data.columns:
            return dead_zones

        # Group by location and find locations with consistently weak signal
        if "location_id" in network_data.columns:
            location_signal = network_data.groupby("location_id")["wifi_signal"].agg(["mean", "min"])
            for loc_id, row in location_signal.iterrows():
                if row["mean"] < self.WEAK_SIGNAL_DBM or row["min"] < -85:
                    dead_zones.append(str(loc_id))

        return dead_zones

    def _calculate_carrier_stats(
        self,
        network_data: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Calculate statistics by carrier."""
        if network_data.empty or "carrier" not in network_data.columns:
            return {}

        stats: dict[str, dict[str, float]] = {}

        for carrier in network_data["carrier"].dropna().unique():
            carrier_data = network_data[network_data["carrier"] == carrier]
            stats[carrier] = {
                "devices": len(carrier_data["device_id"].unique()),
                "avg_signal": float(carrier_data["cell_signal"].mean()) if "cell_signal" in carrier_data.columns else 0,
                "avg_cells": float(carrier_data["unique_cells"].mean()) if "unique_cells" in carrier_data.columns else 0,
            }

        return stats

    def _calculate_network_type_distribution(
        self,
        network_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate distribution of time by network type."""
        if network_data.empty:
            return {}

        total_5g = network_data["time_on_5g"].sum() if "time_on_5g" in network_data.columns else 0
        total_4g = network_data["time_on_4g"].sum() if "time_on_4g" in network_data.columns else 0
        total_3g = network_data["time_on_3g"].sum() if "time_on_3g" in network_data.columns else 0
        total_wifi = network_data["time_on_wifi"].sum() if "time_on_wifi" in network_data.columns else 0

        total = total_5g + total_4g + total_3g + total_wifi
        if total == 0:
            return {}

        return {
            "5G": float(total_5g / total * 100),
            "4G/LTE": float(total_4g / total * 100),
            "3G": float(total_3g / total * 100),
            "WiFi": float(total_wifi / total * 100),
        }

    def _detect_disconnect_pattern(
        self,
        network_data: pd.DataFrame,
    ) -> tuple[bool, str | None, float]:
        """Detect if disconnects follow a predictable pattern."""
        # Simplified pattern detection
        # In production, would use time-series analysis

        if network_data.empty:
            return False, None, 0

        # Check if disconnects cluster at certain times
        if "computed_at" in network_data.columns:
            network_data["hour"] = pd.to_datetime(network_data["computed_at"]).dt.hour
            hourly_disconnects = network_data.groupby("hour")["wifi_disconnect_count"].sum()

            if not hourly_disconnects.empty:
                max_hour = hourly_disconnects.idxmax()
                max_count = hourly_disconnects.max()
                avg_count = hourly_disconnects.mean()

                if max_count > avg_count * 2:  # Significant peak
                    return True, f"Peak disconnects at {max_hour}:00", 0.7

        return False, None, 0

    def _analyze_device_hidden_patterns(
        self,
        device_id: int,
        device_data: pd.DataFrame,
        period_days: int,
    ) -> HiddenDeviceIndicator | None:
        """Analyze if a device shows hidden/off-site patterns."""
        if device_data.empty:
            return None

        # Calculate offline hours
        offline_time = device_data["time_no_network"].sum() if "time_no_network" in device_data.columns else 0
        offline_hours = offline_time / 60  # Assuming minutes

        # Expected hours (8 hrs/day work)
        expected_offline = period_days * 16  # 16 hrs offline per day is normal

        indicators: list[str] = []
        hidden_score = 0

        # Check if significantly more offline than expected
        if offline_hours > expected_offline * 1.2:
            hidden_score += 0.3
            indicators.append(f"High offline time ({offline_hours:.0f}h vs {expected_offline:.0f}h expected)")

        # Check for complete absence
        if offline_hours > period_days * 23:  # Basically always offline
            hidden_score += 0.5
            indicators.append("Device almost never online")

        if hidden_score < 0.5:
            return None

        return HiddenDeviceIndicator(
            device_id=device_id,
            device_name=None,
            assigned_location=device_data["location_id"].iloc[0] if "location_id" in device_data.columns else None,
            offline_hours_last_week=float(offline_hours),
            unusual_offline_periods=0,
            last_seen_location=None,
            never_seen_at_assigned=False,
            hidden_score=min(1.0, hidden_score),
            confidence=0.6,
            indicators=indicators,
        )

    def _generate_wifi_recommendations(
        self,
        issues: list[WifiRoamingIssue],
        dead_zones: list[str],
        hopping_count: int,
        stickiness_count: int,
        total_devices: int,
    ) -> list[str]:
        """Generate WiFi recommendations."""
        recommendations = []

        if hopping_count > total_devices * 0.1:
            recommendations.append(
                f"{hopping_count} devices ({hopping_count/total_devices*100:.0f}%) show excessive AP roaming. "
                "Review AP placement and roaming thresholds."
            )

        if stickiness_count > total_devices * 0.1:
            recommendations.append(
                f"{stickiness_count} devices aren't roaming when they should. "
                "Check roaming sensitivity settings."
            )

        if dead_zones:
            recommendations.append(
                f"Potential dead zones detected: {', '.join(dead_zones)}. "
                "Consider adding AP coverage."
            )

        return recommendations

    def _generate_cellular_recommendations(
        self,
        issues: list[CellularIssue],
        carrier_stats: dict[str, dict[str, float]],
        tower_hopping_count: int,
        total_devices: int,
    ) -> list[str]:
        """Generate cellular recommendations."""
        recommendations = []

        if tower_hopping_count > total_devices * 0.1:
            recommendations.append(
                f"{tower_hopping_count} devices show excessive tower hopping. "
                "May indicate poor cellular coverage."
            )

        # Carrier comparison
        if len(carrier_stats) > 1:
            by_signal = sorted(carrier_stats.items(), key=lambda x: x[1].get("avg_signal", 0), reverse=True)
            best = by_signal[0]
            worst = by_signal[-1]
            if best[1].get("avg_signal", 0) - worst[1].get("avg_signal", 0) > 10:
                recommendations.append(
                    f"{best[0]} has better signal ({best[1]['avg_signal']:.0f} dBm) "
                    f"than {worst[0]} ({worst[1]['avg_signal']:.0f} dBm)."
                )

        return recommendations

    def _generate_disconnect_recommendations(
        self,
        total_disconnects: int,
        has_pattern: bool,
        total_devices: int,
    ) -> list[str]:
        """Generate disconnect recommendations."""
        recommendations = []

        avg_disconnects = total_disconnects / total_devices if total_devices > 0 else 0

        if avg_disconnects > 10:
            recommendations.append(
                f"High disconnect rate ({avg_disconnects:.1f} per device). "
                "Review network infrastructure."
            )

        if has_pattern:
            recommendations.append(
                "Disconnects follow a pattern - may be related to scheduled events or infrastructure issues."
            )

        return recommendations

    def _generate_hidden_recommendations(
        self,
        flagged: list[HiddenDeviceIndicator],
        highly_suspicious: int,
    ) -> list[str]:
        """Generate hidden device recommendations."""
        recommendations = []

        if highly_suspicious > 0:
            recommendations.append(
                f"{highly_suspicious} devices are highly suspicious for being hidden/off-site. "
                "Recommend physical audit."
            )

        if flagged:
            recommendations.append(
                "Review check-in/check-out procedures for flagged devices."
            )

        return recommendations
