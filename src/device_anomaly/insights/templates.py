"""Business language templates for customer-facing insights.

Translates technical anomaly data into plain language that customers
can understand without AM/SE explanation.

Carl's key principle: "XSight has the data. XSight needs the story."

Design principles:
1. Lead with business impact - Never show raw scores
2. Always provide comparison - Context is critical
3. Aggregate to actionable level - Roll up to locations/users
4. One click to action - Pre-populate tickets and actions
5. Tell the story - Complete narrative with what/why/how-bad/what-to-do
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from device_anomaly.insights.categories import InsightCategory


@dataclass
class InsightTemplate:
    """Template for generating customer-facing insight text."""

    # Required fields (no defaults) - must come first
    headline: str
    impact: str
    comparison: str

    # Optional fields (with defaults) - must come after required fields
    headline_device: str | None = None  # For single-device context
    headline_location: str | None = None  # For location-level aggregation
    impact_short: str | None = None  # Brief version for cards
    comparison_fleet: str | None = None  # Compare to entire fleet
    comparison_location: str | None = None  # Compare to other locations
    comparison_historical: str | None = None  # Compare to past
    root_cause_hints: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    ticket_description: str | None = None  # For ServiceNow integration

    # Financial impact templates (for cost-aware insights)
    financial_impact_template: str | None = None  # Template for financial summary
    financial_breakdown_template: str | None = None  # Template for cost breakdown
    roi_template: str | None = None  # Template for ROI messaging
    cost_category: str | None = None  # hardware, labor, downtime, opportunity


# ============================================================================
# BATTERY INSIGHT TEMPLATES
# Carl: "Batteries that don't last a shift"
# ============================================================================

BATTERY_SHIFT_FAILURE_TEMPLATE = InsightTemplate(
    headline="{count} device(s) in {location} won't last a full {shift_name} shift",
    headline_device="This device won't last the {shift_name} shift",
    headline_location="{location} has {count} devices that won't complete their shift",
    impact="At current drain rate ({drain_rate:.1f}%/hr), battery will die by {estimated_dead_time}. "
    "Workers may experience {downtime_hours:.1f} hours of unplanned downtime.",
    impact_short="Battery will die by {estimated_dead_time}",
    comparison="{percent_worse:.0f}% faster drain than peer devices",
    comparison_fleet="Draining {multiplier:.1f}x faster than fleet average ({fleet_avg:.1f}%/hr)",
    comparison_location="{location} devices drain {percent_diff:.0f}% faster than {comparison_location}",
    comparison_historical="Drain rate increased {percent_change:.0f}% vs last week",
    root_cause_hints=[
        "Check if device was fully charged before shift start",
        "Review app usage - power-hungry apps may be running",
        "Battery health may be degraded (check cycle count)",
        "Screen brightness or always-on settings may be too aggressive",
    ],
    actions=[
        "Ensure device is charged to 100% before shift start",
        "Check battery health status - consider replacement if below 80%",
        "Review and restrict background app activity",
        "Adjust screen brightness and timeout settings",
    ],
    ticket_description="""Device Battery Shift Failure Risk

Location: {location}
Affected Devices: {count}
Current Drain Rate: {drain_rate:.1f}%/hour
Estimated Failure Time: {estimated_dead_time}

Impact: Workers may experience {downtime_hours:.1f} hours of unplanned downtime.

Comparison: {percent_worse:.0f}% faster drain than peer devices in the fleet.

Recommended Actions:
1. Verify charging procedures are being followed
2. Check battery health on affected devices
3. Review power-hungry app usage
""",
    financial_impact_template="Estimated downtime cost: ${downtime_impact_usd:,.0f}. "
    "Worker productivity loss: ${labor_impact_usd:,.0f}.",
    roi_template="Proper charging procedures could save ${potential_savings_usd:,.0f}/month.",
    cost_category="downtime",
)

BATTERY_RAPID_DRAIN_TEMPLATE = InsightTemplate(
    headline="{count} device(s) draining battery {multiplier:.1f}x faster than normal",
    headline_device="Battery draining {multiplier:.1f}x faster than normal",
    impact="Current drain rate: {drain_rate:.1f}%/hr (normal: {baseline:.1f}%/hr). "
    "Expect {hours_to_dead:.1f} hours of battery life instead of {expected_hours:.1f} hours.",
    impact_short="Draining {multiplier:.1f}x faster than expected",
    comparison="Normal drain rate for this device type is {baseline:.1f}%/hr",
    comparison_fleet="Fleet average is {fleet_avg:.1f}%/hr - this device is in the worst {percentile}%",
    root_cause_hints=[
        "A specific app may be consuming excessive power",
        "Device may have poor cellular signal causing radio to work harder",
        "Background sync or location services may be overactive",
        "Battery health may have degraded",
    ],
    actions=[
        "Check per-app battery usage to identify power hogs",
        "Review background app restrictions",
        "Check signal strength - poor signal increases power consumption",
        "Consider battery replacement if health is degraded",
    ],
)

BATTERY_CHARGE_INCOMPLETE_TEMPLATE = InsightTemplate(
    headline="{count} device(s) in {location} weren't fully charged this morning",
    headline_device="Device started shift at only {battery_level:.0f}% charge",
    impact="Starting shift at {avg_battery:.0f}% instead of 100% reduces effective work time "
    "by approximately {lost_hours:.1f} hours.",
    impact_short="Started at {avg_battery:.0f}% instead of 100%",
    comparison="{comparison_location} has {comparison_pct:.0f}% fully-charged devices vs {this_pct:.0f}% here",
    comparison_fleet="{fleet_pct:.0f}% of fleet devices start fully charged",
    root_cause_hints=[
        "Charging cradles may be damaged or insufficient",
        "Night shift may be removing devices from chargers prematurely",
        "Charging area may have power issues",
        "Devices may not be properly seated in cradles",
    ],
    actions=[
        "Audit charging cradle availability and condition",
        "Review overnight charging procedures with staff",
        "Check power supply to charging stations",
        "Consider adding charging status indicators",
    ],
)

BATTERY_CHARGE_PATTERN_TEMPLATE = InsightTemplate(
    headline="{count} device(s) showing poor charging patterns",
    headline_device="Device has {bad_ratio:.0f}% poor charge sessions",
    impact="Poor charging (short, interrupted charges) degrades battery health faster. "
    "Expected battery lifespan reduced by approximately {lifespan_reduction:.0f}%.",
    impact_short="{bad_count} of {total_count} charges were suboptimal",
    comparison="Fleet average is {fleet_bad_pct:.0f}% poor charges - this is {percent_worse:.0f}% worse",
    root_cause_hints=[
        "Devices being taken off charger before full charge",
        "Faulty charging cables or adapters",
        "Using USB charging instead of AC (slower, less reliable)",
        "Charging station power fluctuations",
    ],
    actions=[
        "Train staff on proper charging procedures (charge to 100%)",
        "Replace damaged charging cables",
        "Upgrade to AC chargers where possible",
        "Check charging station power stability",
    ],
)

BATTERY_HEALTH_DEGRADED_TEMPLATE = InsightTemplate(
    headline="{count} device(s) with degraded battery health (below {threshold:.0f}%)",
    headline_device="Battery health is only {health:.0f}% - replacement recommended",
    impact="Degraded battery health means reduced capacity. Device may only hold "
    "{effective_capacity:.0f}% of original charge, reducing work time proportionally.",
    impact_short="Battery at {health:.0f}% health - needs replacement",
    comparison="Normal battery health is 80-100%. This device is in the bottom {percentile}%.",
    comparison_fleet="{fleet_count} devices across the fleet need battery replacement",
    root_cause_hints=[
        "Normal battery aging after {cycle_count} charge cycles",
        "Poor charging habits accelerated degradation",
        "Exposure to extreme temperatures",
        "Manufacturing defect (if relatively new device)",
    ],
    actions=[
        "Schedule battery replacement for affected devices",
        "Prioritize devices below 70% health",
        "Review warranty status for potential free replacement",
        "Implement charging best practices to slow future degradation",
    ],
)

# ============================================================================
# DEVICE INSIGHT TEMPLATES
# Carl: "Devices/people/locations with excessive drops/reboots"
# ============================================================================

EXCESSIVE_DROPS_TEMPLATE = InsightTemplate(
    headline="{entity_type} '{entity_name}' has {drop_count} device drops in {period}",
    headline_device="This device has been dropped {drop_count} times in {period}",
    headline_location="{location} has {drop_count} total device drops - {multiplier:.1f}x higher than average",
    impact="Estimated repair costs: ${repair_cost_estimate:,.0f}. "
    "Device downtime: approximately {downtime_hours:.1f} hours. "
    "Increased risk of screen damage and data loss.",
    impact_short="${repair_cost_estimate:,.0f} estimated repair costs",
    comparison="This ranks #{rank} worst out of {total} {entity_type}s",
    comparison_fleet="Fleet average is {fleet_avg:.1f} drops per device - this is {multiplier:.1f}x higher",
    comparison_location="{comparison_location} has {comparison_drops} drops - {percent_diff:.0f}% fewer",
    root_cause_hints=[
        "Check if protective cases are being used",
        "Review handling training for staff",
        "Identify high-risk work areas (near conveyor belts, heights, etc.)",
        "Consider if device form factor is appropriate for the work",
    ],
    actions=[
        "Deploy protective cases on affected devices",
        "Conduct handling training for staff at {location}",
        "Investigate work areas for drop hazards",
        "Consider wrist straps or lanyards for high-risk roles",
    ],
    ticket_description="""Excessive Device Drops Detected

{entity_type}: {entity_name}
Drop Count: {drop_count} in {period}
Compared to Fleet: {multiplier:.1f}x higher than average

Estimated Impact:
- Repair costs: ${repair_cost_estimate:,.0f}
- Downtime: {downtime_hours:.1f} hours

Recommended Actions:
1. Deploy protective cases
2. Conduct handling training
3. Investigate work area hazards
""",
    financial_impact_template="Estimated repair costs: ${hardware_impact_usd:,.0f}. "
    "Productivity loss: ${labor_impact_usd:,.0f}. "
    "Total impact: ${total_impact_usd:,.0f}.",
    financial_breakdown_template="Hardware: ${hardware_impact_usd:,.0f} | "
    "Labor: ${labor_impact_usd:,.0f}",
    roi_template="Investing ${investment_required_usd:,.0f} in protective cases could save "
    "${potential_savings_usd:,.0f}/month.",
    cost_category="hardware",
)

EXCESSIVE_REBOOTS_TEMPLATE = InsightTemplate(
    headline="{count} device(s) experiencing excessive reboots ({reboot_count} in {period})",
    headline_device="Device rebooted {reboot_count} times in {period}",
    impact="Each reboot causes {reboot_downtime:.0f} minutes of downtime. "
    "Total productivity loss: approximately {total_downtime:.0f} minutes.",
    impact_short="{total_downtime:.0f} minutes lost to reboots",
    comparison="Normal is {normal_reboots} reboots per {period} - this is {multiplier:.1f}x higher",
    comparison_fleet="Ranks in worst {percentile}% of devices for stability",
    root_cause_hints=[
        "App crashes causing system instability",
        "Memory pressure from too many apps",
        "OS or firmware bug",
        "Hardware issue (overheating, memory failure)",
    ],
    actions=[
        "Check for app crash patterns preceding reboots",
        "Review memory usage and close unnecessary apps",
        "Check for available OS/firmware updates",
        "If hardware issue suspected, schedule device inspection",
    ],
)

DEVICE_ABUSE_PATTERN_TEMPLATE = InsightTemplate(
    headline="Device abuse pattern detected: {drop_count} drops and {reboot_count} reboots",
    headline_location="{location} showing device abuse pattern across {device_count} devices",
    impact="Combined pattern indicates systematic device mishandling. "
    "Estimated impact: ${total_cost:,.0f} in repairs and {productivity_loss:.0f} hours lost productivity.",
    impact_short="Systematic mishandling detected",
    comparison="This {entity_type} has {multiplier:.1f}x more incidents than average",
    root_cause_hints=[
        "User may need training on proper device handling",
        "Work environment may be hazardous for devices",
        "Device may not be appropriate for this use case",
        "Intentional misuse cannot be ruled out",
    ],
    actions=[
        "Meet with affected user(s) to discuss device handling",
        "Assess work environment for device hazards",
        "Consider rugged device options for this role",
        "Implement device monitoring and accountability",
    ],
)

# ============================================================================
# NETWORK INSIGHT TEMPLATES
# Carl: "AP hopping/stickiness", "Tower hopping", "Carrier patterns"
# ============================================================================

WIFI_AP_HOPPING_TEMPLATE = InsightTemplate(
    headline="Excessive WiFi roaming in {location}: {ap_count} access points per day",
    headline_device="Device connected to {ap_count} different access points in one day",
    impact="Frequent AP switching causes {disconnect_minutes:.0f} minutes of connection gaps. "
    "Data sync delays and potential transaction failures.",
    impact_short="{disconnect_minutes:.0f} minutes of connectivity gaps",
    comparison="{comparison_location} averages {comparison_ap_count} APs/day - {percent_diff:.0f}% less switching",
    comparison_fleet="Fleet average is {fleet_avg:.1f} APs/day",
    root_cause_hints=[
        "AP coverage overlap may be insufficient in {location}",
        "Device WiFi roaming settings may be too aggressive",
        "Interference from warehouse equipment",
        "Some APs may have configuration issues",
    ],
    actions=[
        "Survey WiFi coverage and adjust AP placement",
        "Review device roaming aggressiveness settings",
        "Check for RF interference sources",
        "Verify all APs have consistent configuration",
    ],
)

WIFI_DEAD_ZONE_TEMPLATE = InsightTemplate(
    headline="WiFi dead zone detected in {zone_name} at {location}",
    headline_location="{location} has {dead_zone_count} WiFi dead zones",
    impact="{device_count} devices regularly experience connectivity loss in this area. "
    "Estimated {transaction_failures} failed transactions per day.",
    impact_short="{device_count} devices affected",
    comparison="Signal strength {signal_strength} dBm vs required minimum {min_required} dBm",
    root_cause_hints=[
        "Insufficient AP coverage in this area",
        "Physical obstructions (walls, equipment, shelving)",
        "RF interference from other equipment",
        "AP failure or misconfiguration",
    ],
    actions=[
        "Add access point coverage to {zone_name}",
        "Check for and remove RF interference sources",
        "Consider directional antennas for better coverage",
        "Verify existing APs are functioning properly",
    ],
)

CELLULAR_TOWER_HOPPING_TEMPLATE = InsightTemplate(
    headline="{count} device(s) with excessive cell tower switching",
    headline_device="Device connected to {tower_count} different towers in one day",
    impact="Frequent tower switching indicates poor cellular coverage. "
    "May cause {disconnect_count} disconnections and {data_retry_pct:.0f}% data retry overhead.",
    impact_short="{disconnect_count} cellular disconnections",
    comparison="Normal is {normal_towers} towers/day - this is {multiplier:.1f}x higher",
    root_cause_hints=[
        "Device location has marginal cellular coverage",
        "Carrier network congestion causing forced handoffs",
        "Device antenna or radio issues",
        "SIM card problems",
    ],
    actions=[
        "Check carrier coverage maps for affected areas",
        "Consider WiFi offloading to reduce cellular dependency",
        "Test with different carrier if multi-carrier SIMs available",
        "Inspect device antenna and SIM card",
    ],
)

CELLULAR_CARRIER_ISSUE_TEMPLATE = InsightTemplate(
    headline="{carrier} showing {issue_type} affecting {device_count} devices",
    impact="Carrier-specific issue causing {impact_description}. "
    "Affects {percent_affected:.0f}% of devices on this carrier.",
    impact_short="{carrier}: {issue_type}",
    comparison="Other carriers in this area don't show this issue",
    comparison_location="Same carrier at {comparison_location} performs {percent_better:.0f}% better",
    root_cause_hints=[
        "Carrier network outage or congestion",
        "Local tower maintenance or issues",
        "Carrier-specific device configuration problems",
        "SIM provisioning issues",
    ],
    actions=[
        "Contact {carrier} to report service issues",
        "Consider WiFi as backup connectivity",
        "Check carrier service status page",
        "Review carrier SLA and escalate if needed",
    ],
)

NETWORK_DISCONNECT_PATTERN_TEMPLATE = InsightTemplate(
    headline="{count} device(s) showing predictable disconnect patterns",
    headline_device="Device disconnects regularly at {pattern_time}",
    impact="Predictable offline periods of {offline_duration:.0f} minutes affecting "
    "data sync and remote management capabilities.",
    impact_short="Disconnects at {pattern_time} daily",
    comparison="Fleet average is {fleet_disconnects:.1f} disconnections/day - this is {multiplier:.1f}x higher",
    root_cause_hints=[
        "Device may be moved to area with no coverage at these times",
        "Scheduled maintenance or power-down periods",
        "Device may be taken out of facility (taken home?)",
        "Network infrastructure scheduled maintenance",
    ],
    actions=[
        "Investigate device location during offline periods",
        "Review work schedules for correlation",
        "Check if pattern matches facility operations",
        "Consider policy review for device handling",
    ],
)

DEVICE_HIDDEN_PATTERN_TEMPLATE = InsightTemplate(
    headline="{count} device(s) with suspicious offline patterns",
    headline_device="Device offline for {offline_hours:.1f} hours during expected work time",
    impact="Extended offline periods suggest device may be hidden, taken home, or misused. "
    "Unable to receive updates, policies, or remote commands during this time.",
    impact_short="Suspicious {offline_hours:.1f}+ hour offline periods",
    comparison="Normal devices are offline less than {normal_offline:.1f} hours during work time",
    root_cause_hints=[
        "Device may be taken home by employee",
        "Device may be hidden or powered off intentionally",
        "Device stored in area with no connectivity",
        "Legitimate personal use policy may apply",
    ],
    actions=[
        "Review device location during offline periods",
        "Discuss device usage policy with assigned user",
        "Implement geofencing alerts if policy violation suspected",
        "Consider GPS tracking for high-value devices",
    ],
)

# ============================================================================
# APP INSIGHT TEMPLATES
# Carl: "Crashes", "Apps consuming too much power"
# ============================================================================

APP_CRASH_PATTERN_TEMPLATE = InsightTemplate(
    headline="App '{app_name}' crashing repeatedly: {crash_count} crashes in {period}",
    headline_location="{app_name} crashed {crash_count} times across {device_count} devices",
    impact="Each crash causes {crash_downtime:.0f} minutes of downtime. "
    "Total productivity loss: {total_downtime:.0f} minutes across affected devices.",
    impact_short="{crash_count} crashes, {total_downtime:.0f} min downtime",
    comparison="Normal crash rate is {normal_rate:.1f}/day - this is {multiplier:.1f}x higher",
    comparison_fleet="This app ranks #{rank} in crash frequency across fleet",
    root_cause_hints=[
        "App version may have a bug - check for updates",
        "Incompatibility with device OS version",
        "Insufficient device memory for app requirements",
        "Corrupted app data or configuration",
    ],
    actions=[
        "Check for and install app updates",
        "Clear app cache and data",
        "Verify device meets app requirements",
        "Contact app vendor if issue persists",
    ],
)

APP_POWER_DRAIN_TEMPLATE = InsightTemplate(
    headline="App '{app_name}' consuming {drain_pct:.0f}% battery with only {foreground_pct:.0f}% foreground time",
    impact="This app is using {multiplier:.1f}x more battery than expected for its usage level. "
    "Reducing effective device battery life by approximately {hours_lost:.1f} hours.",
    impact_short="Using {multiplier:.1f}x expected battery",
    comparison="Similar apps average {comparison_drain:.0f}% battery for this usage level",
    comparison_fleet="Fleet average for this app is {fleet_drain:.0f}% - this is {percent_worse:.0f}% higher",
    root_cause_hints=[
        "App may have background sync issues",
        "App version may have a battery bug",
        "App permissions allow excessive background activity",
        "App may be stuck in a loop or error state",
    ],
    actions=[
        "Restrict app background activity",
        "Check for app updates that may fix power issues",
        "Review app permissions (location, sync, etc.)",
        "Consider alternative app if issue persists",
    ],
)

APP_ANR_PATTERN_TEMPLATE = InsightTemplate(
    headline="App '{app_name}' frequently not responding: {anr_count} ANRs in {period}",
    impact="ANR (App Not Responding) events force users to wait or force-close the app. "
    "Estimated {wait_time:.0f} minutes of unproductive waiting time.",
    impact_short="{anr_count} ANRs causing delays",
    comparison="Normal is less than {normal_anr} ANRs per day",
    root_cause_hints=[
        "App performing heavy operations on main thread",
        "Network timeouts causing UI freezes",
        "Database operations taking too long",
        "Insufficient device resources for app",
    ],
    actions=[
        "Update app to latest version",
        "Check network connectivity quality",
        "Ensure device has sufficient free memory",
        "Report issue to app developer",
    ],
)

# ============================================================================
# COHORT INSIGHT TEMPLATES
# Carl: "Performance by manufacturer, model, OS version, firmware"
# ============================================================================

COHORT_PERFORMANCE_ISSUE_TEMPLATE = InsightTemplate(
    headline="{cohort_name} devices performing {percent_worse:.0f}% worse than fleet average",
    impact="{device_count} devices affected. Average {metric_name} of {cohort_value:.1f} "
    "vs fleet average of {fleet_value:.1f}.",
    impact_short="{device_count} devices underperforming",
    comparison="Other {comparison_cohort} devices perform {percent_better:.0f}% better",
    root_cause_hints=[
        "Device model may have hardware limitations",
        "OS version may have performance issues",
        "Firmware may need updating",
        "Device specifications may not meet workload requirements",
    ],
    actions=[
        "Check for firmware/OS updates for this cohort",
        "Review if these devices meet current requirements",
        "Consider device refresh for oldest units",
        "Adjust workload or app configuration for these devices",
    ],
)

PROBLEM_COMBINATION_TEMPLATE = InsightTemplate(
    headline="{manufacturer} {model} on {os_version} with firmware {firmware} showing issues",
    impact="{device_count} devices with this combination affected by {issue_type}. "
    "This specific combination has {multiplier:.1f}x the issue rate of other configurations.",
    impact_short="{device_count} devices with problem config",
    comparison="Other {manufacturer} models don't show this issue",
    comparison_fleet="This combination ranks worst for {issue_type} across fleet",
    root_cause_hints=[
        "Known incompatibility between OS and firmware versions",
        "Manufacturer-specific bug in this configuration",
        "This combination may have skipped a critical update",
        "Hardware revision issue specific to this model",
    ],
    actions=[
        "Check manufacturer knowledge base for known issues",
        "Apply firmware updates if available",
        "Consider OS downgrade/upgrade to stable version",
        "File support ticket with manufacturer",
    ],
)

# ============================================================================
# LOCATION INSIGHT TEMPLATES
# Carl: "Relate any anomalies to location (warehouse 1 vs warehouse 2)"
# ============================================================================

LOCATION_ANOMALY_CLUSTER_TEMPLATE = InsightTemplate(
    headline="{location} has {anomaly_count} devices with {issue_type}",
    impact="This concentration suggests a location-specific cause. "
    "{percent_affected:.0f}% of devices at this location are affected.",
    impact_short="{percent_affected:.0f}% of devices affected",
    comparison="Other locations average {other_avg:.0f}% - this is {multiplier:.1f}x higher",
    comparison_location="{comparison_location} has only {comparison_count} devices with this issue",
    root_cause_hints=[
        "Infrastructure issue at this location",
        "Environmental factors (temperature, interference)",
        "Work practices specific to this location",
        "Equipment or facility issues",
    ],
    actions=[
        "Investigate location infrastructure",
        "Compare practices with better-performing locations",
        "Check environmental conditions",
        "Review location-specific equipment and facilities",
    ],
)

LOCATION_PERFORMANCE_GAP_TEMPLATE = InsightTemplate(
    headline="{location_worse} has {metric_name} {percent_diff:.0f}% worse than {location_better}",
    impact="If {location_worse} matched {location_better}'s performance, "
    "estimated savings of {savings_estimate}.",
    impact_short="{percent_diff:.0f}% worse than best location",
    comparison="Key differences: {key_differences}",
    comparison_fleet="{location_worse} ranks #{rank} out of {total_locations} locations",
    root_cause_hints=[
        "Infrastructure differences between locations",
        "Process or procedure differences",
        "Equipment age or condition differences",
        "Staff training or practices differences",
    ],
    actions=[
        "Review {location_better}'s practices and infrastructure",
        "Conduct site assessment at {location_worse}",
        "Share best practices between locations",
        "Invest in infrastructure improvements",
    ],
    ticket_description="""Location Performance Gap Analysis

Lower Performing: {location_worse}
Better Performing: {location_better}
Performance Gap: {percent_diff:.0f}% on {metric_name}

Key Differences:
{key_differences}

If performance gap is closed:
Estimated Savings: {savings_estimate}

Recommended Actions:
1. Conduct site assessment at {location_worse}
2. Review infrastructure and processes at {location_better}
3. Implement best practice sharing program
""",
)

LOCATION_BASELINE_DEVIATION_TEMPLATE = InsightTemplate(
    headline="{location} deviating from its normal pattern: {metric_name} {direction} by {percent_change:.0f}%",
    impact="Something changed at {location}. Current {metric_name}: {current_value:.1f} "
    "vs baseline: {baseline_value:.1f}.",
    impact_short="{percent_change:.0f}% {direction} from normal",
    comparison="This {direction} started approximately {days_ago} days ago",
    root_cause_hints=[
        "Recent change in location operations",
        "New equipment or infrastructure changes",
        "Seasonal or cyclical patterns",
        "Staff or process changes",
    ],
    actions=[
        "Identify what changed around {change_date}",
        "Review recent location events and changes",
        "Determine if this is expected or concerning",
        "Take corrective action if necessary",
    ],
)


# ============================================================================
# TEMPLATE REGISTRY
# ============================================================================

INSIGHT_TEMPLATES: dict[InsightCategory, InsightTemplate] = {
    # Battery
    InsightCategory.BATTERY_SHIFT_FAILURE: BATTERY_SHIFT_FAILURE_TEMPLATE,
    InsightCategory.BATTERY_RAPID_DRAIN: BATTERY_RAPID_DRAIN_TEMPLATE,
    InsightCategory.BATTERY_CHARGE_INCOMPLETE: BATTERY_CHARGE_INCOMPLETE_TEMPLATE,
    InsightCategory.BATTERY_CHARGE_PATTERN: BATTERY_CHARGE_PATTERN_TEMPLATE,
    InsightCategory.BATTERY_HEALTH_DEGRADED: BATTERY_HEALTH_DEGRADED_TEMPLATE,
    # Device
    InsightCategory.EXCESSIVE_DROPS: EXCESSIVE_DROPS_TEMPLATE,
    InsightCategory.EXCESSIVE_REBOOTS: EXCESSIVE_REBOOTS_TEMPLATE,
    InsightCategory.DEVICE_ABUSE_PATTERN: DEVICE_ABUSE_PATTERN_TEMPLATE,
    # Network
    InsightCategory.WIFI_AP_HOPPING: WIFI_AP_HOPPING_TEMPLATE,
    InsightCategory.WIFI_DEAD_ZONE: WIFI_DEAD_ZONE_TEMPLATE,
    InsightCategory.CELLULAR_TOWER_HOPPING: CELLULAR_TOWER_HOPPING_TEMPLATE,
    InsightCategory.CELLULAR_CARRIER_ISSUE: CELLULAR_CARRIER_ISSUE_TEMPLATE,
    InsightCategory.NETWORK_DISCONNECT_PATTERN: NETWORK_DISCONNECT_PATTERN_TEMPLATE,
    InsightCategory.DEVICE_HIDDEN_PATTERN: DEVICE_HIDDEN_PATTERN_TEMPLATE,
    # Apps
    InsightCategory.APP_CRASH_PATTERN: APP_CRASH_PATTERN_TEMPLATE,
    InsightCategory.APP_POWER_DRAIN: APP_POWER_DRAIN_TEMPLATE,
    InsightCategory.APP_ANR_PATTERN: APP_ANR_PATTERN_TEMPLATE,
    # Cohort
    InsightCategory.COHORT_PERFORMANCE_ISSUE: COHORT_PERFORMANCE_ISSUE_TEMPLATE,
    InsightCategory.PROBLEM_COMBINATION: PROBLEM_COMBINATION_TEMPLATE,
    # Location
    InsightCategory.LOCATION_ANOMALY_CLUSTER: LOCATION_ANOMALY_CLUSTER_TEMPLATE,
    InsightCategory.LOCATION_PERFORMANCE_GAP: LOCATION_PERFORMANCE_GAP_TEMPLATE,
    InsightCategory.LOCATION_BASELINE_DEVIATION: LOCATION_BASELINE_DEVIATION_TEMPLATE,
}


def get_template(category: InsightCategory) -> InsightTemplate | None:
    """Get the template for an insight category."""
    return INSIGHT_TEMPLATES.get(category)


def render_headline(
    category: InsightCategory, data: dict[str, Any], context: str = "default"
) -> str:
    """Render the headline for an insight.

    Args:
        category: The insight category
        data: Data dict with values to substitute
        context: "default", "device", or "location" for context-specific headline

    Returns:
        Rendered headline string
    """
    template = get_template(category)
    if not template:
        return f"Issue detected: {category.value}"

    headline_template = template.headline
    if context == "device" and template.headline_device:
        headline_template = template.headline_device
    elif context == "location" and template.headline_location:
        headline_template = template.headline_location

    try:
        return headline_template.format(**data)
    except KeyError as e:
        return f"Issue detected: {category.value} (missing data: {e})"


def render_impact(category: InsightCategory, data: dict[str, Any], short: bool = False) -> str:
    """Render the impact statement for an insight."""
    template = get_template(category)
    if not template:
        return "Business impact pending analysis"

    impact_template = template.impact_short if short and template.impact_short else template.impact

    try:
        return impact_template.format(**data)
    except KeyError:
        return "Business impact pending analysis"


def render_comparison(
    category: InsightCategory, data: dict[str, Any], comparison_type: str = "default"
) -> str:
    """Render the comparison context for an insight.

    Args:
        category: The insight category
        data: Data dict with values to substitute
        comparison_type: "default", "fleet", "location", or "historical"
    """
    template = get_template(category)
    if not template:
        return ""

    comparison_template = template.comparison
    if comparison_type == "fleet" and template.comparison_fleet:
        comparison_template = template.comparison_fleet
    elif comparison_type == "location" and template.comparison_location:
        comparison_template = template.comparison_location
    elif comparison_type == "historical" and template.comparison_historical:
        comparison_template = template.comparison_historical

    try:
        return comparison_template.format(**data)
    except KeyError:
        return ""


def render_actions(category: InsightCategory, data: dict[str, Any]) -> list[str]:
    """Render the recommended actions for an insight."""
    template = get_template(category)
    if not template:
        return ["Investigate and take appropriate action"]

    actions = []
    for action_template in template.actions:
        try:
            actions.append(action_template.format(**data))
        except KeyError:
            actions.append(action_template)  # Return unformatted if data missing

    return actions


def render_ticket_description(category: InsightCategory, data: dict[str, Any]) -> str | None:
    """Render the ServiceNow ticket description for an insight."""
    template = get_template(category)
    if not template or not template.ticket_description:
        return None

    try:
        return template.ticket_description.format(**data)
    except KeyError:
        return None


def render_financial_impact(
    category: InsightCategory,
    data: dict[str, Any],
    include_breakdown: bool = True,
    include_roi: bool = False,
) -> str | None:
    """Render the financial impact statement for an insight.

    Args:
        category: The insight category
        data: Data dict with financial values to substitute (from CostCalculator)
        include_breakdown: Whether to include cost breakdown
        include_roi: Whether to include ROI messaging

    Returns:
        Rendered financial impact string or None if no template
    """
    template = get_template(category)
    if not template or not template.financial_impact_template:
        return None

    parts = []

    # Main financial impact
    try:
        parts.append(template.financial_impact_template.format(**data))
    except KeyError:
        return None

    # Optional breakdown
    if include_breakdown and template.financial_breakdown_template:
        with contextlib.suppress(KeyError):
            parts.append(template.financial_breakdown_template.format(**data))

    # Optional ROI
    if include_roi and template.roi_template:
        with contextlib.suppress(KeyError):
            parts.append(template.roi_template.format(**data))

    return " ".join(parts) if parts else None


def get_cost_category(category: InsightCategory) -> str | None:
    """Get the cost category for an insight type.

    Returns the primary cost category (hardware, labor, downtime, opportunity)
    for an insight, used to determine which cost calculator to use.
    """
    template = get_template(category)
    return template.cost_category if template else None
