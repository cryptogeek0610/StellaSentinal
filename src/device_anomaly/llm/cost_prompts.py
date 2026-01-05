"""Cost-aware prompt templates for LLM financial impact explanations.

This module provides prompt templates that include pre-calculated financial
figures. The LLM should NEVER calculate financial amounts - all figures
are pre-computed and injected into these templates.

Design principle: Inject exact figures, instruct LLM to use them verbatim.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from device_anomaly.costs.models import InsightFinancialData


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

COST_AWARE_SYSTEM_PROMPT = """
You are an enterprise device management assistant that provides business-focused
analysis with financial impact context.

CRITICAL RULES FOR FINANCIAL FIGURES:
1. ONLY use the exact financial figures provided in the FINANCIAL IMPACT section
2. DO NOT invent, calculate, or estimate any monetary amounts
3. DO NOT round or modify the provided figures
4. If a financial figure is not provided, say "financial impact being calculated"
5. Format currency as provided (e.g., $1,234 not $1234 or 1,234 dollars)
6. Never use phrases like "approximately", "around", or "roughly" for amounts

You help operations teams understand:
- The business impact of device anomalies
- Cost implications of fleet issues
- ROI of addressing problems proactively
"""


# =============================================================================
# ANOMALY EXPLANATION PROMPTS
# =============================================================================

COST_AWARE_ANOMALY_PROMPT = """
You are an assistant explaining device anomalies with BUSINESS IMPACT context.

IMPORTANT FINANCIAL RULES:
1. Use ONLY the pre-calculated financial figures provided below
2. DO NOT invent new amounts or perform calculations
3. If financial data is missing, say "Financial impact being calculated"
4. Present costs as provided without modification

TECHNICAL DATA:
{anomaly_json}

{financial_context}

Please provide:
1. A brief technical summary of the anomaly (2-3 sentences)
2. The business impact using the EXACT figures above
3. A prioritization recommendation based on the impact level

Keep the response under 200 words. Use the exact dollar amounts provided.
"""


COST_AWARE_ANOMALY_PROMPT_MINIMAL = """
Explain this device anomaly with its business impact.

TECHNICAL DATA:
{anomaly_json}

{financial_context}

Provide:
1. What's wrong (1-2 sentences)
2. Business impact (use exact figures above)
3. Recommended action

Keep response under 100 words.
"""


# =============================================================================
# INSIGHT EXPLANATION PROMPTS
# =============================================================================

COST_AWARE_INSIGHT_PROMPT = """
You are an assistant explaining fleet-level insights with FINANCIAL IMPACT.

IMPORTANT FINANCIAL RULES:
1. Use ONLY the pre-calculated financial figures provided below
2. DO NOT invent new amounts or perform calculations
3. Present all costs exactly as provided
4. Do not add percentages or ratios not provided

INSIGHT DATA:
{insight_json}

{financial_context}

Please provide:
1. A clear summary of the fleet issue (2-3 sentences)
2. The total financial impact using EXACT figures provided
3. Breakdown by cost type if available
4. Top 2-3 recommendations from the data

Keep the response under 250 words. Use exact dollar amounts.
"""


INSIGHT_SUMMARY_PROMPT = """
Create a brief executive summary of this fleet insight.

{financial_context}

Write 2-3 sentences for a non-technical audience focusing on:
- What's happening (plain English)
- Financial impact (exact figures only)
- Recommended action

Keep under 75 words.
"""


# =============================================================================
# UNUSED DEVICES PROMPT
# =============================================================================

UNUSED_DEVICES_PROMPT = """
Explain the business impact of unused devices.

FLEET DATA:
- Unused devices: {device_count}
- Days unused: {days_unused}
- Total fleet affected: {percent_affected:.1f}%

{financial_context}

Explain:
1. Why this matters to the business
2. The cost implications (use exact figures above)
3. Suggested reallocation strategy

Keep response under 150 words.
"""


# =============================================================================
# BATTERY HEALTH PROMPT
# =============================================================================

BATTERY_HEALTH_PROMPT = """
Explain the impact of degraded battery health across the fleet.

BATTERY DATA:
- Devices needing replacement: {devices_count}
- Average battery health: {avg_health:.0f}%
- Critical devices (under 20%): {critical_count}

{financial_context}

Explain:
1. The operational risk
2. Cost breakdown (use exact figures above)
3. Recommended replacement schedule

Keep response under 150 words.
"""


# =============================================================================
# DROP DAMAGE PROMPT
# =============================================================================

DROP_DAMAGE_PROMPT = """
Analyze the potential impact of device drop incidents.

DROP DATA:
- Total drops detected: {total_drops}
- Devices with drops: {affected_devices}
- Average drops per device: {drops_per_device:.1f}

{financial_context}

Explain:
1. The damage risk based on drop frequency
2. Estimated repair costs (use exact figures above)
3. Prevention recommendations

Keep response under 150 words.
"""


# =============================================================================
# DOWNTIME PROMPT
# =============================================================================

DOWNTIME_IMPACT_PROMPT = """
Explain the business impact of device downtime incidents.

DOWNTIME DATA:
- Incidents: {incident_count}
- Average duration: {avg_duration_minutes:.0f} minutes
- Devices affected: {affected_devices}
- Total downtime: {total_hours:.1f} hours

{financial_context}

Explain:
1. Productivity impact
2. Cost breakdown (use exact figures above)
3. Root cause investigation priorities

Keep response under 150 words.
"""


# =============================================================================
# TEMPLATE FUNCTIONS
# =============================================================================

def build_financial_context(financial_data: Optional[InsightFinancialData]) -> str:
    """Build the financial context section for prompts.

    Args:
        financial_data: Pre-calculated financial data.

    Returns:
        Formatted financial context string.
    """
    if not financial_data:
        return "FINANCIAL IMPACT: Data being calculated..."

    return financial_data.to_prompt_context()


def build_cost_aware_anomaly_prompt(
    anomaly_json: str,
    financial_data: Optional[InsightFinancialData],
    minimal: bool = False,
) -> str:
    """Build a cost-aware prompt for anomaly explanation.

    Args:
        anomaly_json: JSON string of anomaly data.
        financial_data: Pre-calculated financial data.
        minimal: Whether to use minimal prompt template.

    Returns:
        Complete prompt string.
    """
    financial_context = build_financial_context(financial_data)

    template = COST_AWARE_ANOMALY_PROMPT_MINIMAL if minimal else COST_AWARE_ANOMALY_PROMPT

    return template.format(
        anomaly_json=anomaly_json,
        financial_context=financial_context,
    )


def build_cost_aware_insight_prompt(
    insight_json: str,
    financial_data: Optional[InsightFinancialData],
) -> str:
    """Build a cost-aware prompt for insight explanation.

    Args:
        insight_json: JSON string of insight data.
        financial_data: Pre-calculated financial data.

    Returns:
        Complete prompt string.
    """
    financial_context = build_financial_context(financial_data)

    return COST_AWARE_INSIGHT_PROMPT.format(
        insight_json=insight_json,
        financial_context=financial_context,
    )


def build_unused_devices_prompt(
    device_count: int,
    days_unused: int,
    percent_affected: float,
    financial_data: Optional[InsightFinancialData],
) -> str:
    """Build prompt for unused devices insight.

    Args:
        device_count: Number of unused devices.
        days_unused: Days of inactivity.
        percent_affected: Percentage of fleet affected.
        financial_data: Pre-calculated financial data.

    Returns:
        Complete prompt string.
    """
    return UNUSED_DEVICES_PROMPT.format(
        device_count=device_count,
        days_unused=days_unused,
        percent_affected=percent_affected,
        financial_context=build_financial_context(financial_data),
    )


def build_battery_health_prompt(
    devices_count: int,
    avg_health: float,
    critical_count: int,
    financial_data: Optional[InsightFinancialData],
) -> str:
    """Build prompt for battery health insight.

    Args:
        devices_count: Devices needing replacement.
        avg_health: Average battery health percentage.
        critical_count: Devices with critical battery.
        financial_data: Pre-calculated financial data.

    Returns:
        Complete prompt string.
    """
    return BATTERY_HEALTH_PROMPT.format(
        devices_count=devices_count,
        avg_health=avg_health,
        critical_count=critical_count,
        financial_context=build_financial_context(financial_data),
    )


def build_drop_damage_prompt(
    total_drops: int,
    affected_devices: int,
    financial_data: Optional[InsightFinancialData],
) -> str:
    """Build prompt for drop damage insight.

    Args:
        total_drops: Total drop events.
        affected_devices: Devices with drops.
        financial_data: Pre-calculated financial data.

    Returns:
        Complete prompt string.
    """
    drops_per_device = total_drops / max(affected_devices, 1)

    return DROP_DAMAGE_PROMPT.format(
        total_drops=total_drops,
        affected_devices=affected_devices,
        drops_per_device=drops_per_device,
        financial_context=build_financial_context(financial_data),
    )


def build_downtime_prompt(
    incident_count: int,
    avg_duration_minutes: float,
    affected_devices: int,
    financial_data: Optional[InsightFinancialData],
) -> str:
    """Build prompt for downtime impact insight.

    Args:
        incident_count: Number of incidents.
        avg_duration_minutes: Average incident duration.
        affected_devices: Devices affected.
        financial_data: Pre-calculated financial data.

    Returns:
        Complete prompt string.
    """
    total_hours = (incident_count * avg_duration_minutes) / 60

    return DOWNTIME_IMPACT_PROMPT.format(
        incident_count=incident_count,
        avg_duration_minutes=avg_duration_minutes,
        affected_devices=affected_devices,
        total_hours=total_hours,
        financial_context=build_financial_context(financial_data),
    )


def format_currency(amount: Decimal, include_cents: bool = False) -> str:
    """Format a decimal amount as currency.

    Args:
        amount: Amount in USD.
        include_cents: Whether to include cents.

    Returns:
        Formatted currency string.
    """
    if include_cents:
        return f"${float(amount):,.2f}"
    return f"${float(amount):,.0f}"
