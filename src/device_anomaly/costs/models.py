"""Cost dataclass models for financial impact calculations.

These models are used internally for cost calculations and passing
financial data to the LLM pipeline. They are designed for easy
serialization and integration with the insight generation system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any


class ImpactLevel(StrEnum):
    """Financial impact severity level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CostComponentType(StrEnum):
    """Type of cost component in breakdown."""

    HARDWARE = "hardware"
    LABOR = "labor"
    DOWNTIME = "downtime"
    SUPPORT = "support"
    OPPORTUNITY = "opportunity"
    DEPRECIATION = "depreciation"
    OTHER = "other"


@dataclass
class CostBreakdownItem:
    """A single item in a cost breakdown.

    Represents one component of the total financial impact with
    its source and calculation method for audit purposes.
    """

    type: CostComponentType
    category: str
    description: str
    amount_usd: Decimal
    is_recurring: bool = False
    period: str = "one_time"  # one_time, hourly, daily, monthly, annual
    confidence: float = 0.7
    calculation_method: str = ""
    source_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "category": self.category,
            "description": self.description,
            "amount_usd": float(self.amount_usd),
            "is_recurring": self.is_recurring,
            "period": self.period,
            "confidence": self.confidence,
            "calculation_method": self.calculation_method,
        }


@dataclass
class FinancialImpactSummary:
    """Financial impact summary for insights.

    This is the primary model for attaching financial impact
    data to CustomerInsight objects.
    """

    total_impact_usd: Decimal
    monthly_recurring_usd: Decimal = Decimal("0.00")
    potential_savings_usd: Decimal = Decimal("0.00")
    impact_level: ImpactLevel = ImpactLevel.LOW
    breakdown: list[CostBreakdownItem] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # ROI calculation
    investment_required_usd: Decimal | None = None
    payback_months: float | None = None

    # Metadata
    confidence_score: float = 0.7
    confidence_explanation: str = ""
    calculated_at: datetime | None = None
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_impact_usd": float(self.total_impact_usd),
            "monthly_recurring_usd": float(self.monthly_recurring_usd),
            "potential_savings_usd": float(self.potential_savings_usd),
            "impact_level": self.impact_level.value,
            "breakdown": [item.to_dict() for item in self.breakdown],
            "recommendations": self.recommendations,
            "investment_required_usd": float(self.investment_required_usd)
            if self.investment_required_usd
            else None,
            "payback_months": self.payback_months,
            "confidence_score": self.confidence_score,
            "confidence_explanation": self.confidence_explanation,
            "calculated_at": self.calculated_at.isoformat() if self.calculated_at else None,
            "currency": self.currency,
        }


@dataclass
class DeviceCostContext:
    """Cost context for a specific device.

    Contains all cost-relevant information about a device for
    use in financial impact calculations.
    """

    device_id: int
    device_model: str | None = None
    device_name: str | None = None
    location: str | None = None

    # Hardware costs
    purchase_cost_usd: Decimal | None = None
    replacement_cost_usd: Decimal | None = None
    repair_cost_avg_usd: Decimal | None = None

    # Depreciation
    age_months: int = 0
    depreciation_months: int = 36
    residual_value_percent: int = 10
    current_value_usd: Decimal | None = None

    # Warranty
    warranty_months: int | None = None
    is_under_warranty: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "device_id": self.device_id,
            "device_model": self.device_model,
            "device_name": self.device_name,
            "location": self.location,
            "purchase_cost_usd": float(self.purchase_cost_usd) if self.purchase_cost_usd else None,
            "replacement_cost_usd": float(self.replacement_cost_usd)
            if self.replacement_cost_usd
            else None,
            "repair_cost_avg_usd": float(self.repair_cost_avg_usd)
            if self.repair_cost_avg_usd
            else None,
            "age_months": self.age_months,
            "current_value_usd": float(self.current_value_usd) if self.current_value_usd else None,
            "is_under_warranty": self.is_under_warranty,
        }


@dataclass
class CostContext:
    """Full cost context for financial impact calculation.

    Contains all necessary data for calculating the financial
    impact of an anomaly or insight.
    """

    tenant_id: str

    # Device information
    device_context: DeviceCostContext | None = None
    affected_device_count: int = 1

    # Anomaly information
    anomaly_id: int | None = None
    anomaly_type: str | None = None
    anomaly_severity: str | None = None

    # Time-based factors
    duration_hours: float | None = None
    estimated_resolution_hours: float | None = None
    incident_count: int = 1

    # Operational costs (from tenant config)
    worker_hourly_rate_usd: Decimal = Decimal("25.00")
    it_support_hourly_rate_usd: Decimal = Decimal("50.00")
    downtime_cost_per_hour_usd: Decimal = Decimal("100.00")

    # Additional context
    is_critical: bool = False
    similar_incidents_count: int = 0
    similar_incidents_avg_cost_usd: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tenant_id": self.tenant_id,
            "device_context": self.device_context.to_dict() if self.device_context else None,
            "affected_device_count": self.affected_device_count,
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "anomaly_severity": self.anomaly_severity,
            "duration_hours": self.duration_hours,
            "estimated_resolution_hours": self.estimated_resolution_hours,
            "incident_count": self.incident_count,
            "worker_hourly_rate_usd": float(self.worker_hourly_rate_usd),
            "it_support_hourly_rate_usd": float(self.it_support_hourly_rate_usd),
            "downtime_cost_per_hour_usd": float(self.downtime_cost_per_hour_usd),
            "is_critical": self.is_critical,
        }


@dataclass
class InsightFinancialData:
    """Pre-calculated financial data for insight generation.

    This data is passed to the LLM to ensure consistent financial
    figures without hallucination.
    """

    # Summary figures (for LLM to reference)
    total_impact_usd: Decimal
    monthly_recurring_usd: Decimal
    potential_savings_usd: Decimal

    # Device fleet impact
    affected_devices: int
    total_fleet_value_usd: Decimal

    # Breakdown by category (pre-calculated)
    hardware_impact_usd: Decimal = Decimal("0.00")
    labor_impact_usd: Decimal = Decimal("0.00")
    downtime_impact_usd: Decimal = Decimal("0.00")
    opportunity_cost_usd: Decimal = Decimal("0.00")

    # ROI data
    investment_required_usd: Decimal | None = None
    payback_months: float | None = None

    # Formatted strings for LLM (prevent formatting errors)
    formatted_total: str = ""
    formatted_monthly: str = ""
    formatted_savings: str = ""
    formatted_fleet_value: str = ""

    def __post_init__(self):
        """Generate formatted strings after initialization."""
        if not self.formatted_total:
            self.formatted_total = f"${float(self.total_impact_usd):,.0f}"
        if not self.formatted_monthly:
            self.formatted_monthly = f"${float(self.monthly_recurring_usd):,.0f}"
        if not self.formatted_savings:
            self.formatted_savings = f"${float(self.potential_savings_usd):,.0f}"
        if not self.formatted_fleet_value:
            self.formatted_fleet_value = f"${float(self.total_fleet_value_usd):,.0f}"

    def to_prompt_context(self) -> str:
        """Generate context string for LLM prompt injection.

        Returns a formatted string with all financial figures that
        can be injected into the LLM prompt.
        """
        lines = [
            "FINANCIAL IMPACT (PRE-CALCULATED - USE THESE EXACT FIGURES):",
            f"- Total Impact: {self.formatted_total}",
            f"- Monthly Recurring Cost: {self.formatted_monthly}",
            f"- Potential Savings: {self.formatted_savings}",
            f"- Affected Devices: {self.affected_devices}",
            f"- Total Fleet Value: {self.formatted_fleet_value}",
            "",
            "Cost Breakdown:",
            f"- Hardware Costs: ${float(self.hardware_impact_usd):,.0f}",
            f"- Labor Costs: ${float(self.labor_impact_usd):,.0f}",
            f"- Downtime Costs: ${float(self.downtime_impact_usd):,.0f}",
            f"- Opportunity Costs: ${float(self.opportunity_cost_usd):,.0f}",
        ]

        if self.investment_required_usd and self.payback_months:
            lines.extend(
                [
                    "",
                    "ROI Analysis:",
                    f"- Investment Required: ${float(self.investment_required_usd):,.0f}",
                    f"- Payback Period: {self.payback_months:.1f} months",
                ]
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_impact_usd": float(self.total_impact_usd),
            "monthly_recurring_usd": float(self.monthly_recurring_usd),
            "potential_savings_usd": float(self.potential_savings_usd),
            "affected_devices": self.affected_devices,
            "total_fleet_value_usd": float(self.total_fleet_value_usd),
            "hardware_impact_usd": float(self.hardware_impact_usd),
            "labor_impact_usd": float(self.labor_impact_usd),
            "downtime_impact_usd": float(self.downtime_impact_usd),
            "opportunity_cost_usd": float(self.opportunity_cost_usd),
            "investment_required_usd": float(self.investment_required_usd)
            if self.investment_required_usd
            else None,
            "payback_months": self.payback_months,
        }


@dataclass
class CostCalculationResult:
    """Result of a cost calculation operation.

    Encapsulates the full result of calculating financial impact
    including the summary, breakdown, and metadata.
    """

    success: bool
    impact: FinancialImpactSummary | None = None
    financial_data: InsightFinancialData | None = None
    error_message: str | None = None
    calculation_time_ms: float = 0.0
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "impact": self.impact.to_dict() if self.impact else None,
            "financial_data": self.financial_data.to_dict() if self.financial_data else None,
            "error_message": self.error_message,
            "calculation_time_ms": self.calculation_time_ms,
            "cache_hit": self.cache_hit,
        }
