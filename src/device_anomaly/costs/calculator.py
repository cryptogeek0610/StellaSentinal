"""Cost Calculator Service for financial impact analysis.

This module provides the core calculation logic for determining
financial impact of device anomalies. All calculations are performed
here BEFORE passing data to the LLM - the LLM never calculates costs.

Design principle: Pre-compute all financial figures, inject into prompts.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from device_anomaly.costs.config import CostConfig, get_cost_config
from device_anomaly.costs.models import (
    CostBreakdownItem,
    CostCalculationResult,
    CostComponentType,
    CostContext,
    DeviceCostContext,
    FinancialImpactSummary,
    ImpactLevel,
    InsightFinancialData,
)

logger = logging.getLogger(__name__)


class CostCalculator:
    """Cost calculation service for financial impact analysis.

    Pre-computes all financial figures for anomalies and insights.
    The LLM should NEVER calculate financial figures - all numbers
    are computed here and injected into prompts.
    """

    def __init__(self, config: Optional[CostConfig] = None):
        """Initialize calculator with configuration.

        Args:
            config: Cost configuration. Uses default if not provided.
        """
        self.config = config or get_cost_config()

    def calculate_unused_device_cost(
        self,
        device_count: int,
        days_unused: int,
        device_value_usd: Optional[Decimal] = None,
    ) -> CostCalculationResult:
        """Calculate financial impact of unused devices.

        Calculates depreciation waste and tied capital for devices
        that haven't been used in the specified period.

        Args:
            device_count: Number of unused devices.
            days_unused: Number of days devices have been unused.
            device_value_usd: Optional device value override.

        Returns:
            CostCalculationResult with financial impact.
        """
        start_time = time.time()

        try:
            device_value = device_value_usd or self.config.average_device_cost_usd

            # Calculate tied capital (devices sitting idle)
            tied_capital = device_value * device_count

            # Calculate depreciation waste
            # Monthly depreciation = (value - residual) / depreciation_months
            residual_value = device_value * Decimal(self.config.residual_value_percent) / 100
            monthly_depreciation = (device_value - residual_value) / self.config.depreciation_months
            daily_depreciation = monthly_depreciation / 30
            depreciation_waste = daily_depreciation * days_unused * device_count

            # Calculate opportunity cost (could have been earning)
            # Assume 2% of device value per month in productivity value
            productivity_value_per_month = device_value * Decimal("0.02")
            daily_opportunity = productivity_value_per_month / 30
            opportunity_cost = daily_opportunity * days_unused * device_count

            total_impact = depreciation_waste + opportunity_cost
            monthly_recurring = (monthly_depreciation + productivity_value_per_month) * device_count

            breakdown = [
                CostBreakdownItem(
                    type=CostComponentType.DEPRECIATION,
                    category="depreciation_waste",
                    description=f"Depreciation of {device_count} unused devices over {days_unused} days",
                    amount_usd=depreciation_waste.quantize(Decimal("0.01")),
                    is_recurring=True,
                    period="daily",
                    confidence=0.9,
                    calculation_method=f"({device_value} - {residual_value}) / {self.config.depreciation_months} months / 30 days * {days_unused} days * {device_count} devices",
                ),
                CostBreakdownItem(
                    type=CostComponentType.OPPORTUNITY,
                    category="opportunity_cost",
                    description="Lost productivity from idle devices",
                    amount_usd=opportunity_cost.quantize(Decimal("0.01")),
                    is_recurring=True,
                    period="daily",
                    confidence=0.6,
                    calculation_method="Estimated at 2% of device value per month",
                ),
            ]

            # Recommendations
            recommendations = []
            if device_count >= 5:
                recommendations.append(f"Consider reallocating {device_count} unused devices to active workers")
            if days_unused >= 14:
                recommendations.append("Investigate why these devices are not in use")
            if total_impact > self.config.high_impact_threshold_usd:
                recommendations.append("High financial impact - prioritize device reallocation")

            # Potential savings if addressed
            potential_savings = monthly_recurring

            impact = FinancialImpactSummary(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                monthly_recurring_usd=monthly_recurring.quantize(Decimal("0.01")),
                potential_savings_usd=potential_savings.quantize(Decimal("0.01")),
                impact_level=ImpactLevel(self.config.get_impact_level(total_impact)),
                breakdown=breakdown,
                recommendations=recommendations,
                confidence_score=0.8,
                confidence_explanation="Based on device value and standard depreciation schedule",
                calculated_at=datetime.now(timezone.utc),
            )

            financial_data = InsightFinancialData(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                monthly_recurring_usd=monthly_recurring.quantize(Decimal("0.01")),
                potential_savings_usd=potential_savings.quantize(Decimal("0.01")),
                affected_devices=device_count,
                total_fleet_value_usd=tied_capital.quantize(Decimal("0.01")),
                hardware_impact_usd=depreciation_waste.quantize(Decimal("0.01")),
                opportunity_cost_usd=opportunity_cost.quantize(Decimal("0.01")),
            )

            return CostCalculationResult(
                success=True,
                impact=impact,
                financial_data=financial_data,
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Error calculating unused device cost: {e}")
            return CostCalculationResult(
                success=False,
                error_message=str(e),
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

    def calculate_battery_replacement_cost(
        self,
        devices_needing_replacement: int,
        average_battery_health: float,
        battery_cost_usd: Optional[Decimal] = None,
    ) -> CostCalculationResult:
        """Calculate cost impact of battery replacements.

        Args:
            devices_needing_replacement: Number of devices needing new batteries.
            average_battery_health: Average battery health percentage (0-100).
            battery_cost_usd: Optional battery cost override.

        Returns:
            CostCalculationResult with financial impact.
        """
        start_time = time.time()

        try:
            battery_cost = battery_cost_usd or self.config.battery_replacement_cost_usd

            # Direct battery replacement cost
            replacement_cost = battery_cost * devices_needing_replacement

            # Labor cost for replacements
            replacement_time = self.config.battery_replacement_time_minutes
            labor_hours = (replacement_time * devices_needing_replacement) / 60
            labor_cost = labor_hours * self.config.it_support_hourly_rate_usd

            # Productivity loss during replacement
            # Worker is idle during battery swap + setup
            worker_idle_minutes = replacement_time + self.config.device_setup_time_minutes
            worker_idle_hours = (worker_idle_minutes * devices_needing_replacement) / 60
            productivity_loss = worker_idle_hours * self.config.worker_hourly_rate_usd

            total_impact = replacement_cost + labor_cost + productivity_loss

            breakdown = [
                CostBreakdownItem(
                    type=CostComponentType.HARDWARE,
                    category="battery_replacement",
                    description=f"Battery replacement for {devices_needing_replacement} devices",
                    amount_usd=replacement_cost.quantize(Decimal("0.01")),
                    confidence=0.95,
                    calculation_method=f"${battery_cost} per battery * {devices_needing_replacement} devices",
                ),
                CostBreakdownItem(
                    type=CostComponentType.LABOR,
                    category="it_support",
                    description="IT support time for battery replacements",
                    amount_usd=labor_cost.quantize(Decimal("0.01")),
                    confidence=0.8,
                    calculation_method=f"{labor_hours:.1f} hours * ${self.config.it_support_hourly_rate_usd}/hr",
                ),
                CostBreakdownItem(
                    type=CostComponentType.DOWNTIME,
                    category="productivity_loss",
                    description="Worker idle time during replacement",
                    amount_usd=productivity_loss.quantize(Decimal("0.01")),
                    confidence=0.7,
                    calculation_method=f"{worker_idle_hours:.1f} hours * ${self.config.worker_hourly_rate_usd}/hr",
                ),
            ]

            recommendations = []
            if devices_needing_replacement >= 10:
                recommendations.append("Consider bulk battery purchase for cost savings")
            if average_battery_health < 30:
                recommendations.append("Critical battery health - schedule immediate replacements")
            elif average_battery_health < 50:
                recommendations.append("Plan battery replacements within 30 days")

            # Investment required is the total replacement cost
            investment_required = replacement_cost + labor_cost

            impact = FinancialImpactSummary(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                potential_savings_usd=Decimal("0.00"),  # Required cost, no savings
                impact_level=ImpactLevel(self.config.get_impact_level(total_impact)),
                breakdown=breakdown,
                recommendations=recommendations,
                investment_required_usd=investment_required.quantize(Decimal("0.01")),
                confidence_score=0.85,
                confidence_explanation="Based on standard battery costs and IT labor rates",
                calculated_at=datetime.now(timezone.utc),
            )

            financial_data = InsightFinancialData(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                monthly_recurring_usd=Decimal("0.00"),
                potential_savings_usd=Decimal("0.00"),
                affected_devices=devices_needing_replacement,
                total_fleet_value_usd=Decimal("0.00"),
                hardware_impact_usd=replacement_cost.quantize(Decimal("0.01")),
                labor_impact_usd=labor_cost.quantize(Decimal("0.01")),
                downtime_impact_usd=productivity_loss.quantize(Decimal("0.01")),
                investment_required_usd=investment_required.quantize(Decimal("0.01")),
            )

            return CostCalculationResult(
                success=True,
                impact=impact,
                financial_data=financial_data,
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Error calculating battery replacement cost: {e}")
            return CostCalculationResult(
                success=False,
                error_message=str(e),
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

    def calculate_downtime_cost(
        self,
        incident_count: int,
        avg_duration_minutes: float,
        affected_devices: int,
        is_critical: bool = False,
    ) -> CostCalculationResult:
        """Calculate cost of device downtime incidents.

        Args:
            incident_count: Number of downtime incidents.
            avg_duration_minutes: Average incident duration in minutes.
            affected_devices: Number of devices affected.
            is_critical: Whether these are critical devices.

        Returns:
            CostCalculationResult with financial impact.
        """
        start_time = time.time()

        try:
            # Calculate total downtime hours
            total_downtime_hours = (incident_count * avg_duration_minutes) / 60

            # Base downtime cost
            hourly_rate = self.config.downtime_cost_per_hour_usd
            if is_critical:
                hourly_rate = hourly_rate * self.config.critical_downtime_multiplier

            downtime_cost = total_downtime_hours * hourly_rate

            # Worker productivity loss
            worker_hours = total_downtime_hours * affected_devices
            productivity_loss = worker_hours * self.config.worker_hourly_rate_usd

            # IT investigation time per incident
            investigation_hours = (self.config.it_investigation_time_minutes * incident_count) / 60
            support_cost = investigation_hours * self.config.it_support_hourly_rate_usd

            total_impact = downtime_cost + productivity_loss + support_cost

            breakdown = [
                CostBreakdownItem(
                    type=CostComponentType.DOWNTIME,
                    category="device_downtime",
                    description=f"Device downtime cost ({total_downtime_hours:.1f} hours)",
                    amount_usd=downtime_cost.quantize(Decimal("0.01")),
                    confidence=0.75,
                    calculation_method=f"{total_downtime_hours:.1f} hours * ${hourly_rate}/hr" +
                                       (" (critical multiplier applied)" if is_critical else ""),
                ),
                CostBreakdownItem(
                    type=CostComponentType.LABOR,
                    category="productivity_loss",
                    description=f"Worker productivity loss ({affected_devices} devices)",
                    amount_usd=productivity_loss.quantize(Decimal("0.01")),
                    confidence=0.7,
                    calculation_method=f"{worker_hours:.1f} worker-hours * ${self.config.worker_hourly_rate_usd}/hr",
                ),
                CostBreakdownItem(
                    type=CostComponentType.SUPPORT,
                    category="it_support",
                    description="IT investigation and resolution time",
                    amount_usd=support_cost.quantize(Decimal("0.01")),
                    confidence=0.8,
                    calculation_method=f"{investigation_hours:.1f} hours * ${self.config.it_support_hourly_rate_usd}/hr",
                ),
            ]

            # Monthly projection (assuming current rate continues)
            # Project based on incidents per day * 30
            monthly_impact = total_impact  # Already represents the full period impact

            recommendations = []
            if incident_count >= 5:
                recommendations.append("High incident frequency - investigate root cause")
            if avg_duration_minutes > 60:
                recommendations.append("Long resolution times - review troubleshooting procedures")
            if is_critical:
                recommendations.append("Critical devices affected - prioritize reliability improvements")

            impact = FinancialImpactSummary(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                monthly_recurring_usd=monthly_impact.quantize(Decimal("0.01")),
                potential_savings_usd=(total_impact * Decimal("0.5")).quantize(Decimal("0.01")),  # 50% reduction target
                impact_level=ImpactLevel(self.config.get_impact_level(total_impact)),
                breakdown=breakdown,
                recommendations=recommendations,
                confidence_score=0.7,
                confidence_explanation="Based on average downtime costs and IT labor rates",
                calculated_at=datetime.now(timezone.utc),
            )

            financial_data = InsightFinancialData(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                monthly_recurring_usd=monthly_impact.quantize(Decimal("0.01")),
                potential_savings_usd=(total_impact * Decimal("0.5")).quantize(Decimal("0.01")),
                affected_devices=affected_devices,
                total_fleet_value_usd=Decimal("0.00"),
                downtime_impact_usd=downtime_cost.quantize(Decimal("0.01")),
                labor_impact_usd=(productivity_loss + support_cost).quantize(Decimal("0.01")),
            )

            return CostCalculationResult(
                success=True,
                impact=impact,
                financial_data=financial_data,
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Error calculating downtime cost: {e}")
            return CostCalculationResult(
                success=False,
                error_message=str(e),
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

    def calculate_drop_damage_cost(
        self,
        total_drops: int,
        affected_devices: int,
        avg_device_value_usd: Optional[Decimal] = None,
    ) -> CostCalculationResult:
        """Calculate potential damage cost from device drops.

        Uses industry data on drop-to-failure rates to estimate
        potential repair and replacement costs.

        Args:
            total_drops: Total number of drop events.
            affected_devices: Number of devices with drops.
            avg_device_value_usd: Optional device value override.

        Returns:
            CostCalculationResult with financial impact.
        """
        start_time = time.time()

        try:
            device_value = avg_device_value_usd or self.config.average_device_cost_usd

            # Industry estimates: ~2% of drops result in screen damage
            # ~0.5% result in device failure
            screen_damage_rate = Decimal("0.02")
            device_failure_rate = Decimal("0.005")

            estimated_screen_damages = int(total_drops * float(screen_damage_rate))
            estimated_failures = int(total_drops * float(device_failure_rate))

            # Screen repair cost
            screen_repair_total = estimated_screen_damages * self.config.screen_repair_cost_usd

            # Device replacement cost
            replacement_total = estimated_failures * device_value

            # General repair cost for minor damage
            minor_damage_rate = Decimal("0.05")
            estimated_minor_repairs = int(total_drops * float(minor_damage_rate))
            minor_repair_total = estimated_minor_repairs * self.config.general_repair_cost_usd

            # IT time for assessments and repairs
            assessment_hours = (affected_devices * 15) / 60  # 15 min per device
            it_cost = Decimal(assessment_hours) * self.config.it_support_hourly_rate_usd

            total_impact = screen_repair_total + replacement_total + minor_repair_total + it_cost

            breakdown = [
                CostBreakdownItem(
                    type=CostComponentType.HARDWARE,
                    category="screen_repair",
                    description=f"Estimated screen repairs ({estimated_screen_damages} devices)",
                    amount_usd=screen_repair_total.quantize(Decimal("0.01")),
                    confidence=0.5,
                    calculation_method=f"2% damage rate * {total_drops} drops * ${self.config.screen_repair_cost_usd}",
                ),
                CostBreakdownItem(
                    type=CostComponentType.HARDWARE,
                    category="device_replacement",
                    description=f"Estimated device replacements ({estimated_failures} devices)",
                    amount_usd=replacement_total.quantize(Decimal("0.01")),
                    confidence=0.4,
                    calculation_method=f"0.5% failure rate * {total_drops} drops * ${device_value}",
                ),
                CostBreakdownItem(
                    type=CostComponentType.HARDWARE,
                    category="minor_repairs",
                    description="Estimated minor repairs and assessments",
                    amount_usd=minor_repair_total.quantize(Decimal("0.01")),
                    confidence=0.5,
                    calculation_method=f"5% minor damage rate * {total_drops} drops",
                ),
                CostBreakdownItem(
                    type=CostComponentType.SUPPORT,
                    category="it_assessment",
                    description="IT assessment and processing time",
                    amount_usd=it_cost.quantize(Decimal("0.01")),
                    confidence=0.7,
                    calculation_method=f"{assessment_hours:.1f} hours * ${self.config.it_support_hourly_rate_usd}/hr",
                ),
            ]

            recommendations = []
            drops_per_device = total_drops / max(affected_devices, 1)
            if drops_per_device > 5:
                recommendations.append("High drop rate per device - consider protective cases")
            if total_drops > 50:
                recommendations.append("Review handling procedures and worker training")
            if estimated_failures > 0:
                recommendations.append("Budget for potential device replacements")

            # Potential savings from prevention (50% reduction through cases/training)
            potential_savings = total_impact * Decimal("0.5")

            impact = FinancialImpactSummary(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                potential_savings_usd=potential_savings.quantize(Decimal("0.01")),
                impact_level=ImpactLevel(self.config.get_impact_level(total_impact)),
                breakdown=breakdown,
                recommendations=recommendations,
                confidence_score=0.5,  # Lower confidence due to estimated damage rates
                confidence_explanation="Based on industry estimates for drop damage rates",
                calculated_at=datetime.now(timezone.utc),
            )

            financial_data = InsightFinancialData(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                monthly_recurring_usd=Decimal("0.00"),
                potential_savings_usd=potential_savings.quantize(Decimal("0.01")),
                affected_devices=affected_devices,
                total_fleet_value_usd=(device_value * affected_devices).quantize(Decimal("0.01")),
                hardware_impact_usd=(screen_repair_total + replacement_total + minor_repair_total).quantize(Decimal("0.01")),
                labor_impact_usd=it_cost.quantize(Decimal("0.01")),
            )

            return CostCalculationResult(
                success=True,
                impact=impact,
                financial_data=financial_data,
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Error calculating drop damage cost: {e}")
            return CostCalculationResult(
                success=False,
                error_message=str(e),
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

    def calculate_anomaly_impact(
        self,
        context: CostContext,
    ) -> CostCalculationResult:
        """Calculate financial impact for a specific anomaly.

        General-purpose calculation that considers the anomaly type
        and context to determine appropriate cost impact.

        Args:
            context: Full cost context for the anomaly.

        Returns:
            CostCalculationResult with financial impact.
        """
        start_time = time.time()

        try:
            breakdown = []
            total_impact = Decimal("0.00")
            monthly_recurring = Decimal("0.00")
            potential_savings = Decimal("0.00")

            # Device hardware impact (if applicable)
            if context.device_context and context.device_context.current_value_usd:
                device_value = context.device_context.current_value_usd
                at_risk_value = device_value * context.affected_device_count

                # Estimate risk percentage based on severity
                severity_risk = {
                    "critical": Decimal("0.20"),
                    "high": Decimal("0.10"),
                    "medium": Decimal("0.05"),
                    "low": Decimal("0.02"),
                }.get(context.anomaly_severity or "medium", Decimal("0.05"))

                hardware_risk = at_risk_value * severity_risk
                total_impact += hardware_risk

                breakdown.append(CostBreakdownItem(
                    type=CostComponentType.HARDWARE,
                    category="device_at_risk",
                    description=f"Device value at risk ({context.affected_device_count} devices)",
                    amount_usd=hardware_risk.quantize(Decimal("0.01")),
                    confidence=0.6,
                    calculation_method=f"Device value ${device_value} * {context.affected_device_count} * {severity_risk:.0%} risk",
                ))

            # Downtime impact
            if context.duration_hours:
                downtime_rate = context.downtime_cost_per_hour_usd
                if context.is_critical:
                    downtime_rate = downtime_rate * self.config.critical_downtime_multiplier

                downtime_cost = Decimal(context.duration_hours) * downtime_rate * context.affected_device_count
                total_impact += downtime_cost

                breakdown.append(CostBreakdownItem(
                    type=CostComponentType.DOWNTIME,
                    category="downtime",
                    description=f"Downtime cost ({context.duration_hours:.1f} hours)",
                    amount_usd=downtime_cost.quantize(Decimal("0.01")),
                    is_recurring=True,
                    period="one_time",
                    confidence=0.7,
                    calculation_method=f"{context.duration_hours:.1f} hrs * ${downtime_rate}/hr * {context.affected_device_count} devices",
                ))

            # IT support cost
            if context.estimated_resolution_hours:
                support_cost = Decimal(context.estimated_resolution_hours) * context.it_support_hourly_rate_usd
                total_impact += support_cost

                breakdown.append(CostBreakdownItem(
                    type=CostComponentType.SUPPORT,
                    category="it_support",
                    description="IT investigation and resolution",
                    amount_usd=support_cost.quantize(Decimal("0.01")),
                    confidence=0.75,
                    calculation_method=f"{context.estimated_resolution_hours:.1f} hrs * ${context.it_support_hourly_rate_usd}/hr",
                ))

            # Worker productivity impact
            if context.duration_hours and context.affected_device_count > 0:
                worker_cost = Decimal(context.duration_hours) * context.worker_hourly_rate_usd * context.affected_device_count
                total_impact += worker_cost

                breakdown.append(CostBreakdownItem(
                    type=CostComponentType.LABOR,
                    category="productivity_loss",
                    description="Worker productivity loss",
                    amount_usd=worker_cost.quantize(Decimal("0.01")),
                    confidence=0.65,
                    calculation_method=f"{context.duration_hours:.1f} hrs * ${context.worker_hourly_rate_usd}/hr * {context.affected_device_count}",
                ))

            # Use similar incidents for better estimates
            if context.similar_incidents_count > 5 and context.similar_incidents_avg_cost_usd:
                # Blend current estimate with historical average
                historical_estimate = context.similar_incidents_avg_cost_usd
                total_impact = (total_impact + historical_estimate) / 2

            # Determine impact level
            impact_level = ImpactLevel(self.config.get_impact_level(total_impact))

            # Generate recommendations based on impact
            recommendations = []
            if impact_level == ImpactLevel.HIGH:
                recommendations.append("High financial impact - prioritize immediate resolution")
            if context.incident_count > 3:
                recommendations.append("Recurring issue - investigate root cause to prevent future incidents")
            if context.is_critical:
                recommendations.append("Critical device affected - ensure backup resources available")

            # Potential savings (25% reduction through proactive measures)
            potential_savings = total_impact * Decimal("0.25")

            # Confidence based on data quality
            confidence = 0.7
            if context.similar_incidents_count > 10:
                confidence = 0.85
            elif not context.device_context:
                confidence = 0.5

            impact = FinancialImpactSummary(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                monthly_recurring_usd=monthly_recurring.quantize(Decimal("0.01")),
                potential_savings_usd=potential_savings.quantize(Decimal("0.01")),
                impact_level=impact_level,
                breakdown=breakdown,
                recommendations=recommendations,
                confidence_score=confidence,
                confidence_explanation="Based on anomaly severity, device value, and operational costs",
                calculated_at=datetime.now(timezone.utc),
            )

            financial_data = InsightFinancialData(
                total_impact_usd=total_impact.quantize(Decimal("0.01")),
                monthly_recurring_usd=monthly_recurring.quantize(Decimal("0.01")),
                potential_savings_usd=potential_savings.quantize(Decimal("0.01")),
                affected_devices=context.affected_device_count,
                total_fleet_value_usd=(
                    context.device_context.current_value_usd * context.affected_device_count
                    if context.device_context and context.device_context.current_value_usd
                    else Decimal("0.00")
                ).quantize(Decimal("0.01")),
                hardware_impact_usd=sum(
                    b.amount_usd for b in breakdown if b.type == CostComponentType.HARDWARE
                ).quantize(Decimal("0.01")),
                labor_impact_usd=sum(
                    b.amount_usd for b in breakdown if b.type == CostComponentType.LABOR
                ).quantize(Decimal("0.01")),
                downtime_impact_usd=sum(
                    b.amount_usd for b in breakdown if b.type == CostComponentType.DOWNTIME
                ).quantize(Decimal("0.01")),
            )

            return CostCalculationResult(
                success=True,
                impact=impact,
                financial_data=financial_data,
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Error calculating anomaly impact: {e}")
            return CostCalculationResult(
                success=False,
                error_message=str(e),
                calculation_time_ms=(time.time() - start_time) * 1000,
            )

    def format_for_llm_prompt(
        self,
        result: CostCalculationResult,
    ) -> str:
        """Format calculation result for LLM prompt injection.

        Returns a formatted string with all financial figures that
        can be safely injected into the LLM prompt.

        Args:
            result: Calculation result to format.

        Returns:
            Formatted string for prompt injection.
        """
        if not result.success or not result.financial_data:
            return "FINANCIAL DATA: Not available (calculation failed)"

        return result.financial_data.to_prompt_context()

    def get_allowed_amounts(
        self,
        result: CostCalculationResult,
    ) -> List[Decimal]:
        """Get list of allowed monetary amounts from calculation.

        Used by the anti-hallucination validator to verify LLM output.

        Args:
            result: Calculation result.

        Returns:
            List of Decimal amounts the LLM is allowed to cite.
        """
        if not result.success or not result.impact:
            return []

        amounts = [
            result.impact.total_impact_usd,
            result.impact.monthly_recurring_usd,
            result.impact.potential_savings_usd,
        ]

        # Add breakdown amounts
        for item in result.impact.breakdown:
            amounts.append(item.amount_usd)

        # Add investment if present
        if result.impact.investment_required_usd:
            amounts.append(result.impact.investment_required_usd)

        return [a for a in amounts if a > 0]
