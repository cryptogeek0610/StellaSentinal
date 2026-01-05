"""Cost configuration with default values and tenant overrides.

This module provides cost configuration management for calculating
financial impacts of device anomalies. All monetary values are in USD.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from typing import Dict, Optional

from pydantic import BaseModel, Field


class CostConfig(BaseModel):
    """Cost configuration with sensible defaults.

    All monetary values are in USD. These defaults can be overridden
    per-tenant through operational cost entries in the database.
    """

    # Device Hardware Costs (defaults)
    average_device_cost_usd: Decimal = Field(
        default=Decimal("500.00"),
        description="Default device purchase price when not specified",
    )
    battery_replacement_cost_usd: Decimal = Field(
        default=Decimal("50.00"),
        description="Cost to replace a device battery",
    )
    screen_repair_cost_usd: Decimal = Field(
        default=Decimal("150.00"),
        description="Average cost for screen repair",
    )
    general_repair_cost_usd: Decimal = Field(
        default=Decimal("100.00"),
        description="Average cost for general device repairs",
    )

    # Depreciation
    depreciation_months: int = Field(
        default=36,
        ge=1,
        le=120,
        description="Expected useful life of devices in months",
    )
    residual_value_percent: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Expected residual value as percentage of purchase price",
    )

    # Labor Costs
    worker_hourly_rate_usd: Decimal = Field(
        default=Decimal("25.00"),
        description="Average hourly rate for workers using devices",
    )
    it_support_hourly_rate_usd: Decimal = Field(
        default=Decimal("50.00"),
        description="Hourly rate for IT support staff",
    )
    device_admin_hourly_rate_usd: Decimal = Field(
        default=Decimal("40.00"),
        description="Hourly rate for device administrators",
    )

    # Time Estimates (in minutes)
    device_replacement_time_minutes: int = Field(
        default=15,
        ge=1,
        description="Time to replace a device for a worker",
    )
    battery_replacement_time_minutes: int = Field(
        default=30,
        ge=1,
        description="Time to replace a battery",
    )
    it_investigation_time_minutes: int = Field(
        default=30,
        ge=1,
        description="Average time for IT to investigate an anomaly",
    )
    device_setup_time_minutes: int = Field(
        default=45,
        ge=1,
        description="Time to set up a new/replacement device",
    )

    # Downtime Costs
    downtime_cost_per_hour_usd: Decimal = Field(
        default=Decimal("100.00"),
        description="Cost per hour of device downtime (productivity loss)",
    )
    critical_downtime_multiplier: Decimal = Field(
        default=Decimal("2.0"),
        description="Multiplier for critical device downtime",
    )

    # Impact Thresholds
    high_impact_threshold_usd: Decimal = Field(
        default=Decimal("1000.00"),
        description="Threshold above which impact is considered HIGH",
    )
    medium_impact_threshold_usd: Decimal = Field(
        default=Decimal("250.00"),
        description="Threshold above which impact is considered MEDIUM",
    )

    # Calculation Settings
    confidence_high_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score threshold for high confidence",
    )
    confidence_medium_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score threshold for medium confidence",
    )

    # Currency
    default_currency: str = Field(
        default="USD",
        max_length=3,
        description="Default currency code (ISO 4217)",
    )

    def get_impact_level(self, total_impact: Decimal) -> str:
        """Determine impact level based on total impact amount.

        Args:
            total_impact: Total financial impact in USD.

        Returns:
            Impact level: "high", "medium", or "low".
        """
        if total_impact >= self.high_impact_threshold_usd:
            return "high"
        elif total_impact >= self.medium_impact_threshold_usd:
            return "medium"
        return "low"

    def get_confidence_level(self, confidence_score: float) -> str:
        """Determine confidence level based on score.

        Args:
            confidence_score: Confidence score between 0 and 1.

        Returns:
            Confidence level: "high", "medium", or "low".
        """
        if confidence_score >= self.confidence_high_threshold:
            return "high"
        elif confidence_score >= self.confidence_medium_threshold:
            return "medium"
        return "low"

    def calculate_depreciated_value(
        self,
        purchase_cost: Decimal,
        age_months: int,
        depreciation_months: Optional[int] = None,
        residual_percent: Optional[int] = None,
    ) -> Decimal:
        """Calculate current depreciated value of a device.

        Uses straight-line depreciation to calculate current value.

        Args:
            purchase_cost: Original purchase price.
            age_months: Current age of device in months.
            depreciation_months: Optional custom depreciation period.
            residual_percent: Optional custom residual value percentage.

        Returns:
            Current depreciated value in USD.
        """
        dep_months = depreciation_months or self.depreciation_months
        residual = residual_percent if residual_percent is not None else self.residual_value_percent

        if age_months >= dep_months:
            return purchase_cost * Decimal(residual) / 100

        depreciation_amount = purchase_cost * (100 - Decimal(residual)) / 100
        monthly_depreciation = depreciation_amount / dep_months
        current_depreciation = monthly_depreciation * age_months

        return purchase_cost - current_depreciation

    def calculate_hourly_cost(self, minutes: int, hourly_rate: Decimal) -> Decimal:
        """Calculate cost based on time in minutes and hourly rate.

        Args:
            minutes: Time spent in minutes.
            hourly_rate: Hourly rate in USD.

        Returns:
            Total cost in USD.
        """
        hours = Decimal(minutes) / 60
        return (hours * hourly_rate).quantize(Decimal("0.01"))


@dataclass
class TenantCostOverrides:
    """Tenant-specific cost configuration overrides.

    Loaded from operational_costs table for each tenant.
    """

    tenant_id: str
    overrides: Dict[str, Decimal] = field(default_factory=dict)

    def apply_to_config(self, base_config: CostConfig) -> CostConfig:
        """Apply tenant overrides to base configuration.

        Args:
            base_config: Base cost configuration.

        Returns:
            New configuration with tenant overrides applied.
        """
        config_dict = base_config.model_dump()

        for key, value in self.overrides.items():
            if key in config_dict:
                config_dict[key] = value

        return CostConfig(**config_dict)


@lru_cache(maxsize=1)
def get_cost_config() -> CostConfig:
    """Get the cost configuration.

    Returns cached configuration instance. Override values can be
    loaded from environment variables with COST_ prefix.

    Returns:
        CostConfig instance with applied overrides.
    """
    config = CostConfig()

    # Apply environment variable overrides
    env_overrides = {}

    if os.getenv("COST_AVERAGE_DEVICE_COST_USD"):
        env_overrides["average_device_cost_usd"] = Decimal(
            os.getenv("COST_AVERAGE_DEVICE_COST_USD")
        )

    if os.getenv("COST_WORKER_HOURLY_RATE_USD"):
        env_overrides["worker_hourly_rate_usd"] = Decimal(
            os.getenv("COST_WORKER_HOURLY_RATE_USD")
        )

    if os.getenv("COST_IT_SUPPORT_HOURLY_RATE_USD"):
        env_overrides["it_support_hourly_rate_usd"] = Decimal(
            os.getenv("COST_IT_SUPPORT_HOURLY_RATE_USD")
        )

    if os.getenv("COST_DOWNTIME_COST_PER_HOUR_USD"):
        env_overrides["downtime_cost_per_hour_usd"] = Decimal(
            os.getenv("COST_DOWNTIME_COST_PER_HOUR_USD")
        )

    if os.getenv("COST_HIGH_IMPACT_THRESHOLD_USD"):
        env_overrides["high_impact_threshold_usd"] = Decimal(
            os.getenv("COST_HIGH_IMPACT_THRESHOLD_USD")
        )

    if os.getenv("COST_MEDIUM_IMPACT_THRESHOLD_USD"):
        env_overrides["medium_impact_threshold_usd"] = Decimal(
            os.getenv("COST_MEDIUM_IMPACT_THRESHOLD_USD")
        )

    if env_overrides:
        config_dict = config.model_dump()
        config_dict.update(env_overrides)
        config = CostConfig(**config_dict)

    return config


def clear_config_cache() -> None:
    """Clear the configuration cache.

    Call this when configuration changes and needs to be reloaded.
    """
    get_cost_config.cache_clear()
