"""Pydantic models for Cost Intelligence API requests and responses."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# ENUMERATIONS
# =============================================================================

class CostCategory(StrEnum):
    """Cost category classification."""

    LABOR = "labor"
    DOWNTIME = "downtime"
    SUPPORT = "support"
    INFRASTRUCTURE = "infrastructure"
    MAINTENANCE = "maintenance"
    OTHER = "other"


class CostType(StrEnum):
    """Cost calculation type."""

    HOURLY = "hourly"
    DAILY = "daily"
    PER_INCIDENT = "per_incident"
    FIXED_MONTHLY = "fixed_monthly"
    PER_DEVICE = "per_device"


class ScopeType(StrEnum):
    """Scope type for operational costs."""

    TENANT = "tenant"
    LOCATION = "location"
    DEVICE_GROUP = "device_group"
    DEVICE_MODEL = "device_model"


class ImpactLevel(StrEnum):
    """Financial impact severity level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AuditAction(StrEnum):
    """Audit log action types."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class AlertThresholdType(StrEnum):
    """Threshold types for cost alerts."""

    ANOMALY_COST_DAILY = "anomaly_cost_daily"
    ANOMALY_COST_MONTHLY = "anomaly_cost_monthly"
    BATTERY_FORECAST = "battery_forecast"
    OPERATIONAL_COST = "operational_cost"


# =============================================================================
# BASE MODEL CONFIGURATION
# =============================================================================

class CostBaseModel(BaseModel):
    """Base model for cost payloads with consistent Decimal serialization."""

    model_config = {"from_attributes": True, "json_encoders": {Decimal: float}}


# =============================================================================
# HARDWARE COST MODELS
# =============================================================================

class HardwareCostBase(CostBaseModel):
    """Base model for hardware cost entries."""

    device_model: str = Field(..., min_length=1, max_length=255, description="Device model identifier")
    purchase_cost: Decimal = Field(..., ge=0, description="Purchase cost in dollars")
    replacement_cost: Decimal | None = Field(None, ge=0, description="Replacement cost if different")
    repair_cost_avg: Decimal | None = Field(None, ge=0, description="Average repair cost")
    depreciation_months: int | None = Field(36, ge=1, le=120, description="Depreciation period in months")
    residual_value_percent: int | None = Field(0, ge=0, le=100, description="Residual value as percentage")
    warranty_months: int | None = Field(None, ge=0, le=120, description="Warranty period in months")
    currency_code: str = Field("USD", max_length=3, description="ISO 4217 currency code")
    notes: str | None = Field(None, max_length=2000, description="Additional notes")


class HardwareCostCreate(HardwareCostBase):
    """Request model for creating a hardware cost entry."""

    pass


class HardwareCostUpdate(CostBaseModel):
    """Request model for updating a hardware cost entry (partial update)."""

    device_model: str | None = Field(None, min_length=1, max_length=255)
    purchase_cost: Decimal | None = Field(None, ge=0)
    replacement_cost: Decimal | None = Field(None, ge=0)
    repair_cost_avg: Decimal | None = Field(None, ge=0)
    depreciation_months: int | None = Field(None, ge=1, le=120)
    residual_value_percent: int | None = Field(None, ge=0, le=100)
    warranty_months: int | None = Field(None, ge=0, le=120)
    notes: str | None = Field(None, max_length=2000)


class HardwareCostResponse(HardwareCostBase):
    """Response model for a hardware cost entry."""

    id: int
    tenant_id: str
    device_count: int = Field(0, description="Number of devices using this model")
    total_fleet_value: Decimal = Field(default=Decimal(0), description="Total value of devices with this model")
    valid_from: datetime
    valid_to: datetime | None = None
    created_at: datetime
    updated_at: datetime
    created_by: str | None = None


class HardwareCostListResponse(CostBaseModel):
    """Response model for paginated hardware cost list."""

    costs: list[HardwareCostResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    total_fleet_value: Decimal = Field(description="Sum of all fleet values")


class DeviceModelInfo(CostBaseModel):
    """Device model information from database."""

    device_model: str
    device_count: int
    has_cost_entry: bool


class DeviceModelsResponse(CostBaseModel):
    """Response model for device models list."""

    models: list[DeviceModelInfo]
    total: int


# =============================================================================
# OPERATIONAL COST MODELS
# =============================================================================

class OperationalCostBase(CostBaseModel):
    """Base model for operational cost entries."""

    name: str = Field(..., min_length=1, max_length=255, description="Cost entry name")
    category: CostCategory = Field(CostCategory.OTHER, description="Cost category")
    amount: Decimal = Field(..., ge=0, description="Cost amount in dollars")
    cost_type: CostType = Field(CostType.FIXED_MONTHLY, description="Cost calculation type")
    unit: str | None = Field(None, max_length=50, description="Unit of measurement")
    scope_type: ScopeType = Field(ScopeType.TENANT, description="Scope type")
    scope_id: str | None = Field(None, max_length=100, description="Scope entity ID")
    description: str | None = Field(None, max_length=2000)
    currency_code: str = Field("USD", max_length=3, description="ISO 4217 currency code")
    is_active: bool = Field(True, description="Whether this cost is currently active")
    notes: str | None = Field(None, max_length=2000)


class OperationalCostCreate(OperationalCostBase):
    """Request model for creating an operational cost entry."""

    pass


class OperationalCostUpdate(CostBaseModel):
    """Request model for updating an operational cost entry."""

    name: str | None = Field(None, min_length=1, max_length=255)
    category: CostCategory | None = None
    amount: Decimal | None = Field(None, ge=0)
    cost_type: CostType | None = None
    unit: str | None = Field(None, max_length=50)
    scope_type: ScopeType | None = None
    scope_id: str | None = Field(None, max_length=100)
    description: str | None = Field(None, max_length=2000)
    is_active: bool | None = None
    notes: str | None = Field(None, max_length=2000)


class OperationalCostResponse(OperationalCostBase):
    """Response model for an operational cost entry."""

    id: int
    tenant_id: str
    valid_from: datetime
    valid_to: datetime | None = None
    monthly_equivalent: Decimal = Field(description="Normalized monthly cost")
    annual_equivalent: Decimal = Field(description="Normalized annual cost")
    created_at: datetime
    updated_at: datetime
    created_by: str | None = None


class OperationalCostListResponse(CostBaseModel):
    """Response model for paginated operational cost list."""

    costs: list[OperationalCostResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    total_monthly_cost: Decimal
    total_annual_cost: Decimal


# =============================================================================
# COST SUMMARY MODELS
# =============================================================================

class CategoryCostSummary(CostBaseModel):
    """Cost summary for a single category."""

    category: str
    total_cost: Decimal
    item_count: int
    percentage_of_total: float


class DeviceModelCostSummary(CostBaseModel):
    """Cost summary for a device model."""

    device_model: str
    device_count: int
    unit_cost: Decimal
    total_value: Decimal


class CostSummaryResponse(CostBaseModel):
    """Response model for cost summary."""

    tenant_id: str
    summary_period: str = Field(description="Period covered: 'current_month', 'ytd', 'all_time'")

    # Totals
    total_hardware_value: Decimal = Field(description="Total value of all hardware")
    total_operational_monthly: Decimal = Field(description="Total monthly operational costs")
    total_operational_annual: Decimal = Field(description="Total annual operational costs")
    total_anomaly_impact_mtd: Decimal = Field(description="Month-to-date anomaly financial impact")
    total_anomaly_impact_ytd: Decimal = Field(description="Year-to-date anomaly financial impact")

    # Breakdowns
    by_category: list[CategoryCostSummary]
    by_device_model: list[DeviceModelCostSummary]

    # Trends
    cost_trend_30d: float | None = Field(None, description="Percentage change in costs over 30 days")
    anomaly_cost_trend_30d: float | None = Field(None, description="Percentage change in anomaly costs")

    # Metadata
    calculated_at: datetime
    device_count: int
    anomaly_count_period: int


# =============================================================================
# ANOMALY IMPACT MODELS
# =============================================================================

class ImpactComponent(CostBaseModel):
    """Single component of financial impact."""

    type: str = Field(description="Type: direct, productivity, support, opportunity")
    description: str
    amount: Decimal
    calculation_method: str = Field(description="How this was calculated")
    confidence: float = Field(ge=0, le=1, description="Confidence in this estimate")


class AnomalyImpactResponse(CostBaseModel):
    """Financial impact calculation for a specific anomaly."""

    anomaly_id: int
    device_id: int
    device_model: str | None = None
    anomaly_severity: str

    # Impact breakdown
    total_estimated_impact: Decimal
    impact_components: list[ImpactComponent]

    # Hardware context
    device_unit_cost: Decimal | None = None
    device_replacement_cost: Decimal | None = None
    device_age_months: int | None = None
    device_depreciated_value: Decimal | None = None

    # Productivity impact
    estimated_downtime_hours: float | None = None
    productivity_cost_per_hour: Decimal | None = None
    productivity_impact: Decimal | None = None

    # Resolution context
    average_resolution_time_hours: float | None = None
    support_cost_per_hour: Decimal | None = None
    estimated_support_cost: Decimal | None = None

    # Similar case context
    similar_cases_count: int = 0
    similar_cases_avg_cost: Decimal | None = None

    # Confidence
    overall_confidence: float = Field(ge=0, le=1)
    confidence_explanation: str

    # Impact level
    impact_level: ImpactLevel

    # Configuration status
    using_defaults: bool = Field(
        default=True,
        description="True if using system defaults instead of user-configured costs"
    )

    calculated_at: datetime


class DeviceImpactSummary(CostBaseModel):
    """Cost impact summary for a single device."""

    device_id: int
    device_model: str | None = None
    device_name: str | None = None
    location: str | None = None

    # Hardware value
    unit_cost: Decimal | None = None
    current_value: Decimal | None = None

    # Anomaly history
    total_anomalies: int
    open_anomalies: int
    resolved_anomalies: int

    # Impact totals
    total_estimated_impact: Decimal
    impact_mtd: Decimal
    impact_ytd: Decimal

    # Risk assessment
    risk_score: float = Field(ge=0, le=1, description="Risk based on anomaly frequency and cost")
    risk_level: str = Field(description="low, medium, high, critical")


class DeviceImpactResponse(CostBaseModel):
    """Detailed device-level cost impact response."""

    device_id: int
    device_model: str | None = None
    device_name: str | None = None

    # Summary
    summary: DeviceImpactSummary

    # Recent anomalies with impact
    recent_anomalies: list[AnomalyImpactResponse] = Field(default_factory=list)

    # Historical trend
    monthly_impact_trend: dict[str, Decimal] = Field(default_factory=dict, description="Last 12 months impact")

    # Recommendations
    cost_saving_recommendations: list[str] = Field(default_factory=list)


# =============================================================================
# COST HISTORY / AUDIT MODELS
# =============================================================================

class CostChangeEntry(CostBaseModel):
    """Single entry in cost change history."""

    id: int
    timestamp: datetime
    action: AuditAction
    entity_type: str = Field(description="'device_type_cost' or 'operational_cost'")
    entity_id: int
    entity_name: str
    changed_by: str | None = None

    # Change details
    field_changed: str | None = None
    old_value: str | None = None
    new_value: str | None = None

    # Full snapshots for audit
    before_snapshot: dict[str, Any] | None = None
    after_snapshot: dict[str, Any] | None = None


class CostHistoryResponse(CostBaseModel):
    """Response model for cost change history."""

    changes: list[CostChangeEntry]
    total: int
    page: int
    page_size: int
    total_pages: int

    # Summary
    total_creates: int
    total_updates: int
    total_deletes: int
    unique_users: int


# =============================================================================
# FINANCIAL IMPACT FOR INSIGHTS
# =============================================================================

class CostBreakdownItem(CostBaseModel):
    """A single item in a cost breakdown."""

    category: str
    description: str
    amount: Decimal
    is_recurring: bool = False
    period: str = "one_time"  # one_time, monthly, annual
    confidence: float = Field(0.7, ge=0, le=1)


class FinancialImpactSummary(CostBaseModel):
    """Financial impact summary for insights."""

    total_impact_usd: Decimal
    monthly_recurring_usd: Decimal
    potential_savings_usd: Decimal
    impact_level: ImpactLevel
    breakdown: list[CostBreakdownItem] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    # ROI if addressed
    investment_required_usd: Decimal | None = None
    payback_months: float | None = None

    # Metadata
    confidence_score: float = Field(0.7, ge=0, le=1)
    calculated_at: datetime | None = None


# =============================================================================
# BATTERY FORECASTING MODELS
# =============================================================================

class BatteryForecastEntry(CostBaseModel):
    """Battery replacement forecast for a device model."""

    device_model: str
    device_count: int
    battery_replacement_cost: Decimal
    battery_lifespan_months: int
    devices_due_this_month: int
    devices_due_next_month: int
    devices_due_in_90_days: int
    estimated_cost_30_days: Decimal
    estimated_cost_90_days: Decimal
    avg_battery_age_months: float
    oldest_battery_months: float | None = None
    avg_battery_health_percent: float | None = None
    data_quality: str = "estimated"  # "real", "estimated", "mixed"


class BatteryForecastResponse(CostBaseModel):
    """Response model for battery replacement forecast."""

    forecasts: list[BatteryForecastEntry]
    total_devices_with_battery_data: int
    total_estimated_cost_30_days: Decimal
    total_estimated_cost_90_days: Decimal
    total_replacements_due_30_days: int
    total_replacements_due_90_days: int
    forecast_generated_at: datetime
    data_quality: str = "estimated"  # "real", "estimated", "mixed"
    devices_with_health_data: int = 0


# =============================================================================
# COST ALERT MODELS
# =============================================================================

class CostAlertBase(CostBaseModel):
    """Base model for cost alerts."""

    name: str = Field(..., min_length=1, max_length=255)
    threshold_type: AlertThresholdType
    threshold_value: Decimal = Field(..., ge=0)
    is_active: bool = True
    notify_email: str | None = Field(None, max_length=255)
    notify_webhook: str | None = Field(None, max_length=2000)


class CostAlertCreate(CostAlertBase):
    """Request model for creating a cost alert."""

    pass


class CostAlertUpdate(CostBaseModel):
    """Request model for updating a cost alert."""

    name: str | None = Field(None, min_length=1, max_length=255)
    threshold_type: AlertThresholdType | None = None
    threshold_value: Decimal | None = Field(None, ge=0)
    is_active: bool | None = None
    notify_email: str | None = Field(None, max_length=255)
    notify_webhook: str | None = Field(None, max_length=2000)


class CostAlert(CostBaseModel):
    """Response model for a cost alert."""

    id: int
    name: str
    threshold_type: AlertThresholdType
    threshold_value: Decimal
    is_active: bool
    notify_email: str | None = None
    notify_webhook: str | None = None
    last_triggered: datetime | None = None
    trigger_count: int = 0
    created_at: datetime
    updated_at: datetime


class CostAlertListResponse(CostBaseModel):
    """Response model for cost alert list."""

    alerts: list[CostAlert]
    total: int


# =============================================================================
# NFF SUMMARY MODELS
# =============================================================================

class NFFByDeviceModel(CostBaseModel):
    """NFF summary grouped by device model."""

    device_model: str
    count: int
    total_cost: Decimal


class NFFByResolution(CostBaseModel):
    """NFF summary grouped by resolution."""

    resolution: str
    count: int
    total_cost: Decimal


class NFFSummaryResponse(CostBaseModel):
    """Summary response for No Fault Found analysis."""

    total_nff_count: int
    total_nff_cost: Decimal
    avg_cost_per_nff: Decimal
    nff_rate_percent: float
    by_device_model: list[NFFByDeviceModel]
    by_resolution: list[NFFByResolution]
    trend_30_days: float
