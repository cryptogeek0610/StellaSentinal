"""Pydantic models for Cost Intelligence API requests and responses."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CostCategory(str, Enum):
    """Cost category classification."""

    LABOR = "labor"
    DOWNTIME = "downtime"
    SUPPORT = "support"
    INFRASTRUCTURE = "infrastructure"
    MAINTENANCE = "maintenance"
    OTHER = "other"


class CostType(str, Enum):
    """Cost calculation type."""

    HOURLY = "hourly"
    DAILY = "daily"
    PER_INCIDENT = "per_incident"
    FIXED_MONTHLY = "fixed_monthly"
    PER_DEVICE = "per_device"


class ScopeType(str, Enum):
    """Scope type for operational costs."""

    TENANT = "tenant"
    LOCATION = "location"
    DEVICE_GROUP = "device_group"
    DEVICE_MODEL = "device_model"


class ImpactLevel(str, Enum):
    """Financial impact severity level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AuditAction(str, Enum):
    """Audit log action types."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class AlertThresholdType(str, Enum):
    """Threshold types for cost alerts."""

    ANOMALY_COST_DAILY = "anomaly_cost_daily"
    ANOMALY_COST_MONTHLY = "anomaly_cost_monthly"
    BATTERY_FORECAST = "battery_forecast"
    OPERATIONAL_COST = "operational_cost"


# =============================================================================
# HARDWARE COST MODELS
# =============================================================================

class HardwareCostBase(BaseModel):
    """Base model for hardware cost entries."""

    device_model: str = Field(..., min_length=1, max_length=255, description="Device model identifier")
    purchase_cost: Decimal = Field(..., ge=0, description="Purchase cost in dollars")
    replacement_cost: Optional[Decimal] = Field(None, ge=0, description="Replacement cost if different")
    repair_cost_avg: Optional[Decimal] = Field(None, ge=0, description="Average repair cost")
    depreciation_months: Optional[int] = Field(36, ge=1, le=120, description="Depreciation period in months")
    residual_value_percent: Optional[int] = Field(0, ge=0, le=100, description="Residual value as percentage")
    warranty_months: Optional[int] = Field(None, ge=0, le=120, description="Warranty period in months")
    currency_code: str = Field("USD", max_length=3, description="ISO 4217 currency code")
    notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")


class HardwareCostCreate(HardwareCostBase):
    """Request model for creating a hardware cost entry."""

    pass


class HardwareCostUpdate(BaseModel):
    """Request model for updating a hardware cost entry (partial update)."""

    device_model: Optional[str] = Field(None, min_length=1, max_length=255)
    purchase_cost: Optional[Decimal] = Field(None, ge=0)
    replacement_cost: Optional[Decimal] = Field(None, ge=0)
    repair_cost_avg: Optional[Decimal] = Field(None, ge=0)
    depreciation_months: Optional[int] = Field(None, ge=1, le=120)
    residual_value_percent: Optional[int] = Field(None, ge=0, le=100)
    warranty_months: Optional[int] = Field(None, ge=0, le=120)
    notes: Optional[str] = Field(None, max_length=2000)


class HardwareCostResponse(HardwareCostBase):
    """Response model for a hardware cost entry."""

    id: int
    tenant_id: str
    device_count: int = Field(0, description="Number of devices using this model")
    total_fleet_value: Decimal = Field(default=Decimal(0), description="Total value of devices with this model")
    valid_from: datetime
    valid_to: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None

    model_config = {"from_attributes": True}


class HardwareCostListResponse(BaseModel):
    """Response model for paginated hardware cost list."""

    costs: List[HardwareCostResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    total_fleet_value: Decimal = Field(description="Sum of all fleet values")


class DeviceModelInfo(BaseModel):
    """Device model information from database."""

    device_model: str
    device_count: int
    has_cost_entry: bool


class DeviceModelsResponse(BaseModel):
    """Response model for device models list."""

    models: List[DeviceModelInfo]
    total: int


# =============================================================================
# OPERATIONAL COST MODELS
# =============================================================================

class OperationalCostBase(BaseModel):
    """Base model for operational cost entries."""

    name: str = Field(..., min_length=1, max_length=255, description="Cost entry name")
    category: CostCategory = Field(CostCategory.OTHER, description="Cost category")
    amount: Decimal = Field(..., ge=0, description="Cost amount in dollars")
    cost_type: CostType = Field(CostType.FIXED_MONTHLY, description="Cost calculation type")
    unit: Optional[str] = Field(None, max_length=50, description="Unit of measurement")
    scope_type: ScopeType = Field(ScopeType.TENANT, description="Scope type")
    scope_id: Optional[str] = Field(None, max_length=100, description="Scope entity ID")
    description: Optional[str] = Field(None, max_length=2000)
    currency_code: str = Field("USD", max_length=3, description="ISO 4217 currency code")
    is_active: bool = Field(True, description="Whether this cost is currently active")
    notes: Optional[str] = Field(None, max_length=2000)


class OperationalCostCreate(OperationalCostBase):
    """Request model for creating an operational cost entry."""

    pass


class OperationalCostUpdate(BaseModel):
    """Request model for updating an operational cost entry."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    category: Optional[CostCategory] = None
    amount: Optional[Decimal] = Field(None, ge=0)
    cost_type: Optional[CostType] = None
    unit: Optional[str] = Field(None, max_length=50)
    scope_type: Optional[ScopeType] = None
    scope_id: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=2000)
    is_active: Optional[bool] = None
    notes: Optional[str] = Field(None, max_length=2000)


class OperationalCostResponse(OperationalCostBase):
    """Response model for an operational cost entry."""

    id: int
    tenant_id: str
    valid_from: datetime
    valid_to: Optional[datetime] = None
    monthly_equivalent: Decimal = Field(description="Normalized monthly cost")
    annual_equivalent: Decimal = Field(description="Normalized annual cost")
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None

    model_config = {"from_attributes": True}


class OperationalCostListResponse(BaseModel):
    """Response model for paginated operational cost list."""

    costs: List[OperationalCostResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    total_monthly_cost: Decimal
    total_annual_cost: Decimal


# =============================================================================
# COST SUMMARY MODELS
# =============================================================================

class CategoryCostSummary(BaseModel):
    """Cost summary for a single category."""

    category: str
    total_cost: Decimal
    item_count: int
    percentage_of_total: float


class DeviceModelCostSummary(BaseModel):
    """Cost summary for a device model."""

    device_model: str
    device_count: int
    unit_cost: Decimal
    total_value: Decimal


class CostSummaryResponse(BaseModel):
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
    by_category: List[CategoryCostSummary]
    by_device_model: List[DeviceModelCostSummary]

    # Trends
    cost_trend_30d: Optional[float] = Field(None, description="Percentage change in costs over 30 days")
    anomaly_cost_trend_30d: Optional[float] = Field(None, description="Percentage change in anomaly costs")

    # Metadata
    calculated_at: datetime
    device_count: int
    anomaly_count_period: int


# =============================================================================
# ANOMALY IMPACT MODELS
# =============================================================================

class ImpactComponent(BaseModel):
    """Single component of financial impact."""

    type: str = Field(description="Type: direct, productivity, support, opportunity")
    description: str
    amount: Decimal
    calculation_method: str = Field(description="How this was calculated")
    confidence: float = Field(ge=0, le=1, description="Confidence in this estimate")


class AnomalyImpactResponse(BaseModel):
    """Financial impact calculation for a specific anomaly."""

    anomaly_id: int
    device_id: int
    device_model: Optional[str] = None
    anomaly_severity: str

    # Impact breakdown
    total_estimated_impact: Decimal
    impact_components: List[ImpactComponent]

    # Hardware context
    device_unit_cost: Optional[Decimal] = None
    device_replacement_cost: Optional[Decimal] = None
    device_age_months: Optional[int] = None
    device_depreciated_value: Optional[Decimal] = None

    # Productivity impact
    estimated_downtime_hours: Optional[float] = None
    productivity_cost_per_hour: Optional[Decimal] = None
    productivity_impact: Optional[Decimal] = None

    # Resolution context
    average_resolution_time_hours: Optional[float] = None
    support_cost_per_hour: Optional[Decimal] = None
    estimated_support_cost: Optional[Decimal] = None

    # Similar case context
    similar_cases_count: int = 0
    similar_cases_avg_cost: Optional[Decimal] = None

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


class DeviceImpactSummary(BaseModel):
    """Cost impact summary for a single device."""

    device_id: int
    device_model: Optional[str] = None
    device_name: Optional[str] = None
    location: Optional[str] = None

    # Hardware value
    unit_cost: Optional[Decimal] = None
    current_value: Optional[Decimal] = None

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


class DeviceImpactResponse(BaseModel):
    """Detailed device-level cost impact response."""

    device_id: int
    device_model: Optional[str] = None
    device_name: Optional[str] = None

    # Summary
    summary: DeviceImpactSummary

    # Recent anomalies with impact
    recent_anomalies: List[AnomalyImpactResponse] = Field(default_factory=list)

    # Historical trend
    monthly_impact_trend: Dict[str, Decimal] = Field(default_factory=dict, description="Last 12 months impact")

    # Recommendations
    cost_saving_recommendations: List[str] = Field(default_factory=list)


# =============================================================================
# COST HISTORY / AUDIT MODELS
# =============================================================================

class CostChangeEntry(BaseModel):
    """Single entry in cost change history."""

    id: int
    timestamp: datetime
    action: AuditAction
    entity_type: str = Field(description="'device_type_cost' or 'operational_cost'")
    entity_id: int
    entity_name: str
    changed_by: Optional[str] = None

    # Change details
    field_changed: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None

    # Full snapshots for audit
    before_snapshot: Optional[Dict[str, Any]] = None
    after_snapshot: Optional[Dict[str, Any]] = None


class CostHistoryResponse(BaseModel):
    """Response model for cost change history."""

    changes: List[CostChangeEntry]
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

class CostBreakdownItem(BaseModel):
    """A single item in a cost breakdown."""

    category: str
    description: str
    amount: Decimal
    is_recurring: bool = False
    period: str = "one_time"  # one_time, monthly, annual
    confidence: float = Field(0.7, ge=0, le=1)


class FinancialImpactSummary(BaseModel):
    """Financial impact summary for insights."""

    total_impact_usd: Decimal
    monthly_recurring_usd: Decimal
    potential_savings_usd: Decimal
    impact_level: ImpactLevel
    breakdown: List[CostBreakdownItem] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    # ROI if addressed
    investment_required_usd: Optional[Decimal] = None
    payback_months: Optional[float] = None

    # Metadata
    confidence_score: float = Field(0.7, ge=0, le=1)
    calculated_at: Optional[datetime] = None


# =============================================================================
# BATTERY FORECASTING MODELS
# =============================================================================

class BatteryForecastEntry(BaseModel):
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
    oldest_battery_months: Optional[float] = None


class BatteryForecastResponse(BaseModel):
    """Response model for battery replacement forecast."""

    forecasts: List[BatteryForecastEntry]
    total_devices_with_battery_data: int
    total_estimated_cost_30_days: Decimal
    total_estimated_cost_90_days: Decimal
    total_replacements_due_30_days: int
    total_replacements_due_90_days: int
    forecast_generated_at: datetime


# =============================================================================
# COST ALERT MODELS
# =============================================================================

class CostAlertBase(BaseModel):
    """Base model for cost alerts."""

    name: str = Field(..., min_length=1, max_length=255)
    threshold_type: AlertThresholdType
    threshold_value: Decimal = Field(..., ge=0)
    is_active: bool = True
    notify_email: Optional[str] = Field(None, max_length=255)
    notify_webhook: Optional[str] = Field(None, max_length=2000)


class CostAlertCreate(CostAlertBase):
    """Request model for creating a cost alert."""

    pass


class CostAlertUpdate(BaseModel):
    """Request model for updating a cost alert."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    threshold_type: Optional[AlertThresholdType] = None
    threshold_value: Optional[Decimal] = Field(None, ge=0)
    is_active: Optional[bool] = None
    notify_email: Optional[str] = Field(None, max_length=255)
    notify_webhook: Optional[str] = Field(None, max_length=2000)


class CostAlert(BaseModel):
    """Response model for a cost alert."""

    id: int
    name: str
    threshold_type: AlertThresholdType
    threshold_value: Decimal
    is_active: bool
    notify_email: Optional[str] = None
    notify_webhook: Optional[str] = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    created_at: datetime
    updated_at: datetime


class CostAlertListResponse(BaseModel):
    """Response model for cost alert list."""

    alerts: List[CostAlert]
    total: int


# =============================================================================
# NFF SUMMARY MODELS
# =============================================================================

class NFFByDeviceModel(BaseModel):
    """NFF summary grouped by device model."""

    device_model: str
    count: int
    total_cost: Decimal


class NFFByResolution(BaseModel):
    """NFF summary grouped by resolution."""

    resolution: str
    count: int
    total_cost: Decimal


class NFFSummaryResponse(BaseModel):
    """Summary response for No Fault Found analysis."""

    total_nff_count: int
    total_nff_cost: Decimal
    avg_cost_per_nff: Decimal
    nff_rate_percent: float
    by_device_model: List[NFFByDeviceModel]
    by_resolution: List[NFFByResolution]
    trend_30_days: float
