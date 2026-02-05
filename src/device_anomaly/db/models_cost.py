"""SQLAlchemy models for the Cost Intelligence Module.

This module defines the data model for tracking hardware and operational costs
across the device fleet, with full audit trail and multi-tenant support.

Design principles:
1. All monetary values stored as BigInteger (cents) to avoid floating-point issues
2. All tables include tenant_id for multi-tenant isolation
3. Temporal validity (valid_from/valid_to) for cost history
4. Full audit trail for compliance
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)

from device_anomaly.db.models import Base


def _utc_now() -> datetime:
    """Return current UTC time. Used as default for DateTime columns."""
    return datetime.now(UTC)


class DeviceTypeCost(Base):
    """Hardware asset costs linked to device models.

    Stores purchase, replacement, and depreciation costs for device types.
    Uses temporal validity (valid_from/valid_to) for price change history.

    All monetary values stored in minor currency units (cents).
    Example: $599.99 stored as 59999

    Attributes:
        device_model: Device model string (e.g., "Zebra TC52", "TC75x")
        purchase_cost: Original purchase price in cents
        replacement_cost: Current replacement cost in cents
        repair_cost_avg: Average repair cost in cents
        depreciation_months: Expected useful life in months
        residual_value_percent: Expected value at end of life as percentage
    """
    __tablename__ = "device_type_costs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)
    device_model = Column(String(255), nullable=False)  # Device model string

    # Currency (ISO 4217 code)
    currency_code = Column(String(3), default="USD", nullable=False)

    # Core costs (stored in minor units - cents)
    purchase_cost = Column(BigInteger, nullable=False)  # Original purchase price
    replacement_cost = Column(BigInteger)  # Current replacement cost
    repair_cost_avg = Column(BigInteger)  # Average repair cost

    # Depreciation configuration
    depreciation_months = Column(Integer, default=36)  # Useful life in months
    residual_value_percent = Column(Integer, default=0)  # End-of-life value as %

    # Warranty
    warranty_months = Column(Integer)  # Warranty period in months

    # Temporal validity (for price change history)
    valid_from = Column(DateTime, nullable=False, default=_utc_now)
    valid_to = Column(DateTime)  # NULL = currently active

    # Metadata
    notes = Column(Text)
    created_by = Column(String(100))
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)

    __table_args__ = (
        Index("idx_dtc_tenant_model", "tenant_id", "device_model"),
        Index("idx_dtc_tenant_valid", "tenant_id", "valid_from", "valid_to"),
        Index("idx_dtc_active", "tenant_id", "valid_to"),
        Index("idx_dtc_currency", "tenant_id", "currency_code"),
    )

    def __repr__(self):
        return f"<DeviceTypeCost(model='{self.device_model}', purchase={self.purchase_cost}, currency='{self.currency_code}')>"

    @property
    def purchase_cost_decimal(self) -> float:
        """Return purchase cost as decimal dollars."""
        return self.purchase_cost / 100.0 if self.purchase_cost else 0.0

    @property
    def replacement_cost_decimal(self) -> float:
        """Return replacement cost as decimal dollars."""
        return self.replacement_cost / 100.0 if self.replacement_cost else 0.0

    @property
    def repair_cost_avg_decimal(self) -> float:
        """Return average repair cost as decimal dollars."""
        return self.repair_cost_avg / 100.0 if self.repair_cost_avg else 0.0

    @property
    def is_active(self) -> bool:
        """Check if this cost record is currently active."""
        return self.valid_to is None


class OperationalCost(Base):
    """Custom operational cost entries per tenant.

    Stores tenant-specific operational costs like hourly wages, downtime costs,
    IT support costs, etc. Supports categorization and scoping.

    All monetary values stored in minor currency units (cents).

    Categories:
        - labor: Hourly wages, overtime, contractor rates
        - downtime: Cost per hour of device/worker downtime
        - support: IT support costs, helpdesk, training
        - infrastructure: Network, software licenses
        - maintenance: Regular maintenance costs
        - other: Other operational costs
    """
    __tablename__ = "operational_costs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)

    # Cost identification
    name = Column(String(255), nullable=False)  # Human-readable name
    description = Column(Text)

    # Categorization
    category = Column(String(50), nullable=False)  # labor, downtime, support, infrastructure, maintenance, other

    # Currency
    currency_code = Column(String(3), default="USD", nullable=False)

    # Cost values (in minor units - cents)
    amount = Column(BigInteger, nullable=False)  # Primary cost value

    # Cost type and unit
    cost_type = Column(String(50), nullable=False)  # hourly, daily, per_incident, fixed_monthly, per_device
    unit = Column(String(50))  # hour, day, incident, month, device

    # Scope (what this cost applies to)
    scope_type = Column(String(50), default="tenant")  # tenant, location, device_group, device_model
    scope_id = Column(String(100))  # ID of the scope entity (null for tenant-wide)

    # Active status
    is_active = Column(Boolean, default=True, nullable=False)

    # Temporal validity
    valid_from = Column(DateTime, nullable=False, default=_utc_now)
    valid_to = Column(DateTime)  # NULL = currently active

    # Metadata
    notes = Column(Text)
    created_by = Column(String(100))
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)

    __table_args__ = (
        Index("idx_oc_tenant_name", "tenant_id", "name"),
        Index("idx_oc_tenant_category", "tenant_id", "category"),
        Index("idx_oc_tenant_valid", "tenant_id", "valid_from", "valid_to"),
        Index("idx_oc_active", "tenant_id", "is_active"),
        Index("idx_oc_scope", "tenant_id", "scope_type", "scope_id"),
    )

    def __repr__(self):
        return f"<OperationalCost(name='{self.name}', category='{self.category}', amount={self.amount})>"

    @property
    def amount_decimal(self) -> float:
        """Return amount as decimal dollars."""
        return self.amount / 100.0 if self.amount else 0.0


class CostAuditLog(Base):
    """Audit trail for all cost-related changes.

    Tracks who changed what cost data and when, with full before/after values.
    Required for compliance and cost reconciliation.

    Actions tracked:
        - create: New cost record created
        - update: Existing cost modified
        - delete: Cost record removed
    """
    __tablename__ = "cost_audit_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)

    # What was changed
    entity_type = Column(String(50), nullable=False)  # device_type_cost, operational_cost
    entity_id = Column(BigInteger, nullable=False)  # ID of the changed record

    # Change details
    action = Column(String(20), nullable=False)  # create, update, delete

    # Before/After values (JSON snapshots)
    old_values_json = Column(Text)  # JSON: Previous values (null for create)
    new_values_json = Column(Text)  # JSON: New values (null for delete)

    # Specific changed fields (for quick filtering)
    changed_fields_json = Column(Text)  # JSON array: ["purchase_cost", "replacement_cost"]

    # Who made the change
    user_id = Column(String(100))
    user_email = Column(String(255))

    # Change context
    change_reason = Column(Text)  # User-provided reason for change
    source = Column(String(50), default="manual")  # manual, api, import, system

    # Request metadata
    ip_address = Column(String(45))  # IPv6 max length
    user_agent = Column(String(500))
    request_id = Column(String(100))  # For tracing

    # Timestamp
    timestamp = Column(DateTime, default=_utc_now, nullable=False)

    __table_args__ = (
        Index("idx_cal_tenant_time", "tenant_id", "timestamp"),
        Index("idx_cal_entity", "entity_type", "entity_id", "timestamp"),
        Index("idx_cal_user", "user_id", "timestamp"),
        Index("idx_cal_action", "tenant_id", "action", "timestamp"),
    )

    def __repr__(self):
        return f"<CostAuditLog(entity='{self.entity_type}:{self.entity_id}', action='{self.action}')>"


class CostCalculationCache(Base):
    """Pre-computed cost calculations for anomalies and events.

    Stores calculated cost impact for anomaly events, enabling quick
    reporting without re-calculation. Links costs to specific anomaly events.

    Example calculations:
        - Device replacement cost if failure predicted
        - Downtime cost based on event duration
        - IT support cost for investigation time
    """
    __tablename__ = "cost_calculation_cache"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), ForeignKey("tenants.tenant_id"), nullable=False)

    # What this calculation is for
    entity_type = Column(String(50), nullable=False)  # anomaly, anomaly_event, insight, device
    entity_id = Column(String(100), nullable=False)  # ID of the entity

    # Device context (for cost lookup)
    device_id = Column(String(50))
    device_model = Column(String(255))

    # Currency
    currency_code = Column(String(3), default="USD", nullable=False)

    # Calculated costs (in minor units - cents)
    hardware_cost = Column(BigInteger, default=0)  # Replacement/repair cost
    downtime_cost = Column(BigInteger, default=0)  # Lost productivity
    labor_cost = Column(BigInteger, default=0)  # IT support/investigation
    other_cost = Column(BigInteger, default=0)  # Other costs
    total_cost = Column(BigInteger, default=0)  # Sum of all costs

    # Potential savings
    potential_savings = Column(BigInteger, default=0)  # Monthly savings if addressed
    investment_required = Column(BigInteger, default=0)  # Investment to fix
    payback_months = Column(Float)  # Months to ROI

    # Cost breakdown details (JSON)
    breakdown_json = Column(Text)  # Detailed breakdown of each cost component

    # Impact level
    impact_level = Column(String(20))  # high, medium, low

    # Confidence
    confidence_score = Column(Float, default=0.7)
    confidence_explanation = Column(Text)

    # Calculation metadata
    calculation_version = Column(String(20), default="v1")  # For tracking formula changes
    cost_config_snapshot_json = Column(Text)  # Snapshot of costs used in calculation

    # Timestamps
    calculated_at = Column(DateTime, default=_utc_now, nullable=False)
    expires_at = Column(DateTime)  # When this calculation should be refreshed

    __table_args__ = (
        Index("idx_ccc_tenant_entity", "tenant_id", "entity_type", "entity_id"),
        Index("idx_ccc_device", "tenant_id", "device_id"),
        Index("idx_ccc_calculated", "tenant_id", "calculated_at"),
        Index("idx_ccc_impact", "tenant_id", "impact_level"),
    )

    def __repr__(self):
        return f"<CostCalculationCache(entity='{self.entity_type}:{self.entity_id}', total={self.total_cost})>"

    @property
    def total_cost_decimal(self) -> float:
        """Return total cost as decimal dollars."""
        return self.total_cost / 100.0 if self.total_cost else 0.0

    @property
    def potential_savings_decimal(self) -> float:
        """Return potential savings as decimal dollars."""
        return self.potential_savings / 100.0 if self.potential_savings else 0.0
