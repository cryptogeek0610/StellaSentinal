"""
Cost Intelligence API routes package.

This package contains all cost-related endpoints organized by domain:
- hardware: Device type costs and fleet value
- operational: Custom operational cost tracking
- summary: Aggregated cost summaries
- impact: Anomaly financial impact calculations
- history: Cost audit trail
- alerts: Cost alert configuration
- forecasts: Battery replacement and NFF forecasts
"""
from fastapi import APIRouter

from .alerts import router as alerts_router
from .forecasts import router as forecasts_router
from .hardware import router as hardware_router
from .history import router as history_router
from .impact import router as impact_router
from .operational import router as operational_router
from .summary import router as summary_router
from .utils import (
    calculate_monthly_equivalent,
    cents_to_dollars,
    create_audit_log,
    dollars_to_cents,
)

# Main costs router that combines all sub-routers
router = APIRouter(prefix="/costs", tags=["costs"])

# Include all sub-routers
router.include_router(hardware_router)
router.include_router(operational_router)
router.include_router(summary_router)
router.include_router(impact_router)
router.include_router(history_router)
router.include_router(alerts_router)
router.include_router(forecasts_router)

__all__ = [
    "router",
    "cents_to_dollars",
    "dollars_to_cents",
    "calculate_monthly_equivalent",
    "create_audit_log",
]
