"""Insights module for customer-facing anomaly translation.

This module transforms technical anomaly detection results into
plain-language, actionable insights that customers can understand.

Aligned with Carl's (CEO) direction:
- "XSight has the data. XSight needs the story."
- Insights should be pre-interpreted, contextualized, and actionable
- Comparisons matter more than absolutes

Key components:
- categories: InsightCategory enum for all insight types
- templates: Business language templates for each category
- classifier: Maps raw anomalies to insight categories
- location_mapper: Maps devices to locations for aggregation
- entities: EntityAggregator for location/user/cohort rollup
- comparisons: ComparisonEngine for fleet/cohort/historical context
- battery_shift: BatteryShiftAnalyzer for shift-aware analysis
- network_patterns: NetworkPatternAnalyzer for AP/carrier analysis
- device_abuse: DeviceAbuseAnalyzer for drops/reboots
- app_power: AppPowerAnalyzer for app battery drain
- generator: InsightGenerator orchestrates all components
"""

from device_anomaly.insights.categories import (
    CATEGORY_METADATA,
    EntityType,
    InsightCategory,
    InsightSeverity,
    affects_productivity,
    get_categories_by_domain,
    get_category_metrics,
    get_category_severity,
)
from device_anomaly.insights.classifier import (
    ClassificationEvidence,
    ClassifiedInsight,
    ClassifierConfig,
    InsightClassifier,
)
from device_anomaly.insights.location_mapper import LocationMapper
from device_anomaly.insights.templates import (
    INSIGHT_TEMPLATES,
    InsightTemplate,
    get_template,
    render_actions,
    render_comparison,
    render_headline,
    render_impact,
    render_ticket_description,
)
from device_anomaly.insights.entities import (
    CohortInsight,
    EntityAggregator,
    EntityMetrics,
    LocationInsight,
    UserInsight,
)
from device_anomaly.insights.comparisons import (
    CohortComparison,
    ComparisonEngine,
    ComparisonType,
    FleetComparison,
    HistoricalComparison,
    LocationComparison,
)
from device_anomaly.insights.battery_shift import (
    BatteryProjection,
    BatteryShiftAnalyzer,
    ChargingPatternReport,
    DeviceShiftReadiness,
    PeriodicDrainReport,
    ShiftReadinessReport,
)
from device_anomaly.insights.network_patterns import (
    CellularPatternReport,
    DisconnectPatternReport,
    HiddenDeviceReport,
    NetworkPatternAnalyzer,
    WifiRoamingReport,
)
from device_anomaly.insights.device_abuse import (
    DeviceAbuseAnalyzer,
    DropAnalysisReport,
    ProblemCombinationReport,
    RebootAnalysisReport,
)
from device_anomaly.insights.app_power import (
    AppBatteryCorrelation,
    AppCrashReport,
    AppPowerAnalyzer,
    AppPowerReport,
)
from device_anomaly.insights.generator import (
    CustomerInsight,
    DailyInsightDigest,
    InsightGenerator,
    LocationInsightReport,
    TrendingInsight,
)

__all__ = [
    # Categories
    "InsightCategory",
    "InsightSeverity",
    "EntityType",
    "CATEGORY_METADATA",
    "get_categories_by_domain",
    "get_category_severity",
    "get_category_metrics",
    "affects_productivity",
    # Classifier
    "InsightClassifier",
    "ClassifierConfig",
    "ClassifiedInsight",
    "ClassificationEvidence",
    # Templates
    "InsightTemplate",
    "INSIGHT_TEMPLATES",
    "get_template",
    "render_headline",
    "render_impact",
    "render_comparison",
    "render_actions",
    "render_ticket_description",
    # Location
    "LocationMapper",
    # Entities
    "EntityAggregator",
    "EntityMetrics",
    "LocationInsight",
    "UserInsight",
    "CohortInsight",
    # Comparisons
    "ComparisonEngine",
    "ComparisonType",
    "FleetComparison",
    "CohortComparison",
    "HistoricalComparison",
    "LocationComparison",
    # Battery Shift Analyzer
    "BatteryShiftAnalyzer",
    "BatteryProjection",
    "ShiftReadinessReport",
    "DeviceShiftReadiness",
    "ChargingPatternReport",
    "PeriodicDrainReport",
    # Network Pattern Analyzer
    "NetworkPatternAnalyzer",
    "WifiRoamingReport",
    "CellularPatternReport",
    "DisconnectPatternReport",
    "HiddenDeviceReport",
    # Device Abuse Analyzer
    "DeviceAbuseAnalyzer",
    "DropAnalysisReport",
    "RebootAnalysisReport",
    "ProblemCombinationReport",
    # App Power Analyzer
    "AppPowerAnalyzer",
    "AppPowerReport",
    "AppCrashReport",
    "AppBatteryCorrelation",
    # Insight Generator
    "InsightGenerator",
    "CustomerInsight",
    "DailyInsightDigest",
    "LocationInsightReport",
    "TrendingInsight",
]
