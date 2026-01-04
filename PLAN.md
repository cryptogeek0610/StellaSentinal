# XSight Anomaly Detection: Customer-Facing Transformation Plan

> "XSight has the data. XSight needs the story."

## Executive Summary

This plan transforms the existing StellaSentinel anomaly detection system from a technical ML platform into a customer-centric insights engine that speaks plain language with clear business value and actionable next steps.

---

## Part 1: Current State Assessment

### What Exists
- **Solid ML Foundation**: Hybrid IsolationForest + temporal + heuristic detection
- **Rich Data Pipeline**: Battery, network, app, connectivity, security metrics
- **API Infrastructure**: FastAPI with multi-tenant support, WebSocket streaming
- **React Frontend**: Investigation panels, dashboards, device detail pages
- **LLM Integration**: Framework for explanations (partially implemented)
- **Database Schema**: Comprehensive tables for anomalies, remediations, learning

### Critical Gaps (Mapped to Carl's Feedback)

| Carl's Requirement | Current State | Gap |
|-------------------|---------------|-----|
| "Things customers could understand" | Technical ML scores | No plain-language translation |
| Batteries not lasting a shift | Raw drain metrics | No shift-context analysis |
| Excessive drops by person/location | Device-level only | No aggregation by location/user |
| Performance by manufacturer/model/OS | Basic cohort detection | No comparative insights |
| Apps consuming too much power | Foreground time metrics exist | No drain correlation analysis |
| Network disconnect patterns | Disconnect count only | No carrier/tower/AP analysis |
| Relate anomalies to location | DeviceId focus | No warehouse-level comparison |

---

## Part 2: Architecture for Customer-Facing Insights

### New Layer: Insight Translation Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                    CUSTOMER PRESENTATION LAYER                   │
├─────────────────────────────────────────────────────────────────┤
│  • Plain-language insight cards                                  │
│  • Business impact quantification                                │
│  • Comparative context (vs baseline, vs peers)                   │
│  • One-click remediation actions                                 │
│  • Proactive notification system                                 │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│                    INSIGHT TRANSLATION ENGINE                    │  ← NEW
├─────────────────────────────────────────────────────────────────┤
│  • Anomaly → Business Impact Translator                          │
│  • Cross-entity Aggregator (location, user, cohort)              │
│  • Comparative Analysis Engine                                   │
│  • Natural Language Generator (LLM-powered)                      │
│  • Action Recommendation Engine                                  │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│                    EXISTING ML DETECTION LAYER                   │
├─────────────────────────────────────────────────────────────────┤
│  • IsolationForest + Hybrid Detection                            │
│  • Feature Engineering                                           │
│  • Baseline Computation                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Implementation Steps

### Phase 1: Insight Categories & Business Language

**Goal**: Transform technical anomalies into Carl's customer categories

#### Step 1.1: Define Insight Category Taxonomy

Create `/src/device_anomaly/insights/categories.py`:

```python
class InsightCategory(Enum):
    # Battery Insights
    BATTERY_SHIFT_FAILURE = "battery_shift_failure"
    BATTERY_RAPID_DRAIN = "battery_rapid_drain"
    BATTERY_CHARGE_INCOMPLETE = "battery_charge_incomplete"
    BATTERY_CHARGE_PATTERN = "battery_charge_pattern"

    # Device Insights
    EXCESSIVE_DROPS = "excessive_drops"
    EXCESSIVE_REBOOTS = "excessive_reboots"
    PERFORMANCE_DEGRADATION = "performance_degradation"

    # Utilization Insights
    UNDERUTILIZATION = "underutilization"
    OVERUTILIZATION = "overutilization"

    # App Insights
    APP_CRASHES = "app_crashes"
    APP_POWER_DRAIN = "app_power_drain"

    # Network Insights
    NETWORK_DISCONNECTS = "network_disconnects"
    WIFI_ROAMING_ISSUES = "wifi_roaming_issues"
    CELLULAR_ISSUES = "cellular_issues"
    THROUGHPUT_DEGRADATION = "throughput_degradation"
    SERVER_DISCONNECT_PATTERN = "server_disconnect_pattern"
```

#### Step 1.2: Create Business Impact Templates

Create `/src/device_anomaly/insights/templates.py`:

Define templates that translate technical findings into customer language:

```python
INSIGHT_TEMPLATES = {
    "battery_shift_failure": {
        "headline": "{count} devices in {location} won't last a full shift",
        "detail": "These devices are draining {drain_rate}% faster than normal. At current rates, they'll need charging by {estimated_dead_time}.",
        "impact": "Workers may experience {downtime_estimate} of unplanned downtime",
        "comparison": "{percent_worse}% worse than {comparison_location}",
        "action": "Review charging schedules or consider battery replacement"
    },
    "excessive_drops": {
        "headline": "{entity_type} '{entity_name}' has {count} excessive device drops",
        "detail": "This is {multiplier}x higher than the fleet average of {fleet_avg} drops per shift.",
        "impact": "Increased repair costs estimated at ${cost_estimate}",
        "comparison": "Compared to other {entity_type}s, this ranks #{rank} worst",
        "action": "Investigate handling practices or protective case usage"
    },
    # ... templates for all categories
}
```

#### Step 1.3: Create Anomaly-to-Insight Classifier

Create `/src/device_anomaly/insights/classifier.py`:

Map raw anomaly features to insight categories:

```python
class InsightClassifier:
    def classify(self, anomaly: AnomalyResult) -> List[InsightCategory]:
        """
        Analyze anomaly features and return applicable insight categories.
        Uses both rule-based matching and feature contribution analysis.
        """
        categories = []

        # Battery analysis
        if self._is_shift_battery_failure(anomaly):
            categories.append(InsightCategory.BATTERY_SHIFT_FAILURE)
        if self._is_rapid_drain(anomaly):
            categories.append(InsightCategory.BATTERY_RAPID_DRAIN)

        # Device analysis
        if self._has_excessive_drops(anomaly):
            categories.append(InsightCategory.EXCESSIVE_DROPS)

        # ... classify all categories

        return categories
```

---

### Phase 2: Cross-Entity Aggregation

**Goal**: Enable "Warehouse 1 vs Warehouse 2" comparisons

#### Step 2.1: Entity Hierarchy Model

Create `/src/device_anomaly/insights/entities.py`:

```python
class EntityHierarchy:
    """
    Represents the organizational hierarchy for aggregation:
    Organization → Region → Location → Zone → User → Device
    """

class EntityAggregator:
    """
    Aggregates anomalies up the entity hierarchy:
    - Device anomalies → User patterns
    - User patterns → Location insights
    - Location insights → Regional comparisons
    """

    def aggregate_by_location(self, anomalies: List[AnomalyResult]) -> Dict[str, LocationInsight]:
        """Group and summarize anomalies by location."""

    def aggregate_by_user(self, anomalies: List[AnomalyResult]) -> Dict[str, UserInsight]:
        """Identify users with recurring issues."""

    def compare_locations(self, loc1: str, loc2: str) -> LocationComparison:
        """Generate comparative analysis between locations."""
```

#### Step 2.2: Database Schema Extensions

Add to `/src/device_anomaly/database/schema.py`:

```python
class LocationMetadata(Base):
    __tablename__ = "location_metadata"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String, nullable=False)
    location_id = Column(String, nullable=False)
    location_name = Column(String)
    parent_region = Column(String)
    timezone = Column(String)
    shift_schedule = Column(JSON)  # Define shift hours for this location

class AggregatedInsight(Base):
    __tablename__ = "aggregated_insights"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String, nullable=False)
    entity_type = Column(String)  # location, user, cohort
    entity_id = Column(String)
    insight_category = Column(String)
    insight_data = Column(JSON)  # Structured insight payload
    computed_at = Column(DateTime)
    valid_until = Column(DateTime)
```

#### Step 2.3: Location & User Attribution

Create `/src/device_anomaly/insights/attribution.py`:

```python
class AnomalyAttribution:
    """
    Determines which entity (location, user, device model, etc.)
    is the root cause of a pattern.
    """

    def attribute_to_location(self, anomalies: List) -> Optional[LocationAttribution]:
        """
        If anomalies cluster by location, return location attribution.
        Uses chi-square test to determine if location is significant factor.
        """

    def attribute_to_user(self, anomalies: List) -> Optional[UserAttribution]:
        """Identify if specific users are driving anomaly patterns."""

    def attribute_to_cohort(self, anomalies: List) -> Optional[CohortAttribution]:
        """Identify if device model/OS/firmware is the root cause."""
```

---

### Phase 3: Comparative Analysis Engine

**Goal**: Always show context - "85% is bad because peers are at 95%"

#### Step 3.1: Baseline Comparison Service

Create `/src/device_anomaly/insights/comparisons.py`:

```python
class ComparisonEngine:
    """
    Generates contextual comparisons for any metric.
    """

    def compare_to_fleet(self, device_id: str, metric: str) -> FleetComparison:
        """Compare device metric to fleet average and percentile."""

    def compare_to_cohort(self, device_id: str, metric: str) -> CohortComparison:
        """Compare to similar devices (same model/OS)."""

    def compare_to_historical(self, device_id: str, metric: str) -> HistoricalComparison:
        """Compare to device's own historical baseline."""

    def compare_locations(self, location1: str, location2: str) -> LocationComparison:
        """Side-by-side location comparison with key differentiators."""

    def rank_entities(self, entity_type: str, metric: str, order: str = "worst") -> List[EntityRanking]:
        """Rank locations/users/devices by a metric."""
```

#### Step 3.2: Comparison Data Models

```python
@dataclass
class FleetComparison:
    value: float
    fleet_average: float
    fleet_median: float
    percentile: int  # e.g., "10th percentile = bottom 10%"
    verdict: str  # "significantly below average", "typical", "top performer"
    delta_percent: float  # How much worse/better than average

@dataclass
class LocationComparison:
    location1: LocationSummary
    location2: LocationSummary
    key_differences: List[MetricDifference]
    winner: str  # Which location is performing better overall
    recommendations: List[str]
```

---

### Phase 4: Natural Language Generation

**Goal**: Every insight speaks in plain customer language

#### Step 4.1: Enhanced LLM Prompt Templates

Create `/src/device_anomaly/insights/prompts.py`:

```python
INSIGHT_GENERATION_PROMPT = """
You are an AI assistant helping warehouse managers understand device fleet issues.

CONTEXT:
- Target audience: Non-technical operations managers
- Goal: Explain what's wrong, why it matters, and what to do

ANOMALY DATA:
{anomaly_json}

COMPARATIVE CONTEXT:
{comparison_json}

Generate an insight with:
1. HEADLINE (max 15 words): Plain-language summary of the issue
2. IMPACT (1-2 sentences): Business consequences (downtime, costs, productivity)
3. CONTEXT (1-2 sentences): How this compares to normal/peers
4. ROOT CAUSE (1-2 sentences): Most likely explanation
5. ACTION (1 sentence): Specific next step

RULES:
- Never use technical terms like "anomaly score", "z-score", "percentile"
- Always quantify impact (hours, dollars, percentage)
- Always include comparison ("X worse than Y")
- Use active voice and simple sentences
- Be specific, not vague
"""
```

#### Step 4.2: Insight Generation Service

Create `/src/device_anomaly/insights/generator.py`:

```python
class InsightGenerator:
    """
    Generates customer-facing insights from raw anomalies.
    """

    def __init__(self, llm_client: LLMClient, comparison_engine: ComparisonEngine):
        self.llm = llm_client
        self.comparisons = comparison_engine

    async def generate_insight(self, anomaly: AnomalyResult) -> CustomerInsight:
        """
        Full pipeline: Classify → Aggregate → Compare → Generate
        """
        # 1. Classify into business category
        category = self.classifier.classify(anomaly)

        # 2. Get comparative context
        comparisons = self.comparisons.get_all_comparisons(anomaly)

        # 3. Generate natural language
        insight_text = await self.llm.generate(
            INSIGHT_GENERATION_PROMPT,
            anomaly_json=anomaly.to_dict(),
            comparison_json=comparisons.to_dict()
        )

        # 4. Parse and structure response
        return self._parse_insight(insight_text, category, comparisons)
```

---

### Phase 5: Proactive Insights Dashboard

**Goal**: "The product tells the story, not the AM/SE"

#### Step 5.1: Insights Dashboard API

Create `/src/device_anomaly/api/routes/insights.py`:

```python
@router.get("/insights/daily-digest")
async def get_daily_digest(tenant_id: str) -> DailyDigest:
    """
    Returns a prioritized list of today's most important insights.
    Designed to be the first thing a manager sees.
    """

@router.get("/insights/location/{location_id}")
async def get_location_insights(location_id: str) -> LocationInsightReport:
    """
    Comprehensive insight report for a specific location.
    Includes comparisons to other locations.
    """

@router.get("/insights/trending")
async def get_trending_issues() -> List[TrendingInsight]:
    """
    Issues that are getting worse over time.
    Early warning system for emerging problems.
    """

@router.get("/insights/wins")
async def get_wins() -> List[PositiveInsight]:
    """
    Things that are going well - improvements, resolved issues.
    Balances the narrative beyond just problems.
    """
```

#### Step 5.2: Insight Card Component

Frontend component for `/frontend/src/components/InsightCard.tsx`:

```typescript
interface InsightCard {
  category: InsightCategory;
  headline: string;
  impact: string;
  context: string;
  severity: 'critical' | 'warning' | 'info';
  affectedEntities: Entity[];
  trend: 'worsening' | 'stable' | 'improving';
  actions: Action[];
  relatedInsights: InsightCard[];
}
```

#### Step 5.3: Daily Digest Email/Notification

Create `/src/device_anomaly/notifications/digest.py`:

```python
class DailyDigestGenerator:
    """
    Generates a summary email/notification of the day's insights.
    Designed for managers who don't log in daily.
    """

    def generate_digest(self, tenant_id: str, date: date) -> DigestContent:
        """
        Structure:
        1. Executive Summary (3 sentences)
        2. Critical Issues Requiring Attention (max 3)
        3. Trending Concerns (max 3)
        4. Positive Developments (max 2)
        5. Recommended Actions
        """
```

---

### Phase 6: Close-the-Loop Automation

**Goal**: From insight to action with minimal friction

#### Step 6.1: ServiceNow Integration Enhancement

Extend existing automation in `/src/device_anomaly/api/routes/automation.py`:

```python
class AutomatedAction:
    """
    Actions that can be triggered automatically or with one click.
    """

    # Ticketing
    CREATE_SERVICENOW_TICKET = "create_servicenow_ticket"
    UPDATE_SERVICENOW_TICKET = "update_servicenow_ticket"

    # Device Actions (via MobiControl)
    SEND_MESSAGE_TO_DEVICE = "send_message"
    SCHEDULE_RESTART = "schedule_restart"
    TRIGGER_BATTERY_REPORT = "trigger_battery_report"

    # Notification
    NOTIFY_MANAGER = "notify_manager"
    NOTIFY_USER = "notify_user"

    # Documentation
    CREATE_INCIDENT_REPORT = "create_incident_report"
```

#### Step 6.2: One-Click Action Buttons

Create action templates that pre-populate ServiceNow tickets:

```python
class ActionTemplates:
    """
    Pre-configured actions for common insight types.
    """

    BATTERY_REPLACEMENT_REQUEST = {
        "ticket_type": "hardware_request",
        "category": "battery_replacement",
        "title_template": "Battery replacement needed for {device_count} devices in {location}",
        "description_template": """
        ## Summary
        {count} devices in {location} are experiencing battery issues that prevent
        completing a full shift.

        ## Affected Devices
        {device_list}

        ## Business Impact
        {impact_statement}

        ## Recommended Action
        Replace batteries for the listed devices.

        ---
        Generated by XSight Anomaly Detection
        """
    }
```

---

### Phase 7: Carl's Specific Use Cases

#### Step 7.1: Battery Shift Analysis

Create `/src/device_anomaly/insights/battery_shift.py`:

```python
class BatteryShiftAnalyzer:
    """
    Analyzes whether devices can complete a shift.
    Uses location shift schedules to determine expected battery needs.
    """

    def analyze_shift_readiness(self, location_id: str, shift_start: time) -> ShiftReadinessReport:
        """
        Returns:
        - Devices that will/won't make it through the shift
        - Estimated failure time for at-risk devices
        - Comparison to historical shift completion rates
        """

    def analyze_charging_patterns(self, location_id: str) -> ChargingPatternReport:
        """
        Identifies:
        - Devices not fully charged by shift start
        - Devices with short/interrupted charging
        - Correlation between charging patterns and shift failures
        """
```

#### Step 7.2: Excessive Drops/Reboots by Entity

Create `/src/device_anomaly/insights/device_abuse.py`:

```python
class DeviceAbuseAnalyzer:
    """
    Identifies excessive drops and reboots, attributed to:
    - Specific users
    - Specific locations/zones
    - Device models (manufacturing defect?)
    """

    def analyze_drops(self, tenant_id: str, period: DateRange) -> DropAnalysisReport:
        """
        Returns ranked list of:
        - Users with most drops
        - Locations with most drops
        - Device models with most drops
        With statistical significance testing to avoid false patterns.
        """
```

#### Step 7.3: App Power Drain Correlation

Create `/src/device_anomaly/insights/app_power.py`:

```python
class AppPowerAnalyzer:
    """
    Correlates app usage with battery drain.
    Identifies apps that consume disproportionate power.
    """

    def analyze_app_power_correlation(self, tenant_id: str) -> AppPowerReport:
        """
        Returns:
        - Apps with highest drain per hour of foreground time
        - Comparison to expected drain for app category
        - Devices where specific apps are causing problems
        """
```

#### Step 7.4: Network Pattern Analysis

Create `/src/device_anomaly/insights/network_patterns.py`:

```python
class NetworkPatternAnalyzer:
    """
    Analyzes network disconnect patterns by:
    - Location (warehouse zone)
    - Access point
    - Carrier/tower (for cellular)
    - Device model
    """

    def analyze_wifi_issues(self, location_id: str) -> WifiAnalysisReport:
        """
        Identifies:
        - AP hopping/stickiness issues
        - Dead zones by location
        - Device models with wifi problems
        """

    def analyze_cellular_issues(self, tenant_id: str) -> CellularAnalysisReport:
        """
        Identifies:
        - Carrier-specific issues
        - Tower hopping patterns
        - Network type performance (5G vs 4G vs LTE)
        """

    def detect_hidden_devices(self, tenant_id: str) -> HiddenDeviceReport:
        """
        Identifies devices with suspicious disconnect patterns
        (e.g., regular evening disconnects suggesting devices taken home).
        """
```

---

### Phase 8: Frontend Transformation

#### Step 8.1: New Insights Dashboard Page

Create `/frontend/src/pages/InsightsDashboard.tsx`:

```typescript
const InsightsDashboard: React.FC = () => {
  // Primary view is insight cards, not data tables
  return (
    <div>
      <DailyDigestSummary />

      <InsightSection title="Requires Immediate Attention">
        <InsightCardList category="critical" />
      </InsightSection>

      <InsightSection title="Trending Concerns">
        <InsightCardList category="trending" />
      </InsightSection>

      <InsightSection title="Location Comparison">
        <LocationComparisonWidget />
      </InsightSection>

      <InsightSection title="Recent Wins">
        <InsightCardList category="positive" />
      </InsightSection>
    </div>
  );
};
```

#### Step 8.2: Enhanced Location Comparison View

Create `/frontend/src/components/LocationComparison.tsx`:

```typescript
interface LocationComparisonProps {
  locations: string[];  // Compare these locations
}

const LocationComparison: React.FC<LocationComparisonProps> = ({ locations }) => {
  // Side-by-side comparison with:
  // - Key metrics comparison
  // - Trend lines for each
  // - Highlighting where one differs significantly
  // - Recommendations based on differences
};
```

#### Step 8.3: Action-Oriented Investigation Panel

Enhance `/frontend/src/pages/InvestigationDetail.tsx`:

```typescript
// Add prominent action buttons
<ActionPanel>
  <ActionButton
    label="Create ServiceNow Ticket"
    prefilled={insight.ticketTemplate}
  />
  <ActionButton
    label="Message Affected Users"
    recipients={insight.affectedUsers}
  />
  <ActionButton
    label="Schedule Device Check"
    devices={insight.affectedDevices}
  />
  <ActionButton
    label="Mark as False Positive"
    feedback={true}
  />
</ActionPanel>

// Add comparative context
<ComparisonPanel>
  <FleetComparison data={insight.fleetComparison} />
  <HistoricalTrend data={insight.historicalTrend} />
  <SimilarCases data={insight.similarCases} />
</ComparisonPanel>
```

---

### Phase 9: Learning & Prevention System

**Goal**: "Learn from what went wrong to prevent recurrence"

#### Step 9.1: Pattern Learning Service

Extend `/src/device_anomaly/models/patterns.py`:

```python
class PatternLearningService:
    """
    Learns from resolved anomalies to predict and prevent future issues.
    """

    def learn_from_resolution(self, anomaly_id: str, resolution: Resolution):
        """
        Records what fixed an issue and correlates with anomaly features
        to build predictive patterns.
        """

    def predict_issues(self, device_id: str) -> List[PredictedIssue]:
        """
        Based on current metrics and learned patterns, predict
        issues likely to occur in the next 24-48 hours.
        """

    def suggest_preventive_actions(self, tenant_id: str) -> List[PreventiveAction]:
        """
        Based on fleet patterns, suggest proactive maintenance:
        - "These 15 batteries should be replaced before they fail"
        - "This app update has caused crashes - consider rollback"
        """
```

#### Step 9.2: Predictive Alerts

Create `/src/device_anomaly/insights/predictions.py`:

```python
class PredictiveAlertService:
    """
    Generates alerts for predicted future issues, not just current anomalies.
    """

    def generate_shift_predictions(self, location_id: str, shift_date: date) -> ShiftPrediction:
        """
        Before a shift starts, predict:
        - Devices likely to fail during shift
        - Recommended pre-shift actions
        """
```

---

## Part 4: Implementation Priority

### Sprint 1: Foundation (Weeks 1-2)
1. Insight category taxonomy (`categories.py`)
2. Business language templates (`templates.py`)
3. Anomaly-to-insight classifier (`classifier.py`)
4. Basic comparison engine (`comparisons.py`)

### Sprint 2: Aggregation (Weeks 3-4)
1. Entity hierarchy model (`entities.py`)
2. Location/user attribution (`attribution.py`)
3. Database schema extensions
4. Cross-entity aggregation APIs

### Sprint 3: Language Generation (Weeks 5-6)
1. Enhanced LLM prompts (`prompts.py`)
2. Insight generation service (`generator.py`)
3. Caching for generated insights
4. Insight quality feedback loop

### Sprint 4: Dashboard (Weeks 7-8)
1. Insights dashboard API (`/insights/*`)
2. Insight card components
3. Location comparison view
4. Daily digest generation

### Sprint 5: Carl's Use Cases (Weeks 9-10)
1. Battery shift analyzer
2. Excessive drops analyzer
3. App power correlation
4. Network pattern analysis

### Sprint 6: Automation & Actions (Weeks 11-12)
1. ServiceNow template integration
2. One-click action buttons
3. Device action triggers
4. Notification system

### Sprint 7: Learning & Prevention (Weeks 13-14)
1. Pattern learning service
2. Predictive alerts
3. Preventive action suggestions
4. Feedback loop optimization

---

## Part 5: Success Metrics

### Customer Comprehension
- **Metric**: % of customers who can explain an insight without AM/SE help
- **Target**: 90%+ (from current ~30%)

### Time to Action
- **Metric**: Time from insight generation to action taken
- **Target**: <5 minutes (from current 2+ hours of investigation)

### Proactive Detection
- **Metric**: % of issues detected before customer complaint
- **Target**: 80%+ (from current ~40%)

### False Positive Rate
- **Metric**: % of insights marked as irrelevant/incorrect
- **Target**: <10%

### Business Value Attribution
- **Metric**: Every insight includes quantified impact
- **Target**: 100% of insights include dollar/time impact

---

## Part 6: Key Design Principles

### 1. Lead with Business Impact
- Bad: "Device X has anomaly score 0.92"
- Good: "Device X will die mid-shift, costing 2 hours of downtime ($150)"

### 2. Always Provide Comparison
- Bad: "Battery drain is 45%"
- Good: "Battery drain is 45%, which is 2x worse than similar devices"

### 3. Aggregate to Actionable Level
- Bad: "15 devices have network issues"
- Good: "Warehouse B has 3x more network issues than Warehouse A - check AP coverage"

### 4. One Click to Action
- Every insight should have a clear next step
- Pre-populate tickets, messages, and actions

### 5. Tell the Story
- Insights should be complete narratives, not data points
- Include the what, why, how bad, and what to do

---

## Files to Create/Modify

### New Files
| Path | Purpose |
|------|---------|
| `src/device_anomaly/insights/categories.py` | Insight category taxonomy |
| `src/device_anomaly/insights/templates.py` | Business language templates |
| `src/device_anomaly/insights/classifier.py` | Anomaly-to-insight classifier |
| `src/device_anomaly/insights/entities.py` | Entity hierarchy model |
| `src/device_anomaly/insights/attribution.py` | Root cause attribution |
| `src/device_anomaly/insights/comparisons.py` | Comparative analysis engine |
| `src/device_anomaly/insights/prompts.py` | LLM prompt templates |
| `src/device_anomaly/insights/generator.py` | Insight generation service |
| `src/device_anomaly/insights/battery_shift.py` | Battery shift analysis |
| `src/device_anomaly/insights/device_abuse.py` | Drop/reboot analysis |
| `src/device_anomaly/insights/app_power.py` | App power correlation |
| `src/device_anomaly/insights/network_patterns.py` | Network pattern analysis |
| `src/device_anomaly/insights/predictions.py` | Predictive alerts |
| `src/device_anomaly/api/routes/insights.py` | Insights API endpoints |
| `src/device_anomaly/notifications/digest.py` | Daily digest generator |
| `frontend/src/pages/InsightsDashboard.tsx` | New insights dashboard |
| `frontend/src/components/InsightCard.tsx` | Insight card component |
| `frontend/src/components/LocationComparison.tsx` | Location comparison |
| `frontend/src/components/ActionPanel.tsx` | Action buttons panel |

### Modified Files
| Path | Changes |
|------|---------|
| `src/device_anomaly/database/schema.py` | Add LocationMetadata, AggregatedInsight tables |
| `src/device_anomaly/api/routes/automation.py` | Add action templates, ServiceNow integration |
| `src/device_anomaly/models/patterns.py` | Add pattern learning service |
| `src/device_anomaly/llm/explainer.py` | Use new customer-facing prompts |
| `frontend/src/App.tsx` | Add route to insights dashboard |
| `frontend/src/pages/InvestigationDetail.tsx` | Add action panel, comparison panel |

---

## Summary

This plan transforms XSight from a technical anomaly detection tool into a business insights platform by:

1. **Speaking Customer Language**: Translating ML scores into business impact statements
2. **Providing Context**: Always showing comparisons (vs baseline, vs peers, vs history)
3. **Aggregating Intelligently**: Rolling up device issues to location/user/cohort level
4. **Enabling Action**: One-click to ServiceNow, device actions, notifications
5. **Preventing Recurrence**: Learning from resolutions to predict future issues

The result: "XSight has the data. XSight tells the story."
