# Product Blueprint: From Telemetry to Answers

## Executive Summary

We've unlocked 100M+ telemetry records across two enterprise databases. But users don't want more data—they want fewer decisions that matter more. This blueprint transforms raw signals into a product that feels like a trusted advisor: calm, clear, and always right.

---

# Part 1: Product Narrative

## The Problem We're Solving

Retail operators managing 50,000 devices wake up to 847 anomaly alerts. They resolve 12. The rest? Noise. False positives. Duplicate symptoms of the same root cause. They've learned to ignore the system.

Meanwhile, a firmware update silently drains batteries across 3,000 devices in the Southwest region. No single alert captures it. The pattern hides in the data.

## The Vision

**Login. Breathe. Act.**

The new experience:

1. **10 seconds**: Operator sees one number—"4 issues need your attention today"
2. **30 seconds**: Scans the 4 cards. One is critical: "Battery crisis affecting 847 devices at 23 stores—caused by v3.2.1 firmware update"
3. **60 seconds**: Clicks "View Details." Sees the evidence: battery curves, affected devices, rollback recommendation
4. **90 seconds**: Clicks "Roll back firmware on affected devices." Confirms. Done.
5. **2 minutes**: Moves to next issue. Coffee is still warm.

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Answers, not data** | Every screen shows conclusions first, evidence second |
| **Calm defaults** | Start with 4-6 cards. Never 400. |
| **Explainable AI** | Every insight has "Why this matters" and "What to do" |
| **Time-to-value < 2 min** | One home screen. No navigation required for 80% of work |
| **Business impact** | Dollar signs and device counts, not scores |

---

# Part 2: Information Architecture

## Page Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                         START SCREEN                                │
│  (Unified Command Center - 80% of daily work happens here)          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ FOCUS CARDS │  │ FLEET PULSE │  │ COST IMPACT │  │ TRENDS      │ │
│  │ 4-6 issues  │  │ Health ring │  │ $ at risk   │  │ Sparklines  │ │
│  └──────┬──────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    INSIGHT DETAIL PANEL                         ││
│  │  (Slides in from right - no page navigation)                    ││
│  │  Evidence timeline · Affected devices · Actions · Notes         ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │ FLEET   │          │ EXPLORE │          │ SETTINGS│
    │ BROWSER │          │ (Power) │          │ & ADMIN │
    └─────────┘          └─────────┘          └─────────┘
         │                    │                    │
    Store/Group          Correlation           Baselines
    Device List          Builder               Thresholds
    Device Detail        Custom Queries        Training
    Comparison           Saved Views           Integrations
```

## Start Screen: Unified Command Center

### Layout (1440px viewport)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ◉ Anomaly Detection                    Last 24h ▾   All Stores ▾   🔔 2 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────┐  ┌────────────────────────────┐  │
│  │        NEEDS ATTENTION            │  │      FLEET HEALTH          │  │
│  │                                   │  │                            │  │
│  │   ╭─────────────────────────────╮ │  │     ┌──────────────┐       │  │
│  │   │ 🔴 CRITICAL                 │ │  │     │   94.2%      │       │  │
│  │   │ Battery Crisis              │ │  │     │  ╭───────╮   │       │  │
│  │   │ 847 devices · 23 stores     │ │  │     │ ╱ healthy ╲  │       │  │
│  │   │ Est. impact: $47K/month     │ │  │     │╱           ╲ │       │  │
│  │   │ Firmware v3.2.1 rollout     │ │  │     └──────────────┘       │  │
│  │   │                    View →   │ │  │                            │  │
│  │   ╰─────────────────────────────╯ │  │  48,847 devices monitored  │  │
│  │                                   │  │  2,891 need attention      │  │
│  │   ╭─────────────────────────────╮ │  └────────────────────────────┘  │
│  │   │ 🟠 HIGH                     │ │                                  │
│  │   │ WiFi Dead Zones             │ │  ┌────────────────────────────┐  │
│  │   │ 234 devices · 5 stores      │ │  │      COST IMPACT           │  │
│  │   │ Network config mismatch     │ │  │                            │  │
│  │   │                    View →   │ │  │   $127K                    │  │
│  │   ╰─────────────────────────────╯ │  │   monthly avoidable cost   │  │
│  │                                   │  │                            │  │
│  │   ╭─────────────────────────────╮ │  │   ▪ Battery: $47K          │  │
│  │   │ 🟡 MEDIUM                   │ │  │   ▪ Data overage: $31K     │  │
│  │   │ Storage Pressure            │ │  │   ▪ Downtime: $49K         │  │
│  │   │ 156 devices approaching     │ │  │                            │  │
│  │   │ App cache bloat detected    │ │  └────────────────────────────┘  │
│  │   │                    View →   │ │                                  │
│  │   ╰─────────────────────────────╯ │  ┌────────────────────────────┐  │
│  │                                   │  │      7-DAY TREND           │  │
│  │   ╭─────────────────────────────╮ │  │  ▁▂▃▅▇▅▃                   │  │
│  │   │ ✅ RESOLVED TODAY           │ │  │  ↓ 23% vs last week        │  │
│  │   │ 12 issues · 3 by automation │ │  └────────────────────────────┘  │
│  │   ╰─────────────────────────────╯ │                                  │
│  │                                   │  ┌────────────────────────────┐  │
│  │   + 2 more low priority      ▾   │  │      QUICK WINS            │  │
│  └───────────────────────────────────┘  │  ▪ 34 stale apps to remove │  │
│                                         │  ▪ 12 devices need reboot  │  │
│                                         │  ▪ 5 profiles out of sync  │  │
│                                         └────────────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Recent Activity                                      View all →        │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐         │
│  │ 2:34pm  │ 1:15pm  │ 12:02pm │ 11:45am │ 10:30am │ 9:15am  │         │
│  │ ✓ Resv  │ ● New   │ → Escal │ ✓ Auto  │ ● New   │ ✓ Resv  │         │
│  │ Battery │ WiFi    │ Storage │ Reboot  │ App     │ Network │         │
│  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Focus Card Anatomy

```
╭──────────────────────────────────────────────────────────────╮
│ 🔴 CRITICAL                               BATTERY │ 2h ago  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Battery Crisis Affecting Southwest Region                   │
│                                                              │
│  847 devices · 23 stores · $47,200/mo impact                │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ WHY THIS IS HAPPENING                                  │  │
│  │ Firmware v3.2.1 (rolled out 3 days ago) has a          │  │
│  │ background service that prevents sleep mode.           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ WHAT TO DO                                             │  │
│  │ Roll back to v3.2.0 or wait for hotfix v3.2.2         │  │
│  │ (ETA: 2 days per vendor)                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ View Details │  │ Take Action  │  │ Dismiss (reason) │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
╰──────────────────────────────────────────────────────────────╯
```

## Drill-Down: Insight Detail Panel

When user clicks "View Details," a panel slides in from the right (60% width):

```
┌────────────────────────────────────────────────────────────────┐
│ ← Back                    Battery Crisis             ⋮ Actions │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  EVIDENCE TIMELINE                                    ▾ 7 days │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                │
│  Jan 6         Jan 7         Jan 8         Jan 9              │
│  ┊             ┊             ┊             ┊                   │
│  │             │  ▲ Firmware │  ▲▲▲▲▲▲▲▲▲  │                   │
│  │             │    v3.2.1   │  Battery    │                   │
│  │             │    rollout  │  alerts     │                   │
│  │             │             │  spike      │                   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐              │
│  │ Toggle: Battery     │  │ Toggle: Firmware    │              │
│  │ ████████████████   │  │ ████████████████   │              │
│  └─────────────────────┘  └─────────────────────┘              │
│                                                                │
│  CORRELATION CONFIDENCE: 94%                                   │
│  Battery drain events correlate with firmware v3.2.1 rollout   │
│  within a 24-hour lag window (p < 0.001)                       │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  AFFECTED DEVICES                                    View all → │
│                                                                │
│  Store            Devices    Avg Battery Drop    Status        │
│  ─────────────────────────────────────────────────────────     │
│  Phoenix #142     89         47% → 12%/day       Critical      │
│  Tucson #087      67         52% → 15%/day       Critical      │
│  Mesa #203        54         44% → 11%/day       High          │
│  ... 20 more stores                                            │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  RECOMMENDED ACTIONS                                           │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ ● Roll back firmware to v3.2.0                         │   │
│  │   Affects 847 devices · Estimated time: 4-6 hours      │   │
│  │   [Execute Rollback]                                   │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ ○ Wait for vendor hotfix v3.2.2                        │   │
│  │   ETA: 2 days · Monitor battery levels                 │   │
│  │   [Set Reminder]                                       │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  INVESTIGATION NOTES                                  + Add    │
│  ─────────────────────────────────────────────────────────     │
│  Jan 9, 2:15pm - @sarah: Confirmed with vendor, hotfix coming  │
│  Jan 9, 10:30am - System: Auto-correlated with firmware event  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Page Definitions

### 1. Start Screen (/)
**Responsibility**: Surface the 4-6 most important things. Enable resolution without navigation.
- Focus cards (grouped, prioritized issues)
- Fleet health ring
- Cost impact summary
- 7-day trend
- Quick wins (automatable actions)
- Recent activity timeline

### 2. Fleet Browser (/fleet)
**Responsibility**: Find and compare specific devices, stores, or groups.
- Hierarchical navigation: All → Region → Store → Device
- Comparison view (select 2-5 devices, overlay metrics)
- Device detail page with full telemetry history
- Bulk actions (reboot, sync, quarantine)

### 3. Explore (/explore)
**Responsibility**: Power user correlation builder. Answer custom questions.
- Metric selector (battery, data, WiFi, apps, location)
- Time range picker
- Dimension picker (group by store, model, OS, app)
- Correlation matrix visualization
- Save as custom insight

### 4. Settings (/settings)
**Responsibility**: Configure thresholds, integrations, and automation.
- Baseline thresholds
- Alert rules
- Automation workflows
- Training configuration
- API keys and integrations

---

# Part 3: Backend API Specification

## New Endpoints

### 3.1 Focus Cards (Primary Home Screen Data)

```
GET /api/v2/focus-cards
```

Returns the prioritized, grouped issues for the home screen.

**Query Parameters:**
- `time_range`: `24h` | `7d` | `30d` (default: `24h`)
- `store_ids`: comma-separated store IDs (optional)
- `severity_min`: `low` | `medium` | `high` | `critical` (default: `medium`)
- `limit`: max cards to return (default: 6, max: 20)

**Response:**
```json
{
  "focus_cards": [
    {
      "card_id": "fc_8a7b3c",
      "severity": "critical",
      "category": "battery",
      "title": "Battery Crisis Affecting Southwest Region",
      "subtitle": "847 devices · 23 stores",
      "impact": {
        "device_count": 847,
        "store_count": 23,
        "monthly_cost_usd": 47200,
        "impact_type": "operational"
      },
      "why": "Firmware v3.2.1 (rolled out 3 days ago) has a background service that prevents sleep mode.",
      "what_to_do": "Roll back to v3.2.0 or wait for hotfix v3.2.2 (ETA: 2 days per vendor)",
      "correlation": {
        "primary_factor": "firmware_version",
        "primary_value": "3.2.1",
        "confidence": 0.94,
        "lag_hours": 24
      },
      "actions": [
        {
          "action_id": "rollback_firmware",
          "label": "Roll back firmware",
          "type": "remediation",
          "automatable": true,
          "estimated_duration_minutes": 240
        },
        {
          "action_id": "wait_hotfix",
          "label": "Wait for hotfix",
          "type": "defer",
          "automatable": false
        }
      ],
      "created_at": "2026-01-09T10:30:00Z",
      "updated_at": "2026-01-09T14:15:00Z",
      "source_anomaly_ids": ["a1", "a2", "..."],
      "grouping_method": "correlation_cluster"
    }
  ],
  "summary": {
    "total_issues": 6,
    "critical": 1,
    "high": 2,
    "medium": 2,
    "low": 1,
    "resolved_today": 12,
    "auto_resolved": 3
  },
  "quick_wins": [
    {
      "title": "34 stale apps to remove",
      "action": "bulk_app_cleanup",
      "device_count": 34,
      "estimated_savings_usd": 120
    }
  ],
  "computed_at": "2026-01-09T14:30:00Z"
}
```

### 3.2 Fleet Health Summary

```
GET /api/v2/fleet/health
```

**Response:**
```json
{
  "health_score": 94.2,
  "total_devices": 48847,
  "healthy_devices": 45956,
  "needs_attention": 2891,
  "by_severity": {
    "critical": 847,
    "high": 512,
    "medium": 1032,
    "low": 500
  },
  "by_category": {
    "battery": 1247,
    "network": 892,
    "storage": 456,
    "apps": 296
  },
  "trend": {
    "direction": "improving",
    "change_percent": -23,
    "vs_period": "last_week"
  },
  "computed_at": "2026-01-09T14:30:00Z"
}
```

### 3.3 Cost Impact Summary

```
GET /api/v2/cost/impact
```

**Response:**
```json
{
  "total_monthly_impact_usd": 127400,
  "breakdown": [
    {
      "category": "battery_replacement",
      "monthly_usd": 47200,
      "device_count": 847,
      "trend": "increasing"
    },
    {
      "category": "data_overage",
      "monthly_usd": 31000,
      "device_count": 234,
      "trend": "stable"
    },
    {
      "category": "downtime_productivity",
      "monthly_usd": 49200,
      "device_count": 512,
      "trend": "decreasing"
    }
  ],
  "potential_savings_usd": 89000,
  "if_resolved": [
    "Roll back firmware: $47K",
    "Optimize data plans: $21K",
    "Fix WiFi configs: $21K"
  ],
  "computed_at": "2026-01-09T14:30:00Z"
}
```

### 3.4 Correlation Explorer

```
POST /api/v2/correlations/explore
```

**Request:**
```json
{
  "target_metric": "battery_level_drop",
  "candidate_factors": ["firmware_version", "app_version", "wifi_ssid", "store_id"],
  "time_range": {
    "start": "2026-01-02T00:00:00Z",
    "end": "2026-01-09T00:00:00Z"
  },
  "filters": {
    "store_ids": ["store_142", "store_087"],
    "device_models": ["TC52", "TC57"]
  },
  "min_confidence": 0.7,
  "max_results": 10
}
```

**Response:**
```json
{
  "correlations": [
    {
      "correlation_id": "corr_9x8y7z",
      "target_metric": "battery_level_drop",
      "factor": "firmware_version",
      "factor_value": "3.2.1",
      "correlation_coefficient": 0.89,
      "confidence": 0.94,
      "p_value": 0.0001,
      "lag_hours": 24,
      "device_count": 847,
      "explanation": "Devices with firmware v3.2.1 show 3.2x higher battery drain compared to v3.2.0",
      "causality_likelihood": "high",
      "causality_reasoning": "Temporal precedence established (firmware update preceded battery issues by 24h), dose-response relationship observed (longer time on v3.2.1 = more drain)",
      "confounders_checked": ["store_location", "device_age", "usage_pattern"],
      "visualization_data": {
        "type": "scatter_with_regression",
        "x_label": "Days since firmware update",
        "y_label": "Battery drain rate (%/day)",
        "points": [...]
      }
    }
  ],
  "no_correlation_found": ["wifi_ssid", "store_id"],
  "computed_at": "2026-01-09T14:32:00Z"
}
```

### 3.5 Root Cause Timeline

```
GET /api/v2/insights/{insight_id}/timeline
```

**Response:**
```json
{
  "insight_id": "fc_8a7b3c",
  "timeline": [
    {
      "timestamp": "2026-01-06T14:00:00Z",
      "event_type": "firmware_rollout",
      "title": "Firmware v3.2.1 rollout started",
      "source": "mobicontrol_events",
      "device_count": 3200,
      "is_root_cause_candidate": true
    },
    {
      "timestamp": "2026-01-07T02:00:00Z",
      "event_type": "anomaly_spike",
      "title": "Battery anomalies begin",
      "source": "anomaly_detection",
      "device_count": 47,
      "is_root_cause_candidate": false
    },
    {
      "timestamp": "2026-01-07T14:00:00Z",
      "event_type": "anomaly_spike",
      "title": "Battery anomalies accelerate",
      "source": "anomaly_detection",
      "device_count": 234,
      "is_root_cause_candidate": false
    },
    {
      "timestamp": "2026-01-08T08:00:00Z",
      "event_type": "threshold_breach",
      "title": "Critical threshold crossed",
      "source": "alerting",
      "device_count": 512,
      "is_root_cause_candidate": false
    }
  ],
  "correlation_markers": [
    {
      "from_event_index": 0,
      "to_event_index": 1,
      "label": "24h lag",
      "confidence": 0.94
    }
  ]
}
```

### 3.6 Execute Action

```
POST /api/v2/actions/execute
```

**Request:**
```json
{
  "action_id": "rollback_firmware",
  "insight_id": "fc_8a7b3c",
  "target_device_ids": ["d1", "d2", "..."],
  "parameters": {
    "target_version": "3.2.0",
    "batch_size": 100,
    "delay_between_batches_minutes": 5
  },
  "dry_run": false
}
```

**Response:**
```json
{
  "execution_id": "exec_abc123",
  "status": "queued",
  "target_device_count": 847,
  "estimated_completion": "2026-01-09T20:30:00Z",
  "tracking_url": "/api/v2/actions/executions/exec_abc123"
}
```

### 3.7 Insight Feedback

```
POST /api/v2/insights/{insight_id}/feedback
```

**Request:**
```json
{
  "feedback_type": "accurate" | "inaccurate" | "partially_accurate" | "not_actionable",
  "user_notes": "The correlation was correct but the recommended action didn't apply to our environment",
  "actual_root_cause": "It was actually an app update, not firmware",
  "resolution_taken": "manual_investigation"
}
```

---

# Part 4: Correlation Engine Design

## Philosophy

> "Correlation is not causation, but it's a damn good place to start looking."

Our correlation engine must:
1. Find real patterns (high signal)
2. Avoid false causality (low noise)
3. Explain findings in plain English (transparency)
4. Learn from feedback (continuous improvement)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CORRELATION ENGINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  CANONICAL  │───▶│  FEATURE    │───▶│ CORRELATION │         │
│  │  EVENTS     │    │  STORE      │    │  COMPUTER   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                 │                   │                 │
│         ▼                 ▼                   ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  TIME       │    │  COHORT     │    │  CAUSALITY  │         │
│  │  ALIGNER    │    │  AGGREGATOR │    │  SCORER     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                 │                   │                 │
│         └─────────────────┼───────────────────┘                 │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │  INSIGHT    │                              │
│                    │  GENERATOR  │                              │
│                    └─────────────┘                              │
│                           │                                     │
│                           ▼                                     │
│                    ┌─────────────┐                              │
│                    │  EXPLAINER  │                              │
│                    │  (LLM/Rules)│                              │
│                    └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Correlation Methods

### 4.1 Statistical Correlation (Primary)

```python
class StatisticalCorrelator:
    """
    Compute Pearson/Spearman correlation between metrics.
    """

    def correlate(
        self,
        target_metric: str,           # e.g., "battery_level_drop"
        candidate_factors: List[str], # e.g., ["firmware_version", "app_version"]
        time_range: TimeRange,
        cohort_filter: CohortFilter
    ) -> List[CorrelationResult]:

        # 1. Extract time series for target metric
        target_series = self.feature_store.get_metric_series(
            metric=target_metric,
            time_range=time_range,
            granularity="hour"
        )

        # 2. For each candidate factor, extract and align series
        results = []
        for factor in candidate_factors:
            factor_series = self.feature_store.get_factor_series(factor, time_range)

            # 3. Test multiple lag windows (0h, 6h, 12h, 24h, 48h, 72h)
            for lag_hours in [0, 6, 12, 24, 48, 72]:
                aligned = self._align_with_lag(target_series, factor_series, lag_hours)

                # 4. Compute correlation
                if self._is_categorical(factor):
                    # Use ANOVA F-statistic for categorical factors
                    corr, p_value = self._anova_correlation(aligned)
                else:
                    # Use Spearman for robustness to outliers
                    corr, p_value = spearmanr(aligned.target, aligned.factor)

                # 5. Check significance and effect size
                if p_value < 0.05 and abs(corr) > 0.3:
                    results.append(CorrelationResult(
                        factor=factor,
                        correlation=corr,
                        p_value=p_value,
                        lag_hours=lag_hours,
                        sample_size=len(aligned)
                    ))

        # 6. Return sorted by correlation strength
        return sorted(results, key=lambda r: abs(r.correlation), reverse=True)
```

### 4.2 Temporal Co-occurrence

```python
class TemporalCooccurrenceDetector:
    """
    Detect events that consistently precede anomalies.
    """

    def detect(
        self,
        anomaly_timestamps: List[datetime],
        event_stream: EventStream,
        window_hours: int = 48
    ) -> List[CooccurrenceResult]:

        # 1. For each anomaly, look back for preceding events
        preceding_events = defaultdict(list)

        for anomaly_ts in anomaly_timestamps:
            window_start = anomaly_ts - timedelta(hours=window_hours)
            events = event_stream.query(start=window_start, end=anomaly_ts)

            for event in events:
                key = (event.event_type, event.event_value)
                preceding_events[key].append({
                    'lag': (anomaly_ts - event.timestamp).total_seconds() / 3600,
                    'anomaly_ts': anomaly_ts
                })

        # 2. Filter to events that precede anomalies significantly more than baseline
        results = []
        baseline_rate = event_stream.get_baseline_rate()

        for (event_type, event_value), occurrences in preceding_events.items():
            # Calculate observed rate
            observed_rate = len(occurrences) / len(anomaly_timestamps)
            expected_rate = baseline_rate.get((event_type, event_value), 0.01)

            # Chi-square test
            if observed_rate > expected_rate * 2 and len(occurrences) >= 5:
                avg_lag = np.mean([o['lag'] for o in occurrences])
                results.append(CooccurrenceResult(
                    event_type=event_type,
                    event_value=event_value,
                    occurrence_rate=observed_rate,
                    lift_over_baseline=observed_rate / expected_rate,
                    avg_lag_hours=avg_lag,
                    sample_count=len(occurrences)
                ))

        return sorted(results, key=lambda r: r.lift_over_baseline, reverse=True)
```

### 4.3 Causality Scoring

```python
class CausalityScorer:
    """
    Score likelihood that correlation represents causation.
    Uses Bradford-Hill criteria adapted for device telemetry.
    """

    def score(self, correlation: CorrelationResult) -> CausalityScore:
        score = 0.0
        reasoning = []

        # 1. Temporal precedence (critical)
        if correlation.lag_hours > 0:
            score += 0.25
            reasoning.append(f"Factor precedes outcome by {correlation.lag_hours}h")

        # 2. Strength of association
        if abs(correlation.correlation) > 0.7:
            score += 0.20
            reasoning.append(f"Strong correlation ({correlation.correlation:.2f})")
        elif abs(correlation.correlation) > 0.5:
            score += 0.10
            reasoning.append(f"Moderate correlation ({correlation.correlation:.2f})")

        # 3. Dose-response relationship
        dose_response = self._check_dose_response(correlation)
        if dose_response.gradient_consistent:
            score += 0.15
            reasoning.append("Dose-response relationship observed")

        # 4. Consistency across cohorts
        cohort_consistency = self._check_cohort_consistency(correlation)
        if cohort_consistency.ratio > 0.8:
            score += 0.15
            reasoning.append(f"Consistent across {cohort_consistency.ratio*100:.0f}% of cohorts")

        # 5. Biological/mechanical plausibility
        plausibility = self._check_plausibility(correlation)
        if plausibility.is_plausible:
            score += 0.15
            reasoning.append(f"Mechanically plausible: {plausibility.explanation}")

        # 6. Absence of obvious confounders
        confounders = self._identify_confounders(correlation)
        if len(confounders) == 0:
            score += 0.10
            reasoning.append("No obvious confounders identified")
        else:
            reasoning.append(f"Possible confounders: {', '.join(confounders)}")

        return CausalityScore(
            score=min(score, 1.0),
            likelihood="high" if score > 0.7 else "medium" if score > 0.4 else "low",
            reasoning=reasoning,
            confounders=confounders
        )
```

## Avoiding False Causality

### Safeguards Implemented

1. **Multiple Testing Correction**: Apply Bonferroni correction when testing many hypotheses
2. **Minimum Sample Size**: Require n ≥ 30 devices for any correlation claim
3. **Confounder Checking**: Automatically check for common confounders (time of day, day of week, device age)
4. **Cohort Stratification**: Verify correlation holds across different device models/stores
5. **Confidence Intervals**: Always report uncertainty, not just point estimates
6. **Human Labels**: Track user feedback to detect false positive patterns

### Explainability Requirements

Every correlation must have:
```json
{
  "explanation_plain_english": "Devices with firmware v3.2.1 drain battery 3.2x faster",
  "confidence_statement": "94% confident this is a real pattern (not random chance)",
  "causality_statement": "Likely causal because firmware update preceded issues by 24h",
  "alternative_explanations": [
    "Could be confounded by device age (older devices got update first)"
  ],
  "what_would_disprove": "If battery drain doesn't improve after rollback"
}
```

## Feature Store Design

### Precomputed (Nightly)

| Table | Granularity | Retention | Use Case |
|-------|-------------|-----------|----------|
| `hourly_device_metrics` | device × hour | 90 days | Correlation time series |
| `daily_cohort_rollups` | cohort × day | 1 year | Cohort comparisons |
| `weekly_store_summary` | store × week | 2 years | Store benchmarking |
| `correlation_cache` | correlation × day | 30 days | Pre-computed correlations |

### On-Demand (Query Time)

| Computation | Trigger | Caching |
|-------------|---------|---------|
| Ad-hoc correlation | User exploration | 15 min TTL |
| Custom cohort filter | User filter change | Session-scoped |
| Real-time anomaly detail | Detail panel open | 5 min TTL |

---

# Part 5: Data Model Additions

## New Tables

### 5.1 Focus Cards (Materialized Insights)

```sql
CREATE TABLE focus_cards (
    card_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),

    -- Classification
    severity VARCHAR(20) NOT NULL,  -- critical, high, medium, low
    category VARCHAR(50) NOT NULL,  -- battery, network, storage, apps, security
    status VARCHAR(20) DEFAULT 'active',  -- active, acknowledged, resolved, dismissed

    -- Content
    title TEXT NOT NULL,
    subtitle TEXT,
    why_text TEXT,
    what_to_do_text TEXT,

    -- Impact
    device_count INTEGER NOT NULL,
    store_count INTEGER,
    monthly_cost_usd DECIMAL(10,2),
    impact_type VARCHAR(30),  -- operational, financial, security, compliance

    -- Correlation
    correlation_factor VARCHAR(100),
    correlation_value TEXT,
    correlation_confidence DECIMAL(3,2),
    correlation_lag_hours INTEGER,

    -- Grouping
    source_anomaly_ids UUID[] NOT NULL,
    grouping_method VARCHAR(50),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,

    -- Indexes
    INDEX idx_focus_cards_tenant_status (tenant_id, status),
    INDEX idx_focus_cards_severity (tenant_id, severity, created_at DESC),
    INDEX idx_focus_cards_category (tenant_id, category)
);
```

### 5.2 Correlation Cache

```sql
CREATE TABLE correlation_cache (
    correlation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Correlation definition
    target_metric VARCHAR(100) NOT NULL,
    factor_name VARCHAR(100) NOT NULL,
    factor_value TEXT,
    time_range_start TIMESTAMPTZ NOT NULL,
    time_range_end TIMESTAMPTZ NOT NULL,
    cohort_filter JSONB,

    -- Results
    correlation_coefficient DECIMAL(4,3),
    p_value DECIMAL(10,8),
    lag_hours INTEGER,
    sample_size INTEGER,

    -- Causality assessment
    causality_score DECIMAL(3,2),
    causality_reasoning TEXT[],
    confounders_checked TEXT[],

    -- Explanation
    explanation_text TEXT,
    visualization_config JSONB,

    -- Metadata
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    computation_time_ms INTEGER,

    INDEX idx_correlation_cache_lookup (tenant_id, target_metric, factor_name, time_range_end DESC)
);
```

### 5.3 User Feedback

```sql
CREATE TABLE insight_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    insight_id UUID NOT NULL,  -- References focus_cards or correlation_cache
    user_id UUID NOT NULL,

    -- Feedback
    feedback_type VARCHAR(30) NOT NULL,  -- accurate, inaccurate, partially_accurate, not_actionable
    user_notes TEXT,
    actual_root_cause TEXT,
    resolution_taken VARCHAR(50),

    -- For learning
    was_correlation_correct BOOLEAN,
    was_action_helpful BOOLEAN,
    time_to_resolution_hours INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_insight_feedback_learning (insight_id, feedback_type)
);
```

### 5.4 Action Executions

```sql
CREATE TABLE action_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    insight_id UUID,
    action_type VARCHAR(50) NOT NULL,

    -- Targeting
    target_device_ids UUID[] NOT NULL,
    parameters JSONB,

    -- Status
    status VARCHAR(30) DEFAULT 'queued',  -- queued, running, completed, failed, cancelled
    progress_percent INTEGER DEFAULT 0,
    devices_completed INTEGER DEFAULT 0,
    devices_failed INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Results
    result_summary JSONB,
    error_details TEXT,

    INDEX idx_action_executions_status (tenant_id, status, created_at DESC)
);
```

### 5.5 Feature Store Tables

```sql
-- Hourly device metrics for correlation analysis
CREATE TABLE hourly_device_metrics (
    tenant_id UUID NOT NULL,
    device_id UUID NOT NULL,
    hour_bucket TIMESTAMPTZ NOT NULL,

    -- Metrics
    battery_level_avg DECIMAL(5,2),
    battery_drain_rate DECIMAL(5,2),
    storage_free_mb INTEGER,
    data_upload_mb DECIMAL(10,2),
    data_download_mb DECIMAL(10,2),
    wifi_disconnect_count INTEGER,
    wifi_signal_avg INTEGER,
    app_crash_count INTEGER,
    reboot_count INTEGER,

    -- Context
    firmware_version VARCHAR(50),
    primary_app_version VARCHAR(50),
    store_id VARCHAR(50),

    PRIMARY KEY (tenant_id, device_id, hour_bucket),
    INDEX idx_hourly_metrics_time (tenant_id, hour_bucket DESC)
) PARTITION BY RANGE (hour_bucket);

-- Daily cohort rollups for benchmarking
CREATE TABLE daily_cohort_rollups (
    tenant_id UUID NOT NULL,
    cohort_key VARCHAR(200) NOT NULL,  -- e.g., "manufacturer:Zebra|model:TC52|os:Android11"
    day_bucket DATE NOT NULL,

    -- Aggregates
    device_count INTEGER,
    battery_drain_p50 DECIMAL(5,2),
    battery_drain_p90 DECIMAL(5,2),
    data_usage_p50_mb DECIMAL(10,2),
    wifi_issues_percent DECIMAL(5,2),
    anomaly_rate DECIMAL(5,4),

    PRIMARY KEY (tenant_id, cohort_key, day_bucket),
    INDEX idx_cohort_rollups_time (tenant_id, day_bucket DESC)
);
```

---

# Part 6: Frontend Components

## Component Hierarchy

```
src/components/
├── command-center/
│   ├── FocusCardList.tsx        # Main card container
│   ├── FocusCard.tsx            # Individual insight card
│   ├── FleetHealthRing.tsx      # Donut chart with health score
│   ├── CostImpactPanel.tsx      # Financial summary
│   ├── TrendSparkline.tsx       # 7-day mini chart
│   ├── QuickWinsPanel.tsx       # Automatable actions
│   └── RecentActivityTimeline.tsx
│
├── insight-detail/
│   ├── InsightDetailPanel.tsx   # Slide-in panel container
│   ├── EvidenceTimeline.tsx     # Multi-track timeline viz
│   ├── CorrelationBadge.tsx     # Confidence indicator
│   ├── AffectedDevicesTable.tsx # Paginated device list
│   ├── RecommendedActions.tsx   # Action buttons
│   └── InvestigationNotes.tsx   # Notes and history
│
├── correlation/
│   ├── CorrelationExplorer.tsx  # Power user tool
│   ├── MetricSelector.tsx       # Multi-select metrics
│   ├── CorrelationMatrix.tsx    # Heatmap visualization
│   └── ScatterWithRegression.tsx
│
├── shared/
│   ├── SeverityBadge.tsx        # Colored severity indicator
│   ├── ImpactBadge.tsx          # Dollar/device impact
│   ├── CategoryIcon.tsx         # Battery/Network/Storage icons
│   ├── TimeRangePicker.tsx      # 24h/7d/30d selector
│   └── StoreFilter.tsx          # Multi-select stores
│
└── visualizations/
    ├── TimelineChart.tsx        # Recharts timeline
    ├── HealthDonut.tsx          # Fleet health ring
    └── TrendSparkline.tsx       # Tiny line chart
```

## Key Component Designs

### 6.1 FocusCard.tsx

```tsx
interface FocusCardProps {
  card: FocusCard;
  onViewDetails: (cardId: string) => void;
  onTakeAction: (cardId: string, actionId: string) => void;
  onDismiss: (cardId: string, reason: string) => void;
}

export function FocusCard({ card, onViewDetails, onTakeAction, onDismiss }: FocusCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "rounded-lg border p-4 shadow-sm",
        severityStyles[card.severity]
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <SeverityBadge severity={card.severity} />
          <CategoryIcon category={card.category} />
          <span className="text-xs text-muted-foreground">
            {formatRelativeTime(card.updated_at)}
          </span>
        </div>
      </div>

      {/* Title & Subtitle */}
      <h3 className="mt-2 font-semibold">{card.title}</h3>
      <p className="text-sm text-muted-foreground">{card.subtitle}</p>

      {/* Impact */}
      <div className="mt-3 flex gap-4 text-sm">
        <span>{card.impact.device_count} devices</span>
        <span>{card.impact.store_count} stores</span>
        {card.impact.monthly_cost_usd && (
          <span className="font-medium text-red-600">
            ${formatNumber(card.impact.monthly_cost_usd)}/mo
          </span>
        )}
      </div>

      {/* Why & What To Do */}
      <div className="mt-3 space-y-2 text-sm">
        <div className="rounded bg-muted/50 p-2">
          <span className="font-medium">Why: </span>
          {card.why}
        </div>
        <div className="rounded bg-muted/50 p-2">
          <span className="font-medium">Action: </span>
          {card.what_to_do}
        </div>
      </div>

      {/* Actions */}
      <div className="mt-4 flex gap-2">
        <Button variant="default" size="sm" onClick={() => onViewDetails(card.card_id)}>
          View Details
        </Button>
        {card.actions[0]?.automatable && (
          <Button variant="outline" size="sm" onClick={() => onTakeAction(card.card_id, card.actions[0].action_id)}>
            {card.actions[0].label}
          </Button>
        )}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={() => onDismiss(card.card_id, 'not_actionable')}>
              Dismiss - Not actionable
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onDismiss(card.card_id, 'false_positive')}>
              Dismiss - False positive
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </motion.div>
  );
}
```

### 6.2 EvidenceTimeline.tsx

```tsx
interface EvidenceTimelineProps {
  events: TimelineEvent[];
  correlationMarkers: CorrelationMarker[];
  enabledTracks: string[];
  onToggleTrack: (track: string) => void;
}

export function EvidenceTimeline({
  events,
  correlationMarkers,
  enabledTracks,
  onToggleTrack
}: EvidenceTimelineProps) {
  // Group events by track (firmware, battery, network, etc.)
  const tracks = useMemo(() => groupEventsByTrack(events), [events]);

  return (
    <div className="space-y-4">
      {/* Track toggles */}
      <div className="flex flex-wrap gap-2">
        {Object.keys(tracks).map(track => (
          <Toggle
            key={track}
            pressed={enabledTracks.includes(track)}
            onPressedChange={() => onToggleTrack(track)}
            size="sm"
          >
            <TrackIcon track={track} className="mr-1 h-3 w-3" />
            {track}
          </Toggle>
        ))}
      </div>

      {/* Timeline visualization */}
      <div className="relative h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={prepareTimelineData(events, enabledTracks)}>
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatShortDate}
              tick={{ fontSize: 11 }}
            />
            <YAxis hide />
            <Tooltip content={<TimelineTooltip />} />

            {/* Event markers */}
            {enabledTracks.map((track, i) => (
              <Scatter
                key={track}
                dataKey={track}
                fill={trackColors[track]}
                shape={<EventMarker />}
              />
            ))}

            {/* Correlation arrows */}
            {correlationMarkers.map((marker, i) => (
              <ReferenceLine
                key={i}
                segment={[
                  { x: marker.from_timestamp, y: 0 },
                  { x: marker.to_timestamp, y: 0 }
                ]}
                stroke="#6366f1"
                strokeDasharray="5 5"
                label={marker.label}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Correlation confidence callout */}
      {correlationMarkers.length > 0 && (
        <CorrelationBadge
          confidence={correlationMarkers[0].confidence}
          label="Correlation confidence"
        />
      )}
    </div>
  );
}
```

### 6.3 Performance Strategy

```tsx
// hooks/useFocusCards.ts
export function useFocusCards(filters: FocusCardFilters) {
  return useQuery({
    queryKey: ['focus-cards', filters],
    queryFn: () => api.getFocusCards(filters),
    staleTime: 30_000,           // Fresh for 30 seconds
    cacheTime: 5 * 60_000,       // Keep in cache 5 minutes
    refetchInterval: 60_000,     // Auto-refresh every minute
    refetchOnWindowFocus: true,  // Refresh when user returns
    placeholderData: keepPreviousData,  // Show stale while fetching
  });
}

// hooks/useInsightDetail.ts
export function useInsightDetail(insightId: string | null) {
  return useQuery({
    queryKey: ['insight-detail', insightId],
    queryFn: () => api.getInsightDetail(insightId!),
    enabled: !!insightId,        // Only fetch when panel open
    staleTime: 60_000,           // Fresh for 1 minute
  });
}

// Prefetch on hover
export function FocusCardWithPrefetch({ card, ...props }: FocusCardProps) {
  const queryClient = useQueryClient();

  const handleMouseEnter = () => {
    queryClient.prefetchQuery({
      queryKey: ['insight-detail', card.card_id],
      queryFn: () => api.getInsightDetail(card.card_id),
      staleTime: 60_000,
    });
  };

  return (
    <div onMouseEnter={handleMouseEnter}>
      <FocusCard card={card} {...props} />
    </div>
  );
}
```

### 6.4 Grouping Strategy (Frontend)

```tsx
// utils/grouping.ts
export function groupFocusCards(cards: FocusCard[]): GroupedCards {
  // 1. Separate by severity for display order
  const bySeverity = {
    critical: cards.filter(c => c.severity === 'critical'),
    high: cards.filter(c => c.severity === 'high'),
    medium: cards.filter(c => c.severity === 'medium'),
    low: cards.filter(c => c.severity === 'low'),
  };

  // 2. Within severity, limit display
  const maxPerSeverity = { critical: 3, high: 2, medium: 2, low: 1 };

  const displayed: FocusCard[] = [];
  const collapsed: FocusCard[] = [];

  for (const severity of ['critical', 'high', 'medium', 'low'] as const) {
    const severityCards = bySeverity[severity];
    const max = maxPerSeverity[severity];

    displayed.push(...severityCards.slice(0, max));
    collapsed.push(...severityCards.slice(max));
  }

  return {
    displayed,
    collapsed,
    totalCount: cards.length,
    collapsedCount: collapsed.length,
  };
}

// Component usage
function FocusCardList() {
  const { data } = useFocusCards(filters);
  const [showAll, setShowAll] = useState(false);

  const grouped = useMemo(
    () => groupFocusCards(data?.focus_cards ?? []),
    [data]
  );

  const cardsToShow = showAll
    ? [...grouped.displayed, ...grouped.collapsed]
    : grouped.displayed;

  return (
    <div className="space-y-3">
      {cardsToShow.map(card => (
        <FocusCard key={card.card_id} card={card} {...handlers} />
      ))}

      {grouped.collapsedCount > 0 && !showAll && (
        <Button
          variant="ghost"
          onClick={() => setShowAll(true)}
          className="w-full"
        >
          + {grouped.collapsedCount} more low priority
        </Button>
      )}
    </div>
  );
}
```

---

# Part 7: Operational Rollout

## Feature Flags

```python
# config/feature_flags.py
FEATURE_FLAGS = {
    # Phase 1: Shadow mode (compute but don't display)
    "correlation_engine_shadow": {
        "default": False,
        "description": "Compute correlations in background, log results, don't surface to UI",
        "rollout": "internal_only"
    },

    # Phase 2: Beta display
    "focus_cards_beta": {
        "default": False,
        "description": "Show new Focus Cards UI to beta tenants",
        "rollout": "tenant_allowlist",
        "allowed_tenants": ["tenant_001", "tenant_002"]
    },

    # Phase 3: GA
    "focus_cards_ga": {
        "default": False,
        "description": "Show Focus Cards to all tenants",
        "rollout": "percentage",
        "percentage": 0  # Increment 10% at a time
    },

    # Correlation explorer (power feature)
    "correlation_explorer": {
        "default": False,
        "description": "Enable Explore page with correlation builder",
        "rollout": "tenant_allowlist"
    }
}
```

## Success Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Alert noise reduction | 847 alerts/day | < 50 focus cards/day | `COUNT(focus_cards) / COUNT(raw_anomalies)` |
| Time to root cause | 45 min avg | < 5 min avg | Time from alert to resolution action |
| False positive rate | 34% | < 10% | `COUNT(dismissed_false_positive) / COUNT(total)` |
| User engagement | 12% click-through | > 60% click-through | Focus card views / focus cards shown |
| Action completion | 8% actioned | > 40% actioned | Actions taken / focus cards shown |
| Battery replacement cost | $47K/mo | $20K/mo | Actual replacement invoices |
| Mean time to resolution | 4.2 hours | < 1 hour | Timestamp analysis |

## Monitoring & Alerts

```python
# Correlation engine health
- correlation_compute_latency_p99 < 5s
- correlation_cache_hit_rate > 80%
- focus_card_generation_latency_p99 < 2s

# Data quality
- canonical_event_lag_hours < 1
- feature_store_freshness_hours < 2
- correlation_sample_size_min > 30

# User experience
- focus_card_api_latency_p99 < 500ms
- insight_detail_api_latency_p99 < 1s
- ui_time_to_interactive < 2s
```

---

# Part 8: Prioritized Engineering Backlog

## Sprint 1: Foundation (Weeks 1-2)

### Task 1.1: Focus Cards API
**Scope**: Create `/api/v2/focus-cards` endpoint

**Files**:
- `src/device_anomaly/api/routes/focus_cards.py` (new)
- `src/device_anomaly/api/models.py` (extend)
- `src/device_anomaly/services/focus_card_generator.py` (new)

**Acceptance Criteria**:
- Returns top N grouped insights sorted by severity × impact
- Includes `why`, `what_to_do`, `correlation` fields
- Response time < 500ms for 10K anomalies
- Feature flag `focus_cards_beta` controls access

**Tests**:
- `tests/test_focus_cards_api.py`: Response shape validation
- `tests/test_focus_card_generator.py`: Grouping logic unit tests

---

### Task 1.2: Focus Cards Database Schema
**Scope**: Create tables for focus cards, correlation cache, feedback

**Files**:
- `migrations/versions/xxx_add_focus_cards.py` (new)
- `src/device_anomaly/data_access/focus_card_store.py` (new)

**Acceptance Criteria**:
- Tables created with proper indexes
- Tenant isolation enforced
- Idempotent card generation (no duplicates)

**Tests**:
- `tests/test_focus_card_store.py`: CRUD operations

---

### Task 1.3: Correlation Engine Core
**Scope**: Implement statistical correlation computation

**Files**:
- `src/device_anomaly/correlation/engine.py` (new)
- `src/device_anomaly/correlation/statistical.py` (new)
- `src/device_anomaly/correlation/temporal.py` (new)

**Acceptance Criteria**:
- Computes Spearman correlation for continuous metrics
- Computes ANOVA for categorical factors
- Tests multiple lag windows (0, 6, 12, 24, 48, 72h)
- Returns confidence intervals and p-values
- Handles missing data gracefully

**Tests**:
- `tests/test_correlation_engine.py`: Known correlation detection
- `tests/test_correlation_statistical.py`: Edge cases

---

### Task 1.4: Feature Store Hourly Rollups
**Scope**: Create hourly device metrics table and ETL job

**Files**:
- `migrations/versions/xxx_add_hourly_metrics.py`
- `src/device_anomaly/etl/hourly_rollup.py` (new)
- `src/device_anomaly/workers/rollup_scheduler.py` (new)

**Acceptance Criteria**:
- Rolls up canonical events to hourly granularity
- Includes battery, storage, network, app metrics
- Runs incrementally (watermark-based)
- Backfill capability for historical data

**Tests**:
- `tests/test_hourly_rollup.py`: Aggregation accuracy

---

## Sprint 2: UI Foundation (Weeks 3-4)

### Task 2.1: Start Screen Layout
**Scope**: Create unified command center page

**Files**:
- `frontend/src/pages/CommandCenter.tsx` (new)
- `frontend/src/components/command-center/FocusCardList.tsx`
- `frontend/src/components/command-center/FleetHealthRing.tsx`
- `frontend/src/components/command-center/CostImpactPanel.tsx`

**Acceptance Criteria**:
- Responsive layout (1440px, 1024px, mobile)
- Focus cards render with severity styling
- Fleet health ring shows accurate percentage
- Cost impact panel with breakdown

**Tests**:
- `frontend/tests/CommandCenter.test.tsx`: Render tests
- `frontend/tests/FocusCard.test.tsx`: Interaction tests

---

### Task 2.2: Focus Card Component
**Scope**: Create FocusCard with all states

**Files**:
- `frontend/src/components/command-center/FocusCard.tsx`
- `frontend/src/components/shared/SeverityBadge.tsx`
- `frontend/src/components/shared/ImpactBadge.tsx`

**Acceptance Criteria**:
- Displays title, subtitle, impact, why, what_to_do
- Severity-based border/background colors
- Hover prefetch for detail panel
- Action buttons functional

**Tests**:
- Component snapshot tests
- Accessibility audit (a11y)

---

### Task 2.3: Insight Detail Panel
**Scope**: Slide-in panel with evidence and actions

**Files**:
- `frontend/src/components/insight-detail/InsightDetailPanel.tsx`
- `frontend/src/components/insight-detail/EvidenceTimeline.tsx`
- `frontend/src/components/insight-detail/AffectedDevicesTable.tsx`

**Acceptance Criteria**:
- Slides in from right (60% width)
- Evidence timeline with toggleable tracks
- Correlation confidence badge
- Affected devices paginated
- Actions executable

**Tests**:
- Panel open/close animations
- Timeline rendering with mock data

---

### Task 2.4: API Client Extensions
**Scope**: Add TypeScript types and API methods

**Files**:
- `frontend/src/types/focus-cards.ts` (new)
- `frontend/src/api/client.ts` (extend)
- `frontend/src/hooks/useFocusCards.ts` (new)

**Acceptance Criteria**:
- Full type coverage for new API responses
- React Query hooks with proper caching
- Prefetch on hover
- Optimistic updates for actions

**Tests**:
- `frontend/tests/api/focus-cards.test.ts`

---

## Sprint 3: Intelligence Layer (Weeks 5-6)

### Task 3.1: Causality Scorer
**Scope**: Implement Bradford-Hill causality assessment

**Files**:
- `src/device_anomaly/correlation/causality.py` (new)

**Acceptance Criteria**:
- Scores 0-1 based on 6 criteria
- Returns reasoning array
- Identifies potential confounders
- Generates plain English explanation

**Tests**:
- `tests/test_causality_scorer.py`: Known causal vs spurious

---

### Task 3.2: Explainer Module (LLM + Fallback)
**Scope**: Generate natural language explanations

**Files**:
- `src/device_anomaly/correlation/explainer.py` (new)
- `src/device_anomaly/correlation/templates.py` (new)

**Acceptance Criteria**:
- LLM generates explanation when available
- Rule-based fallback when LLM unavailable
- Templates for common correlation types
- Includes "what would disprove" section

**Tests**:
- `tests/test_explainer.py`: Template coverage

---

### Task 3.3: Root Cause Timeline API
**Scope**: Join events across sources into narrative

**Files**:
- `src/device_anomaly/api/routes/timeline.py` (new)
- `src/device_anomaly/services/timeline_builder.py` (new)

**Acceptance Criteria**:
- Queries canonical events by insight ID
- Identifies root cause candidates
- Adds correlation markers
- Returns in temporal order

**Tests**:
- `tests/test_timeline_api.py`

---

### Task 3.4: Correlation Explorer API
**Scope**: Power user ad-hoc correlation queries

**Files**:
- `src/device_anomaly/api/routes/correlation_explorer.py` (new)

**Acceptance Criteria**:
- Accepts target metric, candidates, filters
- Returns ranked correlations with explanations
- Caches results for 15 minutes
- Respects min_confidence threshold

**Tests**:
- `tests/test_correlation_explorer.py`

---

## Sprint 4: Integration & Polish (Weeks 7-8)

### Task 4.1: Action Execution System
**Scope**: Execute remediation actions from UI

**Files**:
- `src/device_anomaly/api/routes/actions.py` (new)
- `src/device_anomaly/services/action_executor.py` (new)
- `src/device_anomaly/workers/action_worker.py` (new)

**Acceptance Criteria**:
- Queue actions for async execution
- Track progress and completion
- Support dry-run mode
- Integrate with MobiControl API

**Tests**:
- `tests/test_action_executor.py`

---

### Task 4.2: Feedback Loop
**Scope**: Collect and learn from user feedback

**Files**:
- `src/device_anomaly/api/routes/feedback.py` (new)
- `src/device_anomaly/services/feedback_analyzer.py` (new)

**Acceptance Criteria**:
- Store feedback with insight reference
- Track accuracy metrics by correlation type
- Generate weekly feedback report

**Tests**:
- `tests/test_feedback_api.py`

---

### Task 4.3: Feature Flags & Rollout
**Scope**: Implement safe rollout infrastructure

**Files**:
- `src/device_anomaly/config/feature_flags.py` (new)
- `src/device_anomaly/middleware/feature_flags.py` (new)

**Acceptance Criteria**:
- Tenant-level flag overrides
- Percentage-based rollout
- Shadow mode logging
- Admin UI for flag management

**Tests**:
- `tests/test_feature_flags.py`

---

### Task 4.4: Observability & Dashboards
**Scope**: Metrics, logging, alerting

**Files**:
- `src/device_anomaly/observability/metrics.py` (extend)
- `grafana/dashboards/correlation_engine.json` (new)

**Acceptance Criteria**:
- Prometheus metrics for correlation engine
- Grafana dashboard for health monitoring
- Alerts for latency and error rate
- Structured logging for debugging

**Tests**:
- Integration tests for metrics emission

---

## Sprint 5: Advanced Features (Weeks 9-10)

### Task 5.1: Correlation Explorer UI
**Scope**: Power user correlation builder

**Files**:
- `frontend/src/pages/Explore.tsx` (new)
- `frontend/src/components/correlation/CorrelationExplorer.tsx`
- `frontend/src/components/correlation/CorrelationMatrix.tsx`

**Acceptance Criteria**:
- Metric multi-select
- Time range picker
- Dimension filters
- Heatmap visualization
- Save as custom insight

**Tests**:
- `frontend/tests/Explore.test.tsx`

---

### Task 5.2: Automated Insight Refresh
**Scope**: Background job to regenerate focus cards

**Files**:
- `src/device_anomaly/workers/insight_refresh.py` (new)

**Acceptance Criteria**:
- Runs every 15 minutes
- Updates correlation confidence with new data
- Promotes/demotes severity based on trajectory
- Archives resolved insights

**Tests**:
- `tests/test_insight_refresh.py`

---

### Task 5.3: Cost Impact Calculator
**Scope**: Estimate financial impact of issues

**Files**:
- `src/device_anomaly/services/cost_calculator.py` (new)
- `src/device_anomaly/config/cost_models.py` (new)

**Acceptance Criteria**:
- Configurable cost models (battery replacement, downtime, data overage)
- Tenant-specific cost parameters
- Monthly projection based on current trajectory

**Tests**:
- `tests/test_cost_calculator.py`

---

### Task 5.4: Mobile-Responsive Polish
**Scope**: Ensure excellent mobile experience

**Files**:
- `frontend/src/components/command-center/*.tsx` (updates)
- `frontend/src/styles/responsive.css`

**Acceptance Criteria**:
- Touch-friendly tap targets
- Swipe gestures for card actions
- Collapsed panels on mobile
- Performance audit (Lighthouse > 90)

**Tests**:
- Visual regression tests at breakpoints

---

# Summary

This blueprint transforms telemetry chaos into operational clarity:

| Before | After |
|--------|-------|
| 847 alerts | 4 focus cards |
| 45 min to root cause | < 2 min |
| "What does this mean?" | "Here's why and what to do" |
| Individual symptoms | Correlated patterns |
| Reactive firefighting | Proactive prevention |

The architecture is:
- **Simple**: Statistical correlation + heuristics, not complex ML
- **Explainable**: Every insight has why + what to do
- **Shippable**: 10-week incremental delivery
- **Measured**: Clear success metrics

The user feels: *Finally, a system that thinks like I do—but faster.*
