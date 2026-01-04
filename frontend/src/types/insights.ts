/**
 * Types for the Customer Insights API
 *
 * Aligned with Carl's vision: "XSight has the data. XSight needs the story."
 */

export interface CustomerInsight {
  insight_id: string;
  category: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  headline: string;
  impact_statement: string;
  comparison_context: string;
  recommended_actions: string[];
  entity_type: string;
  entity_id: string;
  entity_name: string;
  affected_device_count: number;
  primary_metric: string;
  primary_value: number;
  trend_direction: 'improving' | 'stable' | 'degrading';
  trend_change_percent: number | null;
  detected_at: string;
  confidence_score: number;
}

export interface DailyDigest {
  tenant_id: string;
  digest_date: string;
  generated_at: string;
  total_insights: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  top_insights: CustomerInsight[];
  executive_summary: string;
  trending_issues: CustomerInsight[];
  new_issues: CustomerInsight[];
}

export interface LocationInsightReport {
  location_id: string;
  location_name: string;
  report_date: string;
  total_devices: number;
  devices_with_issues: number;
  issue_rate: number;
  shift_readiness: ShiftReadinessData | null;
  insights: CustomerInsight[];
  top_issues: Array<{ category: string; count: number }>;
  rank_among_locations: number;
  better_than_percent: number;
  recommendations: string[];
}

export interface ShiftReadinessData {
  location_id: string;
  location_name: string;
  shift_name: string;
  shift_date: string;
  readiness_percentage: number;
  total_devices: number;
  devices_ready: number;
  devices_at_risk: number;
  devices_critical: number;
  avg_battery_at_start: number;
  avg_drain_rate: number;
  devices_not_fully_charged: number;
  vs_last_week_readiness: number | null;
  device_details: DeviceReadiness[];
  recommendations: string[];
}

export interface DeviceReadiness {
  device_id: number;
  device_name: string;
  current_battery: number;
  drain_rate_per_hour: number;
  projected_end_battery: number;
  will_complete_shift: boolean;
  estimated_dead_time: string | null;
  was_fully_charged: boolean;
  readiness_score: number;
  recommendations: string[];
}

export interface NetworkAnalysis {
  tenant_id: string;
  analysis_period_days: number;
  wifi_summary: {
    total_devices: number;
    devices_with_roaming_issues: number;
    devices_with_stickiness: number;
    avg_aps_per_device: number;
    potential_dead_zones: number;
  };
  cellular_summary: {
    total_devices: number;
    devices_with_tower_hopping: number;
    devices_with_tech_fallback: number;
    best_carrier: string;
    worst_carrier: string;
    network_type_distribution: Record<string, number>;
  } | null;
  disconnect_summary: {
    total_disconnects: number;
    avg_disconnects_per_device: number;
    total_offline_hours: number;
    has_predictable_pattern: boolean;
    pattern_description: string;
  };
  hidden_devices_count: number;
  recommendations: string[];
}

export interface DeviceAbuseAnalysis {
  tenant_id: string;
  analysis_period_days: number;
  total_devices: number;
  total_drops: number;
  total_reboots: number;
  devices_with_excessive_drops: number;
  devices_with_excessive_reboots: number;
  worst_locations: Array<{
    location_id: string;
    drops: number;
    rate_per_device: number;
  }>;
  worst_cohorts: Array<{
    cohort_id: string;
    reboots: number;
    rate_per_device: number;
  }>;
  problem_combinations: Array<{
    cohort_id: string;
    manufacturer: string;
    model: string;
    os_version: string;
    device_count: number;
    vs_fleet_multiplier: number;
    primary_issue: string;
    severity: string;
  }>;
  recommendations: string[];
}

export interface AppAnalysis {
  tenant_id: string;
  analysis_period_days: number;
  total_apps_analyzed: number;
  apps_with_issues: number;
  total_crashes: number;
  total_anrs: number;
  top_power_consumers: Array<{
    package_name: string;
    app_name: string;
    battery_drain_percent: number;
    drain_per_hour: number;
    foreground_hours: number;
    efficiency_score: number;
  }>;
  top_crashers: Array<{
    package_name: string;
    app_name: string;
    crash_count: number;
    anr_count: number;
    devices_affected: number;
    severity: string;
  }>;
  recommendations: string[];
}

export interface LocationComparison {
  location_a_id: string;
  location_a_name: string;
  location_b_id: string;
  location_b_name: string;
  device_count_a: number;
  device_count_b: number;
  metric_comparisons: Record<string, {
    location_a_value: number;
    location_b_value: number;
    difference_percent: number;
  }>;
  overall_winner: string | null;
  key_differences: string[];
}

// Insight categories aligned with backend
export type InsightCategory =
  | 'battery_shift_failure'
  | 'battery_rapid_drain'
  | 'battery_charge_incomplete'
  | 'battery_charge_pattern'
  | 'battery_health_degraded'
  | 'excessive_drops'
  | 'excessive_reboots'
  | 'device_abuse_pattern'
  | 'wifi_ap_hopping'
  | 'wifi_dead_zone'
  | 'cellular_tower_hopping'
  | 'cellular_carrier_issue'
  | 'network_disconnect_pattern'
  | 'device_hidden_pattern'
  | 'app_crash_pattern'
  | 'app_power_drain'
  | 'app_anr_pattern'
  | 'cohort_performance_issue'
  | 'problem_combination'
  | 'location_anomaly_cluster'
  | 'location_performance_gap'
  | 'location_baseline_deviation';
