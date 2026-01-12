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

// ============================================================================
// System Health Types
// ============================================================================

export interface SystemHealthMetrics {
  avg_cpu_usage: number;
  avg_memory_usage: number;
  avg_storage_available_pct: number;
  avg_device_temp: number;
  avg_battery_temp: number;
  devices_high_cpu: number;
  devices_high_memory: number;
  devices_low_storage: number;
  devices_high_temp: number;
  total_devices: number;
}

export interface CohortHealth {
  cohort_id: string;
  cohort_name: string;
  device_count: number;
  health_score: number;
  avg_cpu: number;
  avg_memory: number;
  avg_storage_pct: number;
  devices_at_risk: number;
}

export interface SystemHealthSummary {
  tenant_id: string;
  fleet_health_score: number;
  health_trend: 'improving' | 'stable' | 'degrading' | 'unknown';
  total_devices: number;
  healthy_count: number;
  warning_count: number;
  critical_count: number;
  metrics: SystemHealthMetrics;
  cohort_breakdown: CohortHealth[];
  recommendations: string[];
  generated_at: string;
}

export interface HealthTrendPoint {
  timestamp: string;
  value: number;
  device_count: number;
}

export interface StorageForecastDevice {
  device_id: number;
  device_name: string;
  current_storage_pct: number;
  storage_trend_gb_per_day: number;
  projected_full_date: string | null;
  days_until_full: number | null;
  confidence: number;
}

export interface StorageForecast {
  tenant_id: string;
  devices_at_risk: StorageForecastDevice[];
  total_at_risk_count: number;
  avg_days_until_full: number | null;
  recommendations: string[];
  generated_at: string;
}

// ============================================================================
// Location Intelligence Types
// ============================================================================

export interface GeoBounds {
  min_lat: number;
  max_lat: number;
  min_long: number;
  max_long: number;
}

export interface HeatmapCell {
  lat: number;
  long: number;
  signal_strength: number;
  reading_count: number;
  is_dead_zone: boolean;
  access_point_id: string | null;
}

export interface WiFiHeatmap {
  tenant_id: string;
  grid_cells: HeatmapCell[];
  bounds: GeoBounds | null;
  total_readings: number;
  avg_signal_strength: number;
  dead_zone_count: number;
  generated_at: string;
}

export interface DeadZone {
  zone_id: string;
  lat: number;
  long: number;
  avg_signal: number;
  affected_devices: number;
  total_readings: number;
  first_detected: string | null;
  last_detected: string | null;
}

export interface DeadZonesResponse {
  tenant_id: string;
  dead_zones: DeadZone[];
  total_count: number;
  recommendations: string[];
  generated_at: string;
}

export interface MovementPoint {
  timestamp: string;
  lat: number;
  long: number;
  speed: number;
  heading: number;
}

export interface DeviceMovement {
  device_id: number;
  movements: MovementPoint[];
  total_distance_km: number;
  avg_speed_kmh: number;
  stationary_time_pct: number;
  active_hours: number[];
}

export interface DwellZone {
  zone_id: string;
  lat: number;
  long: number;
  avg_dwell_minutes: number;
  device_count: number;
  visit_count: number;
  peak_hours: number[];
}

export interface DwellTimeResponse {
  tenant_id: string;
  dwell_zones: DwellZone[];
  total_zones: number;
  recommendations: string[];
  generated_at: string;
}

export interface CoverageSummary {
  tenant_id: string;
  total_readings: number;
  avg_signal: number;
  coverage_distribution: Record<string, number>;
  coverage_percentage: number;
  recommendations: string[];
  generated_at: string;
}

// ============================================================================
// Events & Alerts Types
// ============================================================================

export interface EventEntry {
  log_id: number;
  timestamp: string;
  event_id: number;
  severity: string;
  event_class: string;
  message: string;
  device_id: number | null;
  login_id: string | null;
}

export interface EventTimeline {
  tenant_id: string;
  events: EventEntry[];
  total: number;
  page: number;
  page_size: number;
  severity_distribution: Record<string, number>;
  event_class_distribution: Record<string, number>;
  generated_at: string;
}

export interface AlertEntry {
  alert_id: number;
  alert_key: string;
  alert_name: string;
  severity: string;
  device_id: string | null;
  status: string;
  set_datetime: string | null;
  ack_datetime: string | null;
}

export interface AlertNameCount {
  name: string;
  count: number;
}

export interface AlertSummary {
  tenant_id: string;
  total_active: number;
  total_acknowledged: number;
  total_resolved: number;
  by_severity: Record<string, number>;
  by_alert_name: AlertNameCount[];
  recent_alerts: AlertEntry[];
  avg_acknowledge_time_minutes: number;
  avg_resolution_time_minutes: number;
  generated_at: string;
}

export interface AlertTrendPoint {
  timestamp: string;
  count: number;
  severity: string;
}

export interface AlertTrends {
  tenant_id: string;
  trends: AlertTrendPoint[];
  generated_at: string;
}

export interface CorrelatedEvent {
  event: EventEntry;
  time_before_minutes: number;
  frequency_score: number;
}

export interface EventCorrelation {
  tenant_id: string;
  anomaly_timestamp: string;
  device_id: number;
  correlated_events: CorrelatedEvent[];
  total_events_found: number;
  generated_at: string;
}

export interface EventStatistics {
  tenant_id: string;
  total_events: number;
  events_per_day: number;
  unique_devices: number;
  top_event_classes: Array<{ class: string; count: number }>;
  generated_at: string;
}

// ============================================================================
// Temporal Analysis Types
// ============================================================================

export interface HourlyDataPoint {
  hour: number;
  avg_value: number;
  min_value: number;
  max_value: number;
  std_value: number;
  sample_count: number;
}

export interface HourlyBreakdown {
  tenant_id: string;
  metric: string;
  hourly_data: HourlyDataPoint[];
  peak_hours: number[];
  low_hours: number[];
  day_night_ratio: number;
  generated_at: string;
}

export interface PeakDetection {
  timestamp: string;
  value: number;
  z_score: number;
  is_significant: boolean;
}

export interface PeakDetectionList {
  tenant_id: string;
  metric: string;
  peaks: PeakDetection[];
  total_peaks: number;
  generated_at: string;
}

export interface PeriodStats {
  start: string;
  end: string;
  avg: number;
  median: number;
  std: number;
  sample_count: number;
}

export interface TemporalComparison {
  tenant_id: string;
  metric: string;
  period_a: PeriodStats;
  period_b: PeriodStats;
  change_percent: number;
  is_significant: boolean;
  p_value: number;
  generated_at: string;
}

export interface DailyComparisonPoint {
  date: string;
  value: number;
  sample_count: number;
  change_percent: number;
}

export interface DayOverDay {
  tenant_id: string;
  metric: string;
  comparisons: DailyComparisonPoint[];
  generated_at: string;
}

export interface WeeklyComparisonPoint {
  year: number;
  week: number;
  value: number;
  sample_count: number;
  change_percent: number;
}

export interface WeekOverWeek {
  tenant_id: string;
  metric: string;
  comparisons: WeeklyComparisonPoint[];
  generated_at: string;
}
