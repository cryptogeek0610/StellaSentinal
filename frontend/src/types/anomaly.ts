export interface Anomaly {
  id: number
  device_id: number
  timestamp: string
  anomaly_score: number
  anomaly_label: number
  status: string
  assigned_to: string | null
  total_battery_level_drop: number | null
  total_free_storage_kb: number | null
  download: number | null
  upload: number | null
  offline_time: number | null
  disconnect_count: number | null
  wifi_signal_strength: number | null
  connection_time: number | null
  feature_values_json: string | null
  created_at: string
  updated_at: string
}

export interface AnomalyDetail extends Anomaly {
  notes: string | null
  investigation_notes: InvestigationNote[]
}

export interface InvestigationNote {
  id: number
  user: string
  note: string
  action_type: string | null
  created_at: string
}

export interface AnomalyListResponse {
  anomalies: Anomaly[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface Device {
  device_id: number
  device_model: string | null
  device_name: string | null
  location: string | null
  store_id?: string // Stable identifier for grouping
  status: string
  last_seen: string | null
  os_version?: string | null
  agent_version?: string | null
  custom_attributes?: Record<string, string>
}

export interface DeviceDetail extends Device {
  anomaly_count: number
  recent_anomalies: Anomaly[]
  // Extended Telemetry
  battery_level?: number
  is_charging?: boolean
  wifi_signal?: number // dBm
  storage_used?: number // percentage
  memory_usage?: number // percentage
  cpu_load?: number // percentage
  os_version?: string
  agent_version?: string
}

export interface DashboardStats {
  anomalies_today: number
  devices_monitored: number
  critical_issues: number
  resolved_today: number
  open_cases?: number
  total_anomalies?: number  // Total count of all anomalies in database
}

export interface DashboardTrend {
  date: string
  anomaly_count: number
}

export interface ConnectionStatus {
  connected: boolean
  server: string
  error: string | null
  status: 'connected' | 'disconnected' | 'error' | 'not_configured' | 'disabled' | 'unknown'
}

export interface AllConnectionsStatus {
  backend_db: ConnectionStatus  // PostgreSQL
  dw_sql: ConnectionStatus  // XSight SQL Server
  mc_sql: ConnectionStatus  // MobiControl SQL Server
  mobicontrol_api: ConnectionStatus  // MobiControl REST API
  llm: ConnectionStatus  // LLM service
  redis: ConnectionStatus  // Redis for streaming
  qdrant: ConnectionStatus  // Qdrant vector database
  last_checked: string
}

export interface TroubleshootingAdvice {
  advice: string
  summary?: string
}

export interface IsolationForestConfig {
  n_estimators: number
  contamination: number
  random_state: number
  scale_features: boolean
  min_variance: number
  feature_count?: number
  model_type: string
}

export interface ScoreDistributionBin {
  bin_start: number
  bin_end: number
  count: number
  is_anomaly: boolean
}

export interface ScoreDistribution {
  bins: ScoreDistributionBin[]
  total_normal: number
  total_anomalies: number
  mean_score: number
  median_score: number
  min_score: number
  max_score: number
  std_score: number
}

export interface IsolationForestStats {
  config: IsolationForestConfig
  score_distribution: ScoreDistribution
  total_predictions: number
  anomaly_rate: number
  feedback_stats?: FeedbackStats
}

export interface FeedbackStats {
  total_feedback: number
  false_positives: number
  confirmed_anomalies: number
  projected_accuracy_gain: number
  last_retrain: string
}

export interface BaselineSuggestion {
  level: string
  group_key: string
  feature: string
  baseline_median: number
  observed_median: number
  proposed_new_median: number
  rationale: string
}

export interface BaselineAdjustmentRequest {
  level: string
  group_key: Record<string, any>
  feature: string
  adjustment: number
  reason?: string
  auto_retrain?: boolean
}

export interface BaselineAdjustmentResponse {
  success: boolean
  message: string
  baseline_updated: boolean
  model_retrained: boolean
}

// ==========================================
// AI Insights Types (Stellar Operations)
// ==========================================

export type AIInsightType =
  | 'workload'
  | 'efficiency'
  | 'optimization'
  | 'pattern'
  | 'anomaly'
  | 'warning'
  | 'opportunity'
  | 'trend'
  | 'explanation';

export type AIInsightSeverity =
  | 'critical'
  | 'high'
  | 'medium'
  | 'low'
  | 'info'
  | 'warning';

export interface AIInsightImpact {
  metric?: string;
  value?: string | number;
  change?: string;
  confidence?: number;
  utilizationChange?: number;
  deviceCount?: number;
}

// Financial impact data from Cost Intelligence
export interface FinancialImpact {
  total_impact_usd: number;
  monthly_recurring_usd?: number;
  potential_savings_usd?: number;
  impact_level: 'high' | 'medium' | 'low';
  breakdown?: FinancialBreakdownItem[];
  recommendations?: string[];
  investment_required_usd?: number;
  payback_months?: number;
  confidence_score?: number;
  calculated_at?: string;
}

export interface FinancialBreakdownItem {
  category: string;
  description: string;
  amount: number;
  is_recurring: boolean;
  period?: string;
  confidence?: number;
}

export interface AIInsight {
  id: string;
  type: AIInsightType;
  severity: AIInsightSeverity;
  title: string;
  description: string;
  why?: string;
  how?: string;
  whatToDo?: string;
  recommendation?: string;
  impact?: AIInsightImpact;
  financialImpact?: FinancialImpact;
  affectedStores?: string[];
  affectedDevices?: string[];
  affectedItems?: string[];
  affectedCount?: number;
  createdAt: string;
  status: 'pending' | 'applied' | 'dismissed' | 'scheduled';
  scheduledFor?: string;
  actionable?: boolean;
  actionLabel?: string;
}

// ==========================================
// Store & Fleet Types
// ==========================================

export type DeviceState = 'Active' | 'Idle' | 'IdleRisk' | 'Neutral' | 'Offline' | 'Charging';

export interface StoreData {
  id: string;
  name: string;
  utilization: number;
  baseline: number;
  devices: number;
  activeDevices: number;
  region: string;
  idleRiskCount: number;
  lowBatteryCount: number;
}

export interface DeviceDetails {
  id: string;
  name: string;
  model: string;
  manufacturer: string;
  serial: string;
  macAddress?: string;
  storeName?: string;
  storeId?: string;
  region?: string;
  battery: number;
  isCharging: boolean;
  status: DeviceState;
  networkType?: string;
  signalStrength?: number;
  lastSeen: string;
  osVersion?: string;
  agentVersion?: string;
}

export interface DashboardKPIs {
  healthScore: number;
  storesAbove: number;
  storesBelow: number;
  idleRisk: number;
  totalDevices: number;
  activeDevices: number;
  offlineDevices: number;
  avgBattery: number;
}

export interface FilterOptions {
  profileId?: string;
  customer?: string;
  region?: string;
  storeId?: string;
  dateRange?: '24h' | '7d' | '30d' | '72h';
  status?: DeviceState[];
  batteryMin?: number;
  batteryMax?: number;
  searchQuery?: string;
  storeStatus?: 'above' | 'below' | 'critical';
}

// ==========================================
// AIOps Types
// ==========================================

export type IncidentSeverity = 'critical' | 'warning' | 'info';

export interface AIOpsIncident {
  id: string;
  title: string;
  description: string;
  severity: IncidentSeverity;
  startTime: string;
  impactedCount: number;
  domain: string;
  correlatedSignals: string[];
  trendDirection: 'escalating' | 'improving' | 'stable';
  probableDrivers: {
    cause: string;
    confidence: number;
    explanation: string;
    evidence: string[];
  }[];
  forecast: {
    riskLevel: 'critical' | 'high' | 'medium' | 'low';
    next24h: string;
    next7d?: string;
  };
}

export interface NoiseReductionMetrics {
  rawSignalsCount: number;
  incidentsCreated: number;
  compressionRatio: number;
}

// ==========================================
// LLM Settings Types
// ==========================================

export type LLMProvider = 'ollama' | 'lmstudio' | 'azure' | 'openai' | 'unknown';

export interface LLMConfig {
  base_url: string;
  model_name: string | null;
  api_key_set: boolean;
  api_version: string | null;
  is_connected: boolean;
  available_models: string[];
  active_model: string | null;
  provider: LLMProvider;
}

export interface LLMConfigUpdate {
  base_url?: string;
  model_name?: string;
  api_key?: string;
}

export interface LLMModel {
  id: string;
  name: string;
  size?: string | null;
  modified_at?: string | null;
}

export interface LLMModelsResponse {
  models: LLMModel[];
  active_model: string | null;
}

export interface LLMTestResult {
  success: boolean;
  message: string;
  response_time_ms?: number | null;
  model_used?: string | null;
}

export interface OllamaPullResponse {
  success: boolean;
  message: string;
  model_name: string;
}

export interface PopularModel {
  id: string;
  name: string;
  size: string;
  description: string;
}

// ==========================================
// Location Heatmap Types
// ==========================================

export interface LocationData {
  id: string;
  name: string;
  utilization: number;
  baseline: number;
  deviceCount: number;
  activeDeviceCount: number;
  region?: string;
  anomalyCount?: number;
}

export interface LocationHeatmapResponse {
  locations: LocationData[];
  attributeName: string;
  totalLocations: number;
  totalDevices: number;
}

// ==========================================
// Investigation Panel Types
// ==========================================

export interface FeatureContribution {
  feature_name: string;
  feature_display_name: string;
  contribution_percentage: number;
  contribution_direction: 'positive' | 'negative';
  current_value: number;
  current_value_display: string;
  baseline_value: number;
  baseline_value_display: string;
  deviation_sigma: number;
  percentile: number;
  plain_text_explanation: string;
}

export interface AnomalyExplanation {
  summary_text: string;
  detailed_explanation: string;
  feature_contributions: FeatureContribution[];
  top_contributing_features: string[];
  explanation_method: string;
  explanation_generated_at: string | null;
}

export interface BaselineMetric {
  metric_name: string;
  metric_display_name: string;
  metric_unit: string;
  current_value: number;
  current_value_display: string;
  baseline_mean: number;
  baseline_std: number;
  baseline_min: number;
  baseline_max: number;
  deviation_sigma: number;
  deviation_percentage: number;
  percentile_rank: number;
  is_anomalous: boolean;
  anomaly_direction: 'above' | 'below' | 'normal';
}

export interface BaselineConfig {
  baseline_type: string;
  baseline_period_days: number;
  comparison_window_hours: number;
  statistical_method: string;
  peer_group_name: string | null;
  peer_group_size: number | null;
  baseline_calculated_at: string;
}

export interface BaselineComparison {
  baseline_config: BaselineConfig;
  metrics: BaselineMetric[];
  overall_deviation_score: number;
}

export interface TimeSeriesDataPoint {
  timestamp: string;
  value: number;
  is_anomalous: boolean;
}

export interface HistoricalTimeline {
  metric_name: string;
  data_points: TimeSeriesDataPoint[];
  baseline_mean: number;
  baseline_std: number;
  baseline_upper: number;
  baseline_lower: number;
}

export interface EvidenceEvent {
  event_id: string;
  timestamp: string;
  event_type: string;
  event_category: 'apps' | 'storage' | 'battery' | 'network' | 'security' | 'system';
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  title: string;
  description: string;
  details: Record<string, unknown> | null;
  is_contributing_event: boolean;
  contribution_note: string | null;
}

export interface EvidenceHypothesis {
  statement: string;
  strength: 'strong' | 'moderate' | 'weak';
  source: 'telemetry' | 'pattern_match' | 'inference';
  linked_event_id: string | null;
}

export interface RootCauseHypothesis {
  hypothesis_id: string;
  title: string;
  description: string;
  likelihood: number;
  evidence_for: EvidenceHypothesis[];
  evidence_against: EvidenceHypothesis[];
  recommended_actions: string[];
}

export interface AIAnalysis {
  analysis_id: string;
  generated_at: string;
  model_used: string;
  primary_hypothesis: RootCauseHypothesis;
  alternative_hypotheses: RootCauseHypothesis[];
  confidence_score: number;
  confidence_level: 'high' | 'medium' | 'low' | 'uncertain';
  confidence_explanation: string;
  similar_cases_analyzed: number;
  feedback_received: boolean;
  feedback_rating: 'helpful' | 'not_helpful' | null;
}

export interface RemediationSuggestion {
  remediation_id: string;
  title: string;
  description: string;
  detailed_steps: string[];
  priority: number;
  confidence_score: number;
  confidence_level: 'high' | 'medium' | 'low';
  source: 'learned' | 'ai_generated' | 'policy';
  source_details: string;
  historical_success_rate: number | null;
  historical_sample_size: number | null;
  estimated_impact: string | null;
  is_automated: boolean;
  automation_type: string | null;
}

export interface SimilarCase {
  case_id: string;
  anomaly_id: number;
  device_id: number;
  detected_at: string;
  resolved_at: string | null;
  similarity_score: number;
  similarity_factors: string[];
  anomaly_type: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  resolution_status: string;
  resolution_summary: string | null;
  successful_remediation: string | null;
  time_to_resolution_hours: number | null;
}

export interface InvestigationPanel {
  anomaly_id: number;
  device_id: number;
  anomaly_score: number;
  severity: 'critical' | 'high' | 'medium' | 'low';
  confidence_score: number;
  detected_at: string;
  explanation: AnomalyExplanation;
  baseline_comparison: BaselineComparison | null;
  evidence_events: EvidenceEvent[];
  evidence_event_count: number;
  ai_analysis: AIAnalysis | null;
  suggested_remediations: RemediationSuggestion[];
  similar_cases: SimilarCase[];
}

export interface AIAnalysisFeedback {
  rating: 'helpful' | 'not_helpful';
  feedback_text?: string;
  actual_root_cause?: string;
}

export interface RemediationOutcome {
  outcome: 'resolved' | 'partially_resolved' | 'no_effect' | 'made_worse';
  notes?: string;
}

export interface LearnFromFix {
  remediation_description: string;
  tags: string[];
}

// ==========================================
// Baseline Feature Types
// ==========================================

export type BaselineStatus = 'stable' | 'warning' | 'drift';

export interface BaselineFeature {
  feature: string;
  baseline: number;
  observed: number;
  unit: string;
  status: BaselineStatus;
  drift_percent: number;
  mad: number;
  sample_count: number;
  last_updated: string | null;
}

export interface BaselineHistoryEntry {
  id: number;
  date: string;
  feature: string;
  old_value: number;
  new_value: number;
  type: 'auto' | 'manual';
  reason: string | null;
}

// ==========================================
// Smart Anomaly Grouping Types
// ==========================================

export type AnomalyGroupType =
  | 'remediation_match'
  | 'category_match'
  | 'similarity_cluster'
  | 'temporal_cluster'
  | 'location_cluster';

export type Severity = 'critical' | 'high' | 'medium' | 'low';

export interface AnomalyGroupMember {
  anomaly_id: number;
  device_id: number;
  anomaly_score: number;
  severity: Severity;
  status: string;
  timestamp: string;
  device_model?: string;
  location?: string;
  primary_metric?: string;
}

export interface AnomalyGroup {
  group_id: string;
  group_name: string;
  group_category: string;
  group_type: AnomalyGroupType;
  severity: Severity;
  total_count: number;
  open_count: number;
  device_count: number;
  suggested_remediation?: RemediationSuggestion;
  common_location?: string;
  common_device_model?: string;
  time_range_start: string;
  time_range_end: string;
  sample_anomalies: AnomalyGroupMember[];
  grouping_factors: string[];
  avg_similarity_score?: number;
}

export interface GroupedAnomaliesResponse {
  groups: AnomalyGroup[];
  total_anomalies: number;
  total_groups: number;
  ungrouped_count: number;
  ungrouped_anomalies: AnomalyGroupMember[];
  grouping_method: string;
  computed_at: string;
}

export interface BulkActionRequest {
  action: 'resolve' | 'investigating' | 'dismiss';
  anomaly_ids: number[];
  notes?: string;
}

export interface BulkActionResponse {
  success: boolean;
  affected_count: number;
  failed_ids: number[];
  message: string;
}

