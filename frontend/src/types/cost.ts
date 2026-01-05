/**
 * Types for the Cost Intelligence API
 *
 * Enables financial impact analysis for device anomalies and fleet insights.
 */

// =============================================================================
// ENUMERATIONS
// =============================================================================

export type CostCategory = 'labor' | 'downtime' | 'support' | 'infrastructure' | 'maintenance' | 'nff' | 'other';

export type CostType = 'hourly' | 'daily' | 'per_incident' | 'fixed_monthly' | 'per_device';

export type ScopeType = 'tenant' | 'location' | 'device_group' | 'device_model';

export type ImpactLevel = 'high' | 'medium' | 'low';

export type AuditAction = 'create' | 'update' | 'delete';

// =============================================================================
// HARDWARE COST MODELS
// =============================================================================

export interface HardwareCostBase {
  device_model: string;
  purchase_cost: number;
  replacement_cost?: number;
  repair_cost_avg?: number;
  battery_replacement_cost?: number;
  battery_lifespan_months?: number;
  depreciation_months?: number;
  residual_value_percent?: number;
  warranty_months?: number;
  currency_code: string;
  notes?: string;
}

export interface HardwareCostCreate extends HardwareCostBase {}

export interface HardwareCostUpdate {
  device_model?: string;
  purchase_cost?: number;
  replacement_cost?: number;
  repair_cost_avg?: number;
  battery_replacement_cost?: number;
  battery_lifespan_months?: number;
  depreciation_months?: number;
  residual_value_percent?: number;
  warranty_months?: number;
  notes?: string;
}

export interface HardwareCostResponse extends HardwareCostBase {
  id: number;
  tenant_id: string;
  device_count: number;
  total_fleet_value: number;
  valid_from: string;
  valid_to?: string;
  created_at: string;
  updated_at: string;
  created_by?: string;
}

export interface HardwareCostListResponse {
  costs: HardwareCostResponse[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  total_fleet_value: number;
}

export interface DeviceModelInfo {
  device_model: string;
  device_count: number;
  has_cost_entry: boolean;
}

export interface DeviceModelsResponse {
  models: DeviceModelInfo[];
  total: number;
}

// =============================================================================
// OPERATIONAL COST MODELS
// =============================================================================

export interface OperationalCostBase {
  name: string;
  category: CostCategory;
  amount: number;
  cost_type: CostType;
  unit?: string;
  scope_type: ScopeType;
  scope_id?: string;
  description?: string;
  currency_code: string;
  is_active: boolean;
  notes?: string;
}

export interface OperationalCostCreate extends OperationalCostBase {}

export interface OperationalCostUpdate {
  name?: string;
  category?: CostCategory;
  amount?: number;
  cost_type?: CostType;
  unit?: string;
  scope_type?: ScopeType;
  scope_id?: string;
  description?: string;
  is_active?: boolean;
  notes?: string;
}

export interface OperationalCostResponse extends OperationalCostBase {
  id: number;
  tenant_id: string;
  valid_from: string;
  valid_to?: string;
  monthly_equivalent: number;
  annual_equivalent: number;
  created_at: string;
  updated_at: string;
  created_by?: string;
}

export interface OperationalCostListResponse {
  costs: OperationalCostResponse[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  total_monthly_cost: number;
  total_annual_cost: number;
}

// =============================================================================
// COST SUMMARY MODELS
// =============================================================================

export interface CategoryCostSummary {
  category: string;
  total_cost: number;
  item_count: number;
  percentage_of_total: number;
}

export interface DeviceModelCostSummary {
  device_model: string;
  device_count: number;
  unit_cost: number;
  total_value: number;
}

export interface CostSummaryResponse {
  tenant_id: string;
  summary_period: string;
  total_hardware_value: number;
  total_operational_monthly: number;
  total_operational_annual: number;
  total_anomaly_impact_mtd: number;
  total_anomaly_impact_ytd: number;
  by_category: CategoryCostSummary[];
  by_device_model: DeviceModelCostSummary[];
  cost_trend_30d?: number;
  anomaly_cost_trend_30d?: number;
  calculated_at: string;
  device_count: number;
  anomaly_count_period: number;
}

// =============================================================================
// ANOMALY IMPACT MODELS
// =============================================================================

export interface ImpactComponent {
  type: string;
  description: string;
  amount: number;
  calculation_method: string;
  confidence: number;
}

export interface AnomalyImpactResponse {
  anomaly_id: number;
  device_id: number;
  device_model?: string;
  anomaly_severity: string;
  total_estimated_impact: number;
  impact_components: ImpactComponent[];
  device_unit_cost?: number;
  device_replacement_cost?: number;
  device_age_months?: number;
  device_depreciated_value?: number;
  estimated_downtime_hours?: number;
  productivity_cost_per_hour?: number;
  productivity_impact?: number;
  average_resolution_time_hours?: number;
  support_cost_per_hour?: number;
  estimated_support_cost?: number;
  similar_cases_count: number;
  similar_cases_avg_cost?: number;
  overall_confidence: number;
  confidence_explanation: string;
  impact_level: ImpactLevel;
  calculated_at: string;
}

export interface DeviceImpactSummary {
  device_id: number;
  device_model?: string;
  device_name?: string;
  location?: string;
  unit_cost?: number;
  current_value?: number;
  total_anomalies: number;
  open_anomalies: number;
  resolved_anomalies: number;
  total_estimated_impact: number;
  impact_mtd: number;
  impact_ytd: number;
  risk_score: number;
  risk_level: string;
}

export interface DeviceImpactResponse {
  device_id: number;
  device_model?: string;
  device_name?: string;
  summary: DeviceImpactSummary;
  recent_anomalies: AnomalyImpactResponse[];
  monthly_impact_trend: Record<string, number>;
  cost_saving_recommendations: string[];
}

// =============================================================================
// COST HISTORY / AUDIT MODELS
// =============================================================================

export interface CostChangeEntry {
  id: number;
  timestamp: string;
  action: AuditAction;
  entity_type: string;
  entity_id: number;
  entity_name: string;
  changed_by?: string;
  field_changed?: string;
  old_value?: string;
  new_value?: string;
  before_snapshot?: Record<string, unknown>;
  after_snapshot?: Record<string, unknown>;
}

export interface CostHistoryResponse {
  changes: CostChangeEntry[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  total_creates: number;
  total_updates: number;
  total_deletes: number;
  unique_users: number;
}

// =============================================================================
// FINANCIAL IMPACT MODELS (for insights integration)
// =============================================================================

export interface CostBreakdownItem {
  category: string;
  description: string;
  amount: number;
  is_recurring: boolean;
  period: string;
  confidence: number;
}

export interface FinancialImpactSummary {
  total_impact_usd: number;
  monthly_recurring_usd: number;
  potential_savings_usd: number;
  impact_level: ImpactLevel;
  breakdown: CostBreakdownItem[];
  recommendations: string[];
  investment_required_usd?: number;
  payback_months?: number;
  confidence_score: number;
  calculated_at?: string;
}

// =============================================================================
// BATTERY FORECASTING MODELS
// =============================================================================

export interface BatteryForecastEntry {
  device_model: string;
  device_count: number;
  battery_replacement_cost: number;
  battery_lifespan_months: number;
  devices_due_this_month: number;
  devices_due_next_month: number;
  devices_due_in_90_days: number;
  estimated_cost_30_days: number;
  estimated_cost_90_days: number;
  avg_battery_age_months: number;
  oldest_battery_months?: number;
}

export interface BatteryForecastResponse {
  forecasts: BatteryForecastEntry[];
  total_devices_with_battery_data: number;
  total_estimated_cost_30_days: number;
  total_estimated_cost_90_days: number;
  total_replacements_due_30_days: number;
  total_replacements_due_90_days: number;
  forecast_generated_at: string;
}

// =============================================================================
// COST ALERT MODELS
// =============================================================================

export type AlertThresholdType = 'anomaly_cost_daily' | 'anomaly_cost_monthly' | 'battery_forecast' | 'operational_cost';

export interface CostAlert {
  id: number;
  name: string;
  threshold_type: AlertThresholdType;
  threshold_value: number;
  is_active: boolean;
  notify_email?: string;
  notify_webhook?: string;
  last_triggered?: string;
  trigger_count: number;
  created_at: string;
  updated_at: string;
}

export interface CostAlertCreate {
  name: string;
  threshold_type: AlertThresholdType;
  threshold_value: number;
  is_active: boolean;
  notify_email?: string;
  notify_webhook?: string;
}

export interface CostAlertListResponse {
  alerts: CostAlert[];
  total: number;
}

export interface CostAlertTrigger {
  alert_id: number;
  alert_name: string;
  threshold_type: AlertThresholdType;
  threshold_value: number;
  current_value: number;
  triggered_at: string;
  message: string;
}

// =============================================================================
// NFF (NO FAULT FOUND) TRACKING MODELS
// =============================================================================

export interface NFFEntry {
  id: number;
  anomaly_id: number;
  device_id: number;
  device_model: string;
  reported_issue: string;
  investigation_cost: number;
  shipping_cost?: number;
  labor_hours: number;
  labor_cost: number;
  total_cost: number;
  resolution: 'no_fault_found' | 'user_error' | 'intermittent' | 'software_issue';
  notes?: string;
  created_at: string;
}

export interface NFFSummary {
  total_nff_count: number;
  total_nff_cost: number;
  avg_cost_per_nff: number;
  nff_rate_percent: number;
  by_device_model: Array<{
    device_model: string;
    count: number;
    total_cost: number;
  }>;
  by_resolution: Array<{
    resolution: string;
    count: number;
    total_cost: number;
  }>;
  trend_30_days: number;
}

// =============================================================================
// HELPER TYPES
// =============================================================================

export interface CostFilters {
  category?: CostCategory;
  scope_type?: ScopeType;
  is_active?: boolean;
  search?: string;
}

export interface CostPagination {
  page: number;
  page_size: number;
}

// Formatted cost display helpers
export function formatCurrency(amount: number, includeCents = false): string {
  if (includeCents) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
}

export function getImpactLevelColor(level: ImpactLevel): string {
  switch (level) {
    case 'high':
      return 'text-red-600 bg-red-100';
    case 'medium':
      return 'text-yellow-600 bg-yellow-100';
    case 'low':
      return 'text-green-600 bg-green-100';
    default:
      return 'text-gray-600 bg-gray-100';
  }
}

export function getCategoryIcon(category: CostCategory): string {
  switch (category) {
    case 'labor':
      return '👷';
    case 'downtime':
      return '⏱️';
    case 'support':
      return '🛠️';
    case 'infrastructure':
      return '🏢';
    case 'maintenance':
      return '🔧';
    case 'nff':
      return '❌';
    default:
      return '📦';
  }
}
