/**
 * Correlation Intelligence Types
 *
 * Types for correlation analysis, causal graphs, and cross-metric insights.
 */

// =============================================================================
// Correlation Matrix Types
// =============================================================================

/**
 * Statistics about metrics filtered during correlation computation.
 */
export interface FilterStats {
  total_input: number;      // Total metrics before filtering
  low_variance: number;     // Metrics removed due to low variance
  low_cardinality: number;  // Metrics removed due to low cardinality
  high_null: number;        // Metrics removed due to high null ratio
  passed: number;           // Metrics that passed quality filters
}

export interface CorrelationCell {
  metric_x: string;
  metric_y: string;
  correlation: number;
  p_value: number | null;
  sample_count: number;
  method: string;
  is_significant: boolean;  // Whether correlation is statistically significant (p < 0.05)
}

export interface CorrelationMatrixResponse {
  metrics: string[];
  matrix: number[][];  // N x N correlation values
  p_values: number[][] | null;  // N x N p-values matrix
  strong_correlations: CorrelationCell[];  // |r| > threshold
  method: 'pearson' | 'spearman';
  computed_at: string;
  total_samples: number;
  domain_filter: string | null;
  date_range: { start: string; end: string } | null;  // Date range used for computation
  filter_stats: FilterStats | null;  // Statistics about filtered metrics
}

// =============================================================================
// Scatter Plot Types
// =============================================================================

export interface ScatterDataPoint {
  device_id: number;
  x_value: number;
  y_value: number;
  is_anomaly: boolean;
  cohort: string | null;
  timestamp: string | null;
}

export interface ScatterPlotResponse {
  metric_x: string;
  metric_y: string;
  points: ScatterDataPoint[];
  correlation: number;
  regression_slope: number | null;
  regression_intercept: number | null;
  r_squared: number | null;
  total_points: number;
  anomaly_count: number;
}

// =============================================================================
// Causal Graph Types
// =============================================================================

export interface CausalNode {
  metric: string;
  domain: string;
  is_cause: boolean;
  is_effect: boolean;
  connection_count: number;
}

export interface CausalEdge {
  source: string;
  target: string;
  relationship: string;  // "causes", "correlates_with"
  strength: number;  // 0-1
  evidence: string | null;
}

export interface CausalGraphResponse {
  nodes: CausalNode[];
  edges: CausalEdge[];
  generated_at: string;
}

// =============================================================================
// Correlation Insights Types
// =============================================================================

export type InsightStrength = 'strong' | 'moderate' | 'weak';
export type InsightDirection = 'positive' | 'negative';

export interface CorrelationInsight {
  insight_id: string;
  headline: string;
  description: string;
  metrics_involved: string[];
  correlation_value: number;
  strength: InsightStrength;
  direction: InsightDirection;
  novelty_score: number;  // How unusual this correlation is (0-1)
  confidence: number;
  recommendation: string | null;
}

export interface CorrelationInsightsResponse {
  insights: CorrelationInsight[];
  total_correlations_analyzed: number;
  generated_at: string;
}

// =============================================================================
// Cohort Correlation Patterns Types
// =============================================================================

export interface CohortCorrelationPattern {
  cohort_id: string;
  cohort_name: string;
  metric_pair: string[];  // Array of two metric names (backend returns List[str])
  cohort_correlation: number;
  fleet_correlation: number;
  deviation: number;  // How different from fleet average
  device_count: number;
  is_anomalous: boolean;
  insight: string | null;
}

export interface CohortCorrelationPatternsResponse {
  patterns: CohortCorrelationPattern[];
  anomalous_cohorts: number;
  generated_at: string;
}

// =============================================================================
// Time-Lagged Correlation Types
// =============================================================================

export interface TimeLagCorrelation {
  metric_a: string;
  metric_b: string;
  lag_days: number;
  correlation: number;
  p_value: number | null;
  direction: 'a_predicts_b' | 'b_predicts_a';
  insight: string;
}

export interface TimeLagCorrelationsResponse {
  correlations: TimeLagCorrelation[];
  max_lag_analyzed: number;
  generated_at: string;
}

// =============================================================================
// API Request Parameter Types
// =============================================================================

export interface CorrelationMatrixParams {
  domain?: string;
  method?: 'pearson' | 'spearman';
  threshold?: number;
  max_metrics?: number;
}

export interface ScatterDataParams {
  metric_x: string;
  metric_y: string;
  color_by?: 'anomaly' | 'cohort';
  limit?: number;
}

// =============================================================================
// Scatter Anomaly Explanation Types
// =============================================================================

export interface ScatterAnomalyExplainRequest {
  device_id: number;
  metric_x: string;
  metric_y: string;
  x_value: number;
  y_value: number;
}

export interface ScatterAnomalyExplanation {
  explanation: string;
  what_happened: string;
  key_concerns: string[];
  likely_explanation: string;
  suggested_action: string;
}
