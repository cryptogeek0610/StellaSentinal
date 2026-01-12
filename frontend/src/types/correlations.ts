/**
 * Correlation Intelligence Types
 *
 * Types for correlation analysis, causal graphs, and cross-metric insights.
 */

// =============================================================================
// Correlation Matrix Types
// =============================================================================

export interface CorrelationCell {
  metric_x: string;
  metric_y: string;
  correlation: number;
  p_value: number | null;
  sample_count: number;
  method: string;
}

export interface CorrelationMatrixResponse {
  metrics: string[];
  matrix: number[][];  // N x N correlation values
  strong_correlations: CorrelationCell[];  // |r| > threshold
  method: 'pearson' | 'spearman';
  computed_at: string;
  total_samples: number;
  domain_filter: string | null;
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
  metric_pair: [string, string];
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
