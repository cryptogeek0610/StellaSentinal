/**
 * Types for data discovery, training, and ML observability.
 */

// ============================================================================
// Data Discovery Types
// ============================================================================

export interface ColumnStats {
  column_name: string;
  dtype: string;
  null_count: number;
  null_percent: number;
  unique_count: number;
  min_val: number | null;
  max_val: number | null;
  mean: number | null;
  std: number | null;
  percentiles: Record<string, number>;
}

export interface TableProfile {
  table_name: string;
  row_count: number;
  date_range: [string | null, string | null];
  device_count: number;
  column_stats: Record<string, ColumnStats>;
  profiled_at: string | null;
  source_db?: 'xsight' | 'mobicontrol' | null; // Which database the table is from
}

export interface TemporalPattern {
  metric_name: string;
  hourly_stats: Record<number, { mean: number; std: number; count: number }>;
  daily_stats: Record<number, { mean: number; std: number; count: number }>;
}

export interface MetricDistribution {
  bins: number[];
  counts: number[];
  stats: {
    min: number;
    max: number;
    mean: number;
    std: number;
    median: number;
    total_samples: number;
  };
}

export interface AvailableMetric {
  table: string;
  column: string;
  dtype: string;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  category?: string | null; // 'raw', 'rolling', 'derived', 'temporal', 'cohort', 'volatility'
  domain?: string | null; // 'battery', 'rf', 'throughput', 'usage', 'storage', etc.
  description?: string | null; // Human-readable description
}

export interface DiscoverySummary {
  total_tables_profiled: number;
  total_rows: number;
  total_devices: number;
  metrics_discovered: number;
  patterns_analyzed: number;
  date_range: { start: string; end: string } | null;
  discovery_completed: string | null;
}

export interface DataDiscoveryStatus {
  status: 'idle' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string | null;
  started_at: string | null;
  completed_at: string | null;
  results_available: boolean;
}

// ============================================================================
// Training Types
// ============================================================================

export interface TrainingConfigRequest {
  start_date: string;
  end_date: string;
  validation_days?: number;
  contamination?: number;
  n_estimators?: number;
  export_onnx?: boolean;
}

export interface TrainingMetrics {
  train_rows: number;
  validation_rows: number;
  feature_count: number;
  anomaly_rate_train: number;
  anomaly_rate_validation: number;
  validation_auc?: number | null;
  precision_at_recall_80?: number | null;
  feature_importance?: Record<string, number>;
}

export interface TrainingArtifacts {
  model_path?: string | null;
  onnx_path?: string | null;
  baselines_path?: string | null;
  cohort_stats_path?: string | null;
  metadata_path?: string | null;
}

export interface TrainingStage {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at?: string | null;
  completed_at?: string | null;
  message?: string | null;
}

export interface TrainingRun {
  run_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'idle';
  progress: number;
  message?: string | null;
  stage?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  estimated_completion?: string | null;
  config?: Record<string, unknown> | null;
  metrics?: TrainingMetrics | null;
  artifacts?: TrainingArtifacts | null;
  model_version?: string | null;
  error?: string | null;
  stages?: TrainingStage[] | null;
}

export interface TrainingQueueStatus {
  queue_length: number;
  worker_available: boolean;
  last_job_completed_at?: string | null;
  next_scheduled?: string | null;
}

export interface TrainingHistoryResponse {
  runs: TrainingRun[];
  total: number;
}

// ============================================================================
// Enhanced Baseline Types
// ============================================================================

export interface EnhancedBaselineStats {
  median: number;
  mad: number;
  mean: number;
  std: number;
  sample_count: number;
  percentiles: Record<string, number>;
}

export interface TemporalBaseline {
  hourly_stats: Record<number, EnhancedBaselineStats>;
  daily_stats: Record<number, EnhancedBaselineStats>;
}

export interface AnomalyThresholds {
  warning: { lower: number; upper: number };
  critical: { lower: number; upper: number };
}

export interface DataDrivenBaseline {
  metric_name: string;
  global: EnhancedBaselineStats;
  by_hour?: { hour: number; mean: number; std: number }[];
  by_device_type?: Record<string, EnhancedBaselineStats>;
  anomaly_thresholds: AnomalyThresholds;
}
