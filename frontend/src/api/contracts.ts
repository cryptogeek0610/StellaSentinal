import { z } from 'zod';
import type {
  IsolationForestStats,
  BaselineSuggestion,
  BaselineFeature,
  AnomalyListResponse,
  GroupedAnomaliesResponse,
} from '../types/anomaly';

const isolationForestConfigSchema = z.object({
  n_estimators: z.number(),
  contamination: z.number(),
  random_state: z.number(),
  scale_features: z.boolean(),
  min_variance: z.number(),
  feature_count: z.number().nullable().optional(),
  model_type: z.string(),
});

const scoreDistributionBinSchema = z.object({
  bin_start: z.number(),
  bin_end: z.number(),
  count: z.number(),
  is_anomaly: z.boolean(),
});

const scoreDistributionSchema = z.object({
  bins: z.array(scoreDistributionBinSchema),
  total_normal: z.number(),
  total_anomalies: z.number(),
  mean_score: z.number(),
  median_score: z.number(),
  min_score: z.number(),
  max_score: z.number(),
  std_score: z.number(),
});

const feedbackStatsSchema = z.object({
  total_feedback: z.number(),
  false_positives: z.number(),
  confirmed_anomalies: z.number(),
  projected_accuracy_gain: z.number(),
  last_retrain: z.string(),
});

const isolationForestStatsSchema = z.object({
  config: isolationForestConfigSchema,
  defaults: isolationForestConfigSchema.optional().nullable(),
  score_distribution: scoreDistributionSchema,
  total_predictions: z.number(),
  anomaly_rate: z.number(),
  feedback_stats: feedbackStatsSchema.optional().nullable(),
});

const groupKeySchema = z.union([z.string(), z.record(z.any())]).transform((value) => (
  typeof value === 'string' ? value : JSON.stringify(value)
));

const baselineSuggestionSchema = z.object({
  level: z.string(),
  group_key: groupKeySchema,
  feature: z.string(),
  baseline_median: z.number(),
  observed_median: z.number(),
  proposed_new_median: z.number(),
  rationale: z.string(),
});

const baselineFeatureSchema = z.object({
  feature: z.string(),
  baseline: z.number(),
  observed: z.number(),
  unit: z.string(),
  status: z.enum(['stable', 'warning', 'drift']),
  drift_percent: z.number(),
  mad: z.number(),
  sample_count: z.number(),
  last_updated: z.string().nullable().optional(),
});

const anomalyResponseSchema = z.object({
  id: z.number(),
  device_id: z.number(),
  device_name: z.string().nullable().optional(),
  timestamp: z.string(),
  anomaly_score: z.number(),
  anomaly_label: z.number(),
  status: z.string(),
  assigned_to: z.string().nullable().optional(),
  total_battery_level_drop: z.number().nullable().optional(),
  total_free_storage_kb: z.number().nullable().optional(),
  download: z.number().nullable().optional(),
  upload: z.number().nullable().optional(),
  offline_time: z.number().nullable().optional(),
  disconnect_count: z.number().nullable().optional(),
  wifi_signal_strength: z.number().nullable().optional(),
  connection_time: z.number().nullable().optional(),
  feature_values_json: z.string().nullable().optional(),
  created_at: z.string().optional(),
  updated_at: z.string().optional(),
}).passthrough();

const anomalyListSchema = z.object({
  anomalies: z.array(anomalyResponseSchema),
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
  total_pages: z.number(),
});

const severitySchema = z.enum(['critical', 'high', 'medium', 'low']);
const groupTypeSchema = z.enum([
  'remediation_match',
  'category_match',
  'similarity_cluster',
  'temporal_cluster',
  'location_cluster',
]);

const anomalyGroupMemberSchema = z.object({
  anomaly_id: z.number(),
  device_id: z.number(),
  anomaly_score: z.number(),
  severity: severitySchema,
  status: z.string(),
  timestamp: z.string(),
  device_name: z.string(),
  device_model: z.string().nullable().optional(),
  location: z.string().nullable().optional(),
  primary_metric: z.string().nullable().optional(),
}).passthrough();

const anomalyGroupSchema = z.object({
  group_id: z.string(),
  group_name: z.string(),
  group_category: z.string(),
  group_type: groupTypeSchema,
  severity: severitySchema,
  total_count: z.number(),
  open_count: z.number(),
  device_count: z.number(),
  suggested_remediation: z.any().nullable().optional(),
  common_location: z.string().nullable().optional(),
  common_device_model: z.string().nullable().optional(),
  time_range_start: z.string(),
  time_range_end: z.string(),
  sample_anomalies: z.array(anomalyGroupMemberSchema),
  grouping_factors: z.array(z.string()),
  avg_similarity_score: z.number().nullable().optional(),
}).passthrough();

const groupedAnomaliesSchema = z.object({
  groups: z.array(anomalyGroupSchema),
  total_anomalies: z.number(),
  total_groups: z.number(),
  ungrouped_count: z.number(),
  ungrouped_anomalies: z.array(anomalyGroupMemberSchema),
  grouping_method: z.string(),
  computed_at: z.string(),
  // Impact metrics for hero card
  coverage_percent: z.number(),
  top_impact_group_id: z.string().nullable(),
  top_impact_group_name: z.string().nullable(),
});

const formatZodError = (error: z.ZodError): string =>
  error.errors
    .map((issue) => `${issue.path.join('.') || 'root'}: ${issue.message}`)
    .join('; ');

const parseWithSchema = <T>(schema: z.ZodType<T>, payload: unknown, label: string): T => {
  const result = schema.safeParse(payload);
  if (!result.success) {
    throw new Error(`${label} schema mismatch: ${formatZodError(result.error)}`);
  }
  return result.data;
};

export const parseIsolationForestStats = (payload: unknown): IsolationForestStats =>
  parseWithSchema(isolationForestStatsSchema, payload, 'IsolationForestStats') as IsolationForestStats;

export const parseBaselineSuggestions = (payload: unknown): BaselineSuggestion[] =>
  parseWithSchema(z.array(baselineSuggestionSchema), payload, 'BaselineSuggestion[]') as BaselineSuggestion[];

export const parseBaselineFeatures = (payload: unknown): BaselineFeature[] =>
  parseWithSchema(z.array(baselineFeatureSchema), payload, 'BaselineFeature[]') as BaselineFeature[];

export const parseAnomalyList = (payload: unknown): AnomalyListResponse =>
  parseWithSchema(anomalyListSchema, payload, 'AnomalyListResponse') as AnomalyListResponse;

export const parseGroupedAnomalies = (payload: unknown): GroupedAnomaliesResponse =>
  parseWithSchema(groupedAnomaliesSchema, payload, 'GroupedAnomaliesResponse') as GroupedAnomaliesResponse;
