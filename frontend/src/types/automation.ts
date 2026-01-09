/**
 * Types for ML Automation system - scheduler, real-time scoring, auto-retraining
 * IMPORTANT: These types must match the backend API responses in automation.py
 */

export type ScheduleInterval =
  | 'hourly'
  | 'every_6_hours'
  | 'every_12_hours'
  | 'daily'
  | 'weekly'
  | 'manual';

/**
 * Scheduler configuration - matches SchedulerConfigResponse from backend
 */
export interface SchedulerConfig {
  // Training settings
  training_enabled: boolean;
  training_interval: ScheduleInterval;
  training_hour: number;
  training_day_of_week: number;
  training_lookback_days: number;
  training_validation_days: number;

  // Scoring settings
  scoring_enabled: boolean;
  scoring_interval_minutes: number;

  // Auto-retrain settings
  auto_retrain_enabled: boolean;
  auto_retrain_fp_threshold: number;
  auto_retrain_min_feedback: number;
  auto_retrain_cooldown_hours: number;

  // Alerting
  alerting_enabled: boolean;
  alert_on_high_anomaly_rate: boolean;
  high_anomaly_rate_threshold: number;

  // Insight generation settings
  insights_enabled: boolean;
  daily_digest_hour: number;
  shift_readiness_enabled: boolean;
  shift_readiness_lead_minutes: number;
  shift_schedules: string[];
  location_baseline_enabled: boolean;
  location_baseline_day_of_week: number;
  location_baseline_hour: number;

  // Timestamps
  last_training_time: string | null;
  last_scoring_time: string | null;
  last_auto_retrain_time: string | null;
  last_daily_digest_time: string | null;
  last_shift_readiness_time: string | null;
  last_location_baseline_time: string | null;
}

/**
 * Scheduler configuration update request
 */
export interface SchedulerConfigUpdate {
  training_enabled?: boolean;
  training_interval?: ScheduleInterval;
  training_hour?: number;
  training_day_of_week?: number;
  training_lookback_days?: number;
  training_validation_days?: number;
  scoring_enabled?: boolean;
  scoring_interval_minutes?: number;
  auto_retrain_enabled?: boolean;
  auto_retrain_fp_threshold?: number;
  auto_retrain_min_feedback?: number;
  auto_retrain_cooldown_hours?: number;
  alerting_enabled?: boolean;
  alert_on_high_anomaly_rate?: boolean;
  high_anomaly_rate_threshold?: number;
  insights_enabled?: boolean;
  daily_digest_hour?: number;
  shift_readiness_enabled?: boolean;
  shift_readiness_lead_minutes?: number;
  shift_schedules?: string[];
  location_baseline_enabled?: boolean;
  location_baseline_day_of_week?: number;
  location_baseline_hour?: number;
}

/**
 * Scheduler status - matches SchedulerStatusResponse from backend
 */
export interface SchedulerStatus {
  is_running: boolean;
  training_status: 'idle' | 'running' | 'completed' | 'failed';
  scoring_status: 'idle' | 'running' | 'completed' | 'failed';
  insights_status?: 'idle' | 'running' | 'completed' | 'failed';
  last_training_result: {
    success: boolean;
    timestamp: string;
    metrics?: Record<string, unknown>;
    error?: string;
  } | null;
  last_scoring_result: {
    success: boolean;
    timestamp: string;
    total_scored: number;
    anomalies_detected: number;
    anomaly_rate: number;
  } | null;
  last_insight_result?: Record<string, unknown> | null;
  next_training_time: string | null;
  next_scoring_time: string | null;
  next_insight_time?: string | null;
  total_anomalies_detected: number;
  total_insights_generated?: number;
  false_positive_rate: number;
  uptime_seconds: number;
  errors: string[];
}

/**
 * Alert from the scheduler
 */
export interface AutomationAlert {
  id: string;
  timestamp: string;
  message: string;
  acknowledged: boolean;
}

/**
 * Job in history
 */
export interface AutomationJob {
  type: string;
  timestamp: string;
  triggered_by: string;
  success?: boolean;
  error?: string;
  // Backend returns details; keep this optional for forward compatibility.
  details?: Record<string, unknown>;
}

/**
 * History response - array of jobs
 */
export type AutomationHistory = AutomationJob[];

/**
 * Manual job trigger request
 */
export interface TriggerJobRequest {
  job_type: 'training' | 'scoring' | 'daily_digest' | 'shift_readiness' | 'location_baseline' | 'device_metadata_sync';
  start_date?: string;
  end_date?: string;
  validation_days?: number;
  shift_name?: string;
}

/**
 * Manual job trigger response
 */
export interface TriggerJobResponse {
  success: boolean;
  job_id: string | null;
  message: string;
}

/**
 * Score request for on-demand scoring
 */
export interface ScoreRequest {
  device_ids?: number[];
  start_date: string;
  end_date: string;
}

/**
 * Score response
 */
export interface ScoreResponse {
  success: boolean;
  total_scored: number;
  anomalies_detected: number;
  anomaly_rate: number;
  results: Array<{
    device_id: number;
    timestamp: string;
    anomaly_score: number;
  }> | null;
}
