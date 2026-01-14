# Database Datapoints Reference

This document provides a comprehensive list of all datapoints that can be read from the database.

---

## Table of Contents

1. [Core Tenant & Organizational Tables](#core-tenant--organizational-tables)
2. [Metrics & Telemetry Tables](#metrics--telemetry-tables)
3. [Baseline & Anomaly Detection Tables](#baseline--anomaly-detection-tables)
4. [ML Model & Performance Tables](#ml-model--performance-tables)
5. [Explanations & Investigation Tables](#explanations--investigation-tables)
6. [Alerting & Notification Tables](#alerting--notification-tables)
7. [Audit & Compliance Tables](#audit--compliance-tables)
8. [Location & Insights Tables](#location--insights-tables)
9. [Device Assignment Tables](#device-assignment-tables)
10. [Cost Intelligence Tables](#cost-intelligence-tables)
11. [External Data Discovery Tables](#external-data-discovery-tables)
12. [Summary Statistics](#summary-statistics)

---

## Core Tenant & Organizational Tables

### 1. TENANTS

| Field | Type | Description |
|-------|------|-------------|
| `tenant_id` | String(50) | Primary key |
| `name` | String(255) | Tenant name |
| `tier` | String(20) | Subscription tier (free, standard, enterprise) |
| `created_at` | DateTime | Creation timestamp |
| `metadata` | Text/JSON | Additional tenant metadata |

### 2. USERS

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | String(50) | Primary key |
| `tenant_id` | String(50) | Foreign key to tenants |
| `email` | String(255) | User email address (unique) |
| `name` | String(255) | User full name |
| `role` | String(20) | Role (viewer, analyst, admin) |
| `password_hash` | String(255) | Hashed password |
| `is_active` | Boolean | Active status |
| `last_login` | DateTime | Last login timestamp |
| `created_at` | DateTime | Creation timestamp |
| `metadata` | Text/JSON | User preferences and settings |

### 3. DEVICES

| Field | Type | Description |
|-------|------|-------------|
| `device_id` | String(50) | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `source` | String(20) | Data source (xsight, mobicontrol, synthetic) |
| `external_id` | String(100) | Original ID from source system |
| `name` | String(255) | Device name |
| `device_type` | String(50) | Device type (phone, tablet, laptop, etc.) |
| `os_version` | String(50) | Operating system version |
| `last_seen` | DateTime | Last reported timestamp |
| `device_group_id` | String(50) | Grouping identifier |
| `metadata` | Text/JSON | Additional device properties |

### 4. DEVICE_METADATA

| Field | Type | Description |
|-------|------|-------------|
| `device_id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `device_model` | String(100) | Device model string |
| `device_name` | String(200) | Display name |
| `location` | String(200) | Physical location |
| `status` | String(20) | Status (online, offline, unknown) |
| `last_seen` | DateTime | Last seen timestamp |
| `os_version` | String(50) | OS version |
| `agent_version` | String(50) | Agent software version |
| `created_at` | DateTime | Record creation time |
| `updated_at` | DateTime | Last update time |

---

## Metrics & Telemetry Tables

### 5. METRIC_DEFINITIONS

| Field | Type | Description |
|-------|------|-------------|
| `metric_id` | String(50) | Primary key |
| `name` | String(100) | Metric name |
| `category` | String(50) | Category (battery, network, storage, cpu, memory, custom) |
| `unit` | String(20) | Measurement unit (%, KB, bytes, etc.) |
| `data_type` | String(20) | Data type (int, float, string, bool) |
| `source` | String(50) | Data source (xsight, mobicontrol, custom) |
| `is_standard` | Boolean | Whether this is a standard metric |
| `validation_rules` | Text/JSON | Validation rules (min, max, allowed_values) |

**Standard Metrics Defined:**

| Metric ID | Name | Unit |
|-----------|------|------|
| `battery_drop` | Battery Level Drop | % |
| `free_storage` | Free Storage | KB |
| `download` | Download | bytes |
| `upload` | Upload | bytes |
| `offline_time` | Offline Time | minutes |
| `disconnect_count` | Disconnect Count | count |
| `wifi_signal` | WiFi Signal Strength | dBm |
| `connection_time` | Connection Time | minutes |

### 6. TELEMETRY_POINTS (Time-Series Data)

| Field | Type | Description |
|-------|------|-------------|
| `id` | BigInteger | Primary key (auto-increment) |
| `device_id` | String(50) | Foreign key |
| `tenant_id` | String(50) | Foreign key |
| `timestamp` | DateTime | Data point timestamp |
| `metric_id` | String(50) | Foreign key to metrics |
| `value` | Float | Numeric value |
| `value_str` | String(255) | Non-numeric value (optional) |
| `quality` | Integer | Data quality score (0-100) |
| `ingestion_time` | DateTime | When data was ingested |
| `source_batch_id` | String(50) | Batch tracking identifier |

### 7. ANOMALY_RESULTS

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `device_id` | Integer | Device identifier |
| `timestamp` | DateTime | Anomaly detection time |
| `anomaly_score` | Float | Anomaly score from detector |
| `anomaly_label` | Integer | Classification (-1 for anomaly, 1 for normal) |
| `total_battery_level_drop` | Float | Battery metric value |
| `total_free_storage_kb` | Float | Storage metric value |
| `download` | Float | Download metric value |
| `upload` | Float | Upload metric value |
| `offline_time` | Float | Offline time metric value |
| `disconnect_count` | Float | Disconnect count metric value |
| `wifi_signal_strength` | Float | WiFi signal metric value |
| `connection_time` | Float | Connection time metric value |
| `feature_values_json` | Text/JSON | All feature values |
| `status` | String(20) | Investigation status (open, investigating, resolved, false_positive) |
| `assigned_to` | String(100) | Assigned analyst |
| `notes` | Text | Investigation notes |
| `created_at` | DateTime | Creation timestamp |
| `updated_at` | DateTime | Last update timestamp |

---

## Baseline & Anomaly Detection Tables

### 8. BASELINES

| Field | Type | Description |
|-------|------|-------------|
| `baseline_id` | String(50) | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `name` | String(255) | Baseline name |
| `scope` | String(50) | Scope level (tenant, site, device_group, device) |
| `scope_id` | String(50) | ID of scope entity |
| `metric_id` | String(50) | Foreign key to metric |
| `window_config` | Text/JSON | Configuration (window_days, update_frequency, time_segmentation) |
| `stats` | Text/JSON | Baseline statistics (mean, std, median, p5, p95, etc.) |
| `valid_from` | DateTime | Start of validity period |
| `valid_to` | DateTime | End of validity period (NULL = active) |
| `created_by` | String(50) | Creator user ID or 'system' |

### 9. ANOMALIES

| Field | Type | Description |
|-------|------|-------------|
| `anomaly_id` | String(50) | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `device_id` | String(50) | Foreign key |
| `event_id` | String(50) | Foreign key to anomaly_events |
| `timestamp` | DateTime | When anomaly occurred |
| `detector_name` | String(100) | Detection algorithm (isolation_forest, z_score, etc.) |
| `severity` | String(20) | Severity level (low, medium, high, critical) |
| `score` | Float | Anomaly score (0-1 or higher) |
| `metrics_involved` | Text/JSON | Metric values at anomaly time |
| `explanation` | Text | Human-readable explanation |
| `explanation_cache_key` | String(100) | LLM cache key |
| `user_feedback` | String(20) | User feedback (true_positive, false_positive, unknown) |
| `status` | String(20) | Status (new, acknowledged, resolved, ignored) |
| `created_at` | DateTime | Detection time |
| `updated_at` | DateTime | Last update time |

### 10. ANOMALY_EVENTS

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | String(50) | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `device_id` | String(50) | Foreign key |
| `event_start` | DateTime | Event start time |
| `event_end` | DateTime | Event end time |
| `duration_minutes` | Integer | Event duration |
| `anomaly_score_min` | Float | Minimum anomaly score in event |
| `anomaly_score_max` | Float | Maximum anomaly score in event |
| `anomaly_score_mean` | Float | Mean anomaly score in event |
| `row_count` | Integer | Number of anomalous rows in event |
| `severity` | String(20) | Event severity |
| `metrics_json` | Text/JSON | Aggregated metric values |
| `status` | String(20) | Status (new, acknowledged, resolved, ignored) |
| `model_version` | String(50) | ML model version used |
| `created_at` | DateTime | Creation timestamp |

### 11. DEVICE_PATTERNS

| Field | Type | Description |
|-------|------|-------------|
| `pattern_id` | String(50) | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `device_id` | String(50) | Foreign key |
| `period_start` | DateTime | Period start |
| `period_end` | DateTime | Period end |
| `total_points` | Integer | Total data points in period |
| `total_anomalies` | Integer | Total anomalies detected |
| `anomaly_rate` | Float | Percentage of anomalous points |
| `event_count` | Integer | Number of anomaly events |
| `worst_anomaly_score` | Float | Highest anomaly score |
| `mean_anomaly_score` | Float | Average anomaly score |
| `pattern_json` | Text/JSON | Detailed pattern data |
| `explanation` | Text | LLM-generated pattern explanation |
| `model_version` | String(50) | ML model version |
| `created_at` | DateTime | Creation timestamp |

---

## ML Model & Performance Tables

### 12. ML_MODELS

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | String(50) | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `name` | String(255) | Model name |
| `model_type` | String(50) | Type (isolation_forest, z_score, hybrid, autoencoder, etc.) |
| `version` | String(50) | Model version |
| `config_json` | Text/JSON | Model configuration parameters |
| `feature_cols_json` | Text/JSON | List of feature columns used |
| `model_artifact` | Binary | Serialized model (optional) |
| `status` | String(20) | Status (training, trained, deployed, archived) |
| `trained_at` | DateTime | Training completion time |
| `created_at` | DateTime | Creation timestamp |

### 13. MODEL_REGISTRY

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `model_name` | String(100) | Model name |
| `model_version` | String(50) | Version identifier |
| `training_run_id` | String(50) | Foreign key to training run |
| `dataset_version_id` | Integer | Foreign key to dataset version |
| `model_type` | String(50) | Model type |
| `algorithm` | String(100) | Algorithm name |
| `framework` | String(50) | Framework (sklearn, pytorch, onnx) |
| `model_path` | String(500) | File path to model |
| `onnx_path` | String(500) | ONNX export path |
| `config_path` | String(500) | Config file path |
| `baselines_path` | String(500) | Baselines file path |
| `validation_auc` | Float | AUC score |
| `anomaly_rate` | Float | Detected anomaly rate |
| `feature_count` | Integer | Number of features |
| `train_rows` | Integer | Training data rows |
| `stage` | String(20) | Deployment stage (development, staging, production, archived) |
| `is_active` | Boolean | Currently active flag |
| `deployed_at` | DateTime | Deployment time |
| `deployed_by` | String(100) | Deploying user |
| `created_at` | DateTime | Creation time |
| `created_by` | String(100) | Creator |
| `archived_at` | DateTime | Archive time (if archived) |
| `archive_reason` | Text | Reason for archiving |

### 14. MODEL_DEPLOYMENTS

| Field | Type | Description |
|-------|------|-------------|
| `deployment_id` | String(50) | Primary key |
| `model_id` | String(50) | Foreign key to ml_models |
| `environment` | String(50) | Environment (production, staging, development) |
| `is_active` | Boolean | Active flag |
| `deployed_at` | DateTime | Deployment timestamp |
| `deployed_by` | String(50) | Deploying user ID |

### 15. MODEL_METRICS

| Field | Type | Description |
|-------|------|-------------|
| `metric_id` | BigInteger | Primary key |
| `model_id` | String(50) | Foreign key |
| `timestamp` | DateTime | Metric timestamp |
| `precision_score` | Float | Model precision |
| `recall_score` | Float | Model recall |
| `f1_score` | Float | F1 score |
| `true_positives` | Integer | TP count |
| `false_positives` | Integer | FP count |
| `false_negatives` | Integer | FN count |
| `total_predictions` | Integer | Total predictions made |
| `extra_data` | Text/JSON | Additional metrics |

### 16. TRAINING_RUNS

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `run_id` | String(50) | Run identifier |
| `tenant_id` | String(50) | Foreign key |
| `model_version` | String(50) | Model version trained |
| `status` | String(20) | Run status |
| `config_json` | Text/JSON | Training configuration |
| `metrics_json` | Text/JSON | Training metrics |
| `artifacts_json` | Text/JSON | Artifact locations |
| `started_at` | DateTime | Start time |
| `completed_at` | DateTime | Completion time |
| `error` | Text | Error message if failed |
| `created_at` | DateTime | Creation time |
| `dataset_version_id` | Integer | Training dataset version |

### 17. TRAINING_DATASET_VERSIONS

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `version_tag` | String(100) | Version identifier |
| `data_source` | String(50) | Source (xsight, mobicontrol, unified) |
| `start_date` | DateTime | Data range start |
| `end_date` | DateTime | Data range end |
| `row_count` | Integer | Total rows in dataset |
| `column_count` | Integer | Number of columns |
| `device_count` | Integer | Number of unique devices |
| `feature_columns_json` | Text/JSON | Column names |
| `data_hash` | String(64) | SHA256 dataset hash |
| `schema_hash` | String(64) | SHA256 schema hash |
| `description` | Text | Description |
| `query_used` | Text | SQL query or parameters |
| `parameters_json` | Text/JSON | Loading parameters |
| `is_active` | Boolean | Active flag |
| `superseded_by_id` | Integer | Newer version ID |
| `created_at` | DateTime | Creation time |
| `created_by` | String(100) | Creator |

---

## Explanations & Investigation Tables

### 18. EXPLANATIONS

| Field | Type | Description |
|-------|------|-------------|
| `explanation_id` | String(50) | Primary key |
| `anomaly_id` | String(50) | Foreign key to anomalies |
| `llm_model` | String(100) | LLM model used (ollama/llama3.2, claude-3-haiku, etc.) |
| `prompt_version` | String(20) | Prompt version for reproducibility |
| `generated_text` | Text | Generated explanation |
| `confidence` | Float | LLM confidence score (0-1) |
| `tokens_used` | Integer | Token count for cost tracking |
| `generation_time_ms` | Integer | Generation latency |
| `context_used` | Text/JSON | Retrieved RAG context |
| `created_at` | DateTime | Generation timestamp |

### 19. ANOMALY_EXPLANATION_CACHE

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `anomaly_id` | Integer | Anomaly identifier |
| `summary_text` | Text | One-liner explanation |
| `detailed_explanation` | Text | Full explanation |
| `feature_contributions_json` | Text/JSON | Feature impact analysis |
| `top_contributing_features` | Text/JSON | Top factors |
| `ai_analysis_json` | Text/JSON | Full AI analysis |
| `ai_model_used` | String(100) | AI model name |
| `feedback_rating` | String(20) | User feedback (helpful, not_helpful) |
| `feedback_text` | Text | Feedback comment |
| `actual_root_cause` | Text | Correct root cause if provided |
| `created_at` | DateTime | Creation time |
| `updated_at` | DateTime | Update time |

### 20. INVESTIGATION_NOTES

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `anomaly_id` | Integer | Anomaly identifier |
| `user` | String(100) | User who added note |
| `note` | Text | Note content |
| `action_type` | String(50) | Action type (status_change, assignment, note) |
| `created_at` | DateTime | Creation time |

### 21. TROUBLESHOOTING_CACHE

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `error_signature` | String(255) | Hash of error pattern |
| `error_pattern` | Text/JSON | Error details |
| `advice` | Text | LLM-generated troubleshooting advice |
| `summary` | String(500) | Brief summary |
| `service_type` | String(50) | Service (sql, api, llm) |
| `use_count` | Integer | Reuse count |
| `last_used` | DateTime | Last used timestamp |
| `created_at` | DateTime | Creation time |

### 22. LEARNED_REMEDIATIONS

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `pattern_name` | String(255) | Pattern name |
| `pattern_hash` | String(64) | Pattern hash for matching |
| `anomaly_types` | Text/JSON | Applicable anomaly types |
| `severity_range` | Text/JSON | Severity range |
| `feature_conditions_json` | Text/JSON | Matching conditions |
| `event_patterns_json` | Text/JSON | Event patterns |
| `remediation_title` | String(255) | Remediation title |
| `remediation_description` | Text | Full description |
| `remediation_steps_json` | Text/JSON | Step-by-step instructions |
| `automation_config_json` | Text/JSON | Automation settings |
| `times_suggested` | Integer | Number of times suggested |
| `times_applied` | Integer | Number of times applied |
| `success_count` | Integer | Successful applications |
| `failure_count` | Integer | Failed applications |
| `initial_confidence` | Float | Initial confidence (0-1) |
| `current_confidence` | Float | Current confidence (0-1) |
| `confidence_history_json` | Text/JSON | Confidence over time |
| `learned_from_cases_json` | Text/JSON | Source anomalies |
| `last_successful_case_id` | Integer | Last success reference |
| `is_active` | Boolean | Active flag |
| `deactivation_reason` | Text | Reason if deactivated |
| `created_at` | DateTime | Creation time |
| `updated_at` | DateTime | Update time |

### 23. REMEDIATION_OUTCOMES

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `anomaly_id` | Integer | Anomaly identifier |
| `learned_remediation_id` | Integer | Remediation applied |
| `remediation_title` | String(255) | Title |
| `remediation_source` | String(50) | Source (learned, ai_generated, policy, manual) |
| `applied_at` | DateTime | Application time |
| `applied_by` | String(100) | Applying user |
| `outcome` | String(50) | Outcome (resolved, partially_resolved, no_effect, made_worse) |
| `outcome_recorded_at` | DateTime | Outcome timestamp |
| `outcome_notes` | Text | Notes on outcome |
| `anomaly_context_json` | Text/JSON | Snapshot of anomaly state |
| `created_at` | DateTime | Creation time |

### 24. DEVICE_ACTION_LOGS

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `device_id` | Integer | Device identifier |
| `action_type` | String(50) | Action type (lock, restart, wipe, message, locate, sync) |
| `initiated_by` | String(100) | User who initiated action |
| `reason` | Text | Reason for action |
| `success` | Boolean | Success flag |
| `error_message` | Text | Error if failed |
| `mobicontrol_action_id` | String(100) | MobiControl API ID |
| `timestamp` | DateTime | Action time |
| `created_at` | DateTime | Record creation time |

---

## Alerting & Notification Tables

### 25. ALERT_RULES

| Field | Type | Description |
|-------|------|-------------|
| `rule_id` | String(50) | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `name` | String(255) | Rule name |
| `rule_type` | String(50) | Type (anomaly_severity, anomaly_count, pattern_detected, etc.) |
| `conditions_json` | Text/JSON | Rule conditions |
| `severity` | String(20) | Rule severity (low, medium, high, critical) |
| `actions_json` | Text/JSON | Notification actions (email, webhook, etc.) |
| `is_enabled` | Boolean | Enabled flag |
| `created_at` | DateTime | Creation time |
| `updated_at` | DateTime | Update time |

### 26. ALERTS

| Field | Type | Description |
|-------|------|-------------|
| `alert_id` | String(50) | Primary key |
| `rule_id` | String(50) | Foreign key to alert_rules |
| `tenant_id` | String(50) | Foreign key |
| `anomaly_id` | String(50) | Foreign key (optional) |
| `device_id` | String(50) | Foreign key (optional) |
| `event_id` | String(50) | Foreign key (optional) |
| `severity` | String(20) | Alert severity |
| `status` | String(20) | Status (open, acknowledged, resolved, suppressed) |
| `message` | Text | Alert message |
| `triggered_at` | DateTime | Trigger time |
| `acknowledged_at` | DateTime | Acknowledgment time |
| `acknowledged_by` | String(50) | Acknowledging user |
| `resolved_at` | DateTime | Resolution time |
| `extra_data` | Text/JSON | Additional context |

### 27. ALERT_NOTIFICATIONS

| Field | Type | Description |
|-------|------|-------------|
| `notification_id` | BigInteger | Primary key |
| `alert_id` | String(50) | Foreign key |
| `channel` | String(50) | Channel (email, webhook, sms, slack) |
| `recipient` | String(255) | Email, URL, or phone |
| `status` | String(20) | Status (pending, sent, failed, delivered) |
| `response_json` | Text/JSON | Service response |
| `sent_at` | DateTime | Send time |

---

## Audit & Compliance Tables

### 28. AUDIT_LOGS

| Field | Type | Description |
|-------|------|-------------|
| `log_id` | BigInteger | Primary key |
| `timestamp` | DateTime | Event timestamp |
| `user_id` | String(50) | User identifier |
| `tenant_id` | String(50) | Foreign key |
| `action` | String(50) | Action (view, create, update, delete, export) |
| `resource_type` | String(50) | Resource type (anomaly, device, baseline, etc.) |
| `resource_id` | String(50) | Resource identifier |
| `ip_address` | String(45) | IPv6 address |
| `metadata` | Text/JSON | Request details and changes |

### 29. CHANGE_LOG

| Field | Type | Description |
|-------|------|-------------|
| `change_id` | String(50) | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `timestamp` | DateTime | Change timestamp |
| `change_type` | String(50) | Type (policy, os_version, app_version, ap_added, etc.) |
| `description` | Text | Change description |
| `affected_devices` | Text/JSON | List of affected device IDs |
| `source` | String(50) | Source (xsight, mobicontrol, manual, api) |
| `metadata` | Text/JSON | Additional details |

---

## Location & Insights Tables

### 30. LOCATION_METADATA

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `location_id` | String(100) | Location identifier (warehouse-1, store-a101) |
| `location_name` | String(255) | Display name |
| `parent_region` | String(100) | Parent region (Northeast, Region-1) |
| `timezone` | String(50) | Timezone (default UTC) |
| `mapping_type` | String(50) | Mapping method (custom_attribute, label, device_group, geo_fence) |
| `mapping_attribute` | String(100) | Attribute name (Store, Warehouse) |
| `mapping_value` | String(255) | Attribute value (A101, WH-North) |
| `device_group_id` | Integer | Device group identifier |
| `geo_fence_json` | Text/JSON | Geofence data (lat, lon, radius_m) |
| `shift_schedules_json` | Text/JSON | Shift information |
| `baseline_battery_drain_per_hour` | Float | Computed baseline |
| `baseline_disconnect_rate` | Float | Computed baseline |
| `baseline_drop_rate` | Float | Computed baseline |
| `baseline_computed_at` | DateTime | Baseline computation time |
| `is_active` | Boolean | Active flag |
| `created_at` | DateTime | Creation time |
| `updated_at` | DateTime | Update time |

### 31. AGGREGATED_INSIGHTS

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `entity_type` | String(50) | Entity type (location, user, cohort, device_model, app) |
| `entity_id` | String(100) | Entity identifier |
| `entity_name` | String(255) | Entity display name |
| `insight_category` | String(100) | Insight category |
| `severity` | String(20) | Severity (critical, warning, info) |
| `headline` | Text | Customer-facing headline |
| `impact_statement` | Text | Business impact |
| `comparison_context` | Text | Comparative context |
| `recommended_actions_json` | Text/JSON | Recommended actions |
| `insight_data_json` | Text/JSON | Full insight payload |
| `affected_device_count` | Integer | Number of affected devices |
| `affected_devices_json` | Text/JSON | Device list |
| `trend_direction` | String(20) | Trend (improving, stable, worsening) |
| `previous_value` | Float | Previous metric value |
| `current_value` | Float | Current metric value |
| `change_percent` | Float | Percentage change |
| `confidence_score` | Float | Confidence (0-1) |
| `data_quality_score` | Float | Quality score (0-1) |
| `computed_at` | DateTime | Computation time |
| `valid_until` | DateTime | Validity end |
| `is_active` | Boolean | Active flag |
| `acknowledged_at` | DateTime | Acknowledgment time |
| `acknowledged_by` | String(100) | Acknowledging user |

### 32. DEVICE_FEATURES

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `device_id` | Integer | Device identifier |
| `feature_values_json` | Text/JSON | Computed features |
| `metadata_json` | Text/JSON | Device metadata |
| `computed_at` | DateTime | Computation timestamp |
| `created_at` | DateTime | Creation time |

### 33. SHIFT_PERFORMANCE

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `device_id` | Integer | Device identifier |
| `location_id` | String(100) | Location identifier |
| `shift_date` | DateTime | Shift date |
| `shift_name` | String(50) | Shift name (Morning, Afternoon, Night) |
| `shift_start` | DateTime | Shift start time |
| `shift_end` | DateTime | Shift end time |
| `shift_duration_hours` | Float | Duration in hours |
| `battery_start` | Float | Starting battery percentage |
| `battery_end` | Float | Ending battery percentage |
| `battery_drain_total` | Float | Total drain |
| `battery_drain_rate_per_hour` | Float | Drain rate |
| `will_complete_shift` | Boolean | Predicted completion flag |
| `estimated_dead_time` | DateTime | Predicted failure time |
| `actual_completed_shift` | Boolean | Actual completion |
| `was_fully_charged_at_start` | Boolean | Charging status |
| `charge_events_count` | Integer | Number of charge events |
| `total_charge_time_minutes` | Float | Charging duration |
| `charge_received_during_shift` | Float | Battery % gained |
| `screen_on_time_minutes` | Float | Screen active time |
| `app_foreground_time_minutes` | Float | App foreground time |
| `total_drops` | Integer | Call drops |
| `total_disconnects` | Integer | Connection disconnects |
| `drain_vs_location_baseline` | Float | Comparison to location baseline |
| `drain_vs_device_baseline` | Float | Comparison to device baseline |
| `created_at` | DateTime | Creation time |

---

## Device Assignment Tables

### 34. DEVICE_ASSIGNMENTS

| Field | Type | Description |
|-------|------|-------------|
| `id` | BigInteger | Primary key |
| `tenant_id` | String(100) | Foreign key |
| `device_id` | String(100) | Device identifier |
| `user_id` | String(100) | User identifier |
| `user_name` | String(200) | User display name |
| `user_email` | String(200) | User email |
| `team_id` | String(100) | Team identifier |
| `team_name` | String(200) | Team display name |
| `assignment_type` | String(50) | Type (owner, user, operator, manager) |
| `valid_from` | DateTime | Assignment start date |
| `valid_to` | DateTime | Assignment end date (NULL = current) |
| `source` | String(50) | Source (mc_label, mc_attribute, manual, ad_sync) |
| `source_label_type` | String(100) | Source label type (Owner, AssignedUser) |
| `created_at` | DateTime | Creation time |
| `updated_at` | DateTime | Update time |

---

## Cost Intelligence Tables

### 35. DEVICE_TYPE_COSTS

| Field | Type | Description |
|-------|------|-------------|
| `id` | BigInteger | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `device_model` | String(255) | Device model string |
| `currency_code` | String(3) | Currency (default USD) |
| `purchase_cost` | BigInteger | Purchase cost in cents |
| `replacement_cost` | BigInteger | Replacement cost in cents |
| `repair_cost_avg` | BigInteger | Average repair cost in cents |
| `depreciation_months` | Integer | Depreciation period (default 36) |
| `residual_value_percent` | Integer | End-of-life value % |
| `warranty_months` | Integer | Warranty period |
| `valid_from` | DateTime | Effective start date |
| `valid_to` | DateTime | Effective end date (NULL = active) |
| `notes` | Text | Additional notes |
| `created_by` | String(100) | Creator |
| `created_at` | DateTime | Creation time |
| `updated_at` | DateTime | Update time |

### 36. OPERATIONAL_COSTS

| Field | Type | Description |
|-------|------|-------------|
| `id` | BigInteger | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `name` | String(255) | Cost name |
| `description` | Text | Description |
| `category` | String(50) | Category (labor, downtime, support, infrastructure, maintenance, other) |
| `currency_code` | String(3) | Currency (default USD) |
| `amount` | BigInteger | Cost amount in cents |
| `cost_type` | String(50) | Type (hourly, daily, per_incident, fixed_monthly, per_device) |
| `unit` | String(50) | Unit (hour, day, incident, month, device) |
| `scope_type` | String(50) | Scope (tenant, location, device_group, device_model) |
| `scope_id` | String(100) | Scope entity ID |
| `is_active` | Boolean | Active flag |
| `valid_from` | DateTime | Effective start date |
| `valid_to` | DateTime | Effective end date |
| `notes` | Text | Notes |
| `created_by` | String(100) | Creator |
| `created_at` | DateTime | Creation time |
| `updated_at` | DateTime | Update time |

### 37. COST_CALCULATION_CACHE

| Field | Type | Description |
|-------|------|-------------|
| `id` | BigInteger | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `entity_type` | String(50) | Entity type (anomaly, anomaly_event, insight, device) |
| `entity_id` | String(100) | Entity identifier |
| `device_id` | String(50) | Device identifier |
| `device_model` | String(255) | Device model |
| `currency_code` | String(3) | Currency (default USD) |
| `hardware_cost` | BigInteger | Hardware cost in cents |
| `downtime_cost` | BigInteger | Downtime cost in cents |
| `labor_cost` | BigInteger | Labor cost in cents |
| `other_cost` | BigInteger | Other costs in cents |
| `total_cost` | BigInteger | Total cost in cents |
| `potential_savings` | BigInteger | Potential savings in cents |
| `investment_required` | BigInteger | Investment needed in cents |
| `payback_months` | Float | Payback period in months |
| `breakdown_json` | Text/JSON | Detailed cost breakdown |
| `impact_level` | String(20) | Impact level (high, medium, low) |
| `confidence_score` | Float | Confidence score (0-1) |
| `confidence_explanation` | Text | Confidence reasoning |
| `calculation_version` | String(20) | Formula version |
| `cost_config_snapshot_json` | Text/JSON | Cost configuration snapshot |
| `calculated_at` | DateTime | Calculation time |
| `expires_at` | DateTime | Expiration time |

### 38. COST_AUDIT_LOGS

| Field | Type | Description |
|-------|------|-------------|
| `id` | BigInteger | Primary key |
| `tenant_id` | String(50) | Foreign key |
| `entity_type` | String(50) | Entity type (device_type_cost, operational_cost) |
| `entity_id` | BigInteger | Entity identifier |
| `action` | String(20) | Action (create, update, delete) |
| `old_values_json` | Text/JSON | Previous values |
| `new_values_json` | Text/JSON | New values |
| `changed_fields_json` | Text/JSON | Changed field list |
| `user_id` | String(100) | User who made change |
| `user_email` | String(255) | User email |
| `change_reason` | Text | Reason for change |
| `source` | String(50) | Source (manual, api, import, system) |
| `ip_address` | String(45) | Request IP |
| `user_agent` | String(500) | Browser info |
| `request_id` | String(100) | Request tracking ID |
| `timestamp` | DateTime | Timestamp |

---

## External Data Discovery Tables

The system also supports dynamic schema discovery from external data sources (XSight DW and MobiControl databases):

### XSight Tables (cs_* prefix)

| Table Name | Description |
|------------|-------------|
| `cs_BatteryStat` | Battery statistics |
| `cs_AppUsage` | Application usage metrics |
| `cs_DataUsage` | Data consumption metrics |
| `cs_BatteryAppDrain` | Battery drain by application |
| `cs_Heatmap` | Device activity heatmaps |
| `cs_DataUsageByHour` | Hourly data consumption |
| `cs_BatteryLevelDrop` | Battery level drop events |
| `cs_AppUsageListed` | Listed application usage |
| `cs_WifiHour` | Hourly WiFi statistics |
| `cs_WiFiLocation` | WiFi location data |
| `cs_LastKnown` | Last known device state |
| `cs_DeviceInstalledApp` | Installed applications |
| `cs_PresetApps` | Preset application list |

*Plus additional cs_* tables dynamically discovered*

### MobiControl Tables

| Table Name | Description |
|------------|-------------|
| `DeviceStatInt` | Time-series numeric stats |
| `DeviceStatString` | Time-series string stats |
| `DeviceStatLocation` | Location data |
| `DeviceStatNetTraffic` | Network metrics |
| `DeviceInstalledApp` | Installed applications |
| `MainLog` | Main system log |
| `Alert` | Alert records |
| `Events` | Event records |
| `DevInfo` | Device information |
| `DeviceLastKnownLocation` | Last known location |
| `AndroidDevice` | Android device specifics |
| `iOSDevice` | iOS device specifics |
| `WindowsDevice` | Windows device specifics |
| `MacDevice` | Mac device specifics |
| `LinuxDevice` | Linux device specifics |
| `ZebraAndroidDevice` | Zebra Android device specifics |

*Plus additional tables dynamically discovered*

---

## Summary Statistics

| Category | Tables | Approximate Fields |
|----------|--------|-------------------|
| Core Infrastructure | 4 | 40+ |
| Metrics & Telemetry | 3 | 45+ |
| Baselines & Anomalies | 4 | 65+ |
| ML & Training | 6 | 85+ |
| Explanations & Investigation | 7 | 80+ |
| Alerting | 3 | 30+ |
| Audit & Compliance | 2 | 20+ |
| Locations & Insights | 4 | 75+ |
| Assignments | 1 | 15+ |
| Cost Intelligence | 4 | 55+ |
| **Total** | **38+** | **400+** |

---

## Source Files

The database structure is defined in the following source files:

| File | Description |
|------|-------------|
| `scripts/init_backend_schema.sql` | SQL schema definition |
| `src/device_anomaly/database/schema.py` | Python SQLAlchemy schema |
| `src/device_anomaly/db/models.py` | Core ORM models |
| `src/device_anomaly/db/models_cost.py` | Cost module models |
| `src/device_anomaly/api/models.py` | API response models |
| `src/device_anomaly/api/models_cost.py` | Cost API models |
| `src/device_anomaly/data_access/schema_discovery.py` | External data discovery |
