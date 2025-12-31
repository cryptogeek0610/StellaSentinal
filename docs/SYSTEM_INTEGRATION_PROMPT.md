System integration prompt

Context and goal
You are working on a Device Anomaly Detection System that:
- Detects anomalies in IoT device telemetry using Isolation Forest
- Generates human-readable explanations using an LLM
- Manages statistical baselines that define "normal" behavior
- Provides a React dashboard for visualization and management

Ultimate goal: Automatically detect device anomalies, explain them in business terms, and continuously improve detection accuracy by learning from feedback and adjusting baselines.

System architecture and execution order
Phase 1: Data pipeline (must run in this order)
1. DATA INGESTION
   - Load telemetry from Data Warehouse (DW) or generate synthetic data
   - Data includes: DeviceId, Timestamp, 8 metrics (battery, storage, network, WiFi, etc.)
   - Output: Raw DataFrame with time-series data per device
2. FEATURE ENGINEERING
   - Compute rolling statistics (mean, std) over time windows (default: 12 hours)
   - Calculate deltas (day-over-day changes)
   - Create cohort-normalized features if cohort_id available
   - Output: Feature DataFrame with ~24-35 feature columns
3. BASELINE COMPUTATION
   - Compute statistical baselines (median, MAD) at multiple levels:
     - Global baseline (all devices)
     - Cohort baseline (by ManufacturerId, ModelId, OsVersionId)
     - Store/Customer baseline (if available)
     - Device-level baseline (if enough data)
   - Apply baselines to create z-score normalized features (*_z_* columns)
   - Save baselines to artifacts/ for drift detection
   - Output: Baseline-normalized features + baseline statistics
4. ISOLATION FOREST TRAINING
   - Select features (prefer baseline-normalized columns if available)
   - Handle missing values (impute with median)
   - Apply feature domain weights (prevent single domain from dominating)
   - Scale features (StandardScaler)
   - Train Isolation Forest model:
     - n_estimators: 300
     - contamination: 0.03 (3% expected anomalies)
     - random_state: 42
   - Output: Trained detector model
5. ANOMALY SCORING
   - Score all data points using trained model
   - Apply temporal residual scoring (if using HybridAnomalyDetector)
   - Combine scores and determine threshold
   - Label anomalies (score < threshold = anomaly, label = -1)
   - Output: Scored DataFrame with anomaly_score and anomaly_label
6. PERSISTENCE
   - Save anomaly results to database (anomaly_results table)
   - Save baseline statistics to artifacts/ for drift monitoring
   - Output: Persistent storage of anomalies and baselines

Phase 2: UI and LLM integration (on-demand)
7. DASHBOARD DISPLAY
   - Load anomalies from database
   - Display Isolation Forest statistics (score distribution, model config)
   - Show anomaly trends and KPIs
   - Allow user interaction for investigation
8. LLM EXPLANATION GENERATION (on-demand)
   - User clicks "Generate Explanation" on an anomaly
   - System loads anomaly data (score, metrics, feature values)
   - Build LLM payload with:
     - Device ID, timestamp
     - Anomaly score from Isolation Forest
     - Top 5 most abnormal metrics (by z-score)
     - Typical values (from baseline statistics)
   - LLM generates natural language explanation
   - Store explanation in database for future reference
9. BASELINE ADJUSTMENT (LLM-driven)
   - LLM analyzes patterns in false positives/negatives
   - LLM suggests baseline adjustments using suggest_baseline_adjustments()
   - System proposes new baseline medians/MADs
   - User approves or LLM auto-applies (if configured)
   - Create new baseline version (expire old, create new)
   - Retrain Isolation Forest with updated baselines

Critical execution requirements
1. Order of operations (strict)
Never skip steps or run out of order:
- Baselines must be computed before Isolation Forest training
- Features must be engineered before baseline computation
- Isolation Forest must be trained before scoring
- Scoring must complete before persistence
- UI can only display after persistence

Error handling:
- If baseline computation fails -> fall back to raw features (log warning)
- If Isolation Forest training fails -> raise error (cannot proceed)
- If scoring fails -> raise error (cannot proceed)
- If persistence fails -> log error but continue (data in memory)

2. Baseline management for Isolation Forest
How baselines affect Isolation Forest:
- Baselines create normalized features (z-scores) that Isolation Forest uses
- Better baselines -> better feature normalization -> better anomaly detection
- Baselines are computed from training data before model training

LLM-driven baseline adjustment:
When LLM detects systematic issues:
1. LLM analyzes anomaly patterns and false positives
2. LLM calls suggest_baseline_adjustments() to propose changes
3. System creates BaselineFeedback objects:
   - level: "cohort" or "global"
   - group_key: {"ManufacturerId": "X", "ModelId": "Y"}
   - feature: "battery_level"
   - adjustment: +5.0 (shift median up by 5)
   - reason: "LLM analysis: 80% of anomalies in this cohort are false positives due to low baseline"
4. Apply feedback using apply_feedback() with learning_rate=0.35
5. Retrain Isolation Forest with updated baselines

Baseline update flow:
Current Baseline -> LLM Analysis -> Suggested Adjustments -> User Approval (or auto-apply) -> New Baseline Version -> Retrain Isolation Forest -> Re-score Data -> Update UI

3. UI integration requirements
Dashboard must show:
- Isolation Forest model configuration (n_estimators, contamination, etc.)
- Score distribution histogram (normal vs. anomalies)
- Baseline statistics (current active baselines per level)
- Anomaly list with scores and explanations
- Baseline adjustment suggestions from LLM
- Ability to approve/reject baseline changes

UI workflow:
User views anomaly -> Clicks "Explain" -> LLM generates explanation ->
User reviews -> If false positive -> Click "Adjust Baseline" ->
LLM suggests adjustment -> User approves -> System updates baseline ->
System retrains model -> UI refreshes with new results

LLM integration points
1. Explanation generation
Input to LLM:
- Anomaly score from Isolation Forest
- Device metrics at anomaly time
- Baseline statistics (typical mean/std)
- Z-scores for each metric

LLM prompt structure:
You are explaining an anomaly detected by Isolation Forest (score: {score}).
Device {device_id} at {timestamp} showed unusual behavior:
- Metric X: {value} (typical: {mean} +/- {std}, z-score: {z})
- Metric Y: {value} (typical: {mean} +/- {std}, z-score: {z})
Explain what this means in business terms and suggest next steps.

2. Baseline adjustment suggestions
LLM analyzes:
- Patterns in false positives (anomalies marked as false by users)
- Recurring anomalies in same device/cohort
- Drift in feature distributions over time

LLM output:
{
  "level": "cohort",
  "group_key": {"ManufacturerId": "Samsung", "ModelId": "Galaxy S21"},
  "feature": "battery_level",
  "adjustment": +10.0,
  "reason": "80% of anomalies in this cohort are false positives. Baseline median is too low for this device model."
}

LLM calls:
- suggest_baseline_adjustments() - Analyze and propose changes
- apply_feedback() - Apply approved adjustments
- BaselineRepository.update_stats() - Persist new baseline version

Configuration and settings
Isolation Forest configuration
AnomalyDetectorConfig(
  contamination=0.03,
  n_estimators=300,
  random_state=42,
  scale_features=True,
  min_variance=1e-6
)

LLM can adjust:
- contamination: If too many false positives -> increase, if missing anomalies -> decrease
- Feature weights: If certain domains dominate -> adjust feature_domain_weights

Baseline configuration
BaselineLevel(
  name="cohort",
  group_columns=["ManufacturerId", "ModelId", "OsVersionId"],
  min_rows=25
)

Error handling and validation
Pre-flight checks:
- Verify data quality (no all-NaN columns)
- Check minimum data requirements (at least 25 rows per baseline level)
- Validate feature columns exist before training
- Ensure baselines computed before model training

Runtime validation:
- Monitor Isolation Forest score distribution (should be roughly normal)
- Check baseline drift (compare current vs. saved baseline stats)
- Validate LLM explanations (ensure they reference actual metrics)
- Verify baseline adjustments (ensure they're within reasonable bounds)

Success criteria
System is working correctly when:
- Isolation Forest trains successfully on baseline-normalized features
- Anomaly scores are distributed as expected (most near 0, outliers at extremes)
- LLM generates coherent explanations that reference actual metrics
- Baseline adjustments improve detection accuracy (fewer false positives)
- UI displays all components correctly (model stats, score distribution, baselines)
- Execution order is maintained (no step runs before prerequisites)

Implementation checklist
When implementing or debugging, ensure:
- [ ] Data ingestion completes before feature engineering
- [ ] Feature engineering completes before baseline computation
- [ ] Baseline computation completes before Isolation Forest training
- [ ] Isolation Forest training completes before scoring
- [ ] Scoring completes before persistence
- [ ] UI loads data from persistence (not in-memory)
- [ ] LLM explanations reference actual anomaly scores and metrics
- [ ] Baseline adjustments are applied before retraining
- [ ] Retraining happens after baseline updates
- [ ] UI refreshes after model retraining

Example: Complete execution flow
# 1. Load data
df_raw = load_dw_data()  # or generate_synthetic_data()
# 2. Engineer features
feature_builder = DeviceFeatureBuilder(window=12)
df_feat = feature_builder.transform(df_raw)
# 3. Compute baselines
baseline_levels = [
    BaselineLevel(name="global", group_columns=["__all__"]),
    BaselineLevel(name="cohort", group_columns=["ManufacturerId", "ModelId"])
]
baselines = compute_baselines(df_feat, feature_cols, baseline_levels)
df_feat = apply_baselines(df_feat, baselines, baseline_levels)
# 4. Train Isolation Forest
detector = HybridAnomalyDetector(config)
detector.fit(df_feat)  # Uses baseline-normalized features
# 5. Score data
df_scored = detector.score_dataframe(df_feat)
# 6. Persist
save_anomaly_results(df_scored)
# 7. UI displays results
# User clicks "Explain" -> LLM generates explanation
# 8. LLM suggests baseline adjustments
suggestions = suggest_baseline_adjustments(anomalies_df, baselines, baseline_levels)
# LLM analyzes and creates BaselineFeedback
feedback = BaselineFeedback(level="cohort", feature="battery", adjustment=+5.0)
baselines = apply_feedback(baselines, [feedback])
# 9. Retrain with updated baselines
df_feat = apply_baselines(df_feat, baselines, baseline_levels)
detector.fit(df_feat)  # Retrain with new baselines
df_scored = detector.score_dataframe(df_feat)
save_anomaly_results(df_scored)  # Update results
