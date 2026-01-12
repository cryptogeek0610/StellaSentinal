# FLOW_MAP

## Diagram

[XSight DW + MobiControl]
        |
        v
[Unified Loader] -> [Feature Engineering (base)] -> [Train/Val Split]
        |                                   |                   |
        |                                   v                   v
        |                            [Cohort Stats]       [Train/Val Sets]
        |                                   |                   |
        |                                   v                   v
        |                            [Cohort Z Apply]     [Baselines] -> [Model Training]
        |                                                      |
        |                                                      v
        |                                      [Baselines + Cohort Stats + Feature Spec/Norms]
        |                                                      |
        v                                                      v
[Batch Inference / Scoring] -----------------------------------------------> [Persistence + API] -> [Frontend]
        |
        v
[Streaming Telemetry] -> [Streaming Features (canonical)] -> [Streaming Scoring] -> [Alerts/Websocket]

## Stages

### 1) Data ingestion and cleaning
- Entry points: `RealDataTrainingPipeline.load_training_data` (`src/device_anomaly/pipeline/training.py:169-214`), `run_dw_experiment` CLI path uses `load_unified_device_dataset` (`src/device_anomaly/cli/dw_experiment.py:41-104`).
- Key functions: `load_unified_device_dataset` (`src/device_anomaly/data_access/unified_loader.py:240-415`), DW telemetry query with `Timestamp` and device identifiers (`src/device_anomaly/data_access/dw_loader.py:258-266`).
- Config knobs: `TrainingConfig.start_date`, `end_date`, `row_limit`, `use_multi_source`, `row_limit_per_source` (`src/device_anomaly/pipeline/training.py:61-75`).
- Data schema: DataFrame keyed by `DeviceId` + `Timestamp`, including `ModelId`, `ManufacturerId`, `OsVersionId` (`src/device_anomaly/data_access/dw_loader.py:258-266`; `src/device_anomaly/data_access/unified_loader.py:269-277`).
- Outputs: enriched telemetry with MC metadata and derived fields (`src/device_anomaly/data_access/unified_loader.py:376-415`).
- Failure points: empty DW result short-circuits or raises (`src/device_anomaly/data_access/unified_loader.py:303-305`; `src/device_anomaly/pipeline/training.py:209-210`), MC credentials missing skips enrichment (`src/device_anomaly/data_access/unified_loader.py:313-317`).

### 2) Feature engineering
- Entry point: `DeviceFeatureBuilder.transform` (`src/device_anomaly/features/device_features.py:62-106`).
- Key functions: rolling stats and temporal features (`src/device_anomaly/features/device_features.py:149-200`).
- Cohort z-scores are deferred until after the train/val split and applied with stored cohort stats (`src/device_anomaly/features/cohort_stats.py:1-206`).
- Config knobs: rolling window sizes and `min_periods` plus `feature_spec` stored in metadata (`src/device_anomaly/features/device_features.py:26-92`; `src/device_anomaly/pipeline/training.py:235-560`).
- Data schema: requires `DeviceId` and `Timestamp` (`src/device_anomaly/features/device_features.py:75-78`).
- Outputs: feature-enriched DataFrame with rollups, deltas, and derived metrics; cohort z-scores are added later (`src/device_anomaly/features/device_features.py:149-469`).
- Failure points: missing `DeviceId`/`Timestamp` short-circuits feature engineering (`src/device_anomaly/features/device_features.py:75-78`).

### 3) Baseline computation and corrections
- Entry point: `RealDataTrainingPipeline.compute_baselines` (`src/device_anomaly/pipeline/training.py:241-277`).
- Key functions: `compute_data_driven_baselines` and temporal baselines (`src/device_anomaly/models/baseline.py:479-548`, `src/device_anomaly/models/baseline.py:427-476`).
- Config knobs: `TrainingConfig.device_type_col`, `timestamp_col` (`src/device_anomaly/pipeline/training.py:70-72`).
- Data schema: numeric feature columns + `Timestamp` for temporal baselines (`src/device_anomaly/pipeline/training.py:253-268`; `src/device_anomaly/models/baseline.py:533-538`).
- Outputs: `baselines.json` artifact (data-driven schema) with percentiles and thresholds (`src/device_anomaly/pipeline/training.py:512-523`; `src/device_anomaly/models/baseline.py:551-566`).
- Baseline API resolves production baselines first and falls back to legacy artifacts (`src/device_anomaly/models/baseline_store.py:1-170`; `src/device_anomaly/api/routes/baselines.py:28-156`).
- Failure points: insufficient samples per metric skip baseline entries (`src/device_anomaly/models/baseline.py:517-520`).

### 4) Training sample generation and labeling
- Synthetic generator injects `is_injected_anomaly` labels (`src/device_anomaly/data_access/synthetic_generator.py:5-142`).
- DW and synthetic experiments use the same feature builder and baseline workflow for training samples (`src/device_anomaly/cli/dw_experiment.py:41-120`; `src/device_anomaly/cli/synthetic_experiment.py:36-120`).

### 5) Train/validation split + cohort stats
- Time-based split is applied in `train_validation_split` (`src/device_anomaly/pipeline/training.py:279-325`).
- Cohort statistics are computed on the training split only and applied to train/val features (`src/device_anomaly/pipeline/training.py:250-340`; `src/device_anomaly/features/cohort_stats.py:43-206`).
- Fallback random split uses `random_state` for determinism (`src/device_anomaly/pipeline/training.py:296-303`).

### 6) Model training
- Isolation Forest training lives in `AnomalyDetectorIsolationForest.fit` and feature selection (`src/device_anomaly/models/anomaly_detector.py:52-186`).
- Training pipeline invokes it in `RealDataTrainingPipeline.train_model` (`src/device_anomaly/pipeline/training.py:346-369`).

### 7) Evaluation and validation
- Evaluation uses `score_dataframe` and optional AUC if labels exist (`src/device_anomaly/pipeline/training.py:371-429`).

### 8) Artifact export, versioning, and loading
- Artifacts saved in `export_artifacts` and `save_model_metadata` (`src/device_anomaly/pipeline/training.py:471-547`).
- Cohort stats persisted as `cohort_stats.json` and referenced in training metadata (`src/device_anomaly/pipeline/training.py:516-547`; `src/device_anomaly/features/cohort_stats.py:74-206`).
- Feature spec + normalization norms are stored in `training_metadata.json` for parity across batch and streaming (`src/device_anomaly/pipeline/training.py:235-560`; `src/device_anomaly/features/device_features.py:21-648`).
- Detector saved/loaded via joblib in `save_model`/`load_model` (`src/device_anomaly/models/anomaly_detector.py:187-218`, `src/device_anomaly/models/anomaly_detector.py:364-444`).
- Latest model selection for inference uses file timestamps (`src/device_anomaly/models/anomaly_detector.py:446-484`).

### 9) Inference, scoring, and post-processing
- Batch scoring endpoint `/api/automation/score` loads data, builds features using the stored feature spec/norms, applies cohort stats, and scores with the latest model (`src/device_anomaly/api/routes/automation.py:462-528`; `src/device_anomaly/features/device_features.py:593-675`).
- Streaming features use the canonical `DeviceFeatureBuilder` over the in-memory buffer and apply stored cohort stats; short history and missing norms are logged (`src/device_anomaly/streaming/feature_computer.py:99-375`).
- Persistence writes anomalies + feature snapshots to results DB (`src/device_anomaly/data_access/anomaly_persistence.py:187-289`).

### 10) API serving and frontend consumption
- API app wires routes with `/api` and `/api/v1` prefixes (`src/device_anomaly/api/main.py:200-226`).
- Frontend uses `/dashboard/*`, `/anomalies`, `/devices` via the API client (`frontend/src/api/client.ts:272-520`).

### 11) Monitoring, logging, and ops
- FastAPI middleware adds request context + API key/tenant checks (`src/device_anomaly/api/main.py:123-200`).
- Prometheus and OpenTelemetry instrumentation configured at startup (`src/device_anomaly/api/main.py:106-120`).
- Streaming feature parity metrics emit PSI-like drift logs and record short-history counts (`src/device_anomaly/streaming/drift_monitor.py:1-214`, `src/device_anomaly/streaming/feature_computer.py:183-337`).
