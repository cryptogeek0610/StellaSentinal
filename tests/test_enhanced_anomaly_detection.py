"""
Tests for Enhanced Anomaly Detection modules.

Tests the new feature engineering, ML models, and pipeline components:
- Location features
- Event features
- System health features
- Temporal features
- Ensemble detector
- SHAP explainer
- Weak labeling
- Predictive detection
- Root cause analysis
- Model governance
- Auto-retraining
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_telemetry_df():
    """Create sample telemetry data for testing."""
    np.random.seed(42)
    n_devices = 5
    n_days = 30

    data = []
    for device_id in range(1, n_devices + 1):
        for day in range(n_days):
            timestamp = datetime.now() - timedelta(days=n_days - day)
            data.append(
                {
                    "DeviceId": device_id,
                    "Timestamp": timestamp,
                    "ManufacturerId": device_id % 3,
                    "ModelId": device_id % 5,
                    "OsVersionId": device_id % 2,
                    # Battery metrics
                    "TotalBatteryLevelDrop": np.random.uniform(10, 50),
                    "BatteryDrainPerHour": np.random.uniform(1, 10),
                    "TotalDischargeTime_Sec": np.random.uniform(3600, 36000),
                    # App metrics
                    "AppForegroundTime": np.random.uniform(1000, 10000),
                    "CrashCount": np.random.poisson(1),
                    "ANRCount": np.random.poisson(0.5),
                    # Network metrics
                    "Download": np.random.uniform(1e6, 1e9),
                    "Upload": np.random.uniform(1e5, 1e8),
                    "AvgSignalStrength": np.random.uniform(-100, -50),
                    "TotalDropCnt": np.random.poisson(5),
                    "TotalSignalReadings": np.random.randint(100, 1000),
                    # Location
                    "Latitude": 40.7 + np.random.uniform(-0.1, 0.1),
                    "Longitude": -74.0 + np.random.uniform(-0.1, 0.1),
                    "SignalStrength": np.random.uniform(-100, -50),
                    # System health
                    "CpuUsage": np.random.uniform(10, 90),
                    "AvailableRAM": np.random.uniform(1e9, 4e9),
                    "TotalRAM": 4e9,
                    "AvailableStorage": np.random.uniform(1e10, 5e10),
                    "TotalStorage": 6e10,
                    "Temperature": np.random.uniform(30, 50),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_event_df():
    """Create sample event/log data for testing."""
    np.random.seed(42)
    n_events = 100

    event_types = ["info", "warning", "error", "crash", "debug"]
    severities = ["low", "medium", "high", "critical"]

    data = []
    for i in range(n_events):
        data.append(
            {
                "DeviceId": np.random.randint(1, 6),
                "Timestamp": datetime.now() - timedelta(hours=np.random.randint(0, 720)),
                "EventType": np.random.choice(event_types),
                "Severity": np.random.choice(severities),
                "Message": f"Event message {i}",
            }
        )

    return pd.DataFrame(data)


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================


class TestLocationFeatures:
    """Tests for location feature engineering."""

    def test_location_feature_builder_init(self):
        from device_anomaly.features.location_features import LocationFeatureBuilder

        builder = LocationFeatureBuilder()
        assert builder.signal_threshold_dbm == -80.0

    def test_transform_with_gps_data(self, sample_telemetry_df):
        from device_anomaly.features.location_features import LocationFeatureBuilder

        builder = LocationFeatureBuilder()
        result = builder.transform(sample_telemetry_df)

        assert "point_distance_km" in result.columns
        assert "dead_zone_time_pct" in result.columns

    def test_haversine_distance(self):
        from device_anomaly.features.location_features import haversine_distance

        # NYC to LA approximate distance
        lat1 = np.array([40.7128])
        lon1 = np.array([-74.0060])
        lat2 = np.array([34.0522])
        lon2 = np.array([-118.2437])

        distance = haversine_distance(lat1, lon1, lat2, lon2)
        assert 3900 < distance[0] < 4000  # ~3,944 km


class TestEventFeatures:
    """Tests for event feature engineering."""

    def test_event_feature_builder_init(self):
        from device_anomaly.features.event_features import EventFeatureBuilder

        builder = EventFeatureBuilder(window_days=7)
        assert builder.window_days == 7

    def test_transform_with_events(self, sample_event_df):
        from device_anomaly.features.event_features import EventFeatureBuilder

        builder = EventFeatureBuilder()
        result = builder.transform(sample_event_df)

        assert "total_event_count" in result.columns
        assert "error_event_ratio" in result.columns

    def test_crash_detection(self, sample_event_df):
        from device_anomaly.features.event_features import EventFeatureBuilder

        builder = EventFeatureBuilder()
        result = builder.transform(sample_event_df)

        assert "crash_rate" in result.columns
        assert result["crash_rate"].notna().any()


class TestSystemHealthFeatures:
    """Tests for system health feature engineering."""

    def test_system_health_builder_init(self):
        from device_anomaly.features.system_health_features import SystemHealthFeatureBuilder

        builder = SystemHealthFeatureBuilder(cpu_spike_threshold=95.0)
        assert builder.cpu_spike_threshold == 95.0

    def test_transform_with_health_data(self, sample_telemetry_df):
        from device_anomaly.features.system_health_features import SystemHealthFeatureBuilder

        builder = SystemHealthFeatureBuilder()
        result = builder.transform(sample_telemetry_df)

        assert "cpu_usage_avg" in result.columns
        assert "system_health_score" in result.columns


class TestTemporalFeatures:
    """Tests for temporal feature engineering."""

    def test_temporal_builder_init(self):
        from device_anomaly.features.temporal_features import TemporalFeatureBuilder

        builder = TemporalFeatureBuilder(seasonal_period=24)
        assert builder.seasonal_period == 24

    def test_temporal_pattern_features(self, sample_telemetry_df):
        from device_anomaly.features.temporal_features import TemporalFeatureBuilder

        builder = TemporalFeatureBuilder()
        result = builder.transform(sample_telemetry_df)

        assert "hour_sin" in result.columns
        assert "is_weekend" in result.columns


# =============================================================================
# ML MODEL TESTS
# =============================================================================


class TestEnsembleDetector:
    """Tests for ensemble anomaly detector."""

    def test_ensemble_config(self):
        from device_anomaly.models.ensemble_detector import EnsembleConfig

        config = EnsembleConfig(contamination=0.1)
        assert config.contamination == 0.1

    def test_ensemble_fit_predict(self, sample_telemetry_df):
        from device_anomaly.features.device_features import DeviceFeatureBuilder
        from device_anomaly.models.ensemble_detector import EnsembleAnomalyDetector, EnsembleConfig

        # Build features
        builder = DeviceFeatureBuilder(compute_cohort=False)
        df_features = builder.transform(sample_telemetry_df)

        # Train ensemble
        config = EnsembleConfig(contamination=0.1)
        detector = EnsembleAnomalyDetector(config=config)
        detector.fit(df_features)

        # Score
        result = detector.score_dataframe(df_features)

        assert "ensemble_score" in result.columns
        assert "ensemble_label" in result.columns
        assert (result["ensemble_label"] == -1).sum() > 0

    def test_ensemble_save_load(self, sample_telemetry_df, tmp_path):
        from device_anomaly.features.device_features import DeviceFeatureBuilder
        from device_anomaly.models.ensemble_detector import EnsembleAnomalyDetector

        builder = DeviceFeatureBuilder(compute_cohort=False)
        df_features = builder.transform(sample_telemetry_df)

        detector = EnsembleAnomalyDetector()
        detector.fit(df_features)

        # Save
        model_path = tmp_path / "ensemble_model"
        detector.save_model(model_path)

        # Load
        loaded = EnsembleAnomalyDetector.load_model(model_path.with_suffix(".pkl"))
        assert len(loaded.feature_cols) == len(detector.feature_cols)


class TestSHAPExplainer:
    """Tests for SHAP explainability."""

    def test_explainer_init(self):
        from device_anomaly.models.explainer import AnomalyExplainer, ExplanationConfig

        # Mock model with decision_function
        class MockModel:
            def decision_function(self, X):
                return np.random.randn(len(X))

        config = ExplanationConfig(top_features=3)
        explainer = AnomalyExplainer(MockModel(), ["f1", "f2", "f3"], config)
        assert explainer.config.top_features == 3

    def test_fallback_explanations(self, sample_telemetry_df):
        from sklearn.ensemble import IsolationForest

        from device_anomaly.models.explainer import AnomalyExplainer

        # Train a simple model
        feature_cols = ["TotalBatteryLevelDrop", "CrashCount", "TotalDropCnt"]
        X = sample_telemetry_df[feature_cols].values

        model = IsolationForest(n_estimators=10, random_state=42)
        model.fit(X)

        # Create explainer (SHAP may not be available)
        explainer = AnomalyExplainer(model, feature_cols)
        explanations = explainer.explain(X[:5])

        assert len(explanations) == 5
        assert all(e.device_id is not None for e in explanations)


class TestWeakLabeling:
    """Tests for weak label generation."""

    def test_weak_label_generator_init(self):
        from device_anomaly.models.weak_labeling import WeakLabelGenerator

        generator = WeakLabelGenerator()
        assert len(generator._heuristic_rules) > 0

    def test_heuristic_labels(self, sample_telemetry_df):
        from device_anomaly.models.weak_labeling import WeakLabelGenerator

        # Add derived features
        sample_telemetry_df["DropRate"] = sample_telemetry_df["TotalDropCnt"] / (
            sample_telemetry_df["TotalSignalReadings"] + 1
        )

        generator = WeakLabelGenerator()
        result = generator.generate_heuristic_labels(sample_telemetry_df)

        assert "heuristic_label" in result.columns
        assert "heuristic_confidence" in result.columns


class TestPredictiveDetector:
    """Tests for predictive anomaly detection."""

    def test_predictive_config(self):
        from device_anomaly.models.predictive_detector import PredictiveConfig

        config = PredictiveConfig(battery_critical_level=15.0)
        assert config.battery_critical_level == 15.0

    def test_battery_failure_prediction(self):
        from device_anomaly.models.predictive_detector import PredictiveAnomalyDetector

        detector = PredictiveAnomalyDetector()

        # High drain rate
        drain_history = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
        result = detector.predict_battery_failure(
            drain_history=drain_history,
            current_level=50.0,
            shift_duration_hours=8.0,
        )

        assert result.prediction_type == "battery_failure"
        assert result.will_occur is True  # Will drain ~80% in 8 hours

    def test_storage_exhaustion_prediction(self):
        from device_anomaly.models.predictive_detector import PredictiveAnomalyDetector

        detector = PredictiveAnomalyDetector()

        # Declining storage
        storage_history = np.array([5e9, 4.5e9, 4e9, 3.5e9, 3e9, 2.5e9, 2e9])
        result = detector.predict_storage_exhaustion(
            storage_history=storage_history,
            total_storage=10e9,
        )

        assert result.prediction_type == "storage_exhaustion"
        assert result.time_until is not None


# =============================================================================
# PIPELINE TESTS
# =============================================================================


class TestAutoRetraining:
    """Tests for auto-retraining pipeline."""

    def test_retraining_config(self):
        from device_anomaly.pipeline.auto_retrain import RetrainingConfig

        config = RetrainingConfig(psi_trigger_threshold=0.3)
        assert config.psi_trigger_threshold == 0.3

    def test_evaluate_triggers(self):
        from device_anomaly.pipeline.auto_retrain import AutoRetrainingOrchestrator

        orchestrator = AutoRetrainingOrchestrator()

        # Test no retrain needed
        result = orchestrator.evaluate_triggers(
            drift_metrics={"psi": {"f1": 0.1}},
            current_anomaly_rate=0.05,
        )
        assert result.should_retrain is True  # No previous training

    def test_time_triggers(self):
        from datetime import datetime

        from device_anomaly.pipeline.auto_retrain import AutoRetrainingOrchestrator

        orchestrator = AutoRetrainingOrchestrator()
        orchestrator.last_retrain_date = datetime.now(UTC) - timedelta(days=35)

        result = orchestrator.evaluate_triggers(
            drift_metrics={},
            current_anomaly_rate=0.05,
        )
        assert result.should_retrain is True
        assert "Max interval" in result.reason


class TestModelGovernance:
    """Tests for model governance."""

    def test_registry_init(self, tmp_path):
        from device_anomaly.models.governance import ModelGovernanceRegistry

        registry_path = tmp_path / "registry.json"
        registry = ModelGovernanceRegistry(registry_path)
        assert len(registry._versions) == 0

    def test_register_and_promote(self, tmp_path):
        from device_anomaly.models.governance import ModelGovernanceRegistry

        registry_path = tmp_path / "registry.json"
        registry = ModelGovernanceRegistry(registry_path)

        # Register model
        version = registry.register_model(
            model_path=tmp_path / "model.pkl",
            metrics={"accuracy": 0.95},
            config={"contamination": 0.05},
        )

        assert version.version_id is not None

        # Promote to production
        registry.promote_to_production(version.version_id)
        prod = registry.get_production_version()
        assert prod.version_id == version.version_id

    def test_rollback(self, tmp_path):
        from device_anomaly.models.governance import ModelGovernanceRegistry

        registry_path = tmp_path / "registry.json"
        registry = ModelGovernanceRegistry(registry_path)

        # Register two versions
        v1 = registry.register_model(
            model_path=tmp_path / "model_v1.pkl",
            metrics={},
            config={},
        )
        registry.promote_to_production(v1.version_id)

        v2 = registry.register_model(
            model_path=tmp_path / "model_v2.pkl",
            metrics={},
            config={},
        )
        registry.promote_to_production(v2.version_id)

        # Rollback
        rolled_back = registry.rollback(reason="Test rollback")
        assert rolled_back.version_id == v1.version_id


class TestRootCauseAnalysis:
    """Tests for root cause analysis."""

    def test_analyzer_init(self):
        from device_anomaly.insights.root_cause import RootCauseAnalyzer

        analyzer = RootCauseAnalyzer()
        assert len(analyzer._causal_graph) > 0

    def test_analyze_temporal(self):
        from device_anomaly.insights.root_cause import RootCauseAnalyzer

        analyzer = RootCauseAnalyzer()

        current = {"BatteryDrainPerHour": 15.0, "ScreenOnTime": 5000}
        history = [{"BatteryDrainPerHour": 5.0, "ScreenOnTime": 2000}]

        result = analyzer.analyze(
            anomaly_features=current,
            historical_features=history,
            device_id=1,
            anomaly_type="battery",
        )

        assert len(result.probable_causes) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestExtendedFeatures:
    """Integration tests for extended feature building."""

    def test_build_extended_features(self, sample_telemetry_df):
        from device_anomaly.features.device_features import build_extended_features

        result = build_extended_features(
            sample_telemetry_df,
            include_location=True,
            include_events=False,  # No event data in sample
            include_system_health=True,
            include_temporal=False,  # Skip STL for speed
        )

        assert len(result.columns) > len(sample_telemetry_df.columns)

    def test_feature_summary(self, sample_telemetry_df):
        from device_anomaly.features.device_features import (
            build_extended_features,
            get_extended_feature_summary,
        )

        result = build_extended_features(sample_telemetry_df, include_temporal=False)
        summary = get_extended_feature_summary(result)

        assert "total_columns" in summary
        assert "location_features" in summary
        assert "system_health_features" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
