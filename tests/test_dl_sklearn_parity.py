"""Parity tests between Deep Learning and sklearn anomaly detectors.

These tests ensure that DL models (VAE) produce results that correlate
with sklearn models (IsolationForest) and can be used interchangeably
in the pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Skip tests if PyTorch is not available
torch = pytest.importorskip("torch")


@pytest.fixture
def sample_device_telemetry():
    """Generate sample device telemetry data similar to real data."""
    np.random.seed(42)
    n_devices = 100
    n_days = 30

    data = []
    for device_id in range(n_devices):
        for day in range(n_days):
            row = {
                "DeviceId": device_id,
                "day": day,
                # Battery metrics
                "TotalBatteryLevelDrop": np.random.exponential(20) + 10,
                "TotalDischargeTime_Sec": np.random.exponential(3600) + 1800,
                "ChargePatternBadCount": np.random.poisson(2),
                # Signal metrics
                "AvgSignal": np.random.uniform(-90, -60),
                "TotalDropCnt": np.random.poisson(5),
                "TotalSignalReadings": np.random.randint(100, 500),
                # Usage metrics
                "AppVisitCount": np.random.poisson(50),
                "AppForegroundTime": np.random.exponential(3600),
                "CrashCount": np.random.poisson(0.5),
                # Storage metrics
                "AvailableStorage": np.random.uniform(1e9, 30e9),
                "TotalStorage": 32e9,
            }
            data.append(row)

    df = pd.DataFrame(data)

    # Add derived features
    df["BatteryDrainPerHour"] = df["TotalBatteryLevelDrop"] / (
        df["TotalDischargeTime_Sec"] / 3600 + 1
    )
    df["DropRate"] = df["TotalDropCnt"] / (df["TotalSignalReadings"] + 1)
    df["CrashRate"] = df["CrashCount"] / (df["AppVisitCount"] + 1)
    df["StorageUtilization"] = 1 - (df["AvailableStorage"] / df["TotalStorage"])

    return df


@pytest.fixture
def sample_data_with_anomalies_for_parity():
    """Generate data with clear anomalies for parity testing."""
    np.random.seed(42)
    n_normal = 900
    n_anomalies = 100
    n_features = 30

    # Normal data - centered around 0
    normal = np.random.randn(n_normal, n_features) * 0.5

    # Anomalies - clear outliers
    anomalies = np.random.randn(n_anomalies, n_features) * 2 + 4

    data = np.vstack([normal, anomalies])
    true_labels = np.array([1] * n_normal + [-1] * n_anomalies)

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)

    return df, true_labels


class TestScoreCorrelation:
    """Test that DL and sklearn scores correlate."""

    def test_vae_if_score_correlation(self, sample_data_with_anomalies_for_parity):
        """Test that VAE and IsolationForest scores correlate."""
        from device_anomaly.models.anomaly_detector import (
            AnomalyDetectorConfig,
            AnomalyDetectorIsolationForest,
        )
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        df, true_labels = sample_data_with_anomalies_for_parity

        # Train IsolationForest
        if_config = AnomalyDetectorConfig(contamination=0.1)
        if_detector = AnomalyDetectorIsolationForest(if_config)
        if_detector.fit(df)
        if_scores = if_detector.score(df)

        # Train VAE
        vae_config = VAEDetectorConfig(
            epochs=30,
            batch_size=64,
            contamination=0.1,
            hidden_dims=[64, 32],
            latent_dim=8,
        )
        vae_detector = VAEDetector(vae_config)
        vae_detector.fit(df)
        vae_scores = vae_detector.score(df)

        # Note: IF scores are lower for anomalies, VAE scores are higher
        # So we expect negative correlation
        correlation, p_value = stats.spearmanr(-if_scores, vae_scores)

        # Should have meaningful correlation
        assert abs(correlation) > 0.3, f"Expected correlation > 0.3, got {correlation:.3f}"

    def test_both_detect_clear_anomalies(self, sample_data_with_anomalies_for_parity):
        """Test that both methods detect clear anomalies."""
        from device_anomaly.models.anomaly_detector import (
            AnomalyDetectorConfig,
            AnomalyDetectorIsolationForest,
        )
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        df, true_labels = sample_data_with_anomalies_for_parity
        n_normal = 900

        # Train both on the full dataset (unsupervised)
        if_config = AnomalyDetectorConfig(contamination=0.1)
        if_detector = AnomalyDetectorIsolationForest(if_config)
        if_detector.fit(df)

        vae_config = VAEDetectorConfig(epochs=30, batch_size=64, contamination=0.1)
        vae_detector = VAEDetector(vae_config)
        vae_detector.fit(df)

        # Get predictions
        if_preds = if_detector.predict(df)
        vae_preds = vae_detector.predict(df)

        # Check that both detect more anomalies in the anomaly region
        if_anomaly_rate_normal = (if_preds[:n_normal] == -1).mean()
        if_anomaly_rate_anomaly = (if_preds[n_normal:] == -1).mean()

        vae_anomaly_rate_normal = (vae_preds[:n_normal] == -1).mean()
        vae_anomaly_rate_anomaly = (vae_preds[n_normal:] == -1).mean()

        # Both should detect more anomalies in the anomaly region
        assert if_anomaly_rate_anomaly > if_anomaly_rate_normal, (
            f"IF should detect more anomalies in anomaly region: {if_anomaly_rate_anomaly:.3f} vs {if_anomaly_rate_normal:.3f}"
        )

        assert vae_anomaly_rate_anomaly > vae_anomaly_rate_normal, (
            f"VAE should detect more anomalies in anomaly region: {vae_anomaly_rate_anomaly:.3f} vs {vae_anomaly_rate_normal:.3f}"
        )


class TestInterfaceCompatibility:
    """Test that DL detectors have compatible interfaces."""

    def test_same_methods(self, sample_device_telemetry):
        """Test that VAE has same core methods as IsolationForest detector."""
        from device_anomaly.models.anomaly_detector import AnomalyDetectorIsolationForest
        from device_anomaly.models.vae_detector import VAEDetector

        if_detector = AnomalyDetectorIsolationForest()
        vae_detector = VAEDetector()

        # Check core methods exist
        core_methods = ["fit", "score", "predict", "score_dataframe", "save_model"]

        for method in core_methods:
            assert hasattr(if_detector, method), f"IF missing method: {method}"
            assert hasattr(vae_detector, method), f"VAE missing method: {method}"

    def test_score_dataframe_output_format(self, sample_device_telemetry):
        """Test that score_dataframe returns same columns."""
        from device_anomaly.models.anomaly_detector import AnomalyDetectorIsolationForest
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        # Subset of data for faster test
        df = sample_device_telemetry.head(500)

        if_detector = AnomalyDetectorIsolationForest()
        if_detector.fit(df)
        if_result = if_detector.score_dataframe(df)

        vae_config = VAEDetectorConfig(epochs=5, batch_size=64)
        vae_detector = VAEDetector(vae_config)
        vae_detector.fit(df)
        vae_result = vae_detector.score_dataframe(df)

        # Both should have same anomaly columns
        assert "anomaly_score" in if_result.columns
        assert "anomaly_label" in if_result.columns
        assert "anomaly_score" in vae_result.columns
        assert "anomaly_label" in vae_result.columns

        # Labels should be same type
        assert set(if_result["anomaly_label"].unique()).issubset({-1, 1})
        assert set(vae_result["anomaly_label"].unique()).issubset({-1, 1})

    def test_metadata_compatibility(self, sample_device_telemetry):
        """Test that both provide compatible metadata."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        df = sample_device_telemetry.head(500)

        vae_config = VAEDetectorConfig(epochs=5, batch_size=64)
        vae_detector = VAEDetector(vae_config)
        vae_detector.fit(df)

        metadata = vae_detector.get_metadata()

        # Check required metadata fields
        assert "name" in metadata
        assert "type" in metadata
        assert "is_fitted" in metadata
        assert "feature_count" in metadata


class TestSaveLoadCompatibility:
    """Test model persistence compatibility."""

    def test_loaded_model_same_scores(self, sample_device_telemetry):
        """Test that loaded model produces same scores."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        df = sample_device_telemetry.head(500)

        vae_config = VAEDetectorConfig(epochs=5, batch_size=64)
        vae_detector = VAEDetector(vae_config)
        vae_detector.fit(df)

        original_scores = vae_detector.score(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "vae_model"
            vae_detector.save_model(model_path)

            loaded_detector = VAEDetector.load_model(model_path.with_suffix(".pt"))
            loaded_scores = loaded_detector.score(df)

        np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)


class TestAnomalyDetectionQuality:
    """Test anomaly detection quality compared to sklearn."""

    def test_vae_vs_if_on_synthetic_anomalies(self):
        """Compare VAE and IF on synthetic anomalies."""
        from sklearn.metrics import roc_auc_score

        from device_anomaly.models.anomaly_detector import (
            AnomalyDetectorConfig,
            AnomalyDetectorIsolationForest,
        )
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        np.random.seed(42)

        # Generate data
        n_normal = 800
        n_anomalies = 200
        n_features = 20

        normal = np.random.randn(n_normal, n_features)
        anomalies = np.random.randn(n_anomalies, n_features) * 2.5 + 3

        data = np.vstack([normal, anomalies])
        true_labels = np.array([0] * n_normal + [1] * n_anomalies)  # 0=normal, 1=anomaly

        columns = [f"f{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=columns)

        # Train IsolationForest
        if_config = AnomalyDetectorConfig(contamination=0.2)
        if_detector = AnomalyDetectorIsolationForest(if_config)
        if_detector.fit(df)
        if_scores = -if_detector.score(df)  # Negate so higher = more anomalous

        # Train VAE
        vae_config = VAEDetectorConfig(epochs=50, batch_size=64, contamination=0.2)
        vae_detector = VAEDetector(vae_config)
        vae_detector.fit(df)
        vae_scores = vae_detector.score(df)

        # Compute AUC
        if_auc = roc_auc_score(true_labels, if_scores)
        vae_auc = roc_auc_score(true_labels, vae_scores)

        # Both should be reasonably good (>0.7)
        assert if_auc > 0.7, f"IF AUC should be > 0.7, got {if_auc:.3f}"
        assert vae_auc > 0.7, f"VAE AUC should be > 0.7, got {vae_auc:.3f}"

        # VAE should be competitive (within 0.15 of IF)
        assert abs(if_auc - vae_auc) < 0.15, (
            f"VAE AUC ({vae_auc:.3f}) should be within 0.15 of IF AUC ({if_auc:.3f})"
        )


class TestStreamingCompatibility:
    """Test compatibility with streaming pipeline."""

    def test_single_sample_scoring(self, sample_device_telemetry):
        """Test scoring individual samples (streaming use case)."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        df = sample_device_telemetry.head(500)

        vae_config = VAEDetectorConfig(epochs=5, batch_size=64)
        vae_detector = VAEDetector(vae_config)
        vae_detector.fit(df)

        # Score single sample
        single_sample = df.head(1)
        score = vae_detector.score(single_sample)

        assert len(score) == 1
        assert score[0] >= 0  # Reconstruction error is non-negative

    def test_feature_dict_to_score(self, sample_device_telemetry):
        """Test scoring from feature dictionary (streaming format)."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        df = sample_device_telemetry.head(500)

        vae_config = VAEDetectorConfig(epochs=5, batch_size=64)
        vae_detector = VAEDetector(vae_config)
        vae_detector.fit(df)

        # Convert single row to dict (like streaming format)
        row_dict = df.iloc[0].to_dict()

        # Create single-row DataFrame from dict
        single_df = pd.DataFrame([row_dict])

        # Should be able to score
        score = vae_detector.score(single_df)
        assert len(score) == 1
