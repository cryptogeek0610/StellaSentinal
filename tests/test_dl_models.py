"""Tests for Deep Learning anomaly detection models.

This module tests the VAE and Autoencoder based anomaly detectors,
including training, inference, ONNX export, and integration with
the existing pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Skip all tests if PyTorch is not available
torch = pytest.importorskip("torch")


@pytest.fixture
def sample_tabular_data():
    """Generate sample tabular data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 50

    # Normal data
    normal_data = np.random.randn(n_samples, n_features)

    # Add some structure (correlations)
    normal_data[:, 1] = normal_data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
    normal_data[:, 2] = normal_data[:, 0] * 0.5 + normal_data[:, 1] * 0.5

    # Create DataFrame with feature names
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(normal_data, columns=columns)

    return df


@pytest.fixture
def sample_data_with_anomalies():
    """Generate data with known anomalies for testing detection."""
    np.random.seed(42)
    n_normal = 450
    n_anomalies = 50
    n_features = 50

    # Normal data
    normal = np.random.randn(n_normal, n_features)

    # Anomalies - shifted mean and higher variance
    anomalies = np.random.randn(n_anomalies, n_features) * 3 + 5

    # Combine
    data = np.vstack([normal, anomalies])
    labels = np.array([1] * n_normal + [-1] * n_anomalies)

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    df["is_anomaly"] = labels

    return df, labels


class TestVAEArchitecture:
    """Tests for VAE architecture module."""

    def test_vae_creation(self):
        """Test VAE model creation."""
        from device_anomaly.models.vae_architecture import VAE

        model = VAE(input_dim=50, hidden_dims=[128, 64], latent_dim=16)

        assert model.input_dim == 50
        assert model.latent_dim == 16
        assert model.hidden_dims == [128, 64]

    def test_vae_forward_pass(self):
        """Test VAE forward pass."""
        from device_anomaly.models.vae_architecture import VAE

        model = VAE(input_dim=50, hidden_dims=[128, 64], latent_dim=16)
        x = torch.randn(32, 50)

        recon, mu, log_var = model(x)

        assert recon.shape == (32, 50)
        assert mu.shape == (32, 16)
        assert log_var.shape == (32, 16)

    def test_vae_reconstruction_error(self):
        """Test reconstruction error computation."""
        from device_anomaly.models.vae_architecture import VAE

        model = VAE(input_dim=50)
        model.eval()
        x = torch.randn(32, 50)

        error = model.reconstruction_error(x)

        assert error.shape == (32,)
        assert torch.all(error >= 0)

    def test_vae_loss_function(self):
        """Test VAE loss computation."""
        from device_anomaly.models.vae_architecture import VAE

        model = VAE(input_dim=50)
        x = torch.randn(32, 50)

        recon, mu, log_var = model(x)
        total_loss, recon_loss, kl_loss = model.loss_function(x, recon, mu, log_var)

        assert total_loss.dim() == 0  # Scalar
        assert recon_loss >= 0
        assert kl_loss >= 0

    def test_autoencoder_creation(self):
        """Test Autoencoder model creation."""
        from device_anomaly.models.vae_architecture import Autoencoder

        model = Autoencoder(input_dim=50, hidden_dims=[128, 64], latent_dim=16)

        assert model.input_dim == 50
        assert model.latent_dim == 16

    def test_autoencoder_forward_pass(self):
        """Test Autoencoder forward pass."""
        from device_anomaly.models.vae_architecture import Autoencoder

        model = Autoencoder(input_dim=50, hidden_dims=[128, 64], latent_dim=16)
        x = torch.randn(32, 50)

        recon = model(x)

        assert recon.shape == (32, 50)


class TestVAEDetector:
    """Tests for VAE detector wrapper."""

    def test_vae_detector_creation(self):
        """Test VAE detector creation."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(
            latent_dim=16,
            hidden_dims=[128, 64],
            epochs=5,
        )
        detector = VAEDetector(config)

        assert detector.config.latent_dim == 16
        assert not detector.is_fitted

    def test_vae_detector_fit(self, sample_tabular_data):
        """Test VAE detector training."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(
            latent_dim=16,
            hidden_dims=[64, 32],
            epochs=5,
            batch_size=64,
        )
        detector = VAEDetector(config)

        metrics = detector.fit(sample_tabular_data)

        assert detector.is_fitted
        assert len(metrics.train_loss_history) > 0
        assert metrics.best_val_loss < float("inf")

    def test_vae_detector_score(self, sample_tabular_data):
        """Test VAE detector scoring."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        scores = detector.score(sample_tabular_data)

        assert len(scores) == len(sample_tabular_data)
        assert np.all(scores >= 0)  # Reconstruction error is non-negative

    def test_vae_detector_predict(self, sample_tabular_data):
        """Test VAE detector prediction."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        predictions = detector.predict(sample_tabular_data)

        assert len(predictions) == len(sample_tabular_data)
        assert set(np.unique(predictions)).issubset({-1, 1})

    def test_vae_detector_score_dataframe(self, sample_tabular_data):
        """Test score_dataframe method."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        df_scored = detector.score_dataframe(sample_tabular_data)

        assert "anomaly_score" in df_scored.columns
        assert "anomaly_label" in df_scored.columns

    def test_vae_detector_save_load(self, sample_tabular_data):
        """Test model save and load."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "vae_model"
            detector.save_model(model_path)

            # Load
            loaded = VAEDetector.load_model(model_path.with_suffix(".pt"))

            assert loaded.is_fitted
            assert loaded._feature_cols == detector._feature_cols

            # Scores should match
            original_scores = detector.score(sample_tabular_data)
            loaded_scores = loaded.score(sample_tabular_data)

            np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)

    def test_vae_detector_anomaly_detection(self, sample_data_with_anomalies):
        """Test that VAE detects anomalies better than random."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        df, true_labels = sample_data_with_anomalies

        # Train only on normal data
        normal_df = df[df["is_anomaly"] == 1].drop(columns=["is_anomaly"])

        config = VAEDetectorConfig(epochs=20, batch_size=64, contamination=0.1)
        detector = VAEDetector(config)
        detector.fit(normal_df)

        # Score all data
        test_df = df.drop(columns=["is_anomaly"])
        scores = detector.score(test_df)

        # Anomalies should have higher reconstruction error
        normal_mask = true_labels == 1
        anomaly_mask = true_labels == -1

        mean_normal_score = np.mean(scores[normal_mask])
        mean_anomaly_score = np.mean(scores[anomaly_mask])

        assert mean_anomaly_score > mean_normal_score, (
            f"Anomaly score ({mean_anomaly_score:.4f}) should be higher than normal ({mean_normal_score:.4f})"
        )

    def test_vae_get_latent_representation(self, sample_tabular_data):
        """Test getting latent representations."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(latent_dim=16, epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        latents = detector.get_latent_representation(sample_tabular_data)

        assert latents.shape == (len(sample_tabular_data), 16)

    def test_vae_get_feature_reconstruction_errors(self, sample_tabular_data):
        """Test getting per-feature reconstruction errors."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        feature_errors = detector.get_feature_reconstruction_errors(sample_tabular_data)

        # Should have same shape as input features
        assert feature_errors.shape[0] == len(sample_tabular_data)
        assert len(feature_errors.columns) == len(detector._feature_cols)


class TestDLPreprocessor:
    """Tests for DL feature preprocessor."""

    def test_preprocessor_fit_transform(self, sample_tabular_data):
        """Test preprocessor fit and transform."""
        from device_anomaly.features.dl_preprocessor import DLFeaturePreprocessor

        preprocessor = DLFeaturePreprocessor()
        X = preprocessor.fit_transform(sample_tabular_data)

        assert X.dtype == np.float32
        assert X.shape[0] == len(sample_tabular_data)
        assert preprocessor.is_fitted

    def test_preprocessor_transform_only(self, sample_tabular_data):
        """Test transform without fit raises error."""
        from device_anomaly.features.dl_preprocessor import DLFeaturePreprocessor

        preprocessor = DLFeaturePreprocessor()

        with pytest.raises(RuntimeError):
            preprocessor.transform(sample_tabular_data)

    def test_preprocessor_handles_missing_values(self):
        """Test preprocessor handles NaN values."""
        from device_anomaly.features.dl_preprocessor import DLFeaturePreprocessor

        # Create data with missing values
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan, 4, 5],
                "b": [1, np.nan, 3, 4, 5],
                "c": [1, 2, 3, 4, 5],
            }
        )

        preprocessor = DLFeaturePreprocessor()
        X = preprocessor.fit_transform(df)

        assert not np.any(np.isnan(X))

    def test_preprocessor_save_load(self, sample_tabular_data):
        """Test preprocessor save and load."""
        from device_anomaly.features.dl_preprocessor import DLFeaturePreprocessor

        preprocessor = DLFeaturePreprocessor()
        preprocessor.fit(sample_tabular_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preprocessor.pkl"
            preprocessor.save(str(path))

            loaded = DLFeaturePreprocessor.load(str(path))

            assert loaded.is_fitted
            assert loaded.feature_cols == preprocessor.feature_cols


class TestPyTorchONNXExport:
    """Tests for PyTorch to ONNX export."""

    @pytest.mark.skipif(
        not pytest.importorskip("onnx", reason="ONNX not available"), reason="ONNX not available"
    )
    def test_vae_onnx_export(self, sample_tabular_data):
        """Test VAE export to ONNX."""
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "vae_model"
            paths = detector.save_model(model_path, export_onnx=True)

            assert "onnx" in paths
            assert paths["onnx"].exists()

    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not available"),
        reason="ONNX Runtime not available",
    )
    def test_onnx_inference(self, sample_tabular_data):
        """Test ONNX inference matches PyTorch."""
        from device_anomaly.models.pytorch_onnx_exporter import ONNXVAEInference
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "vae_model"
            paths = detector.save_model(model_path, export_onnx=True)

            # PyTorch scores
            pt_scores = detector.score(sample_tabular_data)

            # ONNX inference
            onnx_inference = ONNXVAEInference(paths["onnx"])
            X = detector._prepare_inference_data(sample_tabular_data)
            onnx_scores = onnx_inference.reconstruction_error(X)

            # Should be very close
            np.testing.assert_allclose(pt_scores, onnx_scores, rtol=1e-4, atol=1e-5)


class TestPyTorchInferenceEngine:
    """Tests for PyTorch inference engine."""

    def test_engine_from_checkpoint(self, sample_tabular_data):
        """Test loading engine from checkpoint."""
        from device_anomaly.models.pytorch_inference import PyTorchInferenceEngine
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "vae_model"
            detector.save_model(model_path)

            engine = PyTorchInferenceEngine.from_checkpoint(model_path.with_suffix(".pt"))

            X = detector._prepare_inference_data(sample_tabular_data)
            scores = engine.score_samples(X)

            assert len(scores) == len(sample_tabular_data)

    def test_engine_predict(self, sample_tabular_data):
        """Test engine prediction."""
        from device_anomaly.models.pytorch_inference import PyTorchInferenceEngine
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "vae_model"
            detector.save_model(model_path)

            engine = PyTorchInferenceEngine.from_checkpoint(model_path.with_suffix(".pt"))

            X = detector._prepare_inference_data(sample_tabular_data)
            predictions = engine.predict(X)

            assert len(predictions) == len(sample_tabular_data)
            assert set(np.unique(predictions)).issubset({-1, 1})

    def test_engine_metrics(self, sample_tabular_data):
        """Test inference metrics collection."""
        from device_anomaly.models.pytorch_inference import PyTorchInferenceEngine
        from device_anomaly.models.vae_detector import VAEDetector, VAEDetectorConfig

        config = VAEDetectorConfig(epochs=5, batch_size=64)
        detector = VAEDetector(config)
        detector.fit(sample_tabular_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "vae_model"
            detector.save_model(model_path)

            engine = PyTorchInferenceEngine.from_checkpoint(model_path.with_suffix(".pt"))

            X = detector._prepare_inference_data(sample_tabular_data)
            engine.score_samples(X)

            metrics = engine.get_metrics()

            assert metrics.total_samples == len(sample_tabular_data)
            assert metrics.total_time_ms > 0


class TestDLConfig:
    """Tests for DL configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        from device_anomaly.config.dl_config import DLConfig

        config = DLConfig()

        assert config.model.model_type == "vae"
        assert config.model.latent_dim == 32
        assert config.training.epochs == 100

    def test_config_to_detector_config(self):
        """Test conversion to detector config."""
        from device_anomaly.config.dl_config import DLConfig

        config = DLConfig()
        config.model.latent_dim = 64
        config.training.epochs = 200

        detector_config = config.to_detector_config()

        assert detector_config.latent_dim == 64
        assert detector_config.epochs == 200

    def test_preset_configs(self):
        """Test preset configurations."""
        from device_anomaly.config.dl_config import DL_PRESETS, get_preset

        for preset_name in DL_PRESETS.keys():
            config = get_preset(preset_name)
            assert config is not None
            assert config.model.model_type in ["vae", "autoencoder"]

    def test_invalid_preset(self):
        """Test invalid preset raises error."""
        from device_anomaly.config.dl_config import get_preset

        with pytest.raises(KeyError):
            get_preset("nonexistent_preset")
