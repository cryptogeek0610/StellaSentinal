"""
Unit tests for ONNX integration

Tests model export, inference engine switching, and performance metrics.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from device_anomaly.models.anomaly_detector import (
    AnomalyDetectorConfig,
    AnomalyDetectorIsolationForest,
)
from device_anomaly.models.inference_engine import (
    EngineConfig,
    EngineType,
    ExecutionProvider,
    FallbackInferenceEngine,
    ONNXInferenceEngine,
    ScikitLearnEngine,
    create_inference_engine,
)
from device_anomaly.models.onnx_exporter import ONNXExportConfig, ONNXModelExporter, ONNXQuantizer


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)

    # Generate normal data
    normal = np.random.randn(200, 5)

    # Generate anomalies
    anomalies = np.random.randn(20, 5) * 3 + 5

    data = np.vstack([normal, anomalies])
    np.random.shuffle(data)

    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(5)])
    return df


@pytest.fixture
def trained_model(sample_data):
    """Train a simple IsolationForest model"""
    detector = AnomalyDetectorIsolationForest(
        config=AnomalyDetectorConfig(n_estimators=50, contamination=0.1)
    )
    detector.fit(sample_data)
    return detector


class TestONNXExport:
    """Test ONNX model export functionality"""

    def test_export_isolation_forest(self, trained_model):
        """Test exporting IsolationForest to ONNX"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.onnx"

            exporter = ONNXModelExporter(ONNXExportConfig())
            onnx_path = exporter.export_model(
                model=trained_model.model,
                feature_count=len(trained_model.feature_cols),
                output_path=output_path,
                validate=True,
            )

            assert onnx_path.exists()
            assert onnx_path.suffix == ".onnx"
            assert onnx_path.stat().st_size > 0

    def test_export_with_metadata(self, trained_model):
        """Test ONNX export with custom metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_model.onnx"

            metadata = {
                "model_version": "1.0.0",
                "n_features": len(trained_model.feature_cols),
                "contamination": str(trained_model.config.contamination),
            }

            exporter = ONNXModelExporter()
            onnx_path = exporter.export_with_metadata(
                model=trained_model.model,
                feature_count=len(trained_model.feature_cols),
                output_path=output_path,
                metadata=metadata,
            )

            assert onnx_path.exists()

            # Verify metadata
            import onnx

            onnx_model = onnx.load(str(onnx_path))
            metadata_dict = {prop.key: prop.value for prop in onnx_model.metadata_props}

            assert "model_version" in metadata_dict
            assert metadata_dict["model_version"] == "1.0.0"

    def test_quantization(self, trained_model):
        """Test ONNX model quantization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp32_path = Path(tmpdir) / "model_fp32.onnx"
            int8_path = Path(tmpdir) / "model_int8.onnx"

            # Export FP32 model
            exporter = ONNXModelExporter()
            exporter.export_model(
                model=trained_model.model,
                feature_count=len(trained_model.feature_cols),
                output_path=fp32_path,
            )

            # Quantize to INT8
            ONNXQuantizer.quantize_dynamic(fp32_path, int8_path)

            assert int8_path.exists()

            # Quantized model should exist and be non-empty
            fp32_size = fp32_path.stat().st_size
            int8_size = int8_path.stat().st_size

            assert fp32_size > 0
            assert int8_size > 0


class TestInferenceEngines:
    """Test inference engine abstraction"""

    def test_sklearn_engine(self, trained_model, sample_data):
        """Test ScikitLearn inference engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            import joblib

            joblib.dump(trained_model.model, model_path)

            # Create sklearn engine
            engine = ScikitLearnEngine(model_path)

            # Test predictions
            X = sample_data[trained_model.feature_cols].values
            predictions = engine.predict(X)

            assert len(predictions) == len(X)
            assert set(predictions).issubset({-1, 1})

            # Test scores
            scores = engine.score_samples(X)
            assert len(scores) == len(X)
            assert scores.dtype == np.float64

            # Test engine info
            info = engine.get_engine_info()
            assert info["engine_type"] == "sklearn"
            assert "IsolationForest" in info["model_class"]

    def test_onnx_engine(self, trained_model, sample_data):
        """Test ONNX inference engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"

            # Export to ONNX
            exporter = ONNXModelExporter()
            exporter.export_model(
                model=trained_model.model,
                feature_count=len(trained_model.feature_cols),
                output_path=onnx_path,
            )

            # Create ONNX engine
            config = EngineConfig(
                engine_type=EngineType.ONNX,
                onnx_provider=ExecutionProvider.CPU,
            )
            engine = ONNXInferenceEngine(onnx_path, config)

            # Test predictions
            X = sample_data[trained_model.feature_cols].values
            predictions = engine.predict(X)

            assert len(predictions) == len(X)
            assert set(predictions).issubset({-1, 1})

            # Test engine info
            info = engine.get_engine_info()
            assert info["engine_type"] == "onnx"
            assert info["provider"] == "CPUExecutionProvider"

    def test_engine_prediction_parity(self, trained_model, sample_data):
        """Test that sklearn and ONNX engines produce same results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sklearn_path = Path(tmpdir) / "model.pkl"
            onnx_path = Path(tmpdir) / "model.onnx"

            # Save sklearn model
            import joblib

            joblib.dump(trained_model.model, sklearn_path)

            # Export ONNX model
            exporter = ONNXModelExporter()
            exporter.export_model(
                model=trained_model.model,
                feature_count=len(trained_model.feature_cols),
                output_path=onnx_path,
            )

            # Create both engines
            sklearn_engine = create_inference_engine(sklearn_path, EngineType.SKLEARN)
            onnx_engine = create_inference_engine(
                onnx_path,
                EngineType.ONNX,
                config=EngineConfig(onnx_provider=ExecutionProvider.CPU),
            )

            # Compare predictions
            X = sample_data[trained_model.feature_cols].values

            sklearn_preds = sklearn_engine.predict(X)
            onnx_preds = onnx_engine.predict(X)

            # Predictions should match exactly
            match_rate = np.mean(sklearn_preds == onnx_preds)
            assert match_rate > 0.99, f"Prediction match rate: {match_rate:.2%}"

    def test_fallback_engine(self, trained_model, sample_data):
        """Test fallback engine (ONNX -> sklearn)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sklearn_path = Path(tmpdir) / "model.pkl"
            onnx_path = Path(tmpdir) / "model.onnx"

            # Save sklearn model
            import joblib

            joblib.dump(trained_model.model, sklearn_path)

            # Export ONNX model
            exporter = ONNXModelExporter()
            exporter.export_model(
                model=trained_model.model,
                feature_count=len(trained_model.feature_cols),
                output_path=onnx_path,
            )

            # Create fallback engine
            fallback_engine = FallbackInferenceEngine(onnx_path, sklearn_path)

            # Should use ONNX by default
            info = fallback_engine.get_engine_info()
            assert info["engine_type"] == "onnx"
            assert not info["is_fallback"]

            # Test predictions work
            X = sample_data[trained_model.feature_cols].values
            predictions = fallback_engine.predict(X)
            assert len(predictions) == len(X)


class TestInferenceMetrics:
    """Test inference metrics collection"""

    def test_metrics_collection(self, trained_model, sample_data):
        """Test that engines collect performance metrics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"

            # Export model
            exporter = ONNXModelExporter()
            exporter.export_model(
                model=trained_model.model,
                feature_count=len(trained_model.feature_cols),
                output_path=onnx_path,
            )

            # Create engine with metrics enabled
            config = EngineConfig(collect_metrics=True)
            engine = ONNXInferenceEngine(onnx_path, config)

            # Run inference
            X = sample_data[trained_model.feature_cols].values
            engine.predict(X)

            # Check metrics
            metrics = engine.get_metrics()
            assert metrics is not None
            assert metrics.num_samples == len(X)
            assert metrics.inference_time_ms > 0
            assert metrics.engine_type == "onnx"
            assert metrics.throughput_samples_per_sec > 0


class TestAnomalyDetectorIntegration:
    """Test integration with AnomalyDetectorIsolationForest"""

    def test_save_with_onnx_export(self, sample_data):
        """Test save_model with ONNX export"""
        detector = AnomalyDetectorIsolationForest(config=AnomalyDetectorConfig(n_estimators=50))
        detector.fit(sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "model"

            # Save with ONNX export
            paths = detector.save_model(base_path, export_onnx=True)

            assert "sklearn" in paths
            assert paths["sklearn"].exists()
            assert paths["sklearn"].suffix == ".pkl"

            assert "onnx" in paths
            assert paths["onnx"].exists()
            assert paths["onnx"].suffix == ".onnx"

    def test_save_without_onnx(self, sample_data):
        """Test save_model without ONNX export"""
        detector = AnomalyDetectorIsolationForest(config=AnomalyDetectorConfig(n_estimators=50))
        detector.fit(sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "model"

            # Save without ONNX
            paths = detector.save_model(base_path, export_onnx=False)

            assert "sklearn" in paths
            assert paths["sklearn"].exists()
            assert "onnx" not in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
