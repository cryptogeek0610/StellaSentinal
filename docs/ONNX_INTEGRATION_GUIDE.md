# ONNX Runtime Integration Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Usage Guide](#usage-guide)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Performance Tuning](#performance-tuning)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Introduction

This guide covers the integration of ONNX Runtime into the StellaSentinal anomaly detection system. ONNX Runtime provides significant performance improvements (3-10x faster inference) and enables cross-platform deployment including mobile devices and edge systems.

### What is ONNX?

**ONNX (Open Neural Network Exchange)** is an open format for representing machine learning models. ONNX Runtime is a high-performance inference engine that executes ONNX models across different hardware platforms.

### Benefits

- **Performance:** 3-10x faster inference than scikit-learn
- **Cross-Platform:** Deploy to servers, mobile devices, browsers, and edge systems
- **Framework Agnostic:** Train in PyTorch/TensorFlow, deploy uniformly
- **Hardware Acceleration:** CPU, GPU, and specialized accelerators
- **Optimization:** Graph optimization, quantization, and operator fusion

---

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/yannickweijenberg/Documents/GitHub/AnomalyDetection

# Install ONNX packages
pip install onnxruntime>=1.16.0 skl2onnx>=1.16.0 onnxmltools>=1.12.0 onnx>=1.15.0

# Or use the updated pyproject.toml
pip install -e .
```

### 2. Train and Export Model

```python
from device_anomaly.models.anomaly_detector import (
    AnomalyDetectorIsolationForest,
    AnomalyDetectorConfig
)
import pandas as pd

# Load training data
df_train = pd.read_csv("data/training_data.csv")

# Train model
detector = AnomalyDetectorIsolationForest(
    config=AnomalyDetectorConfig(n_estimators=300)
)
detector.fit(df_train)

# Save with ONNX export
paths = detector.save_model("models/my_model", export_onnx=True)

print(f"Sklearn model: {paths['sklearn']}")
print(f"ONNX model: {paths['onnx']}")
# Output:
# Sklearn model: models/my_model.pkl
# ONNX model: models/my_model.onnx
```

### 3. Use ONNX for Inference

```python
from device_anomaly.models.inference_engine import (
    create_inference_engine,
    EngineType,
    EngineConfig
)
import numpy as np

# Create ONNX inference engine
config = EngineConfig(engine_type=EngineType.ONNX)
engine = create_inference_engine(
    "models/my_model.onnx",
    engine_type=EngineType.ONNX,
    config=config
)

# Prepare features (25 features as example)
features = np.random.randn(100, 25).astype(np.float32)

# Predict
predictions = engine.predict(features)  # Returns: array([ 1,  1, -1,  1, ...])
scores = engine.score_samples(features)  # Returns: array([0.12, 0.45, -0.82, ...])

print(f"Anomalies detected: {sum(predictions == -1)}")
```

### 4. Benchmark Performance

```bash
python scripts/benchmark_onnx.py --batch-sizes 1,100,1000
```

---

## Installation

### System Requirements

**Minimum:**
- Python 3.10+
- 4GB RAM
- 2 CPU cores

**Recommended:**
- Python 3.11
- 8GB RAM
- 4 CPU cores
- Optional: NVIDIA GPU with CUDA 11.8+ for GPU acceleration

### Install ONNX Runtime

#### CPU-Only (Default)

```bash
pip install onnxruntime>=1.16.0
```

#### GPU Support (NVIDIA CUDA)

```bash
# Requires CUDA 11.8 or 12.x
pip install onnxruntime-gpu>=1.16.0
```

#### Check Installation

```python
import onnxruntime as ort

print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

# Expected output:
# ONNX Runtime version: 1.16.3
# Available providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Project Dependencies

All ONNX-related dependencies are in `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    "onnxruntime>=1.16.0",      # ONNX Runtime for inference
    "skl2onnx>=1.16.0",         # scikit-learn to ONNX converter
    "onnxmltools>=1.12.0",      # Additional ML framework converters
    "onnx>=1.15.0",             # ONNX model format library
]
```

---

## Usage Guide

### Exporting Models

#### Basic Export

```python
from device_anomaly.models.onnx_exporter import ONNXModelExporter
from sklearn.ensemble import IsolationForest
import numpy as np

# Train a model
model = IsolationForest(n_estimators=100)
X_train = np.random.randn(1000, 25)
model.fit(X_train)

# Export to ONNX
exporter = ONNXModelExporter()
onnx_path = exporter.export_model(
    model=model,
    feature_count=25,
    output_path="models/isolation_forest.onnx",
    model_name="isolation_forest_v1",
    validate=True  # Verify export correctness
)
```

#### Export with Metadata

```python
metadata = {
    "version": "1.0.0",
    "created_by": "data_science_team",
    "n_features": 25,
    "contamination": "0.03",
    "training_date": "2025-12-27",
}

onnx_path = exporter.export_with_metadata(
    model=model,
    feature_count=25,
    output_path="models/isolation_forest_v1.onnx",
    metadata=metadata
)
```

#### Quantization (INT8)

```python
from device_anomaly.models.onnx_exporter import ONNXQuantizer

# Quantize to INT8 (reduces size by ~75%)
ONNXQuantizer.quantize_dynamic(
    input_path="models/isolation_forest.onnx",
    output_path="models/isolation_forest_int8.onnx",
    weight_type="int8"
)

# Compare sizes
import os
fp32_size = os.path.getsize("models/isolation_forest.onnx") / 1e6
int8_size = os.path.getsize("models/isolation_forest_int8.onnx") / 1e6

print(f"FP32: {fp32_size:.2f} MB")
print(f"INT8: {int8_size:.2f} MB")
print(f"Reduction: {(1 - int8_size/fp32_size)*100:.1f}%")
```

### Inference Engines

#### Using ONNX Engine

```python
from device_anomaly.models.inference_engine import ONNXInferenceEngine, EngineConfig, ExecutionProvider

# CPU inference
config = EngineConfig(
    onnx_provider=ExecutionProvider.CPU,
    intra_op_num_threads=4,
    collect_metrics=True
)

engine = ONNXInferenceEngine("models/my_model.onnx", config)

# Run inference
predictions = engine.predict(features)

# Get metrics
metrics = engine.get_metrics()
print(f"Inference time: {metrics.inference_time_ms:.2f} ms")
print(f"Throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec")
```

#### GPU Acceleration

```python
# GPU inference (requires CUDA)
config = EngineConfig(
    onnx_provider=ExecutionProvider.CUDA,
    enable_graph_optimization=True
)

engine = ONNXInferenceEngine("models/my_model.onnx", config)

# Check active provider
info = engine.get_engine_info()
print(f"Active providers: {info['active_providers']}")
# Output: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

#### Fallback Engine (Production)

```python
from device_anomaly.models.inference_engine import FallbackInferenceEngine

# Automatically falls back to sklearn if ONNX fails
engine = FallbackInferenceEngine(
    onnx_path="models/my_model.onnx",
    sklearn_path="models/my_model.pkl"
)

# Use normally - engine handles fallback internally
predictions = engine.predict(features)

# Check which engine is active
info = engine.get_engine_info()
if info['is_fallback']:
    print("Using sklearn fallback")
else:
    print("Using ONNX engine")
```

### Configuration

#### Environment Variables

```bash
# Enable ONNX export during training
export ONNX_ENABLED=true

# Set inference engine (sklearn or onnx)
export ONNX_ENGINE=onnx

# Set execution provider (cpu, cuda, tensorrt)
export ONNX_PROVIDER=cuda

# Set thread count
export ONNX_THREADS=8

# Enable quantization
export ONNX_QUANTIZE=true

# Enable profiling
export ONNX_PROFILING=true

# Set model directory
export ONNX_MODEL_DIR=/app/models/onnx
```

#### Programmatic Configuration

```python
from device_anomaly.config.onnx_config import ONNXConfig, set_onnx_config
from device_anomaly.models.inference_engine import EngineType, ExecutionProvider

# Create configuration
config = ONNXConfig()

# Configure export
config.export.enabled = True
config.export.export_quantized = True
config.export.optimize = True

# Configure inference
config.inference.engine_type = EngineType.ONNX
config.inference.onnx_provider = ExecutionProvider.CUDA
config.inference.intra_op_num_threads = 8

# Set global config
set_onnx_config(config)
```

---

## API Reference

### ONNXModelExporter

```python
class ONNXModelExporter:
    """Export scikit-learn models to ONNX format"""

    def __init__(self, config: Optional[ONNXExportConfig] = None)

    def export_model(
        self,
        model: BaseEstimator,
        feature_count: int,
        output_path: str | Path,
        model_name: str = "anomaly_model",
        validate: bool = True,
    ) -> Path
        """
        Export a scikit-learn model to ONNX.

        Args:
            model: Trained scikit-learn model
            feature_count: Number of input features
            output_path: Where to save the .onnx file
            model_name: Name for the ONNX model
            validate: Whether to validate predictions match

        Returns:
            Path to exported ONNX model
        """

    def export_with_metadata(
        self,
        model: BaseEstimator,
        feature_count: int,
        output_path: str | Path,
        metadata: dict[str, Any],
    ) -> Path
        """Export model with embedded metadata"""
```

### InferenceEngine

```python
class InferenceEngine(ABC):
    """Base class for inference engines"""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 = normal, -1 = anomaly)"""

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more normal)"""

    @abstractmethod
    def get_engine_info(self) -> dict[str, Any]:
        """Return engine information"""

    def get_metrics(self) -> Optional[InferenceMetrics]:
        """Get latest inference metrics"""
```

### Factory Function

```python
def create_inference_engine(
    model_path: str | Path,
    engine_type: EngineType = EngineType.SKLEARN,
    config: Optional[EngineConfig] = None,
) -> InferenceEngine:
    """
    Create inference engine based on type.

    Args:
        model_path: Path to model (.pkl or .onnx)
        engine_type: EngineType.SKLEARN or EngineType.ONNX
        config: Engine configuration

    Returns:
        Configured inference engine

    Example:
        engine = create_inference_engine(
            "models/model.onnx",
            engine_type=EngineType.ONNX,
            config=EngineConfig(onnx_provider=ExecutionProvider.CUDA)
        )
    """
```

---

## Performance Tuning

### Thread Configuration

```python
# Optimize for throughput (many parallel requests)
config = EngineConfig(
    intra_op_num_threads=1,   # 1 thread per operation
    inter_op_num_threads=16   # 16 parallel operations
)

# Optimize for latency (single request)
config = EngineConfig(
    intra_op_num_threads=8,   # 8 threads per operation
    inter_op_num_threads=1    # Sequential operations
)
```

### Graph Optimization

```python
# Maximum optimization (slower load, faster inference)
config = EngineConfig(
    enable_graph_optimization=True,
    # Also enable in SessionOptions if using ONNX directly
)
```

### Batch Processing

```python
# Process in batches for better throughput
batch_size = 1000
results = []

for i in range(0, len(features), batch_size):
    batch = features[i:i+batch_size]
    batch_results = engine.predict(batch)
    results.extend(batch_results)
```

### Model Warm-up

```python
# Warm up model for consistent latency
dummy_input = np.zeros((1, feature_count), dtype=np.float32)

for _ in range(5):
    _ = engine.predict(dummy_input)

# Now real inference will be faster
```

---

## Deployment

### Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install ONNX Runtime
RUN pip install onnxruntime>=1.16.0

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -e .

# Environment variables
ENV ONNX_ENABLED=true
ENV ONNX_ENGINE=onnx
ENV ONNX_PROVIDER=cpu

# Run application
CMD ["uvicorn", "device_anomaly.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### GPU Docker

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip

# Install ONNX Runtime GPU
RUN pip3 install onnxruntime-gpu>=1.16.0

# ... rest of Dockerfile
ENV ONNX_PROVIDER=cuda
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detection-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: stellasentinal:onnx-v1
        env:
        - name: ONNX_ENABLED
          value: "true"
        - name: ONNX_ENGINE
          value: "onnx"
        - name: ONNX_PROVIDER
          value: "cpu"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Mobile Deployment

See [Mobile Architecture Guide](mobile_architecture.md) for complete details.

---

## Troubleshooting

### Common Issues

#### 1. Import Error: onnxruntime not found

```bash
# Solution: Install onnxruntime
pip install onnxruntime>=1.16.0
```

#### 2. Predictions Don't Match sklearn

```python
# Check validation during export
exporter.export_model(..., validate=True)

# If validation fails, check:
# - Feature count matches
# - Input data type (should be float32)
# - scikit-learn version compatibility
```

#### 3. CUDA Provider Not Available

```python
import onnxruntime as ort

# Check available providers
print(ort.get_available_providers())

# If CUDA not in list:
# 1. Install onnxruntime-gpu (not onnxruntime)
# 2. Verify CUDA installation
# 3. Check CUDA version compatibility
```

#### 4. Slow Inference Performance

```python
# Enable profiling
config = EngineConfig(enable_profiling=True)
engine = ONNXInferenceEngine("model.onnx", config)

# Run inference
engine.predict(features)

# Get profiling data
profile_file = engine.get_profiling_data()
print(f"Profile saved to: {profile_file}")
```

### Debug Mode

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# ONNX Runtime will log detailed info
engine = ONNXInferenceEngine("model.onnx")
```

---

## Best Practices

### 1. Always Validate Exports

```python
exporter.export_model(..., validate=True)
```

### 2. Use Quantization for Edge Deployment

```python
# For mobile/edge, always use INT8
ONNXQuantizer.quantize_dynamic(fp32_path, int8_path)
```

### 3. Implement Fallback

```python
# Production systems should have fallback
engine = FallbackInferenceEngine(onnx_path, sklearn_path)
```

### 4. Monitor Performance

```python
config = EngineConfig(collect_metrics=True)

# After inference
metrics = engine.get_metrics()
log_metrics(metrics)  # Send to monitoring system
```

### 5. Version Models

```python
# Include version in filename and metadata
metadata = {"version": "1.2.0", "created_at": "2025-12-27"}
exporter.export_with_metadata(model, 25, "model_v1.2.0.onnx", metadata)
```

### 6. Use Environment-Specific Configs

```python
# Development: Use sklearn for debugging
if os.getenv("ENV") == "development":
    engine_type = EngineType.SKLEARN
else:
    # Production: Use ONNX
    engine_type = EngineType.ONNX
```

---

## Additional Resources

- **ONNX Runtime Docs:** https://onnxruntime.ai/docs/
- **skl2onnx Tutorial:** https://onnx.ai/sklearn-onnx/
- **Performance Tuning:** https://onnxruntime.ai/docs/performance/
- **GitHub Issues:** https://github.com/microsoft/onnxruntime/issues

---

## Next Steps

1. **Run Benchmarks:** `python scripts/benchmark_onnx.py`
2. **Run Tests:** `pytest tests/test_onnx_integration.py`
3. **Review Performance Analysis:** See `docs/performance_analysis.md`
4. **Plan Mobile Deployment:** See `docs/mobile_architecture.md`
5. **Calculate ROI:** See `docs/cost_savings_analysis.md`

---

## Support

For questions or issues:
1. Check this documentation
2. Review test cases in `tests/test_onnx_integration.py`
3. Run benchmark script with `--help`
4. File an issue in project repository
