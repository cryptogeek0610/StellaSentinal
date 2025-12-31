#!/usr/bin/env python3
"""
ONNX Runtime Performance Benchmark

Compares inference performance between scikit-learn and ONNX Runtime
across different batch sizes, model configurations, and hardware providers.

Usage:
    python scripts/benchmark_onnx.py
    python scripts/benchmark_onnx.py --samples 10000 --batch-sizes 1,100,1000,10000
    python scripts/benchmark_onnx.py --gpu  # Use CUDA if available
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import tempfile

import numpy as np
import pandas as pd
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from device_anomaly.models.anomaly_detector import AnomalyDetectorIsolationForest, AnomalyDetectorConfig
from device_anomaly.models.inference_engine import (
    create_inference_engine,
    EngineType,
    EngineConfig,
    ExecutionProvider,
)
from device_anomaly.models.onnx_exporter import ONNXModelExporter, ONNXQuantizer


class ONNXBenchmark:
    """Benchmark framework for comparing sklearn vs ONNX performance"""

    def __init__(
        self,
        n_features: int = 25,
        n_estimators: int = 300,
        contamination: float = 0.03,
    ):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.contamination = contamination

        self.model_paths: Dict[str, Path] = {}
        self.results: List[Dict[str, Any]] = []

    def generate_data(self, n_samples: int) -> np.ndarray:
        """Generate synthetic test data"""
        np.random.seed(42)
        return np.random.randn(n_samples, self.n_features).astype(np.float32)

    def train_and_export_models(self, training_samples: int = 1000) -> None:
        """Train IsolationForest and export to all formats"""
        print(f"\n{'='*60}")
        print(f"Training IsolationForest ({self.n_estimators} trees, {self.n_features} features)")
        print(f"{'='*60}\n")

        # Generate training data
        train_data = self.generate_data(training_samples)
        df_train = pd.DataFrame(train_data, columns=[f"feature_{i}" for i in range(self.n_features)])

        # Train model
        print(f"Training on {training_samples:,} samples...")
        start = time.time()

        detector = AnomalyDetectorIsolationForest(
            config=AnomalyDetectorConfig(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
            )
        )
        detector.fit(df_train)

        train_time = time.time() - start
        print(f"Training completed in {train_time:.2f}s\n")

        # Create temp directory for models
        tmpdir = tempfile.mkdtemp()
        base_path = Path(tmpdir) / "benchmark_model"

        # Export models
        print("Exporting models...")

        # sklearn
        import joblib

        sklearn_path = base_path.with_suffix(".pkl")
        joblib.dump(detector.model, sklearn_path)
        sklearn_size = sklearn_path.stat().st_size / 1e6
        self.model_paths["sklearn"] = sklearn_path
        print(f"  ✓ sklearn:          {sklearn_size:.2f} MB")

        # ONNX FP32
        exporter = ONNXModelExporter()
        onnx_fp32_path = base_path.with_suffix(".onnx")
        exporter.export_model(
            model=detector.model,
            feature_count=self.n_features,
            output_path=onnx_fp32_path,
            validate=True,
        )
        onnx_fp32_size = onnx_fp32_path.stat().st_size / 1e6
        self.model_paths["onnx_fp32"] = onnx_fp32_path
        print(f"  ✓ ONNX FP32:        {onnx_fp32_size:.2f} MB")

        # ONNX INT8 (quantized)
        onnx_int8_path = base_path.with_name(base_path.stem + "_int8").with_suffix(".onnx")
        ONNXQuantizer.quantize_dynamic(onnx_fp32_path, onnx_int8_path)
        onnx_int8_size = onnx_int8_path.stat().st_size / 1e6
        self.model_paths["onnx_int8"] = onnx_int8_path
        print(f"  ✓ ONNX INT8:        {onnx_int8_size:.2f} MB ({(1-onnx_int8_size/onnx_fp32_size)*100:.1f}% reduction)")

        print()

    def benchmark_engine(
        self,
        engine_name: str,
        engine_type: EngineType,
        model_path: Path,
        test_data: np.ndarray,
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
        provider: ExecutionProvider = ExecutionProvider.CPU,
    ) -> Dict[str, Any]:
        """Benchmark a single inference engine"""

        # Create engine
        config = EngineConfig(
            engine_type=engine_type,
            onnx_provider=provider,
            collect_metrics=False,  # We'll time manually
        )

        if engine_type == EngineType.SKLEARN:
            from device_anomaly.models.inference_engine import ScikitLearnEngine

            engine = ScikitLearnEngine(model_path, config)
        else:
            from device_anomaly.models.inference_engine import ONNXInferenceEngine

            engine = ONNXInferenceEngine(model_path, config)

        # Warmup
        for _ in range(warmup_runs):
            _ = engine.predict(test_data)

        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            predictions = engine.predict(test_data)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        # Calculate statistics
        times = np.array(times)
        return {
            "engine": engine_name,
            "samples": len(test_data),
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "throughput": len(test_data) / (np.mean(times) / 1000),  # samples/sec
        }

    def run_benchmarks(
        self,
        batch_sizes: List[int],
        use_gpu: bool = False,
    ) -> pd.DataFrame:
        """Run benchmarks across different batch sizes"""

        print(f"\n{'='*60}")
        print(f"Running Benchmarks")
        print(f"{'='*60}\n")

        # Determine which engines to test
        engines_to_test = [
            ("sklearn", EngineType.SKLEARN, self.model_paths["sklearn"], ExecutionProvider.CPU),
            ("onnx_fp32_cpu", EngineType.ONNX, self.model_paths["onnx_fp32"], ExecutionProvider.CPU),
            ("onnx_int8_cpu", EngineType.ONNX, self.model_paths["onnx_int8"], ExecutionProvider.CPU),
        ]

        if use_gpu:
            # Check if CUDA is available
            import onnxruntime as ort

            if "CUDAExecutionProvider" in ort.get_available_providers():
                engines_to_test.extend(
                    [
                        ("onnx_fp32_gpu", EngineType.ONNX, self.model_paths["onnx_fp32"], ExecutionProvider.CUDA),
                        ("onnx_int8_gpu", EngineType.ONNX, self.model_paths["onnx_int8"], ExecutionProvider.CUDA),
                    ]
                )
                print("GPU (CUDA) available - including GPU benchmarks\n")
            else:
                print("GPU requested but CUDA not available - skipping GPU benchmarks\n")

        all_results = []

        for batch_size in batch_sizes:
            print(f"Batch size: {batch_size:,} samples")
            print("-" * 60)

            test_data = self.generate_data(batch_size)

            for engine_name, engine_type, model_path, provider in engines_to_test:
                try:
                    result = self.benchmark_engine(
                        engine_name=engine_name,
                        engine_type=engine_type,
                        model_path=model_path,
                        test_data=test_data,
                        provider=provider,
                    )
                    all_results.append(result)

                    print(
                        f"  {engine_name:20s} | "
                        f"{result['mean_ms']:8.2f} ms | "
                        f"{result['throughput']:10,.0f} samples/sec"
                    )

                except Exception as e:
                    print(f"  {engine_name:20s} | ERROR: {e}")

            print()

        return pd.DataFrame(all_results)

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print summary comparison"""
        print(f"\n{'='*60}")
        print("Performance Summary")
        print(f"{'='*60}\n")

        # Group by batch size
        for batch_size in results_df["samples"].unique():
            batch_results = results_df[results_df["samples"] == batch_size]

            print(f"\nBatch Size: {batch_size:,} samples")
            print("-" * 80)

            # Calculate speedup vs sklearn
            sklearn_time = batch_results[batch_results["engine"] == "sklearn"]["mean_ms"].values[0]

            summary_data = []
            for _, row in batch_results.iterrows():
                speedup = sklearn_time / row["mean_ms"]
                summary_data.append(
                    {
                        "Engine": row["engine"],
                        "Mean (ms)": f"{row['mean_ms']:.2f}",
                        "P95 (ms)": f"{row['p95_ms']:.2f}",
                        "Throughput (samples/s)": f"{row['throughput']:,.0f}",
                        "Speedup": f"{speedup:.2f}x",
                    }
                )

            print(tabulate(summary_data, headers="keys", tablefmt="grid"))

    def save_results(self, results_df: pd.DataFrame, output_path: str = "benchmark_results.csv") -> None:
        """Save results to CSV"""
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime vs scikit-learn")
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of samples to generate for benchmarking",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,10,100,1000,10000",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in IsolationForest",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=25,
        help="Number of features",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Include GPU benchmarks if CUDA is available",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file for results",
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print("\n" + "=" * 60)
    print("ONNX Runtime Performance Benchmark")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Features: {args.n_features}")
    print(f"  Trees: {args.n_estimators}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  GPU enabled: {args.gpu}")

    # Install tabulate if needed
    try:
        import tabulate
    except ImportError:
        print("\nInstalling tabulate for pretty tables...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])

    # Run benchmark
    benchmark = ONNXBenchmark(
        n_features=args.n_features,
        n_estimators=args.n_estimators,
    )

    # Train and export
    benchmark.train_and_export_models()

    # Run benchmarks
    results_df = benchmark.run_benchmarks(batch_sizes, use_gpu=args.gpu)

    # Print summary
    benchmark.print_summary(results_df)

    # Save results
    benchmark.save_results(results_df, args.output)

    print("\n✓ Benchmark complete!\n")


if __name__ == "__main__":
    main()
