import argparse
import logging
from pathlib import Path

import pandas as pd

from device_anomaly import __version__
from device_anomaly.config.logging_config import setup_logging
from device_anomaly.config.experiment_config import SyntheticExperimentConfig
from device_anomaly.config.model_config import make_model_version
from device_anomaly.config.config_loader import load_synthetic_config

from device_anomaly.data_access.synthetic_generator import generate_synthetic_device_telemetry
from device_anomaly.data_access.persistence import (
    build_anomaly_results_df,
    save_anomaly_results,
    save_anomaly_events,
    save_device_patterns,
)
from device_anomaly.features.device_features import DeviceFeatureBuilder
from device_anomaly.models.anomaly_detector import AnomalyDetectorConfig
from device_anomaly.models.hybrid import HybridAnomalyDetector, HybridAnomalyDetectorConfig
from device_anomaly.models.calibration import IsoScoreCalibrator, IsoScoreCalibratorConfig
from device_anomaly.models.drift_monitor import compute_feature_stats, save_stats, load_stats, compare_stats
from device_anomaly.models.baseline import (
    BaselineLevel,
    compute_baselines,
    apply_baselines,
    save_baselines,
    suggest_baseline_adjustments,
)
from device_anomaly.models.events import (
    build_event_results,
    select_top_anomalous_devices,
)
from device_anomaly.models.patterns import build_device_pattern_results
from device_anomaly.pipeline import (
    PipelineStage,
    PipelineTracker,
    drop_all_nan_columns,
    ensure_min_rows,
    ensure_required_columns,
    save_model_metadata,
)

def run_synthetic_experiment(
    config: SyntheticExperimentConfig | None = None,
) -> None:
    if config is None:
        config = SyntheticExperimentConfig()

    logger = logging.getLogger(__name__)
    logger.info(
        "Running synthetic experiment: n_devices=%d, n_days=%d, window=%d, anomaly_rate=%.3f",
        config.n_devices,
        config.n_days,
        config.detection.window,
        config.anomaly_rate,
    )

    tracker = PipelineTracker()

    # 1) Generate synthetic data
    df_raw = generate_synthetic_device_telemetry(
        n_devices=config.n_devices,
        n_days=config.n_days,
        freq="1h",
        anomaly_rate=config.anomaly_rate,
    )
    ensure_required_columns(df_raw, ["DeviceId", "Timestamp"], "Synthetic raw data")
    ensure_min_rows(df_raw, 25, "Synthetic raw data")
    df_raw = drop_all_nan_columns(df_raw, "Synthetic raw data")
    tracker.advance(PipelineStage.INGESTION)
    logger.info("Synthetic raw data shape: %s", df_raw.shape)
    logger.info("Sample:\n%s", df_raw.head())

    # 2) Build features
    feature_builder = DeviceFeatureBuilder(window=config.detection.window)
    df_feat = feature_builder.transform(df_raw)
    ensure_required_columns(df_feat, ["DeviceId", "Timestamp"], "Synthetic features")
    ensure_min_rows(df_feat, 25, "Synthetic features")
    df_feat = drop_all_nan_columns(df_feat, "Synthetic features")
    tracker.advance(PipelineStage.FEATURES)
    logger.info("Feature data shape: %s", df_feat.shape)

    # 2b) Baseline-aware z-scores (global baseline)
    baselines = {}
    baseline_applied = False
    baseline_levels = [BaselineLevel(name="global", group_columns=["__all__"], min_rows=25)]
    df_feat["__all__"] = "all"

    exclude_baseline_cols = {
        "DeviceId",
        "Timestamp",
        "anomaly_score",
        "anomaly_label",
        "is_injected_anomaly",
    }
    baseline_features = [
        col for col in df_feat.columns
        if pd.api.types.is_numeric_dtype(df_feat[col]) and col not in exclude_baseline_cols
    ]
    baselines_path = Path("artifacts/synthetic_baselines.json")

    try:
        baselines = compute_baselines(
            df=df_feat,
            feature_cols=baseline_features,
            levels=baseline_levels,
        )
        df_feat = apply_baselines(df_feat, baselines, baseline_levels)
        save_baselines(baselines, baselines_path)
        baseline_applied = True
    except Exception as exc:
        logger.warning("Baseline computation failed; falling back to raw features. Error: %s", exc)
    finally:
        df_feat = df_feat.drop(columns=["__all__"], errors="ignore")
    tracker.advance(PipelineStage.BASELINES)

    # 3) Train anomaly detector
    detector = HybridAnomalyDetector(
        HybridAnomalyDetectorConfig(
            iso_config=AnomalyDetectorConfig(contamination=config.detection.contamination),
            feature_overrides=config.detection.feature_overrides,
        )
    )
    try:
        detector.fit(df_feat)
    except Exception:
        logger.exception("Isolation Forest training failed; aborting pipeline.")
        raise
    tracker.advance(PipelineStage.TRAINING)

    # 4) Score all points
    try:
        df_scored = detector.score_dataframe(df_feat)
    except Exception:
        logger.exception("Scoring failed; aborting pipeline.")
        raise
    tracker.advance(PipelineStage.SCORING)

    calibrator = IsoScoreCalibrator(
        IsoScoreCalibratorConfig(model_path=Path("artifacts/iso_calibrator.pkl"))
    )
    calibrator.fit(df_scored)

    stats = compute_feature_stats(
        df=df_feat,
        feature_cols=detector.feature_columns,
        anomaly_scores=df_scored["anomaly_score"],
    )
    stats_path = Path("artifacts/synthetic_stats.json")
    baseline = load_stats(stats_path)
    if baseline:
        for warning in compare_stats(stats, baseline):
            logger.warning("Drift warning: %s", warning)
    save_stats(stats, stats_path)

    save_model_metadata(
        path=Path("artifacts/synthetic_model_config.json"),
        payload={
            "source": "synthetic",
            "service_version": __version__,
            "detection": {
                "window": config.detection.window,
                "contamination": config.detection.contamination,
                "feature_overrides": config.detection.feature_overrides,
            },
            "iso_config": {
                "n_estimators": detector.global_detector.config.n_estimators,
                "random_state": detector.global_detector.config.random_state,
                "scale_features": detector.global_detector.config.scale_features,
                "min_variance": detector.global_detector.config.min_variance,
            },
            "hybrid_config": {
                "iso_weight": detector.config.iso_weight,
                "temporal_weight": detector.config.temporal_weight,
                "heuristic_weight": detector.config.heuristic_weight,
                "use_cohort_models": detector.config.use_cohort_models,
            },
            "baseline_applied": baseline_applied,
            "baseline_levels": [level.__dict__ for level in baseline_levels],
            "row_count": int(len(df_feat)),
            "feature_count": int(len(detector.feature_columns)),
            "feature_columns": detector.feature_columns,
        },
    )

    # 5) Evaluation on synthetic ground truth (unchanged)
    if "is_injected_anomaly" in df_scored.columns:
        ground_truth = df_scored["is_injected_anomaly"]
        predicted_anomaly = df_scored["anomaly_label"] == -1

        true_positives = int(((ground_truth == True) & predicted_anomaly).sum())   # noqa: E712
        false_positives = int(((ground_truth == False) & predicted_anomaly).sum()) # noqa: E712
        false_negatives = int(((ground_truth == True) & ~predicted_anomaly).sum()) # noqa: E712

        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)

        logger.info("Evaluation on synthetic ground truth:")
        logger.info("  TP=%d, FP=%d, FN=%d", true_positives, false_positives, false_negatives)
        logger.info("  Precision=%.3f, Recall=%.3f", precision, recall)

    anomalies_df = df_scored[df_scored["anomaly_label"] == -1].copy()
    if anomalies_df.empty:
        logger.info("No anomalies detected in synthetic experiment.")
        return

    # Persist ALL row-level anomalies (no row-level LLM)
    anomalies_for_rows = anomalies_df.sort_values("anomaly_score")
    model_version = make_model_version("synthetic")

    results_df = build_anomaly_results_df(
        df_scored=anomalies_for_rows,
        explanations=None,
        source="synthetic",
        model_version=model_version,
    )
    try:
        saved_rows = save_anomaly_results(results_df)
        logger.info("Synthetic experiment persisted %d row-level anomalies.", saved_rows)
    except Exception as exc:
        logger.error("Failed to persist row-level anomalies: %s", exc, exc_info=True)
    tracker.advance(PipelineStage.PERSISTENCE)

    # Generate baseline suggestions
    if baseline_applied and baselines:
        baseline_suggestions = suggest_baseline_adjustments(
            anomalies_df=anomalies_df,
            baselines=baselines,
            levels=baseline_levels,
        )
        if baseline_suggestions:
            import json
            suggestion_path = Path("artifacts/synthetic_baseline_suggestions.json")
            suggestion_path.parent.mkdir(parents=True, exist_ok=True)
            suggestion_path.write_text(json.dumps(baseline_suggestions, indent=2, default=float))
            logger.info("Captured %d baseline adjustment suggestions at %s", len(baseline_suggestions), suggestion_path)
    else:
        logger.info("Baseline suggestions skipped (baselines unavailable).")

    # ---- Focus on devices with pattern of anomalies ----
    top_devices = select_top_anomalous_devices(
        df_scored=df_scored,
        top_n=config.events.top_n_devices,
        min_total_points=config.events.min_total_points,
        min_anomalies=config.events.min_anomalies,
    )
    if not top_devices:
        logger.info("No devices meet the criteria for repeated anomalies.")
        return

    logger.info("Top anomalous devices (synthetic): %s", top_devices)

    anomalies_for_events = anomalies_df[anomalies_df["DeviceId"].isin(top_devices)]
    if anomalies_for_events.empty:
        logger.info("No anomalies for selected devices; skipping event-level LLM.")
        return

    # Build and persist event-level anomalies
    event_results_df = build_event_results(
        anomalies_df=anomalies_for_events,
        source="synthetic",
        model_version=model_version,
        max_gap_hours=config.events.max_gap_hours,
    )

    try:
        saved_events = save_anomaly_events(event_results_df)
        logger.info(
            "Synthetic experiment persisted %d anomaly events for top devices.",
            saved_events,
        )
    except Exception as exc:
        saved_events = 0
        logger.error("Failed to persist anomaly events: %s", exc, exc_info=True)

    # ---- Device-level pattern summaries ----
    if saved_events > 0 and not event_results_df.empty:
        period_start = df_raw["Timestamp"].min()
        period_end = df_raw["Timestamp"].max()
        device_ids = sorted(anomalies_for_events["DeviceId"].unique())

        pattern_df = build_device_pattern_results(
            df_scored=df_scored,
            events_df=event_results_df,
            source="synthetic",
            model_version=model_version,
            period_start=period_start,
            period_end=period_end,
            device_ids=device_ids,
        )

        try:
            saved_patterns = save_device_patterns(pattern_df)
            logger.info(
                "Synthetic experiment persisted %d device anomaly patterns.",
                saved_patterns,
            )
        except Exception as exc:
            logger.error("Failed to persist device patterns: %s", exc, exc_info=True)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run synthetic anomaly detection experiment."
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to YAML/JSON config file for the synthetic experiment",
        default=None,
    )

    args = parser.parse_args()

    if args.config:
        cfg = load_synthetic_config(args.config)
    else:
        cfg = SyntheticExperimentConfig()

    run_synthetic_experiment(config=cfg)


if __name__ == "__main__":
    main()
