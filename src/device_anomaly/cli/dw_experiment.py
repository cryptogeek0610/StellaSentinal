from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from device_anomaly import __version__
from device_anomaly.config.config_loader import load_dw_config
from device_anomaly.config.experiment_config import DWExperimentConfig
from device_anomaly.config.logging_config import setup_logging
from device_anomaly.config.model_config import make_model_version
from device_anomaly.data_access.persistence import (
    build_anomaly_results_df,
    save_anomaly_events,
    save_anomaly_results,
    save_device_patterns,
)
from device_anomaly.data_access.unified_loader import load_unified_device_dataset
from device_anomaly.features.device_features import DeviceFeatureBuilder
from device_anomaly.models.actionable import build_actionable_outputs
from device_anomaly.models.anomaly_detector import AnomalyDetectorConfig
from device_anomaly.models.baseline import (
    BaselineLevel,
    apply_baselines,
    compute_baselines,
    save_baselines,
    suggest_baseline_adjustments,
)
from device_anomaly.models.calibration import IsoScoreCalibrator, IsoScoreCalibratorConfig
from device_anomaly.models.drift_monitor import (
    compare_stats,
    compute_feature_stats,
    load_stats,
    save_stats,
)
from device_anomaly.models.events import (
    build_event_results,
    select_top_anomalous_devices,
)
from device_anomaly.models.heuristics import apply_heuristics, build_rules_from_dicts
from device_anomaly.models.hybrid import HybridAnomalyDetector, HybridAnomalyDetectorConfig
from device_anomaly.models.patterns import build_device_pattern_results
from device_anomaly.pipeline import (
    PipelineStage,
    PipelineTracker,
    drop_all_nan_columns,
    ensure_min_rows,
    ensure_required_columns,
    save_model_metadata,
)


def run_dw_experiment(
    config: DWExperimentConfig | None = None,
) -> None:
    if config is None:
        config = DWExperimentConfig()

    logger = logging.getLogger(__name__)
    logger.info(
        "Running DW experiment: %s to %s, devices=%s, window=%d",
        config.start_date,
        config.end_date,
        config.device_ids if config.device_ids else "ALL (sampled)",
        config.detection.window,
    )

    tracker = PipelineTracker()

    # 1) Load telemetry from DW
    df_raw = load_unified_device_dataset(
        start_date=config.start_date,
        end_date=config.end_date,
        device_ids=config.device_ids,
        row_limit=config.row_limit,
        include_mc_labels=True,
    )
    logger.info("Loaded raw DW data shape: %s", df_raw.shape)
    if df_raw.empty:
        logger.warning("No data returned for given parameters.")
        return

    ensure_required_columns(df_raw, ["DeviceId", "Timestamp"], "DW raw data")
    ensure_min_rows(df_raw, 25, "DW raw data")
    df_raw = drop_all_nan_columns(df_raw, "DW raw data")
    tracker.advance(PipelineStage.INGESTION)

    logger.info("Sample DW data:\n%s", df_raw.head())

    # 2) Build features
    feature_builder = DeviceFeatureBuilder(window=config.detection.window)
    df_feat = feature_builder.transform(df_raw)
    ensure_required_columns(df_feat, ["DeviceId", "Timestamp"], "DW features")
    ensure_min_rows(df_feat, 25, "DW features")
    df_feat = drop_all_nan_columns(df_feat, "DW features")
    tracker.advance(PipelineStage.FEATURES)
    logger.info("Feature data shape: %s", df_feat.shape)

    # 2b) Baseline-aware z-scores (global + store/customer if present)
    baselines = {}
    baseline_applied = False
    baseline_levels = [BaselineLevel(name="global", group_columns=["__all__"], min_rows=25)]
    df_feat["__all__"] = "all"

    if {"StoreId"}.issubset(df_feat.columns):
        baseline_levels.append(BaselineLevel(name="store", group_columns=["StoreId"]))
    if {"CustomerId"}.issubset(df_feat.columns):
        baseline_levels.append(BaselineLevel(name="customer", group_columns=["CustomerId"]))
    # always include hardware cohort baseline
    if {"ManufacturerId", "ModelId", "OsVersionId"} <= set(df_feat.columns):
        baseline_levels.append(
            BaselineLevel(name="hardware", group_columns=["ManufacturerId", "ModelId", "OsVersionId"])
        )
    # optional device-level baseline (only when enough rows)
    if {"DeviceId"}.issubset(df_feat.columns):
        baseline_levels.append(
            BaselineLevel(name="device", group_columns=["DeviceId"], min_rows=50)
        )

    exclude_baseline_cols = {
        "DeviceId",
        "Timestamp",
        "ModelId",
        "ManufacturerId",
        "OsVersionId",
        "anomaly_score",
        "anomaly_label",
    }
    baseline_features = [
        col for col in df_feat.columns
        if pd.api.types.is_numeric_dtype(df_feat[col]) and col not in exclude_baseline_cols
    ]
    baselines_path = Path("artifacts/dw_baselines.json")

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

    # 2c) Optional heuristics (threshold-based flags that feed into ML)
    heuristic_rules_cfg = []
    if "TotalBatteryLevelDrop_roll_mean" in df_feat.columns:
        threshold = df_feat["TotalBatteryLevelDrop_roll_mean"].median() + 2 * df_feat["TotalBatteryLevelDrop_roll_mean"].mad()
        heuristic_rules_cfg.append(
            {
                "name": "battery_drain_consistent",
                "column": "TotalBatteryLevelDrop_roll_mean",
                "threshold": threshold,
                "op": ">=",
                "min_consecutive": 3,
                "severity": 0.4,
                "description": "Battery drain persistently above baseline",
            }
        )
    if "Rssi" in df_feat.columns:
        heuristic_rules_cfg.append(
            {
                "name": "low_signal",
                "column": "Rssi",
                "threshold": df_feat["Rssi"].median() - 2 * df_feat["Rssi"].mad(),
                "op": "<=",
                "min_consecutive": 2,
                "severity": 0.6,
                "description": "RSSI persistently low",
            }
        )

    heuristic_flags = apply_heuristics(df_feat, build_rules_from_dicts(heuristic_rules_cfg))

    # 3) Train anomaly detector on this slice
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

    # 4) Score data
    try:
        df_scored = detector.score_dataframe(df_feat, heuristic_flags=heuristic_flags)
    except Exception:
        logger.exception("Scoring failed; aborting pipeline.")
        raise
    tracker.advance(PipelineStage.SCORING)

    calibrator = IsoScoreCalibrator(
        IsoScoreCalibratorConfig(model_path=Path("artifacts/iso_calibrator.pkl"))
    )
    calibrator.predict(df_scored)

    stats = compute_feature_stats(
        df=df_feat,
        feature_cols=detector.feature_columns,
        anomaly_scores=df_scored["anomaly_score"],
    )
    stats_path = Path("artifacts/dw_stats.json")
    baseline = load_stats(stats_path)
    if baseline:
        for warning in compare_stats(stats, baseline):
            logger.warning("Drift warning: %s", warning)
    save_stats(stats, stats_path)

    save_model_metadata(
        path=Path("artifacts/dw_model_config.json"),
        payload={
            "source": "dw",
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

    # 5) Extract anomalies
    anomalies_df = df_scored[df_scored["anomaly_label"] == -1].copy()
    if anomalies_df.empty:
        logger.info("No anomalies detected in DW experiment.")
        return

    model_version = make_model_version("dw")

    # Persist all row-level anomalies (no row-level LLM)
    anomalies_for_rows = anomalies_df.sort_values("anomaly_score")
    results_df = build_anomaly_results_df(
        df_scored=anomalies_for_rows,
        explanations=None,  # explanations generated later in UI if needed
        source="dw",
        model_version=model_version,
    )
    try:
        saved_rows = save_anomaly_results(results_df)
        logger.info("DW experiment persisted %d row-level anomalies.", saved_rows)
    except Exception as exc:
        saved_rows = 0
        logger.error("Failed to persist row-level anomalies: %s", exc, exc_info=True)
    tracker.advance(PipelineStage.PERSISTENCE)

    actionable_path = Path("artifacts/dw_actionable.json")
    actionable_df = build_actionable_outputs(df_scored=anomalies_for_rows, model_version=model_version, top_k_factors=3)
    if not actionable_df.empty:
        actionable_path.parent.mkdir(parents=True, exist_ok=True)
        actionable_path.write_text(actionable_df.to_json(orient="records", date_format="iso"))
        logger.info("Wrote actionable outputs for dashboard to %s", actionable_path)

    if baseline_applied and baselines:
        baseline_suggestions = suggest_baseline_adjustments(
            anomalies_df=anomalies_df,
            baselines=baselines,
            levels=baseline_levels,
        )
        if baseline_suggestions:
            suggestion_path = Path("artifacts/dw_baseline_suggestions.json")
            suggestion_path.parent.mkdir(parents=True, exist_ok=True)
            suggestion_path.write_text(json.dumps(baseline_suggestions, indent=2, default=float))
            logger.info("Captured %d baseline adjustment suggestions at %s", len(baseline_suggestions), suggestion_path)
    else:
        logger.info("Baseline suggestions skipped (baselines unavailable).")

    # ---- Focus on devices with highest anomaly rate ----
    top_devices = select_top_anomalous_devices(
        df_scored=df_scored,
        top_n=config.events.top_n_devices,
        min_total_points=config.events.min_total_points,
        min_anomalies=config.events.min_anomalies,
    )
    if not top_devices:
        logger.info("No devices meet the criteria for repeated anomalies (DW).")
        return

    logger.info("Top anomalous devices (DW): %s", top_devices)

    anomalies_for_events = anomalies_df[anomalies_df["DeviceId"].isin(top_devices)]
    if anomalies_for_events.empty:
        logger.info("No anomalies for selected devices; skipping event-level aggregation.")
        return

    # Build and persist event-level anomalies (no LLM here)
    event_results_df = build_event_results(
        anomalies_df=anomalies_for_events,
        source="dw",
        model_version=model_version,
        max_gap_hours=config.events.max_gap_hours,
    )

    if event_results_df.empty:
        logger.info("No event-level anomalies were built.")
        return

    try:
        saved_events = save_anomaly_events(event_results_df)
        logger.info("DW experiment persisted %d anomaly events.", saved_events)
    except Exception as exc:
        saved_events = 0
        logger.error("Failed to persist anomaly events: %s", exc, exc_info=True)

    logger.info(
        "Sample anomaly events:\n%s",
        event_results_df[
            ["DeviceId", "EventStart", "EventEnd", "RowCount", "AnomalyScoreMin"]
        ]
        .head(5)
        .to_string(index=False),
    )

    # ---- Device-level pattern summaries ----
    if saved_events > 0 and not event_results_df.empty:
        period_start = df_raw["Timestamp"].min()
        period_end = df_raw["Timestamp"].max()
        device_ids = sorted(anomalies_for_events["DeviceId"].unique())

        pattern_df = build_device_pattern_results(
            df_scored=df_scored,
            events_df=event_results_df,
            source="dw",
            model_version=model_version,
            period_start=period_start,
            period_end=period_end,
            device_ids=device_ids,
        )

        if not pattern_df.empty:
            try:
                saved_patterns = save_device_patterns(pattern_df)
                logger.info(
                    "DW experiment persisted %d device anomaly patterns.",
                    saved_patterns,
                )
            except Exception as exc:
                logger.error("Failed to persist device anomaly patterns: %s", exc, exc_info=True)
        else:
            logger.info("No device-level anomaly patterns were built.")


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run DW anomaly detection experiment."
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to YAML/JSON config file for the DW experiment",
        default=None,
    )

    args = parser.parse_args()

    cfg = load_dw_config(args.config) if args.config else DWExperimentConfig()

    run_dw_experiment(config=cfg)


if __name__ == "__main__":
    main()
