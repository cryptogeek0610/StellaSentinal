from __future__ import annotations

import json
import logging

import pandas as pd

from device_anomaly.config.feature_config import FeatureConfig
from device_anomaly.data_access.db_connection import create_dw_engine

logger = logging.getLogger(__name__)


def build_anomaly_results_df(
    df_scored: pd.DataFrame,
    explanations: list[str] | None,
    source: str,
    model_version: str,
) -> pd.DataFrame:
    """
    Build a DataFrame aligned with dbo.ml_AnomalyResults (without Id).

    Assumes:
        - df_scored is already filtered to anomalies you want to persist
        - len(df_scored) == len(explanations)
        - df_scored has DeviceId, Timestamp, anomaly_score, anomaly_label columns
    """
    if explanations is not None and len(df_scored) != len(explanations):
        raise ValueError(
            f"len(df_scored)={len(df_scored)} != len(explanations)={len(explanations)}"
        )

    records = []
    explanations_iter = iter(explanations) if explanations is not None else None

    for _, row in df_scored.iterrows():
        explanation = next(explanations_iter) if explanations_iter is not None else None

        metrics = {
            col: row[col] for col in FeatureConfig.genericFeatures if col in df_scored.columns
        }
        # Keep heuristic context inside metrics payload so dashboards can display it without schema changes.
        if "heuristic_reasons" in df_scored.columns or "heuristic_score" in df_scored.columns:
            metrics["heuristics"] = {
                "reasons": row.get("heuristic_reasons"),
                "score": row.get("heuristic_score"),
            }

        records.append(
            {
                "Source": source,
                "DeviceId": int(row["DeviceId"]),
                "Timestamp": row["Timestamp"],
                "AnomalyScore": float(row["anomaly_score"]),
                "AnomalyLabel": int(row["anomaly_label"]),
                "ModelVersion": model_version,
                "MetricsJson": json.dumps(metrics, default=float),
                "Explanation": explanation,
            }
        )

    return pd.DataFrame.from_records(records)


def save_anomaly_results(df_results: pd.DataFrame) -> int:
    """
    Append anomaly results rows to dbo.ml_AnomalyResults.

    Returns:
        Number of rows written.
    """
    if df_results.empty:
        logger.info("No anomaly results to save.")
        return 0

    engine = create_dw_engine()
    with engine.begin() as conn:
        df_results.to_sql(
            name="ml_AnomalyResults",
            con=conn,
            schema="dbo",
            if_exists="append",
            index=False,
        )

    logger.info("Saved %d anomaly results to dbo.ml_AnomalyResults.", len(df_results))
    return len(df_results)


def save_anomaly_events(df_events: pd.DataFrame) -> int:
    """
    Append anomaly event rows to dbo.ml_AnomalyEvents.

    Returns:
        Number of rows written.
    """
    if df_events.empty:
        logger.info("No anomaly events to save.")
        return 0

    engine = create_dw_engine()
    with engine.begin() as conn:
        df_events.to_sql(
            name="ml_AnomalyEvents",
            con=conn,
            schema="dbo",
            if_exists="append",
            index=False,
        )

    logger.info("Saved %d anomaly events to dbo.ml_AnomalyEvents.", len(df_events))
    return len(df_events)


def save_device_patterns(df_patterns: pd.DataFrame) -> int:
    """
    Append device-level pattern rows to dbo.ml_DeviceAnomalyPatterns.
    """
    if df_patterns.empty:
        logger.info("No device patterns to save.")
        return 0

    engine = create_dw_engine()
    with engine.begin() as conn:
        df_patterns.to_sql(
            name="ml_DeviceAnomalyPatterns",
            con=conn,
            schema="dbo",
            if_exists="append",
            index=False,
        )

    logger.info(
        "Saved %d device anomaly patterns to dbo.ml_DeviceAnomalyPatterns.",
        len(df_patterns),
    )
    return len(df_patterns)
