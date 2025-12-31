from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
import streamlit as st

from device_anomaly.data_access.db_connection import create_dw_engine
from device_anomaly.models.drift_monitor import (
    load_stats,
    compute_feature_stats,
    compare_stats,
)
from device_anomaly.models.baseline import load_baselines


# -------------------------- Engine + loaders -------------------------- #


@st.cache_resource
def get_engine():
    return create_dw_engine()


@st.cache_data
def load_anomaly_results(
    _engine,
    source: Optional[str] = None,
    date_range: Optional[Tuple[dt.date, dt.date]] = None,
) -> pd.DataFrame:
    sql = """
        SELECT
            Id,
            Source,
            DeviceId,
            Timestamp,
            AnomalyScore,
            AnomalyLabel,
            ModelVersion,
            MetricsJson,
            Explanation
        FROM dbo.ml_AnomalyResults
        WHERE 1 = 1
    """

    params: dict[str, object] = {}
    if source:
        sql += " AND Source = :source"
        params["source"] = source

    if date_range is not None:
        start, end = date_range
        sql += " AND Timestamp BETWEEN :start_ts AND :end_ts"
        params["start_ts"] = dt.datetime.combine(start, dt.time.min)
        params["end_ts"] = dt.datetime.combine(end, dt.time.max)

    with _engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if not df.empty:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@st.cache_data
def load_anomaly_events(
    _engine,
    source: Optional[str] = None,
    date_range: Optional[Tuple[dt.date, dt.date]] = None,
) -> pd.DataFrame:
    sql = """
        SELECT
            Id,
            Source,
            DeviceId,
            EventStart,
            EventEnd,
            DurationMinutes,
            AnomalyScoreMin,
            AnomalyScoreMax,
            AnomalyScoreMean,
            [RowCount],
            ModelVersion,
            MetricsJson,
            Explanation
        FROM dbo.ml_AnomalyEvents
        WHERE 1 = 1
    """

    params: dict[str, object] = {}
    if source:
        sql += " AND Source = :source"
        params["source"] = source

    if date_range is not None:
        start, end = date_range
        sql += " AND EventStart BETWEEN :start_ts AND :end_ts"
        params["start_ts"] = dt.datetime.combine(start, dt.time.min)
        params["end_ts"] = dt.datetime.combine(end, dt.time.max)

    with _engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if not df.empty:
        df["EventStart"] = pd.to_datetime(df["EventStart"])
        df["EventEnd"] = pd.to_datetime(df["EventEnd"])
    return df


@st.cache_data
def load_device_patterns(
    _engine,
    source: Optional[str] = None,
) -> pd.DataFrame:
    sql = """
        SELECT
            Id,
            Source,
            DeviceId,
            PeriodStart,
            PeriodEnd,
            TotalPoints,
            TotalAnomalies,
            AnomalyRate,
            EventCount,
            WorstAnomalyScore,
            MeanAnomalyScore,
            PatternJson,
            Explanation,
            ModelVersion
        FROM dbo.ml_DeviceAnomalyPatterns
        WHERE 1 = 1
    """

    params: dict[str, object] = {}
    if source:
        sql += " AND Source = :source"
        params["source"] = source

    with _engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if not df.empty:
        df["PeriodStart"] = pd.to_datetime(df["PeriodStart"])
        df["PeriodEnd"] = pd.to_datetime(df["PeriodEnd"])
    return df


def load_model_config(source: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    if source == "synthetic":
        candidates = ["artifacts/synthetic_model_config.json"]
    elif source == "dw":
        candidates = ["artifacts/dw_model_config.json"]
    else:
        candidates = ["artifacts/dw_model_config.json", "artifacts/synthetic_model_config.json"]

    for path_str in candidates:
        path = Path(path_str)
        if path.exists():
            return json.loads(path.read_text()), path_str
    return None, None


def load_baseline_suggestions(source: Optional[str]) -> tuple[Optional[list], Optional[str]]:
    if source == "synthetic":
        candidates = ["artifacts/synthetic_baseline_suggestions.json"]
    elif source == "dw":
        candidates = ["artifacts/dw_baseline_suggestions.json"]
    else:
        candidates = ["artifacts/dw_baseline_suggestions.json", "artifacts/synthetic_baseline_suggestions.json"]

    for path_str in candidates:
        path = Path(path_str)
        if path.exists():
            return json.loads(path.read_text()), path_str
    return None, None


def load_baseline_stats(source: Optional[str]) -> tuple[dict, Optional[str]]:
    if source == "synthetic":
        path = Path("artifacts/synthetic_baselines.json")
    elif source == "dw":
        path = Path("artifacts/dw_baselines.json")
    else:
        path = Path("artifacts/dw_baselines.json")

    if not path.exists():
        return {}, None
    return load_baselines(path), str(path)


# ------------------------- Metric utilities ------------------------- #


def expand_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "MetricsJson" not in df.columns:
        return pd.DataFrame(index=df.index)

    metrics = []
    for val in df["MetricsJson"]:
        if isinstance(val, dict):
            metrics.append(val)
        elif isinstance(val, str) and val.strip():
            try:
                metrics.append(json.loads(val))
            except json.JSONDecodeError:
                metrics.append({})
        else:
            metrics.append({})

    metrics_df = pd.json_normalize(metrics)
    metrics_df.index = df.index
    return metrics_df


def compute_feature_contributions(df_results: pd.DataFrame) -> pd.DataFrame:
    if df_results.empty:
        return pd.DataFrame()

    metrics_df = expand_metrics(df_results)
    if metrics_df.empty:
        return pd.DataFrame()

    df_metrics = metrics_df.apply(pd.to_numeric, errors="coerce")
    df_metrics["IsAnomaly"] = df_results["AnomalyLabel"] == -1

    contributions = []
    for col in df_metrics.columns:
        if col == "IsAnomaly":
            continue
        series = df_metrics[col]
        if series.dropna().empty:
            continue
        overall_mean = float(series.mean())
        anomaly_mean = float(series[df_metrics["IsAnomaly"]].mean())
        delta = anomaly_mean - overall_mean
        contributions.append(
            {
                "metric": col,
                "overall_mean": overall_mean,
                "anomaly_mean": anomaly_mean,
                "delta": delta,
                "delta_abs": abs(delta),
            }
        )

    if not contributions:
        return pd.DataFrame()

    contrib_df = pd.DataFrame(contributions).sort_values("delta_abs", ascending=False)
    return contrib_df


def load_stats_for_source(source: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    candidates: list[str] = []
    if source == "synthetic":
        candidates = ["artifacts/synthetic_stats.json"]
    elif source == "dw":
        candidates = ["artifacts/dw_stats.json"]
    else:
        candidates = ["artifacts/dw_stats.json", "artifacts/synthetic_stats.json"]

    for path_str in candidates:
        stats = load_stats(Path(path_str))
        if stats:
            return stats, path_str
    return None, None


def compute_current_stats(df_results: pd.DataFrame) -> Optional[dict]:
    metrics_df = expand_metrics(df_results)
    if metrics_df.empty:
        return None
    return compute_feature_stats(
        df=metrics_df,
        feature_cols=list(metrics_df.columns),
        anomaly_scores=df_results.get("AnomalyScore"),
    )
