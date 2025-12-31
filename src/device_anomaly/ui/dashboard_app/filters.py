from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st
import pandas as pd


@dataclass
class DashboardFilters:
    source: str | None
    date_range: Tuple[dt.date, dt.date] | None
    device_ids: List[int] | None
    top_n_devices: int
    min_total_points: int
    min_anomalies: int
    anomaly_score_max: float | None
    event_min_rows: int
    event_min_duration: int


def parse_device_ids(raw: str) -> List[int]:
    ids: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError:
            continue
    return ids


def parse_feature_list(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def sidebar_filters(df_results: pd.DataFrame) -> DashboardFilters:
    st.sidebar.header("Filters & configuration")

    sources = sorted(df_results["Source"].dropna().unique().tolist()) if not df_results.empty else []
    source = st.sidebar.selectbox(
        "Source",
        options=["(all)"] + sources,
        index=0,
    )
    source = None if source == "(all)" else source

    if not df_results.empty:
        default_start = df_results["Timestamp"].min().date()
        default_end = df_results["Timestamp"].max().date()
    else:
        default_end = dt.date.today()
        default_start = default_end - dt.timedelta(days=30)

    st.sidebar.write("Date range")
    date_range_input = st.sidebar.date_input(
        "From / To",
        value=(default_start, default_end),
        min_value=min(dt.date(2018, 1, 1), default_start),
        max_value=max(dt.date.today() + dt.timedelta(days=365), default_end),
        help="Pick any custom window to pull anomalies from the warehouse.",
    )
    if isinstance(date_range_input, (list, tuple)) and len(date_range_input) == 2:
        date_range = (date_range_input[0], date_range_input[1])
    else:
        date_range = (default_start, default_end)

    device_filter_raw = st.sidebar.text_input(
        "Limit to device IDs (comma separated)",
        value="",
        help="Leave blank to include all devices in the selected window.",
    )
    device_ids = parse_device_ids(device_filter_raw) or None

    st.sidebar.header("Overview settings")
    top_n_devices = st.sidebar.slider(
        "Top N devices to display",
        min_value=5,
        max_value=100,
        value=10,
        step=5,
    )
    min_total_points = st.sidebar.number_input(
        "Minimum rows per device",
        min_value=1,
        value=50,
        help="Devices with fewer rows than this are excluded from the overview rankings.",
    )
    min_anomalies = st.sidebar.number_input(
        "Minimum anomalies per device",
        min_value=1,
        value=3,
        help="Require this many anomalies before ranking a device.",
    )

    score_filter_enabled = st.sidebar.checkbox(
        "Filter by anomaly score threshold",
        value=False,
        help="Only show anomalies whose score is below the specified threshold.",
    )
    anomaly_score_max: float | None = None
    if score_filter_enabled:
        anomaly_score_max = st.sidebar.number_input(
            "Max anomaly score (lower = more anomalous)",
            value=-0.1,
            step=0.1,
        )

    st.sidebar.header("Event visibility")
    event_min_rows = st.sidebar.number_input(
        "Minimum rows per event",
        min_value=1,
        value=1,
    )
    event_min_duration = st.sidebar.number_input(
        "Minimum duration (days)",
        min_value=0,
        value=0,
    )

    return DashboardFilters(
        source=source,
        date_range=date_range,
        device_ids=device_ids,
        top_n_devices=top_n_devices,
        min_total_points=int(min_total_points),
        min_anomalies=int(min_anomalies),
        anomaly_score_max=anomaly_score_max,
        event_min_rows=int(event_min_rows),
        event_min_duration=int(event_min_duration),
    )


def apply_result_filters(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    if df.empty:
        return df

    df_filtered = df.copy()

    if filters.device_ids:
        df_filtered = df_filtered[df_filtered["DeviceId"].isin(filters.device_ids)]

    if filters.anomaly_score_max is not None and "AnomalyScore" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["AnomalyScore"] <= filters.anomaly_score_max]

    return df_filtered


def apply_event_filters(df: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
    if df.empty:
        return df
    df_filtered = df.copy()

    if {"EventStart", "EventEnd"} <= set(df_filtered.columns):
        start = pd.to_datetime(df_filtered["EventStart"])
        end = pd.to_datetime(df_filtered["EventEnd"])
        duration = (end - start).dt.days.fillna(0).astype(int) + 1
        df_filtered["DurationDays"] = duration.clip(lower=1)
    else:
        df_filtered["DurationDays"] = 1

    if filters.device_ids:
        df_filtered = df_filtered[df_filtered["DeviceId"].isin(filters.device_ids)]

    if filters.event_min_rows > 1 and "RowCount" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["RowCount"] >= filters.event_min_rows]

    if filters.event_min_duration > 0:
        df_filtered = df_filtered[df_filtered["DurationDays"] >= filters.event_min_duration]

    return df_filtered


__all__ = [
    "DashboardFilters",
    "sidebar_filters",
    "apply_result_filters",
    "apply_event_filters",
    "parse_feature_list",
    "parse_device_ids",
]
