from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from device_anomaly.llm.pattern_explanations import generate_and_save_pattern_explanation

from .filters import DashboardFilters
from .data import get_engine


def render_pattern_tab(df_patterns: pd.DataFrame, filters: DashboardFilters) -> None:
    st.subheader("Device pattern summaries")

    if df_patterns.empty:
        st.info("No device pattern records found (ml_DevicePatterns).")
        return

    df_patterns = df_patterns.copy()
    if filters.device_ids:
        df_patterns = df_patterns[df_patterns["DeviceId"].isin(filters.device_ids)]
        if df_patterns.empty:
            st.info("No pattern records match the selected device IDs.")
            return
    df_patterns["AnomalyRatePct"] = df_patterns["AnomalyRate"] * 100.0

    st.markdown("### Per-device anomaly rate and event count")

    fig = px.scatter(
        df_patterns,
        x="EventCount",
        y="AnomalyRatePct",
        hover_name="DeviceId",
        size="TotalAnomalies",
        title="Anomaly rate vs. number of events",
        labels={"AnomalyRatePct": "Anomaly rate (%)"},
    )
    st.plotly_chart(fig, width="stretch")

    device_ids = sorted(df_patterns["DeviceId"].unique().tolist())
    if not device_ids:
        st.info("No devices available for pattern summaries.")
        return
    device_id = st.selectbox("Select device for pattern summary", device_ids)

    df_dev = df_patterns[df_patterns["DeviceId"] == device_id].iloc[0]

    st.markdown(f"### Pattern explanation – Device {device_id}")
    st.write(f"Period: {df_dev['PeriodStart']} → {df_dev['PeriodEnd']}")
    st.write(
        f"Anomaly rate: {df_dev['AnomalyRate']:.3%}, "
        f"Events: {df_dev['EventCount']}, "
        f"Total anomalies: {int(df_dev['TotalAnomalies'])}"
    )

    with st.expander("Raw pattern JSON"):
        st.json(df_dev["PatternJson"])

    st.markdown("#### Pattern explanation")
    engine = get_engine()

    if df_dev.get("Explanation"):
        st.write(df_dev["Explanation"])
    else:
        if st.button("Generate pattern explanation", key=f"gen_pattern_{df_dev['Id']}"):
            exp = generate_and_save_pattern_explanation(engine, int(df_dev["Id"]))
            st.write(exp)
        else:
            st.info("No explanation yet. Click the button to generate and persist it.")


__all__ = ["render_pattern_tab"]
