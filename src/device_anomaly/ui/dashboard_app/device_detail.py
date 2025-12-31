from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from device_anomaly.llm.event_explanations import generate_and_save_event_explanation
from device_anomaly.llm.explainer import generate_and_save_row_explanation

from .filters import DashboardFilters
from .data import get_engine


def render_device_detail_tab(
    df_results: pd.DataFrame,
    df_events: pd.DataFrame,
    filters: DashboardFilters,
) -> None:
    engine = get_engine()
    st.subheader("Device detail")

    if df_results.empty:
        st.info("No anomaly results to show.")
        return

    device_ids = sorted(df_results["DeviceId"].unique().tolist())
    device_id = st.selectbox("Select device", device_ids)

    df_dev = df_results[df_results["DeviceId"] == device_id].copy()
    df_dev = df_dev.sort_values("Timestamp")

    st.markdown(f"### Device {device_id} – anomaly timeline")

    df_dev["DateOnly"] = df_dev["Timestamp"].dt.date
    daily_device = (
        df_dev.groupby("DateOnly")
        .agg(
            TotalRows=("DeviceId", "size"),
            Anomalies=("AnomalyLabel", lambda s: (s == -1).sum()),
        )
        .reset_index()
    )
    if not daily_device.empty:
        fig_daily_device = px.bar(
            daily_device,
            x="DateOnly",
            y=["TotalRows", "Anomalies"],
            title="Daily rows vs anomalies for this device",
            labels={"value": "Count", "DateOnly": "Date", "variable": "Series"},
        )
        fig_daily_device.update_layout(barmode="overlay", legend_title="")
        st.plotly_chart(fig_daily_device, width="stretch")

    fig_ts = px.scatter(
        df_dev,
        x="Timestamp",
        y="AnomalyScore",
        color=df_dev["AnomalyLabel"].map({1: "Normal", -1: "Anomaly"}),
        size=np.clip(-df_dev["AnomalyScore"], 0.01, None),
        hover_data=["ModelVersion"],
        labels={"color": "Status"},
        title="Anomaly severity dots (color = label, size = severity)",
    )
    fig_ts.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    st.plotly_chart(fig_ts, width="stretch")

    df_anom = df_dev[df_dev["AnomalyLabel"] == -1]
    st.markdown("### Anomalous points")
    display_cols = ["Timestamp", "AnomalyScore", "ModelVersion"]
    if "Explanation" in df_anom.columns:
        display_cols.append("Explanation")
    st.dataframe(
        df_anom[display_cols],
        width="stretch",
        height=300,
    )

    if "Id" in df_anom.columns and not df_anom.empty:
        st.markdown("### Row-level explanations")
        df_candidates = df_anom.sort_values("AnomalyScore").head(20).copy()
        df_candidates["Label"] = (
            df_candidates["Timestamp"].astype(str)
            + " | score="
            + df_candidates["AnomalyScore"].round(4).astype(str)
        )
        selection = st.selectbox(
            "Select an anomalous row",
            df_candidates["Label"].tolist(),
        )
        selected_row = df_candidates[df_candidates["Label"] == selection].iloc[0]

        if selected_row.get("Explanation"):
            st.write(selected_row["Explanation"])
        else:
            if st.button("Generate row explanation", key=f"gen_row_exp_{selected_row['Id']}"):
                exp = generate_and_save_row_explanation(engine, int(selected_row["Id"]))
                st.write(exp)
            else:
                st.info("No explanation yet. Click the button to generate and store it.")

    df_ev_dev = df_events[df_events["DeviceId"] == device_id].copy()
    if not df_ev_dev.empty:
        st.markdown("### Events for this device")

        df_ev_dev = df_ev_dev.sort_values("EventStart")
        st.dataframe(
            df_ev_dev[
                [
                    "EventStart",
                    "EventEnd",
                    "DurationDays",
                    "RowCount",
                    "AnomalyScoreMin",
                    "AnomalyScoreMean",
                ]
            ],
            width="stretch",
            height=300,
        )

        st.markdown("### Event explanations")
        for _, row in df_ev_dev.iterrows():
            title = (
                f"Device {row['DeviceId']} – {row['EventStart']} → {row['EventEnd']} "
                f"(rows={row['RowCount']}, worst={row['AnomalyScoreMin']:.3f}, "
                f"duration={row.get('DurationDays', 1)} days)"
            )
            with st.expander(title):
                if row.get("Explanation"):
                    st.write(row["Explanation"])
                else:
                    if st.button("Generate explanation", key=f"gen_event_exp_{row['Id']}"):
                        exp = generate_and_save_event_explanation(engine, int(row["Id"]))
                        st.write(exp)
                    else:
                        st.info("No explanation yet. Click the button to generate and store it.")
    else:
        st.info("No events recorded for this device in the selected period.")


__all__ = ["render_device_detail_tab"]
