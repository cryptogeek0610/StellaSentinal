from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .filters import DashboardFilters
from .data import compute_feature_contributions


def render_overview_tab(
    df_results: pd.DataFrame,
    df_events: pd.DataFrame,
    filters: DashboardFilters,
) -> None:
    st.subheader("Overview")

    if df_results.empty:
        st.info("No anomaly results found for the selected filters.")
        return

    df_view = df_results.copy()
    df_view["Date"] = df_view["Timestamp"].dt.floor("D")
    df_view["IsAnomaly"] = df_view["AnomalyLabel"] == -1

    total_rows = len(df_view)
    total_devices = df_view["DeviceId"].nunique()
    anomalies_total = int(df_view["IsAnomaly"].sum())
    anomaly_rate = anomalies_total / max(total_rows, 1)
    event_devices = df_events["DeviceId"].nunique() if not df_events.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows scored", f"{total_rows:,}")
    col2.metric("Devices observed", f"{total_devices:,}")
    col3.metric("Anomalies detected", f"{anomalies_total:,}", f"{anomaly_rate:.1%}")
    col4.metric("Devices with events", f"{event_devices:,}")

    daily_summary = (
        df_view.groupby("Date")
        .agg(
            total_rows=("DeviceId", "size"),
            anomalies=("IsAnomaly", "sum"),
        )
        .reset_index()
    )
    if not daily_summary.empty:
        daily_long = daily_summary.melt(
            id_vars="Date",
            value_vars=["total_rows", "anomalies"],
            var_name="Metric",
            value_name="Count",
        )
        fig_daily = px.area(
            daily_long,
            x="Date",
            y="Count",
            color="Metric",
            title="Daily activity vs. anomalies",
        )
        fig_daily.update_layout(legend_title="")
        st.plotly_chart(fig_daily, width="stretch")

    fig_hist = px.histogram(
        df_view,
        x="AnomalyScore",
        color=df_view["IsAnomaly"].map({True: "Anomaly", False: "Normal"}),
        nbins=30,
        title="Anomaly score distribution",
        barmode="overlay",
        opacity=0.65,
    )
    fig_hist.update_layout(legend_title="", xaxis_title="IsolationForest score")
    st.plotly_chart(fig_hist, width="stretch")

    df_anom = df_view[df_view["IsAnomaly"]]
    if not df_anom.empty:
        fig_scatter = px.scatter(
            df_anom,
            x="Timestamp",
            y="DeviceId",
            color="AnomalyScore",
            color_continuous_scale="Turbo",
            title="Anomalies over time (color = score severity)",
            hover_data=["ModelVersion"],
        )
        st.plotly_chart(fig_scatter, width="stretch")

    counts = (
        df_view.groupby("DeviceId")
        .agg(
            TotalRows=("DeviceId", "size"),
            AnomalyCount=("IsAnomaly", "sum"),
        )
        .reset_index()
    )
    counts["AnomalyRate"] = counts["AnomalyCount"] / counts["TotalRows"].clip(lower=1)
    counts = counts[
        (counts["TotalRows"] >= filters.min_total_points)
        & (counts["AnomalyCount"] >= filters.min_anomalies)
    ]

    if counts.empty:
        st.info("No devices meet the minimum thresholds. Adjust the filters on the left.")
        return

    st.markdown("### Top devices by anomaly rate")

    counts_top = (
        counts.sort_values(["AnomalyRate", "AnomalyCount"], ascending=[False, False])
        .head(filters.top_n_devices)
    )

    st.caption(
        f"Top {filters.top_n_devices} devices "
        f"(≥{filters.min_total_points} rows, ≥{filters.min_anomalies} anomalies)."
    )

    fig = px.bar(
        counts_top,
        x="DeviceId",
        y="AnomalyRate",
        hover_data=["AnomalyCount", "TotalRows"],
        title="Anomaly rate per device",
    )
    st.plotly_chart(fig, width="stretch")

    if not df_events.empty:
        st.markdown("### Events per device")

        ev_counts = (
            df_events.groupby("DeviceId")
            .agg(
                EventCount=("DeviceId", "size"),
                WorstScore=("AnomalyScoreMin", "min"),
            )
            .reset_index()
        )
        fig2 = px.bar(
            ev_counts.sort_values("EventCount", ascending=False).head(filters.top_n_devices),
            x="DeviceId",
            y="EventCount",
            hover_data=["WorstScore"],
            title="Number of anomaly events per device",
        )
        st.plotly_chart(fig2, width="stretch")

        timeline_devices = counts_top["DeviceId"].tolist() if not counts_top.empty else []
        df_timeline = (
            df_events[df_events["DeviceId"].isin(timeline_devices)].copy()
            if timeline_devices
            else df_events.copy()
        )
        if not df_timeline.empty:
            df_timeline["Device"] = "Device " + df_timeline["DeviceId"].astype(str)
            fig_timeline = px.timeline(
                df_timeline.sort_values("EventStart"),
                x_start="EventStart",
                x_end="EventEnd",
                y="Device",
                color="AnomalyScoreMin",
                hover_data=["DurationDays", "RowCount"],
                title="Event timeline (color = worst anomaly score)",
                color_continuous_scale="Turbo",
            )
            fig_timeline.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_timeline, width="stretch")

    if "ModelVersion" in df_results.columns and not df_results.empty:
        mv_stats = (
            df_view.groupby("ModelVersion")
            .agg(
                TotalRows=("DeviceId", "size"),
                Anomalies=("IsAnomaly", "sum"),
            )
            .reset_index()
        )
        mv_stats["AnomalyRate"] = mv_stats["Anomalies"] / mv_stats["TotalRows"].clip(lower=1)
        mv_stats = mv_stats.sort_values("AnomalyRate", ascending=False).head(15)
        if not mv_stats.empty:
            st.markdown("### Top model versions by anomaly rate")
            fig_mv = px.bar(
                mv_stats,
                x="ModelVersion",
                y="AnomalyRate",
                hover_data=["Anomalies", "TotalRows"],
                title="Anomaly rate by ModelVersion",
            )
            fig_mv.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_mv, width="stretch")

    contrib_df = compute_feature_contributions(df_results)
    if not contrib_df.empty:
        st.markdown("### Feature contributions (anomalous vs typical)")
        top_contrib = contrib_df.head(12)
        fig_contrib = px.bar(
            top_contrib,
            x="delta",
            y="metric",
            orientation="h",
            color="delta",
            color_continuous_scale="RdBu",
            title="Difference between anomaly mean and overall mean",
            labels={"delta": "Anomaly - Overall"},
        )
        fig_contrib.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig_contrib, width="stretch")


__all__ = ["render_overview_tab"]
