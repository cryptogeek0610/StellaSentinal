"""Stella Sentinel - Streamlit Dashboard.

A legacy Python UI for power users to interact with the anomaly detection system.
Connects to the FastAPI backend API for data.
"""

import os
import time
from datetime import datetime, timedelta

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "30"))

# Page configuration
st.set_page_config(
    page_title="Stella Sentinel",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme matching the React frontend
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0a0e1a;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Status indicators */
    .status-healthy { color: #22c55e; }
    .status-warning { color: #f59e0b; }
    .status-critical { color: #ef4444; }

    /* Section headers */
    .section-header {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }
</style>
""", unsafe_allow_html=True)


def api_get(endpoint: str, params: dict = None) -> dict | list | None:
    """Make a GET request to the API."""
    try:
        url = f"{API_BASE_URL}/api/v1{endpoint}"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def render_kpi_cards(stats: dict):
    """Render KPI metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Anomalies Today",
            value=stats.get("anomalies_today", 0),
            delta=None,
        )

    with col2:
        st.metric(
            label="Critical Issues",
            value=stats.get("critical_issues", 0),
            delta=None,
        )

    with col3:
        st.metric(
            label="Open Cases",
            value=stats.get("open_cases", 0),
            delta=None,
        )

    with col4:
        st.metric(
            label="Resolved Today",
            value=stats.get("resolved_today", 0),
            delta=None,
        )

    with col5:
        st.metric(
            label="Devices Monitored",
            value=stats.get("devices_monitored", 0),
            delta=None,
        )


def render_trend_chart(trends: list):
    """Render anomaly trend chart."""
    if not trends:
        st.info("No trend data available")
        return

    df = pd.DataFrame(trends)
    df["date"] = pd.to_datetime(df["date"])

    fig = px.area(
        df,
        x="date",
        y="anomaly_count",
        title="Anomaly Trends (Last 7 Days)",
        labels={"date": "Date", "anomaly_count": "Anomalies"},
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        title_font_color="#f8fafc",
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
    )

    fig.update_traces(
        fill="tozeroy",
        fillcolor="rgba(245, 158, 11, 0.2)",
        line_color="#f59e0b",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_connection_status(connections: dict):
    """Render connection status panel."""
    st.subheader("System Connections")

    status_emoji = {
        "connected": "",
        "disconnected": "",
        "error": "",
        "not_configured": "",
        "disabled": "",
    }

    services = [
        ("Backend DB", connections.get("backend_db", {})),
        ("XSight SQL", connections.get("dw_sql", {})),
        ("MobiControl SQL", connections.get("mc_sql", {})),
        ("MobiControl API", connections.get("mobicontrol_api", {})),
        ("LLM Service", connections.get("llm", {})),
        ("Redis", connections.get("redis", {})),
        ("Qdrant", connections.get("qdrant", {})),
    ]

    for name, conn in services:
        status = conn.get("status", "disconnected")
        emoji = status_emoji.get(status, "")
        server = conn.get("server", "N/A")
        error = conn.get("error")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"{emoji} **{name}**")
        with col2:
            if status == "connected":
                st.success(f"Connected to {server}")
            elif status == "disabled":
                st.info("Feature disabled")
            elif status == "not_configured":
                st.warning("Not configured")
            else:
                st.error(error or "Disconnected")


def render_score_distribution(if_stats: dict):
    """Render Isolation Forest score distribution chart."""
    score_dist = if_stats.get("score_distribution", {})
    bins = score_dist.get("bins", [])

    if not bins:
        st.info("No score distribution data available")
        return

    # Prepare data for the histogram
    normal_bins = [b for b in bins if not b.get("is_anomaly", False)]
    anomaly_bins = [b for b in bins if b.get("is_anomaly", False)]

    fig = go.Figure()

    if normal_bins:
        fig.add_trace(go.Bar(
            x=[(b["bin_start"] + b["bin_end"]) / 2 for b in normal_bins],
            y=[b["count"] for b in normal_bins],
            name="Normal",
            marker_color="rgba(34, 197, 94, 0.7)",
            width=[(b["bin_end"] - b["bin_start"]) * 0.9 for b in normal_bins],
        ))

    if anomaly_bins:
        fig.add_trace(go.Bar(
            x=[(b["bin_start"] + b["bin_end"]) / 2 for b in anomaly_bins],
            y=[b["count"] for b in anomaly_bins],
            name="Anomaly",
            marker_color="rgba(239, 68, 68, 0.7)",
            width=[(b["bin_end"] - b["bin_start"]) * 0.9 for b in anomaly_bins],
        ))

    fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        barmode="overlay",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        title_font_color="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Score", f"{score_dist.get('mean_score', 0):.3f}")
    with col2:
        st.metric("Median Score", f"{score_dist.get('median_score', 0):.3f}")
    with col3:
        st.metric("Total Normal", score_dist.get("total_normal", 0))
    with col4:
        st.metric("Total Anomalies", score_dist.get("total_anomalies", 0))


def render_model_config(if_stats: dict):
    """Render Isolation Forest model configuration."""
    config = if_stats.get("config", {})

    st.subheader("Model Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Estimators", config.get("n_estimators", "N/A"))
        st.metric("Contamination", f"{config.get('contamination', 0) * 100:.1f}%")

    with col2:
        st.metric("Feature Count", config.get("feature_count", "N/A"))
        st.metric("Random State", config.get("random_state", "N/A"))

    with col3:
        st.metric("Total Predictions", if_stats.get("total_predictions", 0))
        st.metric("Anomaly Rate", f"{if_stats.get('anomaly_rate', 0) * 100:.1f}%")


def main():
    """Main dashboard entry point."""
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x50?text=Stella+Sentinel", width=200)
        st.markdown("---")

        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Anomalies", "Model Stats", "Connections"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            st.write(f"Refreshing every {REFRESH_INTERVAL}s")
            time.sleep(REFRESH_INTERVAL)
            st.rerun()

        # Manual refresh button
        if st.button("Refresh Now"):
            st.rerun()

        st.markdown("---")
        st.caption(f"API: {API_BASE_URL}")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Main content
    st.title("Stella Sentinel")
    st.caption("Device Anomaly Detection Dashboard")

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Anomalies":
        render_anomalies_page()
    elif page == "Model Stats":
        render_model_stats_page()
    elif page == "Connections":
        render_connections_page()


def render_dashboard_page():
    """Render the main dashboard page."""
    # Fetch dashboard stats
    stats = api_get("/dashboard/stats")
    if stats:
        render_kpi_cards(stats)

    st.markdown("---")

    # Two column layout for charts
    col1, col2 = st.columns(2)

    with col1:
        # Anomaly trends
        trends = api_get("/dashboard/trends", {"days": 7})
        if trends:
            render_trend_chart(trends)

    with col2:
        # AI Summary
        st.subheader("AI Summary")
        ai_summary = api_get("/dashboard/ai-summary")
        if ai_summary:
            health_status = ai_summary.get("health_status", "unknown")
            status_colors = {
                "healthy": "success",
                "degraded": "warning",
                "critical": "error",
            }

            if health_status == "healthy":
                st.success(f"Fleet Status: {health_status.upper()}")
            elif health_status == "degraded":
                st.warning(f"Fleet Status: {health_status.upper()}")
            else:
                st.error(f"Fleet Status: {health_status.upper()}")

            st.write(ai_summary.get("summary", "No summary available"))

            priority_actions = ai_summary.get("priority_actions", [])
            if priority_actions:
                st.markdown("**Priority Actions:**")
                for action in priority_actions:
                    st.markdown(f"- {action}")


def render_anomalies_page():
    """Render the anomalies list page."""
    st.header("Recent Anomalies")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "open", "investigating", "resolved", "false_positive"],
        )
    with col2:
        days = st.slider("Days", 1, 30, 7)
    with col3:
        limit = st.number_input("Limit", 10, 100, 50)

    # Build query params
    params = {"limit": limit}
    if status_filter != "All":
        params["status"] = status_filter

    # Fetch anomalies
    anomalies = api_get("/anomalies/", params)

    if anomalies:
        # Convert to DataFrame
        df = pd.DataFrame(anomalies)

        if not df.empty:
            # Format columns
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
            if "anomaly_score" in df.columns:
                df["anomaly_score"] = df["anomaly_score"].round(3)

            # Display columns
            display_cols = ["device_id", "timestamp", "anomaly_score", "status", "explanation"]
            available_cols = [c for c in display_cols if c in df.columns]

            st.dataframe(df[available_cols], use_container_width=True, height=500)
        else:
            st.info("No anomalies found matching the filters")
    else:
        st.error("Failed to fetch anomalies")


def render_model_stats_page():
    """Render the model statistics page."""
    st.header("Isolation Forest Model Statistics")

    # Fetch IF stats
    if_stats = api_get("/dashboard/isolation-forest/stats", {"days": 30})

    if if_stats:
        render_model_config(if_stats)
        st.markdown("---")
        render_score_distribution(if_stats)

        # Feedback stats if available
        feedback_stats = if_stats.get("feedback_stats")
        if feedback_stats:
            st.markdown("---")
            st.subheader("Feedback & Model Adaptation")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Feedback", feedback_stats.get("total_feedback", 0))
            with col2:
                st.metric("False Positives", feedback_stats.get("false_positives", 0))
            with col3:
                st.metric("Confirmed Anomalies", feedback_stats.get("confirmed_anomalies", 0))
            with col4:
                st.metric("Projected Accuracy Gain", f"+{feedback_stats.get('projected_accuracy_gain', 0)}%")

            last_retrain = feedback_stats.get("last_retrain")
            if last_retrain:
                st.caption(f"Last model retrain: {last_retrain}")
    else:
        st.error("Failed to fetch model statistics")


def render_connections_page():
    """Render the connections status page."""
    st.header("System Connections")

    # Fetch connection status
    connections = api_get("/dashboard/connections")

    if connections:
        render_connection_status(connections)

        st.markdown("---")

        # Troubleshooting section
        failed_services = []
        for service_name, service_key in [
            ("XSight SQL", "dw_sql"),
            ("MobiControl SQL", "mc_sql"),
            ("MobiControl API", "mobicontrol_api"),
            ("LLM Service", "llm"),
        ]:
            conn = connections.get(service_key, {})
            if conn.get("error") and conn.get("status") not in ["disabled", "not_configured"]:
                failed_services.append(service_name)

        if failed_services:
            st.warning(f"Failed services: {', '.join(failed_services)}")

            if st.button("Get Troubleshooting Advice"):
                with st.spinner("Analyzing connection issues..."):
                    advice = requests.post(
                        f"{API_BASE_URL}/api/v1/dashboard/connections/troubleshoot",
                        json=connections,
                        timeout=30,
                    )
                    if advice.status_code == 200:
                        result = advice.json()
                        st.markdown("### Troubleshooting Advice")
                        st.markdown(result.get("advice", "No advice available"))
                    else:
                        st.error("Failed to get troubleshooting advice")
        else:
            st.success("All services are healthy!")

        st.caption(f"Last checked: {connections.get('last_checked', 'N/A')}")
    else:
        st.error("Failed to fetch connection status")


if __name__ == "__main__":
    main()
