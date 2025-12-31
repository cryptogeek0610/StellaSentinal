from __future__ import annotations

import pandas as pd
import streamlit as st

from device_anomaly.ui.dashboard_app.data import (
    get_engine,
    load_anomaly_results,
    load_anomaly_events,
    load_device_patterns,
)
from device_anomaly.ui.dashboard_app.filters import (
    DashboardFilters,
    sidebar_filters,
    apply_result_filters,
    apply_event_filters,
)
from device_anomaly.ui.dashboard_app.training import render_training_controls
from device_anomaly.ui.dashboard_app.overview import render_overview_tab
from device_anomaly.ui.dashboard_app.device_detail import render_device_detail_tab
from device_anomaly.ui.dashboard_app.patterns_tab import render_pattern_tab
from device_anomaly.ui.dashboard_app.diagnostics import render_diagnostics_tab


def main() -> None:
    st.set_page_config(
        page_title="Stella Sentinel",
        layout="wide",
    )

    st.title("📊 SOTI Stella Sentinel")

    engine = get_engine()

    df_results_all = load_anomaly_results(engine)
    filters = sidebar_filters(df_results_all)

    df_results = load_anomaly_results(
        engine,
        source=filters.source,
        date_range=filters.date_range,
    )
    df_results = apply_result_filters(df_results, filters)

    df_events = load_anomaly_events(
        engine,
        source=filters.source,
        date_range=filters.date_range,
    )
    df_events = apply_event_filters(df_events, filters)

    try:
        df_patterns = load_device_patterns(engine, source=filters.source)
    except Exception:
        df_patterns = pd.DataFrame()

    render_training_controls()

    tab_overview, tab_device, tab_patterns, tab_diag = st.tabs(
        ["Overview", "Device detail", "Patterns", "Diagnostics"]
    )

    with tab_overview:
        render_overview_tab(df_results, df_events, filters)

    with tab_device:
        render_device_detail_tab(df_results, df_events, filters)

    with tab_patterns:
        render_pattern_tab(df_patterns, filters)

    with tab_diag:
        render_diagnostics_tab(df_results, df_events, df_patterns, filters)


if __name__ == "__main__":
    main()
