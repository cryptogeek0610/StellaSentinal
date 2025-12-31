from __future__ import annotations

import datetime as dt

import streamlit as st

from device_anomaly.cli.synthetic_experiment import run_synthetic_experiment
from device_anomaly.cli.dw_experiment import run_dw_experiment
from device_anomaly.config.experiment_config import (
    DetectionConfig,
    EventConfig,
    SyntheticExperimentConfig,
    DWExperimentConfig,
)

from .filters import parse_feature_list, parse_device_ids


def render_training_controls() -> None:
    st.subheader("Train / Refresh anomaly detector")
    with st.expander("Launch training runs directly from the dashboard", expanded=False):
        st.write(
            "Kick off synthetic experiments for quick smoke tests or run DW slices "
            "without leaving the UI. Results persist to the same DW tables the charts read from."
        )

        tab_synth, tab_dw = st.tabs(["Synthetic data", "DW data"])

        with tab_synth:
            with st.form("train_synth_form"):
                n_devices = st.number_input("Devices", min_value=5, max_value=1000, value=20, step=5)
                n_days = st.number_input("Days", min_value=3, max_value=60, value=14)
                anomaly_rate = st.slider(
                    "Injected anomaly rate",
                    min_value=0.01,
                    max_value=0.2,
                    value=0.05,
                    step=0.01,
                )

                st.markdown("**Detection settings**")
                synth_window = st.slider("Rolling window (days)", min_value=6, max_value=48, value=24, step=2)
                synth_contamination = st.slider(
                    "IsolationForest contamination",
                    min_value=0.005,
                    max_value=0.2,
                    value=0.02,
                    step=0.005,
                )
                synth_features_raw = st.text_input(
                    "Feature columns (comma separated; blank = auto)",
                    value="",
                    help="Provide specific feature names from the feature table if you want to restrict training columns.",
                )
                synth_features = parse_feature_list(synth_features_raw)

                st.markdown("**Event aggregation**")
                synth_max_gap = st.number_input(
                    "Max gap between anomalies (hours)", min_value=1, max_value=24, value=3
                )
                synth_top_n = st.number_input("Top N devices for events", min_value=1, max_value=50, value=5)
                synth_min_points = st.number_input(
                    "Minimum rows per device", min_value=10, max_value=1000, value=100, step=10
                )
                synth_min_anoms = st.number_input("Minimum anomalies per device", min_value=1, max_value=50, value=5)

                submit_synth = st.form_submit_button("Run synthetic experiment")
                if submit_synth:
                    try:
                        config = SyntheticExperimentConfig(
                            n_devices=int(n_devices),
                            n_days=int(n_days),
                            anomaly_rate=float(anomaly_rate),
                            detection=DetectionConfig(
                                window=int(synth_window),
                                contamination=float(synth_contamination),
                                feature_overrides=synth_features or None,
                            ),
                            events=EventConfig(
                                max_gap_hours=int(synth_max_gap),
                                top_n_devices=int(synth_top_n),
                                min_total_points=int(synth_min_points),
                                min_anomalies=int(synth_min_anoms),
                            ),
                        )
                        with st.spinner("Training on synthetic data..."):
                            run_synthetic_experiment(config=config)
                        st.success(
                            "Synthetic experiment completed and anomalies persisted. "
                            "Click 'Reload dashboard data' below to refresh charts."
                        )
                    except Exception as exc:  # pragma: no cover
                        st.error(f"Failed to run synthetic experiment: {exc}")

        with tab_dw:
            today = dt.date.today()
            default_start = today - dt.timedelta(days=30)
            earliest = dt.date(2018, 1, 1)
            with st.form("train_dw_form"):
                dw_dates = st.date_input(
                    "DW date range",
                    value=(default_start, today),
                    min_value=earliest,
                    max_value=today,
                )
                if isinstance(dw_dates, (list, tuple)) and len(dw_dates) == 2:
                    dw_start, dw_end = dw_dates
                else:
                    dw_start, dw_end = default_start, today

                dw_row_limit = st.number_input(
                    "Row limit (0 = no limit)",
                    min_value=0,
                    max_value=2_000_000,
                    value=200_000,
                    step=50_000,
                )
                dw_devices_raw = st.text_input(
                    "Optional device ID list (comma separated)",
                    value="",
                    help="Leave blank to sample all devices that match the other filters.",
                )

                st.markdown("**Detection settings**")
                dw_window = st.slider(
                    "Rolling window (days)",
                    min_value=6,
                    max_value=60,
                    value=24,
                    step=2,
                    key="dw_window_slider",
                )
                dw_contamination = st.slider(
                    "IsolationForest contamination",
                    min_value=0.005,
                    max_value=0.2,
                    value=0.03,
                    step=0.005,
                    key="dw_contamination_slider",
                )
                dw_features_raw = st.text_input(
                    "Feature columns (comma separated; blank = auto)",
                    value="",
                    help="Provide explicit DW feature names to use for training or leave blank to auto-select.",
                )
                dw_features = parse_feature_list(dw_features_raw)

                st.markdown("**Event aggregation**")
                dw_max_gap = st.number_input(
                    "Max gap between anomalies (hours)", min_value=1, max_value=24, value=4
                )
                dw_top_n = st.number_input("Top N devices for events", min_value=1, max_value=50, value=10)
                dw_min_points = st.number_input(
                    "Minimum rows per device", min_value=10, max_value=5000, value=200, step=10
                )
                dw_min_anoms = st.number_input("Minimum anomalies per device", min_value=1, max_value=50, value=5)

                submit_dw = st.form_submit_button("Run DW experiment")
                if submit_dw:
                    dw_device_ids = parse_device_ids(dw_devices_raw)
                    try:
                        config = DWExperimentConfig(
                            start_date=str(dw_start),
                            end_date=str(dw_end),
                            detection=DetectionConfig(
                                window=int(dw_window),
                                contamination=float(dw_contamination),
                                feature_overrides=dw_features or None,
                            ),
                            events=EventConfig(
                                max_gap_hours=int(dw_max_gap),
                                top_n_devices=int(dw_top_n),
                                min_total_points=int(dw_min_points),
                                min_anomalies=int(dw_min_anoms),
                            ),
                            row_limit=int(dw_row_limit) if dw_row_limit > 0 else None,
                            device_ids=dw_device_ids or None,
                        )
                        with st.spinner("Training on DW slice..."):
                            run_dw_experiment(config=config)
                        st.success(
                            "DW experiment completed and anomalies persisted. "
                            "Click 'Reload dashboard data' below to pull the latest results."
                        )
                    except Exception as exc:  # pragma: no cover
                        st.error(f"Failed to run DW experiment: {exc}")

        if st.button("Reload dashboard data"):
            _rerun_app()


def _rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):  # pragma: no cover
        st.experimental_rerun()


__all__ = ["render_training_controls"]
