from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from device_anomaly.models.baseline import BaselineFeedback, apply_feedback, save_baselines_versioned
from device_anomaly.models.drift_monitor import compare_stats

from .filters import DashboardFilters
from .data import (
    expand_metrics,
    load_stats_for_source,
    compute_current_stats,
    load_model_config,
    load_baseline_suggestions,
    load_baseline_stats,
)


def render_diagnostics_tab(
    df_results: pd.DataFrame,
    df_events: pd.DataFrame,
    df_patterns: pd.DataFrame,
    filters: DashboardFilters,
) -> None:
    st.subheader("Diagnostics & Drift")

    latest_result = df_results["Timestamp"].max() if not df_results.empty and "Timestamp" in df_results.columns else None
    latest_event = df_events["EventStart"].max() if not df_events.empty and "EventStart" in df_events.columns else None
    latest_pattern = df_patterns["PeriodEnd"].max() if not df_patterns.empty and "PeriodEnd" in df_patterns.columns else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Last anomaly timestamp", str(latest_result) if latest_result else "n/a")
    col2.metric("Last event start", str(latest_event) if latest_event else "n/a")
    col3.metric("Last pattern period end", str(latest_pattern) if latest_pattern else "n/a")

    if not df_events.empty:
        size_values = df_events["AnomalyScoreMean"].abs().clip(lower=1e-6)
        fig_corr = px.scatter(
            df_events,
            x="DurationDays",
            y="RowCount",
            color="AnomalyScoreMin",
            size=size_values,
            hover_data=["DeviceId"],
            title="Event duration vs. rows (color = worst score)",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_corr, width="stretch")

    st.markdown("### Model configuration")
    model_config, model_path = load_model_config(filters.source)
    if model_config:
        st.caption(f"Model config file: {model_path}")
        st.json(model_config)
    else:
        st.info("No model configuration artifact found. Run a training job to persist config.")

    st.markdown("### Baseline statistics")
    baselines, baselines_path = load_baseline_stats(filters.source)
    if baselines:
        summary_rows = []
        for level_name, df_level in baselines.items():
            groups = df_level["__group_key__"].nunique() if "__group_key__" in df_level.columns else 0
            features = df_level["feature"].nunique() if "feature" in df_level.columns else 0
            summary_rows.append(
                {
                    "level": level_name,
                    "group_count": int(groups),
                    "feature_count": int(features),
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), width="stretch")
        if baselines_path:
            st.caption(f"Baseline file: {baselines_path}")
    else:
        st.info("No baseline statistics found. Run a training job to persist baselines.")

    st.markdown("### Drift diagnostics")
    metrics_df = expand_metrics(df_results)
    baseline_stats, baseline_path = load_stats_for_source(filters.source)
    if baseline_stats and not metrics_df.empty:
        current_stats = compute_current_stats(df_results)
        warnings = compare_stats(current_stats, baseline_stats) if current_stats else []
        if warnings:
            for warning in warnings[:5]:
                st.warning(warning)
        else:
            st.success("No significant drift detected compared to saved stats.")

        diff_rows = []
        if current_stats:
            for feat, base_vals in baseline_stats.get("features", {}).items():
                cur_vals = current_stats.get("features", {}).get(feat)
                if not cur_vals:
                    continue
                shift = cur_vals["median"] - base_vals.get("median", 0.0)
                diff_rows.append({"feature": feat, "median_shift": shift})

        if diff_rows:
            diff_df = (
                pd.DataFrame(diff_rows)
                .sort_values("median_shift", key=lambda s: s.abs(), ascending=False)
                .head(10)
            )
            st.dataframe(diff_df, width="stretch")

        st.caption(f"Baseline stats file: {baseline_path}")
    elif baseline_stats:
        st.info("Baseline stats available but MetricsJson columns were empty for this slice.")
    else:
        st.info("No saved drift statistics found yet. Run a training job to persist stats.")

    st.markdown("### Baseline adjustment suggestions")
    suggestions, suggestions_path = load_baseline_suggestions(filters.source)
    if suggestions:
        st.caption(f"Suggestions file: {suggestions_path}")
        df_suggestions = pd.DataFrame(suggestions)
        st.dataframe(df_suggestions, width="stretch")

        if st.button("Apply baseline suggestions"):
            if not baselines or not baselines_path:
                st.error("Baseline file not found; run a training job to generate baselines first.")
            else:
                feedback_items = []
                for row in suggestions:
                    try:
                        adjustment = float(row["proposed_new_median"]) - float(row["baseline_median"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    feedback_items.append(
                        BaselineFeedback(
                            level=row.get("level", ""),
                            group_key=row.get("group_key", ""),
                            feature=row.get("feature", ""),
                            adjustment=adjustment,
                            reason=row.get("rationale"),
                        )
                    )

                if not feedback_items:
                    st.error("No valid adjustments found in suggestions.")
                else:
                    updated = apply_feedback(baselines, feedback_items, learning_rate=0.35)
                    backup = save_baselines_versioned(updated, Path(baselines_path))
                    applied_path = Path("artifacts/baseline_suggestions_applied.json")
                    applied_path.write_text(json.dumps(suggestions, indent=2, default=float))
                    if backup:
                        st.success(f"Applied suggestions. Previous baseline backed up to {backup}.")
                    else:
                        st.success("Applied suggestions and updated baseline file.")
                    st.info("Retrain the model to apply updated baselines.")
        if st.button("Dismiss baseline suggestions"):
            rejected_path = Path("artifacts/baseline_suggestions_rejected.json")
            rejected_path.write_text(json.dumps(suggestions, indent=2, default=float))
            st.success("Suggestions archived as rejected.")
    else:
        st.info("No baseline suggestions available yet.")


__all__ = ["render_diagnostics_tab"]
