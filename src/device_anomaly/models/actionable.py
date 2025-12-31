from __future__ import annotations

from typing import List

import pandas as pd

from device_anomaly.config.feature_config import FeatureConfig


def _top_factors(row: pd.Series, top_k: int = 3) -> List[dict]:
    z_cols = [c for c in row.index if "_z_" in c]
    if not z_cols:
        return []
    ranked = sorted(z_cols, key=lambda c: abs(row[c] if pd.notna(row[c]) else 0), reverse=True)
    factors = []
    for col in ranked[:top_k]:
        base = col.split("_z_")[0]
        factors.append(
            {
                "feature": base,
                "z": float(row[col]),
                "domain": FeatureConfig.feature_domains.get(base),
            }
        )
    return factors


def _confidence(row: pd.Series) -> float:
    if "calibrated_probability" in row and pd.notna(row["calibrated_probability"]):
        return float(row["calibrated_probability"])
    score = float(row.get("hybrid_score", row.get("anomaly_score", 0)))
    return float(min(1.0, max(0.0, -score)))


def build_actionable_outputs(
    df_scored: pd.DataFrame,
    model_version: str,
    top_k_factors: int = 3,
    top_n_rows: int = 200,
) -> pd.DataFrame:
    """
    Convert scored anomalies into dashboard-ready records with clear "what/where/why/cause/confidence".
    """
    anomalies = df_scored[df_scored["anomaly_label"] == -1].copy()
    if anomalies.empty:
        return pd.DataFrame()

    records = []
    for _, row in anomalies.head(top_n_rows).iterrows():
        factors = _top_factors(row, top_k=top_k_factors)
        where_parts = []
        for col in ["StoreId", "SiteId", "CustomerId"]:
            if col in anomalies.columns and pd.notna(row.get(col)):
                where_parts.append(f"{col}={row.get(col)}")
        where = ", ".join(where_parts) if where_parts else "global"

        likely_cause = ""
        if row.get("heuristic_reasons"):
            likely_cause = str(row.get("heuristic_reasons"))
        elif factors:
            fw_factor = next((f for f in factors if f["domain"] in {"firmware", "hardware"}), None)
            if fw_factor:
                likely_cause = f"Firmware/hardware divergence on {fw_factor['feature']}"

        records.append(
            {
                "DeviceId": int(row["DeviceId"]),
                "Timestamp": row.get("Timestamp"),
                "What": f"Anomalous behavior across {', '.join(f['feature'] for f in factors[:2])}" if factors else "Anomaly flagged",
                "Where": where,
                "WhyFactors": factors,
                "LikelyCause": likely_cause or "Correlated feature shifts",
                "Confidence": _confidence(row),
                "ModelVersion": model_version,
            }
        )

    return pd.DataFrame.from_records(records)
