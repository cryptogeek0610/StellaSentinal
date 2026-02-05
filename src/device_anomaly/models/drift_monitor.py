from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def compute_feature_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    anomaly_scores: pd.Series | None = None,
) -> dict:
    stats = {
        "row_count": len(df),
        "features": {},
    }
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        stats["features"][col] = {
            "median": float(series.median(skipna=True)),
            "mad": float((series - series.median()).abs().median() or 1e-6),
        }

    if anomaly_scores is not None:
        series = pd.to_numeric(anomaly_scores, errors="coerce")
        stats["anomaly_score"] = {
            "mean": float(series.mean(skipna=True)),
            "std": float(series.std(ddof=0) or 1e-6),
        }
    return stats


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, default=float))


def load_stats(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compare_stats(current: dict, baseline: dict, z_threshold: float = 3.0) -> list[str]:
    warnings: list[str] = []
    for feature, cur_stats in current.get("features", {}).items():
        base_stats = (baseline.get("features") or {}).get(feature)
        if not base_stats:
            continue
        mad = base_stats.get("mad") or 1e-6
        z_shift = abs(cur_stats["median"] - base_stats["median"]) / mad
        if z_shift > z_threshold:
            warnings.append(
                f"Feature '{feature}' median shifted by {z_shift:.1f} MADs "
                "relative to previous run."
            )
    return warnings
