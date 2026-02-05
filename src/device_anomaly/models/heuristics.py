from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass
class HeuristicRule:
    name: str
    column: str
    threshold: float
    op: Callable[[pd.Series, float], pd.Series]
    min_consecutive: int = 1
    severity: float = 0.5
    description: str | None = None


def _op_map() -> dict[str, Callable[[pd.Series, float], pd.Series]]:
    return {
        ">": lambda s, t: s > t,
        ">=": lambda s, t: s >= t,
        "<": lambda s, t: s < t,
        "<=": lambda s, t: s <= t,
    }


def build_rules_from_dicts(configs: Iterable[dict]) -> list[HeuristicRule]:
    rules: list[HeuristicRule] = []
    for cfg in configs:
        op_str = cfg.get("op", ">=")
        op = _op_map().get(op_str)
        if op is None:
            continue
        rules.append(
            HeuristicRule(
                name=cfg.get("name", cfg.get("column", "rule")),
                column=cfg["column"],
                threshold=float(cfg.get("threshold", 0)),
                op=op,
                min_consecutive=int(cfg.get("min_consecutive", 1)),
                severity=float(cfg.get("severity", 0.5)),
                description=cfg.get("description"),
            )
        )
    return rules


def apply_heuristics(df: pd.DataFrame, rules: Iterable[HeuristicRule]) -> pd.DataFrame:
    """
    Evaluate threshold-style heuristics and return a long-form dataframe:
      DeviceId, Timestamp (if present), HeuristicName, Reason, HeuristicScore
    """
    if df.empty:
        return pd.DataFrame()

    rules = list(rules)
    if not rules:
        return pd.DataFrame()

    records = []
    for rule in rules:
        if rule.column not in df.columns:
            continue
        series = pd.to_numeric(df[rule.column], errors="coerce")
        mask = rule.op(series, rule.threshold)

        if "DeviceId" in df.columns and "Timestamp" in df.columns:
            df_sorted = df.sort_values(["DeviceId", "Timestamp"])
            series_sorted = pd.to_numeric(df_sorted[rule.column], errors="coerce")
            mask_sorted = rule.op(series_sorted, rule.threshold)
            rolling = (
                mask_sorted.groupby(df_sorted["DeviceId"])
                .rolling(rule.min_consecutive)
                .sum()
                .reset_index(level=0, drop=True)
            )
            mask = (rolling >= rule.min_consecutive)
            mask.index = df_sorted.index
            mask = mask.reindex(df.index).fillna(False)

        for idx in df.index[mask]:
            rec = {
                "DeviceId": int(df.loc[idx, "DeviceId"]) if "DeviceId" in df.columns else None,
                "Timestamp": df.loc[idx, "Timestamp"] if "Timestamp" in df.columns else None,
                "HeuristicName": rule.name,
                "HeuristicScore": float(rule.severity),
                "Reason": rule.description or f"{rule.column} threshold {rule.threshold}",
            }
            records.append(rec)

    return pd.DataFrame.from_records(records)


def summarize_heuristics(flags: pd.DataFrame) -> pd.DataFrame:
    if flags.empty:
        return flags

    grouped = []
    for device_id, grp in flags.groupby("DeviceId"):
        reasons = sorted(set(grp["Reason"].dropna().astype(str)))
        score = float(grp["HeuristicScore"].max())
        grouped.append(
            {
                "DeviceId": int(device_id),
                "HeuristicReasons": "; ".join(reasons),
                "HeuristicScore": score,
            }
        )
    return pd.DataFrame.from_records(grouped)
