from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from device_anomaly.models.baseline import load_baselines, load_data_driven_baselines_payload
from device_anomaly.models.model_registry import resolve_artifact_path, resolve_model_artifacts


@dataclass
class BaselineResolution:
    kind: str
    path: Path
    payload: dict[str, Any]
    schema_version: str
    model_version: str | None = None
    device_type_col: str | None = None


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("baseline_type") == "data_driven" and "baselines" in payload:
        return payload
    return {
        "schema_version": "legacy_v0",
        "baseline_type": "legacy",
        "baselines": payload,
    }


def _detect_baseline_kind(payload: dict[str, Any]) -> str:
    if payload.get("baseline_type") == "data_driven":
        return "data_driven"
    # Heuristic: legacy baseline files are keyed by level with list records
    if payload and all(isinstance(v, list) for v in payload.values()):
        return "legacy"
    return "data_driven"


def resolve_production_baselines(models_dir: Path | None = None) -> BaselineResolution | None:
    artifacts = resolve_model_artifacts(models_dir)
    metadata = artifacts.metadata or {}

    baselines_path = resolve_artifact_path(
        artifacts.model_dir,
        (metadata.get("artifacts") or {}).get("baselines_path"),
    )
    if baselines_path is None:
        candidate = artifacts.model_dir / "baselines.json"
        baselines_path = candidate if candidate.exists() else None

    if baselines_path is None or not baselines_path.exists():
        return None

    payload = load_data_driven_baselines_payload(baselines_path)
    payload = _normalize_payload(payload)
    kind = _detect_baseline_kind(payload)
    schema_version = payload.get("schema_version", "unknown")
    model_version = metadata.get("model_version")
    device_type_col = (metadata.get("config") or {}).get("device_type_col")

    return BaselineResolution(
        kind=kind,
        path=baselines_path,
        payload=payload,
        schema_version=schema_version,
        model_version=model_version,
        device_type_col=device_type_col,
    )


def resolve_legacy_baselines(source: str) -> BaselineResolution | None:
    path = (Path("artifacts") / f"{source}_baselines.json").resolve()
    if not path.exists():
        return None
    payload = load_baselines(path)
    if not payload:
        return None
    return BaselineResolution(
        kind="legacy",
        path=path,
        payload={"baselines": payload, "baseline_type": "legacy", "schema_version": "legacy_v0"},
        schema_version="legacy_v0",
    )


def resolve_baselines(source: str, models_dir: Path | None = None) -> BaselineResolution | None:
    if source == "dw":
        production = resolve_production_baselines(models_dir)
        if production is not None:
            return production
        logging.getLogger(__name__).warning(
            "Production baselines not found; falling back to legacy artifacts."
        )
    return resolve_legacy_baselines(source)


def data_driven_to_legacy(
    payload: dict[str, Any], device_type_col: str | None
) -> dict[str, pd.DataFrame]:
    baselines = payload.get("baselines", {})
    rows = []
    for metric, data in baselines.items():
        global_stats = data.get("global") or {}
        rows.append(
            {
                "__group_key__": "all",
                "feature": metric,
                "median": float(global_stats.get("median", 0.0)),
                "mad": float(global_stats.get("mad", 1e-6) or 1e-6),
            }
        )
    result = {"global": pd.DataFrame(rows)} if rows else {}

    if device_type_col and any(data.get("by_device_type") for data in baselines.values()):
        dtype_rows = []
        for metric, data in baselines.items():
            by_device = data.get("by_device_type") or {}
            for dtype, stats in by_device.items():
                dtype_rows.append(
                    {
                        "__group_key__": str(dtype),
                        "feature": metric,
                        "median": float(stats.get("median", 0.0)),
                        "mad": float(stats.get("mad", 1e-6) or 1e-6),
                    }
                )
        if dtype_rows:
            result["device_type"] = pd.DataFrame(dtype_rows)

    return result


def load_legacy_frames(resolution: BaselineResolution) -> dict[str, pd.DataFrame]:
    if resolution.kind == "legacy":
        return resolution.payload.get("baselines", {})
    return data_driven_to_legacy(resolution.payload, resolution.device_type_col)


def update_data_driven_baseline(
    payload: dict[str, Any],
    level: str,
    group_key: str,
    feature: str,
    adjustment: float,
    device_type_col: str | None,
) -> dict[str, Any]:
    baselines = payload.get("baselines", {})
    if feature not in baselines:
        return payload

    if level == "global":
        global_stats = baselines[feature].setdefault("global", {})
        current = float(global_stats.get("median", 0.0))
        global_stats["median"] = current + adjustment
        return payload

    if level in {"device_type", device_type_col or ""}:
        by_device = baselines[feature].setdefault("by_device_type", {})
        if group_key in by_device:
            current = float(by_device[group_key].get("median", 0.0))
            by_device[group_key]["median"] = current + adjustment
        return payload

    return payload


def save_baseline_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
