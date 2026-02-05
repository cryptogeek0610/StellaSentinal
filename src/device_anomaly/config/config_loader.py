from __future__ import annotations

import json
import os
from typing import Any

from device_anomaly.config.experiment_config import (
    DetectionConfig,
    DWExperimentConfig,
    EventConfig,
    SyntheticExperimentConfig,
)


def _load_raw_config(path: str) -> dict[str, Any]:
    """
    Load a dict from a YAML or JSON config file.

    - .yaml / .yml -> requires PyYAML (pip install pyyaml)
    - .json        -> uses built-in json
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            ) from e

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config extension '{ext}'. Use .yaml / .yml / .json."
        )

    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON/YAML object (mapping).")

    return data


# ------------ Synthetic config ------------


def _apply_synthetic_config_from_dict(
    data: dict[str, Any],
    base: SyntheticExperimentConfig | None = None,
) -> SyntheticExperimentConfig:
    """
    Overlay values from dict onto a SyntheticExperimentConfig,
    preserving defaults for anything not specified.
    """
    cfg = base or SyntheticExperimentConfig()

    # Top-level fields
    if "n_devices" in data:
        cfg.n_devices = int(data["n_devices"])
    if "n_days" in data:
        cfg.n_days = int(data["n_days"])
    if "anomaly_rate" in data:
        cfg.anomaly_rate = float(data["anomaly_rate"])

    # Detection nested config
    det_data = data.get("detection") or {}
    if det_data:
        det_cfg = cfg.detection or DetectionConfig()
        if "window" in det_data:
            det_cfg.window = int(det_data["window"])
        if "contamination" in det_data:
            det_cfg.contamination = float(det_data["contamination"])
        if "feature_overrides" in det_data:
            fval = det_data["feature_overrides"]
            if fval is None:
                det_cfg.feature_overrides = None
            elif isinstance(fval, str):
                det_cfg.feature_overrides = [c.strip() for c in fval.split(",") if c.strip()]
            else:
                det_cfg.feature_overrides = [str(c).strip() for c in fval if str(c).strip()]
        cfg.detection = det_cfg

    # Events nested config
    ev_data = data.get("events") or {}
    if ev_data:
        ev_cfg = cfg.events or EventConfig()
        if "max_gap_hours" in ev_data:
            ev_cfg.max_gap_hours = int(ev_data["max_gap_hours"])
        if "top_n_devices" in ev_data:
            ev_cfg.top_n_devices = int(ev_data["top_n_devices"])
        if "min_total_points" in ev_data:
            ev_cfg.min_total_points = int(ev_data["min_total_points"])
        if "min_anomalies" in ev_data:
            ev_cfg.min_anomalies = int(ev_data["min_anomalies"])
        cfg.events = ev_cfg

    return cfg


def load_synthetic_config(path: str) -> SyntheticExperimentConfig:
    raw = _load_raw_config(path)
    return _apply_synthetic_config_from_dict(raw)


# ------------ DW config ------------


def _apply_dw_config_from_dict(
    data: dict[str, Any],
    base: DWExperimentConfig | None = None,
) -> DWExperimentConfig:
    """
    Overlay values from dict onto a DWExperimentConfig,
    preserving defaults for anything not specified.
    """
    cfg = base or DWExperimentConfig()

    # Top-level fields
    if "start_date" in data:
        cfg.start_date = str(data["start_date"])
    if "end_date" in data:
        cfg.end_date = str(data["end_date"])
    if "row_limit" in data:
        cfg.row_limit = int(data["row_limit"]) if data["row_limit"] is not None else None
    if "device_ids" in data:
        # device_ids can be null or a list
        devs = data["device_ids"]
        if devs is None:
            cfg.device_ids = None
        else:
            cfg.device_ids = [int(d) for d in devs]

    # Detection nested config
    det_data = data.get("detection") or {}
    if det_data:
        det_cfg = cfg.detection or DetectionConfig()
        if "window" in det_data:
            det_cfg.window = int(det_data["window"])
        if "contamination" in det_data:
            det_cfg.contamination = float(det_data["contamination"])
        if "feature_overrides" in det_data:
            fval = det_data["feature_overrides"]
            if fval is None:
                det_cfg.feature_overrides = None
            elif isinstance(fval, str):
                det_cfg.feature_overrides = [c.strip() for c in fval.split(",") if c.strip()]
            else:
                det_cfg.feature_overrides = [str(c).strip() for c in fval if str(c).strip()]
        cfg.detection = det_cfg

    # Events nested config
    ev_data = data.get("events") or {}
    if ev_data:
        ev_cfg = cfg.events or EventConfig()
        if "max_gap_hours" in ev_data:
            ev_cfg.max_gap_hours = int(ev_data["max_gap_hours"])
        if "top_n_devices" in ev_data:
            ev_cfg.top_n_devices = int(ev_data["top_n_devices"])
        if "min_total_points" in ev_data:
            ev_cfg.min_total_points = int(ev_data["min_total_points"])
        if "min_anomalies" in ev_data:
            ev_cfg.min_anomalies = int(ev_data["min_anomalies"])
        cfg.events = ev_cfg

    return cfg


def load_dw_config(path: str) -> DWExperimentConfig:
    raw = _load_raw_config(path)
    return _apply_dw_config_from_dict(raw)
