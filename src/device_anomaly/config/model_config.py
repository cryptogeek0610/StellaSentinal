from __future__ import annotations

from device_anomaly import __version__


DEFAULT_MODEL_FAMILY = "iforest_v1"


def make_model_version(source: str, model_family: str = DEFAULT_MODEL_FAMILY) -> str:
    """
    Build a model_version string like:
        dw_iforest_v1_0.1.0
        synthetic_iforest_v1_0.1.0
    """
    source = source.strip().lower()
    return f"{source}_{model_family}_{__version__}"
