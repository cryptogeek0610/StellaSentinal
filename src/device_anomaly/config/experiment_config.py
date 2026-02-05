from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    """Config for the anomaly detector itself."""
    window: int = 12
    contamination: float = 0.03
    feature_overrides: list[str] | None = None


@dataclass
class EventConfig:
    """
    Config for grouping anomalies into events
    and picking which devices to send to the LLM.
    """
    max_gap_hours: int = 2

    top_n_devices: int = 5
    min_total_points: int = 50
    min_anomalies: int = 3



@dataclass
class SyntheticExperimentConfig:
    # synthetic data generation
    n_devices: int = 10
    n_days: int = 14
    anomaly_rate: float = 0.05

    detection: DetectionConfig = field(
        default_factory=lambda: DetectionConfig(window=12, contamination=0.03)
    )
    events: EventConfig = field(
        # for synthetic we usually look at smaller top_n
        default_factory=lambda: EventConfig(
            max_gap_hours=2,
            top_n_devices=5,
            min_total_points=50,
            min_anomalies=3,
        )
    )


@dataclass
class DWExperimentConfig:
    start_date: str = ""
    end_date: str = ""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    events: EventConfig = field(default_factory=EventConfig)

    row_limit: int | None = 1_000_000
    device_ids: list[int] | None = None
