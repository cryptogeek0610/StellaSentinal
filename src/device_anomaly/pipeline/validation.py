from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PipelineStage:
    INGESTION = "ingestion"
    FEATURES = "features"
    BASELINES = "baselines"
    TRAINING = "training"
    SCORING = "scoring"
    PERSISTENCE = "persistence"


PIPELINE_ORDER = [
    PipelineStage.INGESTION,
    PipelineStage.FEATURES,
    PipelineStage.BASELINES,
    PipelineStage.TRAINING,
    PipelineStage.SCORING,
    PipelineStage.PERSISTENCE,
]


@dataclass
class PipelineTracker:
    last_stage: str | None = None

    def advance(self, stage: str) -> None:
        if stage not in PIPELINE_ORDER:
            raise ValueError(f"Unknown pipeline stage: {stage}")

        if self.last_stage is None:
            expected = PIPELINE_ORDER[0]
        else:
            last_idx = PIPELINE_ORDER.index(self.last_stage)
            expected = PIPELINE_ORDER[min(last_idx + 1, len(PIPELINE_ORDER) - 1)]

        if stage != expected:
            raise RuntimeError(
                f"Pipeline stage out of order. Expected '{expected}', got '{stage}'."
            )
        self.last_stage = stage
        logger.info("Pipeline stage advanced: %s", stage)


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def ensure_min_rows(df: pd.DataFrame, min_rows: int, name: str) -> None:
    if len(df) < min_rows:
        raise ValueError(f"{name} requires at least {min_rows} rows; got {len(df)}.")


def drop_all_nan_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return df
    nan_cols = [c for c in df.columns if df[c].isna().all()]
    if nan_cols:
        logger.warning(
            "%s has %d all-NaN columns; dropping: %s", name, len(nan_cols), nan_cols[:20]
        )
        return df.drop(columns=nan_cols)
    return df


def save_model_metadata(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("generated_at", datetime.now(UTC).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
