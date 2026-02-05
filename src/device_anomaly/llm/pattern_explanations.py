from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from device_anomaly.llm.client import get_default_llm_client, strip_thinking_tags
from device_anomaly.models.patterns import _build_pattern_prompt


def generate_and_save_pattern_explanation(
    engine: Engine,
    pattern_id: int,
) -> str:
    """
    Load a single pattern row (ml_DeviceAnomalyPatterns), generate an LLM
    explanation based on PatternJson, save it, and return it.
    """
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT *
                FROM dbo.ml_DeviceAnomalyPatterns
                WHERE Id = :pattern_id
                """
            ),
            conn,
            params={"pattern_id": pattern_id},
        )

        if df.empty:
            raise ValueError(f"No pattern found with Id={pattern_id}")

        row = df.iloc[0]

        if row.get("Explanation"):
            return row["Explanation"]

        payload = json.loads(row["PatternJson"])
        prompt = _build_pattern_prompt(payload)

        llm = get_default_llm_client()
        raw_explanation = llm.generate(prompt)
        explanation = strip_thinking_tags(raw_explanation)

        conn.execute(
            text(
                """
                UPDATE dbo.ml_DeviceAnomalyPatterns
                SET Explanation = :exp
                WHERE Id = :pattern_id
                """
            ),
            {"exp": explanation, "pattern_id": pattern_id},
        )

    return explanation
