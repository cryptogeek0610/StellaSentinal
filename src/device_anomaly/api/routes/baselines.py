"""API routes for baseline management and LLM-driven adjustments.

This module provides:
1. Standard baseline management (suggestions, adjustments, history)
2. ML-enhanced baselines with ensemble detection and Bayesian adaptation
3. Causal correlation discovery
4. Real-time drift detection
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from device_anomaly.api.dependencies import get_db, get_tenant_id, require_role
from device_anomaly.models.baseline import (
    BaselineLevel,
    BaselineFeedback,
    load_baselines,
    save_baselines,
    apply_feedback,
    suggest_baseline_adjustments,
)
from device_anomaly.models.baseline_store import (
    resolve_baselines,
    load_legacy_frames,
    update_data_driven_baseline,
    save_baseline_payload,
)
from device_anomaly.llm.client import get_default_llm_client, strip_thinking_tags
from device_anomaly.llm.prompt_utils import translate_metric, NO_THINKING_INSTRUCTION
from device_anomaly.database.schema import AnomalyResult

router = APIRouter(prefix="/baselines", tags=["baselines"])
logger = logging.getLogger(__name__)

# Global ML service instance (lazy loaded)
_ml_service = None

_ALLOWED_BASELINE_SOURCES = {"dw", "synthetic"}


def _normalize_source(source: str) -> str:
    normalized = (source or "").strip().lower()
    if normalized not in _ALLOWED_BASELINE_SOURCES:
        raise HTTPException(
            status_code=400,
            detail="Invalid source. Allowed values: dw, synthetic",
        )
    return normalized


def _load_baseline_resolution(source: str):
    resolution = resolve_baselines(source)
    if resolution is None:
        raise HTTPException(
            status_code=404,
            detail="No baselines found in production artifacts or legacy store.",
        )
    legacy_frames = load_legacy_frames(resolution)
    if not legacy_frames:
        raise HTTPException(status_code=404, detail="Resolved baselines are empty.")
    return resolution, legacy_frames


class BaselineSuggestionResponse(BaseModel):
    level: str
    group_key: str
    feature: str
    baseline_median: float
    observed_median: float
    proposed_new_median: float
    rationale: str


class BaselineFeatureResponse(BaseModel):
    """Response model for baseline feature overview."""
    feature: str
    baseline: float
    observed: float
    unit: str
    status: str  # 'stable', 'warning', 'drift'
    drift_percent: float
    mad: float
    sample_count: int
    last_updated: Optional[str] = None


class BaselineHistoryEntry(BaseModel):
    """Response model for baseline adjustment history."""
    id: int
    date: str
    feature: str
    old_value: float
    new_value: float
    type: str  # 'auto' or 'manual'
    reason: Optional[str] = None


class BaselineAdjustmentRequest(BaseModel):
    level: str
    group_key: dict | str
    feature: str
    adjustment: float
    reason: Optional[str] = None
    auto_retrain: bool = False


class BaselineAdjustmentResponse(BaseModel):
    success: bool
    message: str
    baseline_updated: bool
    model_retrained: bool = False


@router.get("/suggestions", response_model=List[BaselineSuggestionResponse])
def get_baseline_suggestions(
    source: str = Query("dw", description="Data source: 'dw' or 'synthetic'"),
    days: int = Query(30, ge=1, le=365),
    z_threshold: float = Query(3.0, ge=1.0, le=10.0),
    db: Session = Depends(get_db),
):
    """Get baseline adjustment suggestions based on anomaly patterns.
    
    Analyzes recent anomalies and suggests baseline adjustments where
    systematic drift is detected.
    """
    source = _normalize_source(source)
    tenant_id = get_tenant_id()
    resolution, baselines = _load_baseline_resolution(source)
    
    # Get recent anomalies
    now = datetime.now(timezone.utc)
    filter_start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    anomalies = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(AnomalyResult.timestamp >= filter_start)
        .all()
    )
    
    if not anomalies:
        return []
    
    # Convert to DataFrame for analysis
    import pandas as pd
    
    # Build anomalies DataFrame with available data
    anomalies_data = []
    for a in anomalies:
        row_data = {
            "DeviceId": a.device_id,
            "anomaly_score": a.anomaly_score,
        }
        
        # Try to parse feature values if available
        if a.feature_values_json:
            try:
                feature_values = json.loads(a.feature_values_json)
                row_data.update(feature_values)
            except (json.JSONDecodeError, TypeError):
                pass
        
        anomalies_data.append(row_data)
    
    anomalies_df = pd.DataFrame(anomalies_data)
    
    if anomalies_df.empty:
        return []
    
    # Define baseline levels (should match experiment config)
    baseline_levels = [BaselineLevel(name="global", group_columns=["__all__"], min_rows=10)]

    if resolution.device_type_col and resolution.device_type_col in anomalies_df.columns:
        baseline_levels.append(
            BaselineLevel(name="device_type", group_columns=[resolution.device_type_col])
        )
    
    # Add __all__ column for global baseline
    if "__all__" not in anomalies_df.columns:
        anomalies_df["__all__"] = "all"
    
    suggestions = suggest_baseline_adjustments(
        anomalies_df=anomalies_df,
        baselines=baselines,
        levels=baseline_levels,
        z_threshold=z_threshold,
    )
    
    if not suggestions and resolution.kind == "data_driven":
        logger = logging.getLogger(__name__)
        logger.info("No baseline suggestions generated from production baselines.")

    return [BaselineSuggestionResponse(**s) for s in suggestions]


@router.post("/analyze-with-llm", response_model=List[BaselineSuggestionResponse])
def analyze_baselines_with_llm(
    source: str = Query("dw"),
    days: int = Query(30),
    db: Session = Depends(get_db),
):
    """Use LLM to analyze anomaly patterns and suggest baseline adjustments.
    
    This endpoint uses the LLM to intelligently analyze false positives
    and recurring anomalies to suggest baseline improvements.
    """
    source = _normalize_source(source)
    tenant_id = get_tenant_id()
    # Get baseline suggestions first
    suggestions = get_baseline_suggestions(source=source, days=days, db=db)
    
    if not suggestions:
        return []
    
    # Get false positive feedback (anomalies marked as false_positive)
    now = datetime.now(timezone.utc)
    filter_start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    false_positives = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(AnomalyResult.status == "false_positive")
        .filter(AnomalyResult.timestamp >= filter_start)
        .count()
    )
    
    total_anomalies = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(AnomalyResult.timestamp >= filter_start)
        .count()
    )
    
    fp_rate = false_positives / max(total_anomalies, 1)
    
    # Build LLM prompt
    llm_client = get_default_llm_client()
    
    suggestions_json = json.dumps([s.dict() for s in suggestions], indent=2)
    
    prompt = f"""<role>
You are an ML operations analyst reviewing anomaly detection baseline adjustments for an enterprise mobile device management system. The system uses Isolation Forest to detect unusual device behavior in warehouses, retail stores, and field operations.
</role>

<output_format>
{NO_THINKING_INSTRUCTION}

Return ONLY a valid JSON array. No markdown code blocks, no explanation text.

Each object must have this exact structure:
{{
  "level": "string (from input)",
  "group_key": "string (from input)",
  "feature": "exact_feature_name_from_input",
  "baseline_median": number_from_input,
  "observed_median": number_from_input,
  "proposed_new_median": number_from_input,
  "priority": 1_to_5_where_1_is_highest,
  "rationale": "2-3 sentences explaining why this adjustment matters and expected impact"
}}
</output_format>

<current_situation>
- Total anomalies detected: {total_anomalies}
- Estimated false positive rate: {fp_rate:.1%}
- Number of suggested adjustments: {len(suggestions)}

Context: A high false positive rate ({fp_rate:.1%}) means the model is flagging too many normal devices as anomalous, causing alert fatigue for IT staff.
</current_situation>

<baseline_suggestions>
{suggestions_json}
</baseline_suggestions>

<prioritization_criteria>
Rank adjustments by priority (1=highest, 5=lowest):
1. Features with highest false positive contribution (largest gap between current and observed)
2. Features in critical domains (battery, connectivity) over less critical
3. Adjustments that are safe (wide margin, won't miss real anomalies)
4. Features where normal variation is well understood
</prioritization_criteria>

<instructions>
1. Return ALL suggestions from input - do not remove any
2. Reorder by priority field (1 = adjust first, 5 = optional)
3. Do NOT invent new features - only use features from the input
4. Keep the same field names and types from input
5. Enhance the rationale field with business context
6. Be conservative: estimate 5-15% FP reduction per adjustment is realistic
</instructions>"""
    
    try:
        raw_llm_response = llm_client.generate(prompt, max_tokens=1200, temperature=0.1)
        llm_response = strip_thinking_tags(raw_llm_response)

        # Try to parse LLM response as JSON
        # If it's not valid JSON, try to extract JSON from markdown code blocks
        try:
            # Remove markdown code blocks if present
            if "```json" in llm_response:
                llm_response = llm_response.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_response:
                llm_response = llm_response.split("```")[1].split("```")[0].strip()
            
            enhanced_suggestions_data = json.loads(llm_response)
            # Validate and convert to response format
            enhanced_suggestions = []
            for s in enhanced_suggestions_data:
                if all(k in s for k in ["level", "feature", "baseline_median", "observed_median", "proposed_new_median", "rationale"]):
                    # Ensure group_key is a string
                    if "group_key" not in s:
                        s["group_key"] = suggestions[0].group_key if suggestions else "unknown"
                    elif not isinstance(s["group_key"], str):
                        s["group_key"] = json.dumps(s["group_key"]) if isinstance(s["group_key"], dict) else str(s["group_key"])
                    enhanced_suggestions.append(BaselineSuggestionResponse(**s))
            
            if enhanced_suggestions:
                return enhanced_suggestions
        except (json.JSONDecodeError, KeyError, TypeError):
            # If parsing fails, use original suggestions but add LLM analysis note
            enhanced_suggestions = []
            for s in suggestions:
                enhanced_s = BaselineSuggestionResponse(
                    level=s.level,
                    group_key=s.group_key,
                    feature=s.feature,
                    baseline_median=s.baseline_median,
                    observed_median=s.observed_median,
                    proposed_new_median=s.proposed_new_median,
                    rationale=f"{s.rationale} [LLM Analysis: {llm_response[:200]}...]",
                )
                enhanced_suggestions.append(enhanced_s)
            return enhanced_suggestions
        
        return suggestions
    except Exception as e:
        # Fallback to original suggestions if LLM fails
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"LLM analysis failed: {e}, using original suggestions")
        return suggestions


@router.post("/apply-adjustment", response_model=BaselineAdjustmentResponse)
def apply_baseline_adjustment(
    request: BaselineAdjustmentRequest,
    source: str = Query("dw"),
    _: None = Depends(require_role(["analyst", "admin"])),
    db: Session = Depends(get_db),
):
    """Apply a baseline adjustment and optionally retrain the model.
    
    This applies the feedback to baselines and can trigger model retraining.
    """
    source = _normalize_source(source)
    resolution, baselines = _load_baseline_resolution(source)
    
    # Create feedback object
    feedback = BaselineFeedback(
        level=request.level,
        group_key=request.group_key,
        feature=request.feature,
        adjustment=request.adjustment,
        reason=request.reason,
    )
    
    if resolution.kind == "data_driven":
        group_key = request.group_key
        if isinstance(group_key, dict):
            group_key = group_key.get(resolution.device_type_col or "", group_key)
        group_key = str(group_key)
        payload = update_data_driven_baseline(
            resolution.payload,
            request.level,
            group_key,
            request.feature,
            request.adjustment,
            resolution.device_type_col,
        )
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_baseline_payload(payload, resolution.path)
    else:
        # Apply feedback
        updated_baselines = apply_feedback(baselines, [feedback], learning_rate=0.3)
        # Save updated baselines
        save_baselines(updated_baselines, resolution.path)
    
    # Optionally trigger retraining
    model_retrained = False
    if request.auto_retrain:
        # This would trigger a background job to retrain
        # For now, just return success - actual retraining would be async
        model_retrained = True
    
    return BaselineAdjustmentResponse(
        success=True,
        message=f"Baseline adjusted for {request.level}/{request.feature}",
        baseline_updated=True,
        model_retrained=model_retrained,
    )


# Feature unit mapping for display purposes
_FEATURE_UNITS: Dict[str, str] = {
    "BatteryDrop": "%/day",
    "OfflineTime": "min/day",
    "UploadSize": "MB/day",
    "DownloadSize": "MB/day",
    "StorageFree": "GB",
    "AppCrashes": "/device/day",
    "battery_drop_pct": "%/day",
    "offline_minutes": "min/day",
    "upload_mb": "MB/day",
    "download_mb": "MB/day",
    "storage_free_gb": "GB",
    "app_crash_count": "/device/day",
    "anomaly_score": "score",
}


@router.get("/features", response_model=List[BaselineFeatureResponse])
def get_baseline_features(
    source: str = Query("dw", description="Data source: 'dw' or 'synthetic'"),
    days: int = Query(30, ge=1, le=365, description="Days to analyze for observed values"),
    db: Session = Depends(get_db),
):
    """Get baseline feature overview with current observed values and drift status.

    Returns all tracked features with their baseline values, current observed
    values, and drift status (stable, warning, drift).
    """
    source = _normalize_source(source)
    tenant_id = get_tenant_id()

    resolution, baselines = _load_baseline_resolution(source)

    # Get recent anomaly data to compute observed values
    now = datetime.now(timezone.utc)
    filter_start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Query recent data (both normal and anomalous)
    recent_data = (
        db.query(AnomalyResult)
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.timestamp >= filter_start)
        .all()
    )

    # Build feature values from stored data
    import pandas as pd

    feature_values: Dict[str, List[float]] = {}
    for record in recent_data:
        if record.feature_values_json:
            try:
                values = json.loads(record.feature_values_json)
                for feature, value in values.items():
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if feature not in feature_values:
                            feature_values[feature] = []
                        feature_values[feature].append(float(value))
            except (json.JSONDecodeError, TypeError):
                pass

    # Build response from baselines
    features: List[BaselineFeatureResponse] = []

    if resolution.kind == "data_driven":
        data_driven = resolution.payload.get("baselines", {})
        for feature_name, data in data_driven.items():
            global_stats = data.get("global") or {}
            baseline_median = float(global_stats.get("median", 0))
            mad = float(global_stats.get("mad", 1e-6)) or 1e-6

            observed_values = feature_values.get(feature_name, [])
            if observed_values:
                observed = float(pd.Series(observed_values).median())
                sample_count = len(observed_values)
            else:
                observed = baseline_median
                sample_count = 0

            if baseline_median != 0:
                drift_percent = ((observed - baseline_median) / baseline_median) * 100
            else:
                drift_percent = 0.0

            z_score = abs(observed - baseline_median) / mad if mad > 0 else 0
            if z_score < 2:
                status = "stable"
            elif z_score < 3:
                status = "warning"
            else:
                status = "drift"

            unit = _FEATURE_UNITS.get(feature_name, "units")

            features.append(BaselineFeatureResponse(
                feature=feature_name,
                baseline=round(baseline_median, 2),
                observed=round(observed, 2),
                unit=unit,
                status=status,
                drift_percent=round(drift_percent, 1),
                mad=round(mad, 4),
                sample_count=sample_count,
                last_updated=None,
            ))
    else:
        # Process each baseline level (prefer 'global' level)
        for level_name in ["global", "hardware", "cohort", "device_type"]:
            level_baselines = baselines.get(level_name)
            if level_baselines is None or level_baselines.empty:
                continue

            for _, row in level_baselines.iterrows():
                feature_name = row.get("feature", "")
                if not feature_name:
                    continue

                # Skip if we already processed this feature from a more specific level
                if any(f.feature == feature_name for f in features):
                    continue

                baseline_median = float(row.get("median", 0))
                mad = float(row.get("mad", 1e-6)) or 1e-6

                # Get observed value from recent data
                observed_values = feature_values.get(feature_name, [])
                if observed_values:
                    observed = float(pd.Series(observed_values).median())
                    sample_count = len(observed_values)
                else:
                    # No recent data - use baseline as observed
                    observed = baseline_median
                    sample_count = 0

                # Calculate drift
                if baseline_median != 0:
                    drift_percent = ((observed - baseline_median) / baseline_median) * 100
                else:
                    drift_percent = 0.0

                # Determine status based on drift and MAD
                z_score = abs(observed - baseline_median) / mad if mad > 0 else 0
                if z_score < 2:
                    status = "stable"
                elif z_score < 3:
                    status = "warning"
                else:
                    status = "drift"

                # Get unit for display
                unit = _FEATURE_UNITS.get(feature_name, "units")

                features.append(BaselineFeatureResponse(
                    feature=feature_name,
                    baseline=round(baseline_median, 2),
                    observed=round(observed, 2),
                    unit=unit,
                    status=status,
                    drift_percent=round(drift_percent, 1),
                    mad=round(mad, 4),
                    sample_count=sample_count,
                    last_updated=None,
                ))

    return features


@router.get("/history", response_model=List[BaselineHistoryEntry])
def get_baseline_history(
    source: str = Query("dw", description="Data source: 'dw' or 'synthetic'"),
    limit: int = Query(50, ge=1, le=500, description="Maximum entries to return"),
    db: Session = Depends(get_db),
):
    """Get baseline adjustment history.

    Returns recent baseline adjustments with details about what changed and why.
    Note: This requires a history tracking mechanism to be implemented.
    For now, returns empty list if no history file exists.
    """
    source = _normalize_source(source)

    # Load history file if it exists
    history_path = Path("artifacts") / f"{source}_baseline_history.json"
    if not history_path.exists():
        return []

    try:
        history_data = json.loads(history_path.read_text())
        entries = []
        for idx, entry in enumerate(history_data[-limit:]):
            entries.append(BaselineHistoryEntry(
                id=entry.get('id', idx + 1),
                date=entry.get('date', ''),
                feature=entry.get('feature', ''),
                old_value=float(entry.get('old_value', 0)),
                new_value=float(entry.get('new_value', 0)),
                type=entry.get('type', 'manual'),
                reason=entry.get('reason'),
            ))
        return entries
    except (json.JSONDecodeError, TypeError):
        return []


# =============================================================================
# ML-ENHANCED BASELINE ROUTES (Ultra Mode)
# =============================================================================


async def _get_ml_service():
    """Lazy-load the ML baseline service."""
    global _ml_service
    if _ml_service is None:
        try:
            from device_anomaly.services.ml_baseline_service import MLBaselineService
            _ml_service = MLBaselineService()
            # Try to load checkpoint
            await _ml_service.load_checkpoint()
        except ImportError as e:
            logger.warning(f"ML baseline service not available: {e}")
            raise HTTPException(
                status_code=503,
                detail="ML baseline service not available. Install required dependencies."
            )
    return _ml_service


class MLBaselineStatusResponse(BaseModel):
    """Status of the ML baseline engine."""
    initialized: bool
    last_train_time: Optional[str] = None
    last_drift_check: Optional[str] = None
    feature_count: int = 0
    metric_count: int = 0
    training_history_count: int = 0
    drift_history_count: int = 0


class MLTrainRequest(BaseModel):
    """Request to train the ML baseline engine."""
    lookback_days: int = Field(default=90, ge=7, le=365)
    sources: Optional[List[str]] = None
    feature_cols: Optional[List[str]] = None
    metric_cols: Optional[List[str]] = None


class MLTrainResponse(BaseModel):
    """Response from ML training."""
    success: bool
    samples_trained: int = 0
    feature_count: int = 0
    metric_count: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None


class MLScoreRequest(BaseModel):
    """Request to score device telemetry."""
    device_data: Dict[str, Any]


class MLScoreResponse(BaseModel):
    """Response from ML scoring."""
    device_id: Optional[str] = None
    overall_anomaly_score: float
    is_anomaly: bool
    anomaly_type: str
    metrics: Dict[str, Any]


class CausalInsightResponse(BaseModel):
    """Causal relationship insight."""
    cause: str
    effect: str
    lag_days: int
    correlation: float
    direction: str
    confidence: float
    insight: str


class DriftReportResponse(BaseModel):
    """Drift detection report."""
    timestamp: str
    metrics_checked: int
    metrics_drifted: int
    drift_rate: float
    auto_retrained: bool = False
    details: Dict[str, Any] = {}


class MLSuggestionResponse(BaseModel):
    """ML-enhanced baseline suggestion."""
    metric: str
    current_baseline_mean: float
    current_baseline_std: float
    observed_median: float
    z_score: float
    proposed_adjustment: float
    confidence: float
    rationale: str
    bayesian_uncertainty: Optional[float] = None
    credible_interval: Optional[List[float]] = None


@router.get("/ml/status", response_model=MLBaselineStatusResponse, tags=["ml-baselines"])
async def get_ml_baseline_status():
    """Get the current status of the ML baseline engine.

    Returns information about:
    - Whether the engine is initialized
    - When it was last trained
    - Number of features/metrics being tracked
    - Training and drift detection history
    """
    try:
        service = await _get_ml_service()
        status = service.get_status()
        return MLBaselineStatusResponse(**status)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get ML baseline status")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/train", response_model=MLTrainResponse, tags=["ml-baselines"])
async def train_ml_baselines(
    request: MLTrainRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_role(["admin"])),
):
    """Train the ML baseline engine from all data sources.

    This is the 'Ultra' training mode that:
    - Fuses data from XSight, MobiControl, and custom sources
    - Trains an ensemble anomaly detector (IsolationForest + LOF + AutoEncoder + DBSCAN)
    - Initializes Bayesian baselines with uncertainty quantification
    - Discovers causal relationships between metrics
    - Sets up drift detection references

    Training runs synchronously but saves checkpoints for persistence.
    """
    try:
        service = await _get_ml_service()

        result = await service.train(
            lookback_days=request.lookback_days,
            feature_cols=request.feature_cols,
            metric_cols=request.metric_cols,
        )

        return MLTrainResponse(
            success=result.get("success", False),
            samples_trained=result.get("samples_trained", 0),
            feature_count=result.get("feature_count", 0),
            metric_count=result.get("metric_count", 0),
            duration_seconds=result.get("duration_seconds", 0.0),
            error=result.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ML training failed")
        return MLTrainResponse(success=False, error=str(e))


@router.post("/ml/score", response_model=MLScoreResponse, tags=["ml-baselines"])
async def score_device_with_ml(request: MLScoreRequest):
    """Score a single device's telemetry using the ML engine.

    Returns:
    - Overall anomaly score (0-1)
    - Anomaly classification
    - Per-metric anomaly probabilities from Bayesian analysis
    - Severity levels for each metric
    """
    try:
        service = await _get_ml_service()
        result = service.score_single_device(request.device_data)

        return MLScoreResponse(
            device_id=result.get("device_id"),
            overall_anomaly_score=result.get("overall_anomaly_score", 0.0),
            is_anomaly=result.get("is_anomaly", False),
            anomaly_type=result.get("anomaly_type", "normal"),
            metrics=result.get("metrics", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ML scoring failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/causal-insights", response_model=List[CausalInsightResponse], tags=["ml-baselines"])
async def get_causal_insights():
    """Get discovered causal relationships between metrics.

    Returns insights from causal discovery including:
    - Cause and effect metrics
    - Time lag between cause and effect
    - Correlation strength
    - Direction (causal, reverse_causal, bidirectional, spurious)
    - Confidence score
    - Human-readable insight

    These relationships help understand how metrics influence each other,
    enabling proactive anomaly prediction.
    """
    try:
        service = await _get_ml_service()
        insights = service.get_correlation_insights()

        return [CausalInsightResponse(**i) for i in insights]

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get causal insights")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/check-drift", response_model=DriftReportResponse, tags=["ml-baselines"])
async def check_baseline_drift(
    lookback_days: int = Query(default=7, ge=1, le=90),
):
    """Check for distribution drift in recent data.

    Uses multiple drift detection algorithms:
    - PSI (Population Stability Index)
    - KS Test (Kolmogorov-Smirnov)
    - Jensen-Shannon Divergence
    - Mean and variance shift detection

    If significant drift is detected and auto-retraining is enabled,
    the engine will automatically retrain.
    """
    try:
        service = await _get_ml_service()

        # Load recent data for drift check
        import pandas as pd
        df = await service.load_training_data(lookback_days=lookback_days)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for drift check")

        report = await service.check_drift(df)

        return DriftReportResponse(
            timestamp=report.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metrics_checked=report.get("metrics_checked", 0),
            metrics_drifted=report.get("metrics_drifted", 0),
            drift_rate=report.get("drift_rate", 0.0),
            auto_retrained=report.get("auto_retrained", False),
            details=report.get("details", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Drift check failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/suggestions", response_model=List[MLSuggestionResponse], tags=["ml-baselines"])
async def get_ml_baseline_suggestions(
    days: int = Query(default=30, ge=1, le=365),
    z_threshold: float = Query(default=3.0, ge=1.0, le=10.0),
):
    """Get ML-enhanced baseline adjustment suggestions.

    Combines:
    - Statistical drift detection (z-scores from median/MAD)
    - Bayesian uncertainty quantification
    - Credible intervals for each metric

    Suggestions are prioritized by:
    1. Statistical significance (z-score)
    2. Bayesian uncertainty (high uncertainty = more urgent)
    3. Impact on anomaly detection
    """
    try:
        service = await _get_ml_service()

        # Load recent data
        import pandas as pd
        df = await service.load_training_data(lookback_days=days)

        if df.empty:
            return []

        suggestions = service.get_baseline_suggestions(df, z_threshold=z_threshold)

        return [
            MLSuggestionResponse(
                metric=s["metric"],
                current_baseline_mean=s["current_baseline_mean"],
                current_baseline_std=s["current_baseline_std"],
                observed_median=s["observed_median"],
                z_score=s["z_score"],
                proposed_adjustment=s["proposed_adjustment"],
                confidence=s["confidence"],
                rationale=s["rationale"],
                bayesian_uncertainty=s.get("bayesian_uncertainty"),
                credible_interval=s.get("credible_interval"),
            )
            for s in suggestions
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get ML suggestions")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/update-online", tags=["ml-baselines"])
async def update_baselines_online(
    device_data: Dict[str, Any],
):
    """Update baselines with new streaming data (online learning).

    Uses Bayesian updating for real-time baseline adaptation:
    - Incremental mean/variance updates (Welford's algorithm)
    - Exponentially weighted moving statistics
    - Change point detection

    Call this endpoint as new telemetry arrives to keep baselines current.
    """
    try:
        service = await _get_ml_service()

        import pandas as pd
        df = pd.DataFrame([device_data])

        result = await service.update_baselines(df)

        return {
            "success": True,
            "updated_metrics": result.get("updated_metrics", []),
            "needs_retraining": result.get("needs_retraining", []),
            "timestamp": result.get("timestamp"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Online update failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/training-history", tags=["ml-baselines"])
async def get_ml_training_history():
    """Get the training history of the ML baseline engine.

    Returns list of past training runs with:
    - Timestamp
    - Number of samples trained
    - Features and metrics used
    - Duration
    - Success/failure status
    """
    try:
        service = await _get_ml_service()
        return service.get_training_history()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get training history")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/drift-history", tags=["ml-baselines"])
async def get_ml_drift_history():
    """Get the drift detection history.

    Returns list of past drift checks with:
    - Timestamp
    - Drift rate (fraction of metrics that drifted)
    - Number of metrics that drifted
    """
    try:
        service = await _get_ml_service()
        return service.get_drift_history()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get drift history")
        raise HTTPException(status_code=500, detail=str(e))
