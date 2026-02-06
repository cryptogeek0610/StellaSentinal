"""
StellaSentinal API Server
=========================
FastAPI application providing REST endpoints for the Scranton Digital workforce,
kanban board, watercooler, and SAAP anomaly detection / NBA recommendation services.

Run with:
    uvicorn api.server:app --reload
"""

import importlib
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKSPACE = Path(
    os.environ.get("STELLA_WORKSPACE", str(Path(__file__).resolve().parent.parent))
)

logger = logging.getLogger("scranton.api")

# ---------------------------------------------------------------------------
# Import MissionControl (normal package under office/scripts)
# ---------------------------------------------------------------------------
_scripts_dir = str(WORKSPACE / "office" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from mission_control_daemon import MissionControl  # noqa: E402

# ---------------------------------------------------------------------------
# Import SAAP services (hyphenated directory requires importlib tricks)
# ---------------------------------------------------------------------------
_ai_service_dir = str(
    WORKSPACE / "projects" / "SOTI-Advanced-Analytics-Plus" / "ai-service"
)
if _ai_service_dir not in sys.path:
    sys.path.insert(0, _ai_service_dir)

from anomaly_predictor import SAAPAnomalyPredictor  # noqa: E402
from nba_engine import NextBestActionEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Pydantic Models -- Requests
# ---------------------------------------------------------------------------

class KanbanTaskCreate(BaseModel):
    """Payload for creating a new kanban task."""
    title: str = Field(..., min_length=1, max_length=200, description="Task title")
    status: str = Field(
        default="todo",
        description="Task status (e.g. todo, in_progress, done)",
    )
    owner: str = Field(default="", description="Agent ID of the task owner")
    project: str = Field(default="", description="Associated project name")


class KanbanTaskUpdate(BaseModel):
    """Payload for updating a kanban task's status."""
    status: str = Field(
        ..., min_length=1, max_length=50, description="New status value"
    )


class WatercoolerMessageCreate(BaseModel):
    """Payload for posting a watercooler message."""
    agent_id: str = Field(..., min_length=1, description="ID of the posting agent")
    message: str = Field(
        ..., min_length=1, max_length=1000, description="Message content"
    )


class AnomalyPredictRequest(BaseModel):
    """Feature data for anomaly prediction."""
    feature_matrix: List[List[float]] = Field(
        ..., min_length=1, description="2-D feature matrix (list of sample vectors)"
    )
    training_data: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional training data. If provided the model is (re)trained before prediction.",
    )


class AnomalyRecommendRequest(BaseModel):
    """Anomaly dict passed to the NBA engine."""
    type: str = Field(default="unknown", description="Anomaly type")
    root_cause_hint: str = Field(default="", description="Root-cause hint")
    summary: str = Field(default="", description="Anomaly summary")
    affected_cohort: str = Field(default="Unknown", description="Affected cohort")


# ---------------------------------------------------------------------------
# Pydantic Models -- Responses
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str


class AgentResponse(BaseModel):
    id: str
    name: str
    role: str
    specialty: str


class WorkforceResponse(BaseModel):
    squad_name: str
    lead_agent: str
    agents: List[AgentResponse]
    kanban: List[Dict[str, Any]]
    watercooler: List[Dict[str, Any]]


class KanbanTaskResponse(BaseModel):
    index: int
    task: Dict[str, Any]


class WatercoolerMessageResponse(BaseModel):
    index: int
    entry: Dict[str, Any]


class AnomalyPredictResponse(BaseModel):
    predictions: List[int]
    scores: List[float]
    anomaly_count: int
    total: int


class AnomalyRecommendResponse(BaseModel):
    recommendation: Dict[str, str]


# ---------------------------------------------------------------------------
# Application state (populated during lifespan)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {}
_start_time: float = 0.0

APP_VERSION = "0.4.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI app."""
    global _start_time

    logging.basicConfig(
        level=os.environ.get("STELLA_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Initializing MissionControl...")
    _state["mc"] = MissionControl()
    logger.info("MissionControl loaded with squad: %s", _state["mc"].state.get("squad_name"))

    _state["anomaly_predictor"] = SAAPAnomalyPredictor()
    _state["nba_engine"] = NextBestActionEngine()
    logger.info("SAAP AI services initialised.")

    _start_time = time.monotonic()
    logger.info("StellaSentinal API ready (v%s)", APP_VERSION)

    yield  # application is running

    logger.info("Shutting down StellaSentinal API...")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="StellaSentinal API",
    description="REST API for the Scranton Digital workforce, kanban, watercooler, and SAAP AI services.",
    version=APP_VERSION,
    lifespan=lifespan,
)

# CORS -- allow all origins so the static dashboard can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _mc() -> MissionControl:
    """Return the shared MissionControl instance, or raise 500 if not ready."""
    mc = _state.get("mc")
    if mc is None:
        raise HTTPException(status_code=500, detail="MissionControl not initialised")
    return mc


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Return service health, uptime and version."""
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.monotonic() - _start_time, 2),
        version=APP_VERSION,
    )


# ---------------------------------------------------------------------------
# 2-4. Workforce
# ---------------------------------------------------------------------------
@app.get("/api/workforce", response_model=WorkforceResponse)
async def get_workforce():
    """Return the full workforce state."""
    mc = _mc()
    return mc.state


@app.get("/api/workforce/agents", response_model=List[AgentResponse])
async def get_agents():
    """Return the list of agents."""
    mc = _mc()
    return mc.state.get("agents", [])


@app.get("/api/workforce/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Return a single agent by ID."""
    mc = _mc()
    for agent in mc.state.get("agents", []):
        if agent.get("id") == agent_id:
            return agent
    raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")


# ---------------------------------------------------------------------------
# 5-7. Kanban
# ---------------------------------------------------------------------------
@app.get("/api/kanban", response_model=List[Dict[str, Any]])
async def get_kanban():
    """Return all kanban tasks."""
    mc = _mc()
    return mc.state.get("kanban", [])


@app.post("/api/kanban", response_model=KanbanTaskResponse, status_code=201)
async def create_kanban_task(task: KanbanTaskCreate):
    """Add a new kanban task and persist state."""
    mc = _mc()
    new_task = task.model_dump()
    kanban = mc.state.setdefault("kanban", [])
    kanban.append(new_task)
    mc.save_state()
    logger.info("Kanban task created: %s", new_task["title"])
    return KanbanTaskResponse(index=len(kanban) - 1, task=new_task)


@app.patch("/api/kanban/{index}", response_model=KanbanTaskResponse)
async def update_kanban_task(index: int, update: KanbanTaskUpdate):
    """Update the status of a kanban task by its index."""
    mc = _mc()
    kanban = mc.state.get("kanban", [])
    if index < 0 or index >= len(kanban):
        raise HTTPException(
            status_code=404,
            detail=f"Kanban task at index {index} not found (board has {len(kanban)} items)",
        )
    kanban[index]["status"] = update.status
    mc.save_state()
    logger.info("Kanban task %d updated to status '%s'", index, update.status)
    return KanbanTaskResponse(index=index, task=kanban[index])


# ---------------------------------------------------------------------------
# 8-9. Watercooler
# ---------------------------------------------------------------------------
@app.get("/api/watercooler", response_model=List[Dict[str, Any]])
async def get_watercooler():
    """Return all watercooler messages."""
    mc = _mc()
    return mc.state.get("watercooler", [])


@app.post("/api/watercooler", response_model=WatercoolerMessageResponse, status_code=201)
async def create_watercooler_message(msg: WatercoolerMessageCreate):
    """Post a new watercooler message and persist state."""
    mc = _mc()

    # Validate that the agent_id exists
    agents = mc.state.get("agents", [])
    agent_ids = {a.get("id") for a in agents}
    if msg.agent_id not in agent_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{msg.agent_id}' not found in workforce",
        )

    entry = {
        "agent_id": msg.agent_id,
        "message": msg.message,
    }
    watercooler = mc.state.setdefault("watercooler", [])
    watercooler.append(entry)
    mc.save_state()
    logger.info("Watercooler message from %s", msg.agent_id)
    return WatercoolerMessageResponse(index=len(watercooler) - 1, entry=entry)


# ---------------------------------------------------------------------------
# 10. Anomaly Prediction
# ---------------------------------------------------------------------------
@app.post("/api/anomaly/predict", response_model=AnomalyPredictResponse)
async def anomaly_predict(req: AnomalyPredictRequest):
    """Run anomaly prediction on the supplied feature data.

    If ``training_data`` is provided the model will be (re)trained first.
    Otherwise the model must have been previously trained or a 422 is returned.
    """
    predictor: SAAPAnomalyPredictor = _state.get("anomaly_predictor")
    if predictor is None:
        raise HTTPException(status_code=500, detail="Anomaly predictor not initialised")

    try:
        # Optionally train / retrain
        if req.training_data is not None:
            predictor.train(req.training_data)

        predictions, scores = predictor.predict(req.feature_matrix)
        preds_list = predictions.tolist()
        scores_list = [round(float(s), 6) for s in scores.tolist()]
        anomaly_count = sum(1 for p in preds_list if p == -1)

        return AnomalyPredictResponse(
            predictions=preds_list,
            scores=scores_list,
            anomaly_count=anomaly_count,
            total=len(preds_list),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Anomaly prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


# ---------------------------------------------------------------------------
# 11. Anomaly Recommendation (NBA)
# ---------------------------------------------------------------------------
@app.post("/api/anomaly/recommend", response_model=AnomalyRecommendResponse)
async def anomaly_recommend(req: AnomalyRecommendRequest):
    """Return a Next-Best-Action recommendation for the described anomaly."""
    nba: NextBestActionEngine = _state.get("nba_engine")
    if nba is None:
        raise HTTPException(status_code=500, detail="NBA engine not initialised")

    anomaly_dict = req.model_dump()
    recommendation = nba.recommend(anomaly_dict)
    return AnomalyRecommendResponse(recommendation=recommendation)


# ---------------------------------------------------------------------------
# Static files -- serve the Mission Control dashboard at /
# Must be mounted LAST so it does not shadow /api/* routes.
# ---------------------------------------------------------------------------
_dashboard_dir = WORKSPACE / "office" / "mission-control"
if _dashboard_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard")
    logger.info("Serving dashboard from %s", _dashboard_dir)
else:
    logger.warning("Dashboard directory not found at %s; static files will not be served.", _dashboard_dir)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("STELLA_API_PORT", "8000")),
        reload=True,
    )
