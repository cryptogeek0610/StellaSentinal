"""FastAPI application main entry point."""
from __future__ import annotations

import hmac
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from device_anomaly.api.routes import anomalies, dashboard, devices, baselines, llm_settings, data_discovery, training, automation, streaming, setup, investigation, device_actions, insights, costs, system_health, location_intelligence, events_alerts, temporal, correlations, cross_device, data_quality, network, security, action_center, scranton_bridge
from device_anomaly.api.request_context import clear_request_context, set_request_context
from device_anomaly.config.logging_config import setup_logging
from device_anomaly.observability.otel import setup_tracing

logger = logging.getLogger(__name__)

setup_logging(force=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup

    # Initialize MobiControl device metadata cache for streaming enrichment
    try:
        from device_anomaly.services.device_metadata_sync import (
            load_mc_device_metadata_cache,
            get_mc_cache_stats,
        )

        cache_count = load_mc_device_metadata_cache()
        stats = get_mc_cache_stats()
        if cache_count > 0:
            logger.info(
                "MobiControl device cache initialized: %d devices loaded",
                cache_count,
            )
        else:
            logger.warning(
                "MobiControl device cache empty - streaming enrichment will not have device metadata. "
                "Check MC_DB_* environment variables and MobiControl database connectivity."
            )
    except Exception as e:
        logger.warning("MobiControl cache initialization failed: %s", e)

    # Initialize streaming system
    if os.getenv("ENABLE_STREAMING", "false").lower() == "true":
        try:
            await streaming.initialize_streaming()
            logger.info("Streaming system initialized")
        except Exception as e:
            logger.warning("Streaming system initialization failed: %s", e)

    yield

    # Shutdown
    try:
        await streaming.shutdown_streaming()
    except Exception as e:
        logger.warning("Streaming system shutdown failed: %s", e)


app = FastAPI(
    title="Stella Sentinel API",
    description="API for enterprise device anomaly detection and investigation",
    version="0.1.0",
    lifespan=lifespan,
)

# Environment-based CORS configuration
# In production, set CORS_ORIGINS environment variable (comma-separated)
# Example: CORS_ORIGINS=https://app.example.com,https://admin.example.com
_env = os.getenv("APP_ENV", "local")
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
_require_api_key = os.getenv("REQUIRE_API_KEY", "false").lower() == "true" or _env == "production"
_require_tenant = os.getenv("REQUIRE_TENANT_HEADER", "false").lower() == "true" or _env == "production"
_api_key = os.getenv("API_KEY")
_default_tenant = os.getenv("DEFAULT_TENANT_ID", "default")
_default_user_role = os.getenv("DEFAULT_USER_ROLE", "viewer")
_api_key_role = os.getenv("API_KEY_ROLE", _default_user_role)
_trust_client_headers = os.getenv("TRUST_CLIENT_HEADERS", "false").lower() == "true"
_enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
_tenant_allowlist = {
    tenant.strip()
    for tenant in os.getenv("TENANT_ID_ALLOWLIST", "").split(",")
    if tenant.strip()
}

if _cors_origins_env:
    # Use explicitly configured origins
    CORS_ORIGINS = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
elif _env in ("local", "development"):
    # Development defaults
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"]
else:
    # Production: no default origins - must be explicitly configured
    CORS_ORIGINS = []
    logger.warning("No CORS_ORIGINS configured for production. Set CORS_ORIGINS environment variable.")

# Restrict methods and headers based on environment
if _env in ("local", "development"):
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    CORS_HEADERS = ["*"]
else:
    # Production: only allow necessary methods and headers
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
    CORS_HEADERS = [
        "Content-Type",
        "Authorization",
        "X-Request-Id",
        "X-Tenant-Id",
        "X-Api-Key",
        "X-User-Id",
        "X-User-Role",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

if setup_tracing("stella-sentinel-api"):
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:
        logger.warning("OpenTelemetry FastAPI instrumentation failed: %s", exc)

if _enable_metrics:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, include_in_schema=False)
    except Exception as exc:
        logger.warning("Prometheus metrics instrumentation failed: %s", exc)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    bypass_auth = request.url.path in {
        "/health",
        "/health/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
    request_id = (
        request.headers.get("X-Request-Id")
        or request.headers.get("X-Request-ID")
        or request.headers.get("X-Correlation-Id")
        or str(uuid.uuid4())
    )

    tenant_id = request.headers.get("X-Tenant-Id") or request.headers.get("X-Tenant-ID")
    if _require_tenant and not tenant_id and not bypass_auth:
        return JSONResponse(
            status_code=400,
            content={"detail": "X-Tenant-Id header is required"},
        )
    tenant_id = tenant_id or _default_tenant
    if _tenant_allowlist and tenant_id not in _tenant_allowlist:
        return JSONResponse(
            status_code=403,
            content={"detail": "Tenant is not allowed"},
        )

    provided_key = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        provided_key = auth_header.split(" ", 1)[1].strip()
    if not provided_key:
        provided_key = request.headers.get("X-Api-Key") or request.headers.get("X-API-Key")

    if _require_api_key and not bypass_auth:
        if not _api_key:
            logger.error("REQUIRE_API_KEY is enabled but API_KEY is not configured.")
            return JSONResponse(
                status_code=500,
                content={"detail": "Authentication not configured"},
            )
        # Use constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(provided_key or "", _api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )

    user_id = None
    user_role = _default_user_role
    if _trust_client_headers:
        user_id = request.headers.get("X-User-Id") or request.headers.get("X-User-ID")
        user_role = (
            request.headers.get("X-User-Role")
            or request.headers.get("X-User-ROLE")
            or _default_user_role
        )
    if provided_key and not user_id:
        user_id = "api_key"
    if provided_key and not _trust_client_headers:
        user_role = _api_key_role

    set_request_context(
        request_id=request_id,
        tenant_id=tenant_id,
        user_id=user_id,
        user_role=user_role,
    )

    try:
        response = await call_next(request)
    finally:
        clear_request_context()

    response.headers["X-Request-Id"] = request_id
    return response

# Include routers with /api prefix
app.include_router(anomalies.router, prefix="/api")
app.include_router(devices.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(baselines.router, prefix="/api")
app.include_router(llm_settings.router, prefix="/api")
app.include_router(data_discovery.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(automation.router, prefix="/api")
app.include_router(streaming.router, prefix="/api")
app.include_router(setup.router, prefix="/api")
app.include_router(investigation.router, prefix="/api")
app.include_router(device_actions.router, prefix="/api")
app.include_router(insights.router, prefix="/api")
app.include_router(costs.router, prefix="/api")
app.include_router(system_health.router, prefix="/api")
app.include_router(location_intelligence.router, prefix="/api")
app.include_router(events_alerts.router, prefix="/api")
app.include_router(temporal.router, prefix="/api")
app.include_router(correlations.router, prefix="/api")
app.include_router(cross_device.router, prefix="/api")
app.include_router(data_quality.router, prefix="/api")
app.include_router(network.router, prefix="/api")
app.include_router(security.router, prefix="/api")
app.include_router(action_center.router, prefix="/api")
app.include_router(scranton_bridge.router, prefix="/api")


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Stella Sentinel API", "version": "0.1.0"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/ready")
def readiness():
    """Readiness check endpoint with service status.

    Returns status of all dependent services.
    """
    import os

    checks = {
        "api": True,
        "results_db": False,
        "backend_db": False,
        "llm": False,
        "qdrant": False,
    }

    # Check results database (SQLite)
    try:
        from sqlalchemy import text
        from device_anomaly.database.connection import get_results_db_session
        session = get_results_db_session()
        session.execute(text("SELECT 1"))
        session.close()
        checks["results_db"] = True
    except Exception as e:
        logger.warning("Results database health check failed: %s", e)

    # Check backend database (PostgreSQL)
    try:
        from device_anomaly.db.session import DatabaseSession
        db = DatabaseSession()
        checks["backend_db"] = db.test_connection()
    except Exception as e:
        logger.warning("Backend database health check failed: %s", e)

    # Check LLM service - optional (OpenAI-compatible API)
    try:
        import requests
        llm_base_url = os.getenv("LLM_BASE_URL", "")
        if llm_base_url:
            # Try OpenAI-compatible models endpoint
            resp = requests.get(f"{llm_base_url}/v1/models", timeout=2)
            checks["llm"] = resp.status_code == 200
    except Exception as e:
        logger.debug("LLM health check failed (optional service): %s", e)

    # Check Qdrant (vector database) - optional
    try:
        import requests
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        resp = requests.get(f"http://{qdrant_host}:{qdrant_port}/healthz", timeout=2)
        checks["qdrant"] = resp.status_code == 200
    except Exception as e:
        logger.debug("Qdrant health check failed (optional service): %s", e)

    # Determine overall status
    # Core services: api, results_db
    # Optional services: backend_db, ollama, qdrant
    core_healthy = checks["api"] and checks["results_db"]

    return {
        "status": "ready" if core_healthy else "degraded",
        "checks": checks,
    }
