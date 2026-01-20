"""
Setup API Routes

Endpoints for environment configuration and connection testing.
This module provides a UI-driven way to configure the .env file.
"""

import os
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from device_anomaly.api.dependencies import require_role
from device_anomaly.config.settings import reload_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/setup",
    tags=["setup"],
    dependencies=[Depends(require_role(["admin"]))],
)


class TestConnectionRequest(BaseModel):
    """Request model for testing a connection."""

    type: Literal["backend_db", "dw_db", "mc_db", "llm", "redis", "mobicontrol_api"]
    config: dict[str, Any]


class TestConnectionResponse(BaseModel):
    """Response model for connection test results."""

    success: bool
    message: str
    latency_ms: float | None = None


class SaveConfigResponse(BaseModel):
    """Response model for save configuration."""

    success: bool
    message: str
    file_path: str | None = None


class EnvironmentConfig(BaseModel):
    """Complete environment configuration model."""

    # Application Environment
    app_env: str = "local"

    # API Security & Tenant Context
    require_api_key: bool = False
    api_key: str = ""
    api_key_role: str = "viewer"
    trust_client_headers: bool = False
    require_tenant_header: bool = False
    default_tenant_id: str = "default"
    default_user_role: str = "viewer"
    tenant_id_allowlist: str = ""
    allow_in_process_training: bool = False

    # Observability
    enable_otel: bool = False
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_otlp_insecure: bool = True
    otel_service_name: str = "stella-sentinel-api"
    enable_metrics: bool = True
    enable_db_metrics: bool = True
    enable_tenant_metrics: bool = False
    tenant_metrics_allowlist: str = ""
    results_db_use_migrations: bool = False

    # Frontend Configuration
    frontend_port: int = 3000
    frontend_dev_port: int = 5173
    vite_tenant_id: str = "default"
    vite_api_key: str = ""
    vite_user_id: str = ""
    vite_user_role: str = "viewer"

    # PostgreSQL Backend Database
    backend_db_host: str = "postgres"
    backend_db_port: int = 5432
    backend_db_name: str = "anomaly_detection"
    backend_db_user: str = "postgres"
    backend_db_pass: str = "postgres"
    backend_db_connect_timeout: int = 5
    backend_db_statement_timeout_ms: int = 30000

    # SOTI XSight SQL Server Configuration
    dw_db_host: str = ""
    dw_db_port: int = 1433
    dw_db_name: str = "XSight"
    dw_db_user: str = ""
    dw_db_pass: str = ""
    dw_db_driver: str = "ODBC Driver 18 for SQL Server"
    dw_db_connect_timeout: int = 5
    dw_db_query_timeout: int = 30
    dw_trust_server_cert: bool = False

    # SOTI MobiControl SQL Server Configuration
    mc_db_host: str = ""
    mc_db_port: int = 1433
    mc_db_name: str = "MobiControlDB"
    mc_db_user: str = ""
    mc_db_pass: str = ""
    mc_db_driver: str = "ODBC Driver 18 for SQL Server"
    mc_db_connect_timeout: int = 5
    mc_db_query_timeout: int = 30
    mc_trust_server_cert: bool = False

    # LLM Configuration
    enable_llm: bool = False
    llm_base_url: str = "http://ollama:11434"
    llm_api_key: str = "not-needed"
    llm_model_name: str = "llama3.2"
    llm_api_version: str = ""
    llm_base_url_allowlist: str = ""

    # Real-Time Streaming Configuration
    enable_streaming: bool = False
    redis_url: str = "redis://redis:6379"
    redis_db: int = 0
    stream_buffer_size: int = 1000
    stream_flush_interval_ms: int = 100

    # SOTI MobiControl API Configuration
    mobicontrol_server_url: str = ""
    mobicontrol_client_id: str = ""
    mobicontrol_client_secret: str = ""
    mobicontrol_username: str = ""
    mobicontrol_password: str = ""
    mobicontrol_tenant_id: str = ""


def _test_postgres_connection(config: dict[str, Any]) -> TestConnectionResponse:
    """Test PostgreSQL connection."""
    import time

    try:
        import psycopg2

        start = time.time()
        conn = psycopg2.connect(
            host=config.get("backend_db_host", "postgres"),
            port=config.get("backend_db_port", 5432),
            dbname=config.get("backend_db_name", "anomaly_detection"),
            user=config.get("backend_db_user", "postgres"),
            password=config.get("backend_db_pass", ""),
            connect_timeout=config.get("backend_db_connect_timeout", 5),
        )
        conn.close()
        latency = (time.time() - start) * 1000

        return TestConnectionResponse(
            success=True,
            message="PostgreSQL connection successful",
            latency_ms=round(latency, 2),
        )
    except ImportError:
        return TestConnectionResponse(
            success=False,
            message="psycopg2 not installed. Install with: pip install psycopg2-binary",
        )
    except Exception as e:
        return TestConnectionResponse(
            success=False,
            message=f"PostgreSQL connection failed: {e!s}",
        )


def _test_sqlserver_connection(
    config: dict[str, Any], prefix: str
) -> TestConnectionResponse:
    """Test SQL Server connection."""
    import time

    try:
        import pyodbc

        host = config.get(f"{prefix}_host", "")
        port = config.get(f"{prefix}_port", 1433)
        database = config.get(f"{prefix}_name", "")
        user = config.get(f"{prefix}_user", "")
        password = config.get(f"{prefix}_pass", "")
        driver = config.get(f"{prefix}_driver", "ODBC Driver 18 for SQL Server")
        trust_cert = config.get(f"{prefix.replace('_db', '')}_trust_server_cert", False)

        if not host:
            return TestConnectionResponse(
                success=False,
                message="Host not configured",
            )

        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={host},{port};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={password};"
        )
        if trust_cert:
            conn_str += "TrustServerCertificate=yes;"

        start = time.time()
        conn = pyodbc.connect(conn_str, timeout=5)
        conn.close()
        latency = (time.time() - start) * 1000

        return TestConnectionResponse(
            success=True,
            message=f"SQL Server connection to {database} successful",
            latency_ms=round(latency, 2),
        )
    except ImportError:
        return TestConnectionResponse(
            success=False,
            message="pyodbc not installed. Install with: pip install pyodbc",
        )
    except Exception as e:
        return TestConnectionResponse(
            success=False,
            message=f"SQL Server connection failed: {e!s}",
        )


def _test_llm_connection(config: dict[str, Any]) -> TestConnectionResponse:
    """Test LLM connection."""
    import time

    try:
        import httpx

        base_url = config.get("llm_base_url", "http://ollama:11434")
        api_key = config.get("llm_api_key", "")

        # Determine the test endpoint based on the URL
        if "ollama" in base_url or ":11434" in base_url:
            test_url = f"{base_url.rstrip('/')}/api/tags"
            headers = {}
        elif "openai" in base_url:
            test_url = f"{base_url.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {api_key}"}
        else:
            # Generic OpenAI-compatible endpoint
            test_url = f"{base_url.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        start = time.time()
        with httpx.Client(timeout=10.0) as client:
            response = client.get(test_url, headers=headers)
            response.raise_for_status()
        latency = (time.time() - start) * 1000

        return TestConnectionResponse(
            success=True,
            message="LLM connection successful",
            latency_ms=round(latency, 2),
        )
    except ImportError:
        return TestConnectionResponse(
            success=False,
            message="httpx not installed",
        )
    except Exception as e:
        return TestConnectionResponse(
            success=False,
            message=f"LLM connection failed: {e!s}",
        )


def _test_redis_connection(config: dict[str, Any]) -> TestConnectionResponse:
    """Test Redis connection."""
    import time

    try:
        import redis

        redis_url = config.get("redis_url", "redis://redis:6379")
        redis_db = config.get("redis_db", 0)

        start = time.time()
        client = redis.from_url(redis_url, db=redis_db, socket_timeout=5)
        client.ping()
        client.close()
        latency = (time.time() - start) * 1000

        return TestConnectionResponse(
            success=True,
            message="Redis connection successful",
            latency_ms=round(latency, 2),
        )
    except ImportError:
        return TestConnectionResponse(
            success=False,
            message="redis not installed. Install with: pip install redis",
        )
    except Exception as e:
        return TestConnectionResponse(
            success=False,
            message=f"Redis connection failed: {e!s}",
        )


def _test_mobicontrol_api(config: dict[str, Any]) -> TestConnectionResponse:
    """Test MobiControl API connection."""
    import time

    try:
        import httpx

        server_url = config.get("mobicontrol_server_url", "")
        if not server_url:
            return TestConnectionResponse(
                success=False,
                message="MobiControl server URL not configured",
            )

        # Try to reach the server
        start = time.time()
        with httpx.Client(timeout=10.0, verify=False) as client:
            response = client.get(f"{server_url.rstrip('/')}/api/health")
            # Even a 401/403 means the server is reachable
            if response.status_code in [200, 401, 403]:
                latency = (time.time() - start) * 1000
                return TestConnectionResponse(
                    success=True,
                    message="MobiControl API server reachable",
                    latency_ms=round(latency, 2),
                )
            response.raise_for_status()

        return TestConnectionResponse(
            success=False,
            message="Could not reach MobiControl API",
        )
    except ImportError:
        return TestConnectionResponse(
            success=False,
            message="httpx not installed",
        )
    except Exception as e:
        return TestConnectionResponse(
            success=False,
            message=f"MobiControl API connection failed: {e!s}",
        )


@router.post("/test-connection", response_model=TestConnectionResponse)
async def test_connection(request: TestConnectionRequest) -> TestConnectionResponse:
    """
    Test a connection to a service.

    Supports testing:
    - backend_db: PostgreSQL backend database
    - dw_db: SOTI XSight SQL Server
    - mc_db: SOTI MobiControl SQL Server
    - llm: LLM/AI service
    - redis: Redis for streaming
    - mobicontrol_api: MobiControl REST API
    """
    try:
        if request.type == "backend_db":
            return _test_postgres_connection(request.config)
        elif request.type == "dw_db":
            return _test_sqlserver_connection(request.config, "dw_db")
        elif request.type == "mc_db":
            return _test_sqlserver_connection(request.config, "mc_db")
        elif request.type == "llm":
            return _test_llm_connection(request.config)
        elif request.type == "redis":
            return _test_redis_connection(request.config)
        elif request.type == "mobicontrol_api":
            return _test_mobicontrol_api(request.config)
        else:
            return TestConnectionResponse(
                success=False,
                message=f"Unknown connection type: {request.type}",
            )
    except Exception as e:
        logger.exception("Error testing connection")
        return TestConnectionResponse(
            success=False,
            message=f"Error testing connection: {e!s}",
        )


def _find_project_root() -> Path | None:
    """
    Find the project root directory by searching for marker files.

    Walks up from the current file and also checks the current working directory.
    """
    # Project root indicators (in order of priority)
    markers = ["docker-compose.yml", "docker-compose.yaml", "pyproject.toml", ".git"]

    # Strategy 1: Walk up from the current file location
    current_dir = Path(__file__).resolve().parent
    for _ in range(10):  # Limit search depth
        for marker in markers:
            if (current_dir / marker).exists():
                return current_dir
        if current_dir.parent == current_dir:
            break  # Reached filesystem root
        current_dir = current_dir.parent

    # Strategy 2: Check current working directory
    cwd = Path.cwd()
    for marker in markers:
        if (cwd / marker).exists():
            return cwd

    # Strategy 3: Check common paths relative to cwd
    for subdir in [".", "..", "../.."]:
        check_dir = (cwd / subdir).resolve()
        for marker in markers:
            if (check_dir / marker).exists():
                return check_dir

    # Strategy 4: Use /app if running in Docker
    docker_app = Path("/app")
    if docker_app.exists():
        for marker in markers:
            if (docker_app / marker).exists():
                return docker_app
        # Even without markers, /app is likely the right place in Docker
        return docker_app

    return None


@router.post("/save-config", response_model=SaveConfigResponse)
async def save_config(config: EnvironmentConfig) -> SaveConfigResponse:
    """
    Save environment configuration to .env file.

    This will create or overwrite the .env file in the project root.
    """
    try:
        # Find the project root
        project_root = _find_project_root()

        if project_root is None:
            raise HTTPException(
                status_code=500,
                detail="Could not locate project root directory. Ensure docker-compose.yml, pyproject.toml, or .git exists.",
            )

        env_path = project_root / ".env"

        # Generate .env content
        env_content = _generate_env_content(config)

        # Write the file
        with open(env_path, "w") as f:
            f.write(env_content)

        logger.info(f"Configuration saved to {env_path}")

        # Hot-reload settings so changes take effect immediately
        new_settings = reload_settings(env_path)
        logger.info(
            f"Settings hot-reloaded: ENABLE_LLM={new_settings.enable_llm}, "
            f"LLM_BASE_URL={new_settings.llm.base_url}"
        )

        return SaveConfigResponse(
            success=True,
            message="Configuration saved and applied successfully",
            file_path=str(env_path),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error saving configuration")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save configuration: {e!s}",
        ) from e


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """
    Get current environment configuration from .env file.

    Returns the current configuration if a .env file exists.
    """
    try:
        # Find the project root
        project_root = _find_project_root()

        if project_root is None:
            return {}

        env_path = project_root / ".env"

        if not env_path.exists():
            return {}

        # Parse the .env file
        config: dict[str, Any] = {}
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    # Convert to appropriate types
                    if value.lower() in ("true", "false"):
                        config[key] = value.lower() == "true"
                    elif value.isdigit():
                        config[key] = int(value)
                    else:
                        config[key] = value

        return config
    except Exception as e:
        logger.exception("Error reading configuration")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read configuration: {e!s}",
        ) from e


def _generate_env_content(config: EnvironmentConfig) -> str:
    """Generate .env file content from configuration."""
    return f"""# =============================================================================
# SOTI Stella Sentinel - Environment Configuration
# =============================================================================
# Generated by Setup Wizard
# NEVER commit this .env file to version control.
# =============================================================================

# Application Environment
# Options: local, development, staging, production
APP_ENV={config.app_env}

# =============================================================================
# API Security & Tenant Context
# =============================================================================
REQUIRE_API_KEY={str(config.require_api_key).lower()}
API_KEY={config.api_key}
API_KEY_ROLE={config.api_key_role}
TRUST_CLIENT_HEADERS={str(config.trust_client_headers).lower()}
REQUIRE_TENANT_HEADER={str(config.require_tenant_header).lower()}
DEFAULT_TENANT_ID={config.default_tenant_id}
DEFAULT_USER_ROLE={config.default_user_role}
TENANT_ID_ALLOWLIST={config.tenant_id_allowlist}
ALLOW_IN_PROCESS_TRAINING={str(config.allow_in_process_training).lower()}

# Observability
ENABLE_OTEL={str(config.enable_otel).lower()}
OTEL_EXPORTER_OTLP_ENDPOINT={config.otel_exporter_otlp_endpoint}
OTEL_EXPORTER_OTLP_INSECURE={str(config.otel_exporter_otlp_insecure).lower()}
OTEL_SERVICE_NAME={config.otel_service_name}
ENABLE_METRICS={str(config.enable_metrics).lower()}
ENABLE_DB_METRICS={str(config.enable_db_metrics).lower()}
ENABLE_TENANT_METRICS={str(config.enable_tenant_metrics).lower()}
TENANT_METRICS_ALLOWLIST={config.tenant_metrics_allowlist}
RESULTS_DB_USE_MIGRATIONS={str(config.results_db_use_migrations).lower()}

# =============================================================================
# Frontend Configuration
# =============================================================================
FRONTEND_PORT={config.frontend_port}
FRONTEND_DEV_PORT={config.frontend_dev_port}
VITE_TENANT_ID={config.vite_tenant_id}
VITE_API_KEY={config.vite_api_key}
VITE_USER_ID={config.vite_user_id}
VITE_USER_ROLE={config.vite_user_role}

# =============================================================================
# PostgreSQL Backend Database
# =============================================================================
BACKEND_DB_HOST={config.backend_db_host}
BACKEND_DB_PORT={config.backend_db_port}
BACKEND_DB_NAME={config.backend_db_name}
BACKEND_DB_USER={config.backend_db_user}
BACKEND_DB_PASS={config.backend_db_pass}
BACKEND_DB_CONNECT_TIMEOUT={config.backend_db_connect_timeout}
BACKEND_DB_STATEMENT_TIMEOUT_MS={config.backend_db_statement_timeout_ms}

# =============================================================================
# SOTI XSight SQL Server Configuration
# =============================================================================
DW_DB_HOST={config.dw_db_host}
DW_DB_PORT={config.dw_db_port}
DW_DB_NAME={config.dw_db_name}
DW_DB_USER={config.dw_db_user}
DW_DB_PASS={config.dw_db_pass}
DW_DB_DRIVER={config.dw_db_driver}
DW_DB_CONNECT_TIMEOUT={config.dw_db_connect_timeout}
DW_DB_QUERY_TIMEOUT={config.dw_db_query_timeout}
DW_TRUST_SERVER_CERT={str(config.dw_trust_server_cert).lower()}

# =============================================================================
# SOTI MobiControl SQL Server Configuration (Optional)
# =============================================================================
MC_DB_HOST={config.mc_db_host}
MC_DB_PORT={config.mc_db_port}
MC_DB_NAME={config.mc_db_name}
MC_DB_USER={config.mc_db_user}
MC_DB_PASS={config.mc_db_pass}
MC_DB_DRIVER={config.mc_db_driver}
MC_DB_CONNECT_TIMEOUT={config.mc_db_connect_timeout}
MC_DB_QUERY_TIMEOUT={config.mc_db_query_timeout}
MC_TRUST_SERVER_CERT={str(config.mc_trust_server_cert).lower()}

# =============================================================================
# LLM Configuration
# =============================================================================
ENABLE_LLM={str(config.enable_llm).lower()}
LLM_BASE_URL={config.llm_base_url}
LLM_API_KEY={config.llm_api_key}
LLM_MODEL_NAME={config.llm_model_name}
LLM_API_VERSION={config.llm_api_version}
LLM_BASE_URL_ALLOWLIST={config.llm_base_url_allowlist}

# =============================================================================
# Real-Time Streaming Configuration
# =============================================================================
ENABLE_STREAMING={str(config.enable_streaming).lower()}
REDIS_URL={config.redis_url}
REDIS_DB={config.redis_db}
STREAM_BUFFER_SIZE={config.stream_buffer_size}
STREAM_FLUSH_INTERVAL_MS={config.stream_flush_interval_ms}

# =============================================================================
# SOTI MobiControl API Configuration (Optional)
# =============================================================================
MOBICONTROL_SERVER_URL={config.mobicontrol_server_url}
MOBICONTROL_CLIENT_ID={config.mobicontrol_client_id}
MOBICONTROL_CLIENT_SECRET={config.mobicontrol_client_secret}
MOBICONTROL_USERNAME={config.mobicontrol_username}
MOBICONTROL_PASSWORD={config.mobicontrol_password}
MOBICONTROL_TENANT_ID={config.mobicontrol_tenant_id}
"""


# =============================================================================
# Location Sync Endpoints
# =============================================================================


class LocationSyncRequest(BaseModel):
    """Request model for location sync."""

    tenant_id: str = "default"
    label_types: list[str] | None = None
    include_device_groups: bool = True


class LocationSyncResponse(BaseModel):
    """Response model for location sync."""

    success: bool
    message: str
    synced_count: int
    labels_synced: int = 0
    groups_synced: int = 0
    duration_seconds: float
    errors: list[str] = []
    label_types_searched: list[str] = []


@router.post("/sync-locations", response_model=LocationSyncResponse)
async def sync_locations(request: LocationSyncRequest) -> LocationSyncResponse:
    """
    Sync locations from MobiControl to PostgreSQL.

    This endpoint imports location data from MobiControl labels and device groups
    to enable "Warehouse A vs Warehouse B" comparisons in insights.

    Label types that can represent locations:
    - Store, Warehouse, Site, Location, Building, Branch, Office, Facility, Region

    The sync creates location_metadata records that map devices to semantic location names.
    """
    try:
        from device_anomaly.services.location_sync import sync_all_locations

        result = sync_all_locations(
            tenant_id=request.tenant_id,
            label_types=request.label_types,
            include_device_groups=request.include_device_groups,
            raise_on_error=False,
        )

        errors = result.get("errors", [])
        success = result.get("success", False)
        if success and errors:
            message = f"Location sync completed with {len(errors)} warning(s)."
        elif success:
            message = "Locations synced successfully."
        else:
            message = "Location sync failed."

        return LocationSyncResponse(
            success=success,
            message=message,
            synced_count=result.get("synced_count", 0),
            labels_synced=result.get("labels_synced", 0),
            groups_synced=result.get("groups_synced", 0),
            duration_seconds=result.get("duration_seconds", 0.0),
            errors=errors,
            label_types_searched=request.label_types or [],
        )
    except Exception as e:
        logger.exception("Error syncing locations")
        return LocationSyncResponse(
            success=False,
            message=f"Location sync failed: {e}",
            synced_count=0,
            duration_seconds=0.0,
            errors=[str(e)],
        )


@router.get("/location-sync-stats")
async def get_location_sync_stats(tenant_id: str = "default") -> dict[str, Any]:
    """
    Get statistics about synced locations.

    Returns counts of locations from different sources (labels vs device groups).
    """
    try:
        from device_anomaly.services.location_sync import get_location_sync_stats as get_stats

        return get_stats(tenant_id=tenant_id)
    except Exception as e:
        logger.exception("Error getting location sync stats")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get location sync stats: {e!s}",
        ) from e
