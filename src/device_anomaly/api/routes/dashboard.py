"""API routes for dashboard endpoints."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, text
from sqlalchemy.orm import Session
import numpy as np

from device_anomaly.api.dependencies import get_db, get_mock_mode, get_tenant_id
from device_anomaly.api.mock_mode import (
    get_mock_dashboard_stats,
    get_mock_dashboard_trends,
    get_mock_connection_status,
    get_mock_isolation_forest_stats,
    get_mock_location_heatmap,
    get_mock_custom_attributes,
)
from device_anomaly.database.schema import TroubleshootingCache
from device_anomaly.api.models import (
    AllConnectionsStatusResponse,
    ConnectionStatusResponse,
    DashboardStatsResponse,
    DashboardTrendResponse,
    TroubleshootingAdviceResponse,
    IsolationForestConfigResponse,
    IsolationForestStatsResponse,
    ScoreDistributionResponse,
    ScoreDistributionBin,
    LocationHeatmapResponse,
    LocationDataResponse,
)
from device_anomaly.config.settings import get_settings
from device_anomaly.database.schema import AnomalyResult, AnomalyStatus, DeviceMetadata
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _parse_connection_error(error_str: str) -> str:
    """Parse SQL Server connection errors into user-friendly messages."""
    error_lower = error_str.lower()
    
    # Timeout errors
    if "login timeout" in error_lower or "timeout expired" in error_lower:
        return "Connection timed out — server may be unreachable or firewall is blocking"
    
    # Network/connectivity errors
    if "unable to connect" in error_lower or "connection refused" in error_lower:
        return "Cannot reach server — check if server is running and network is accessible"
    
    if "network-related" in error_lower or "instance-specific" in error_lower:
        return "Network error — server not found or SQL Browser service not running"
    
    # Authentication errors
    if "login failed for user" in error_lower:
        return "Login failed — invalid username or password"
    
    if "login failed" in error_lower and "not associated with a trusted" in error_lower:
        return "Login failed — Windows authentication not configured"
    
    # Database errors
    if "cannot open database" in error_lower:
        # Extract database name if possible
        if "requested by the login" in error_lower:
            return "Database not found or access denied — check database name and permissions"
        return "Cannot open database — database may not exist"
    
    # SSL/Certificate errors
    if "ssl" in error_lower or "certificate" in error_lower:
        return "SSL/Certificate error — server certificate not trusted"
    
    # Driver errors
    if "odbc driver" in error_lower and "not found" in error_lower:
        return "Database driver not installed"
    
    # Permission errors
    if "permission" in error_lower or "access denied" in error_lower:
        return "Access denied — insufficient permissions"
    
    # Generic connection errors
    if "connection" in error_lower and ("failed" in error_lower or "error" in error_lower):
        return "Connection failed — check server address and credentials"
    
    # Fallback: return a truncated, cleaned version
    # Remove the Python exception prefix if present
    if ")" in error_str and error_str.startswith("("):
        # Find the actual message after the error code
        parts = error_str.split("]")
        if len(parts) > 1:
            message = parts[-1].strip().rstrip(")")
            if message:
                return message[:100] + ("..." if len(message) > 100 else "")
    
    return error_str[:100] + ("..." if len(error_str) > 100 else "")


def _parse_api_error(error_str: str) -> str:
    """Parse API connection errors into user-friendly messages."""
    error_lower = error_str.lower()
    
    # Timeout
    if "timeout" in error_lower:
        return "Request timed out — server may be slow or unreachable"
    
    # Authentication errors
    if "401" in error_str:
        return "Authentication failed — check client ID and secret"
    if "403" in error_str:
        return "Access forbidden — insufficient API permissions"
    
    # Connection errors
    if "connection" in error_lower and ("refused" in error_lower or "error" in error_lower):
        return "Cannot connect to API — check server URL"
    
    if "name or service not known" in error_lower or "nodename nor servname" in error_lower:
        return "Server not found — check the server URL"
    
    # SSL errors
    if "ssl" in error_lower or "certificate" in error_lower:
        return "SSL error — certificate verification failed"
    
    # HTTP errors
    if "404" in error_str:
        return "API endpoint not found — check server URL"
    if "500" in error_str or "502" in error_str or "503" in error_str:
        return "Server error — the API server is having issues"
    
    return error_str[:100] + ("..." if len(error_str) > 100 else "")


@router.get("/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics (KPIs)."""
    # Return mock data if Mock Mode is enabled
    if mock_mode:
        mock_data = get_mock_dashboard_stats()
        return DashboardStatsResponse(**mock_data)
    
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    tenant_id = get_tenant_id()

    # Anomalies today
    anomalies_today = (
        db.query(func.count(AnomalyResult.id))
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(AnomalyResult.timestamp >= today_start)
        .scalar()
    ) or 0

    # Devices monitored (count distinct device_ids with anomalies)
    devices_monitored = (
        db.query(func.count(func.distinct(AnomalyResult.device_id)))
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .scalar()
    ) or 0

    # Critical issues (anomalies with very low scores, status open or investigating)
    critical_threshold = -0.7
    critical_issues = (
        db.query(func.count(AnomalyResult.id))
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(AnomalyResult.anomaly_score <= critical_threshold)
        .filter(
            AnomalyResult.status.in_([AnomalyStatus.OPEN.value, AnomalyStatus.INVESTIGATING.value])
        )
        .scalar()
    ) or 0

    # Resolved today
    resolved_today = (
        db.query(func.count(AnomalyResult.id))
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(AnomalyResult.status == AnomalyStatus.RESOLVED.value)
        .filter(AnomalyResult.updated_at >= today_start)
        .scalar()
    ) or 0

    # Open cases (anomalies with status 'open' or 'investigating')
    open_cases = (
        db.query(func.count(AnomalyResult.id))
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(
            AnomalyResult.status.in_([AnomalyStatus.OPEN.value, AnomalyStatus.INVESTIGATING.value])
        )
        .scalar()
    ) or 0

    # Total anomalies (all anomalies regardless of status)
    total_anomalies = (
        db.query(func.count(AnomalyResult.id))
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .scalar()
    ) or 0

    return DashboardStatsResponse(
        anomalies_today=anomalies_today,
        devices_monitored=devices_monitored,
        critical_issues=critical_issues,
        resolved_today=resolved_today,
        open_cases=open_cases,
        total_anomalies=total_anomalies,
    )


@router.get("/trends", response_model=list[DashboardTrendResponse])
def get_dashboard_trends(
    days: Optional[int] = Query(None, ge=1, le=90),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Get anomaly trends over time.
    
    Can use either 'days' parameter (for backward compatibility) or 'start_date'/'end_date' for custom ranges.
    If both are provided, start_date/end_date take precedence.
    """
    # Return mock data if Mock Mode is enabled
    if mock_mode:
        mock_data = get_mock_dashboard_trends(days or 7, start_date, end_date)
        return [DashboardTrendResponse(**item) for item in mock_data]
    
    now = datetime.now(timezone.utc)
    
    # Determine date range: prioritize start_date/end_date over days
    if start_date and end_date:
        # Use provided date range
        filter_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        filter_end = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif days:
        # Use days parameter (backward compatibility)
        filter_start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        filter_end = now
    else:
        # Default to 7 days if nothing specified
        filter_start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        filter_end = now

    # Group anomalies by date
    tenant_id = get_tenant_id()
    results = (
        db.query(
            func.date(AnomalyResult.timestamp).label("date"),
            func.count(AnomalyResult.id).label("count"),
        )
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.anomaly_label == -1)
        .filter(AnomalyResult.timestamp >= filter_start)
        .filter(AnomalyResult.timestamp <= filter_end)
        .group_by(func.date(AnomalyResult.timestamp))
        .order_by(func.date(AnomalyResult.timestamp))
        .all()
    )

    trends = []
    for result in results:
        # Convert date to datetime at midnight; SQLite may return a string.
        date_val = result.date
        if isinstance(date_val, str):
            try:
                date_val = datetime.fromisoformat(date_val).date()
            except ValueError:
                date_val = datetime.strptime(date_val, "%Y-%m-%d").date()
        elif isinstance(date_val, datetime):
            date_val = date_val.date()

        date_dt = datetime.combine(date_val, datetime.min.time())
        trends.append(DashboardTrendResponse(date=date_dt, anomaly_count=result.count))

    return trends


@router.get("/connections", response_model=AllConnectionsStatusResponse)
def get_connection_status(mock_mode: bool = Depends(get_mock_mode)):
    """Get connection status for all data sources and services.

    This endpoint tests each connection and returns the current status.
    The frontend auto-retries by polling this endpoint periodically.

    Services checked:
    - backend_db: PostgreSQL backend database
    - dw_sql: SOTI XSight SQL Server
    - mc_sql: SOTI MobiControl SQL Server
    - mobicontrol_api: MobiControl REST API
    - llm: LLM service (Ollama, LM Studio, etc.)
    - redis: Redis for real-time streaming
    - qdrant: Qdrant vector database
    """
    # Return mock data if Mock Mode is enabled
    if mock_mode:
        mock_data = get_mock_connection_status()
        return AllConnectionsStatusResponse(
            backend_db=ConnectionStatusResponse(**mock_data.get("backend_db", {"connected": True, "server": "postgres:5432", "error": None, "status": "connected"})),
            dw_sql=ConnectionStatusResponse(**mock_data["dw_sql"]),
            mc_sql=ConnectionStatusResponse(**mock_data["mc_sql"]),
            mobicontrol_api=ConnectionStatusResponse(**mock_data["mobicontrol_api"]),
            llm=ConnectionStatusResponse(**mock_data["llm"]),
            redis=ConnectionStatusResponse(**mock_data.get("redis", {"connected": True, "server": "redis:6379", "error": None, "status": "connected"})),
            qdrant=ConnectionStatusResponse(**mock_data.get("qdrant", {"connected": True, "server": "qdrant:6333", "error": None, "status": "connected"})),
            last_checked=datetime.fromisoformat(mock_data["last_checked"]),
        )

    settings = get_settings()

    # Check PostgreSQL Backend Database
    backend_db_connected = False
    backend_db_server = f"{settings.backend_db.host}:{settings.backend_db.port}"
    backend_db_error = None
    backend_db_status = "disconnected"
    try:
        from device_anomaly.db.session import DatabaseSession
        db = DatabaseSession()
        if db.test_connection():
            backend_db_connected = True
            backend_db_status = "connected"
        else:
            backend_db_error = "Connection test failed"
            backend_db_status = "error"
    except Exception as e:
        backend_db_error = str(e)[:100]
        backend_db_status = "error"

    # Check DW SQL Server connection
    dw_connected = False
    dw_server = settings.dw.host
    dw_error = None
    dw_status = "disconnected"
    try:
        from device_anomaly.data_access.db_connection import create_dw_engine

        engine = create_dw_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        dw_connected = True
        dw_status = "connected"
    except Exception as e:
        dw_error = _parse_connection_error(str(e))
        dw_status = "error"

    # Check MC SQL Server connection
    mc_connected = False
    mc_server = settings.mc.host or "Not configured"
    mc_error = None
    mc_status = "not_configured" if not settings.mc.host else "disconnected"
    try:
        if settings.mc.host:
            from device_anomaly.data_access.db_connection import create_mc_engine

            engine = create_mc_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            mc_connected = True
            mc_status = "connected"
        else:
            mc_error = "Database host not configured"
    except Exception as e:
        mc_error = _parse_connection_error(str(e))
        mc_status = "error"

    # Check MobiControl API connection
    mc_api_connected = False
    mc_api_server = settings.mobicontrol.server_url or "Not configured"
    mc_api_error = None

    # Check if feature is enabled
    if not settings.enable_mobicontrol:
        mc_api_status = "disabled"
        mc_api_error = "Feature disabled (set ENABLE_MOBICONTROL=true to enable)"
    elif not settings.mobicontrol.server_url:
        mc_api_status = "not_configured"
        mc_api_error = "Server URL not configured"
    elif not (settings.mobicontrol.client_id or settings.mobicontrol.username):
        mc_api_status = "not_configured"
        mc_api_error = "Authentication credentials not configured"
    else:
        mc_api_status = "disconnected"
        try:
            from device_anomaly.data_access.mobicontrol_client import MobiControlClient

            client = MobiControlClient()
            client._ensure_authenticated()
            mc_api_connected = True
            mc_api_status = "connected"
        except Exception as e:
            mc_api_error = _parse_api_error(str(e))
            mc_api_status = "error"

    # Check LLM connection (OpenAI-compatible API - LM Studio, vLLM, etc.)
    from device_anomaly.llm.client import get_llm_config_snapshot

    llm_connected = False
    llm_config = get_llm_config_snapshot()
    llm_base_url = llm_config["resolved_base_url"]
    llm_model = llm_config["resolved_model"] or ""
    llm_server = llm_base_url
    llm_error = None

    # Check if LLM feature is enabled
    if not settings.enable_llm:
        llm_status = "disabled"
        llm_error = "Feature disabled (set ENABLE_LLM=true to enable)"
    else:
        llm_status = "disconnected"
        try:
            import requests

            models_url = f"{llm_config['resolved_base_url_api']}/models"
            resp = requests.get(models_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    llm_connected = True
                    llm_status = "connected"
                    # Show the loaded model name
                    loaded_model = models[0].get("id", "unknown")
                    llm_server = f"{llm_base_url} ({loaded_model})"
                else:
                    llm_error = "No models loaded — load a model in LM Studio"
                    llm_status = "error"
            else:
                llm_error = f"LLM service returned error {resp.status_code}"
                llm_status = "error"

        except requests.exceptions.ConnectionError:
            llm_error = "Cannot connect to LLM — is LM Studio running?"
            llm_status = "error"
        except requests.exceptions.Timeout:
            llm_error = "Connection timed out — LLM may be starting up"
            llm_status = "error"
        except Exception as e:
            llm_error = _parse_api_error(str(e))
            llm_status = "error"

    # Check Redis connection (always check - used for ML automation, job queues, and streaming)
    redis_connected = False
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    redis_server = redis_url
    redis_error = None
    redis_status = "disconnected"

    try:
        import redis as redis_client
        r = redis_client.from_url(redis_url, socket_timeout=3)
        r.ping()
        r.close()
        redis_connected = True
        redis_status = "connected"
    except ImportError:
        redis_error = "Redis client not installed"
        redis_status = "error"
    except Exception as e:
        redis_error = f"Cannot connect to Redis — {str(e)[:50]}"
        redis_status = "error"

    # Check Qdrant connection
    qdrant_connected = False
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    qdrant_server = f"{qdrant_host}:{qdrant_port}"
    qdrant_error = None
    qdrant_status = "disconnected"

    try:
        import requests
        # Qdrant uses root endpoint for health check (returns version info)
        resp = requests.get(f"http://{qdrant_host}:{qdrant_port}/", timeout=3)
        if resp.status_code == 200:
            qdrant_connected = True
            qdrant_status = "connected"
        else:
            qdrant_error = f"Qdrant returned status {resp.status_code}"
            qdrant_status = "error"
    except requests.exceptions.ConnectionError:
        qdrant_error = "Cannot connect to Qdrant — is it running?"
        qdrant_status = "error"
    except requests.exceptions.Timeout:
        qdrant_error = "Connection timed out"
        qdrant_status = "error"
    except Exception as e:
        qdrant_error = str(e)[:50]
        qdrant_status = "error"

    return AllConnectionsStatusResponse(
        backend_db=ConnectionStatusResponse(
            connected=backend_db_connected,
            server=backend_db_server,
            error=backend_db_error,
            status=backend_db_status,
        ),
        dw_sql=ConnectionStatusResponse(
            connected=dw_connected,
            server=dw_server,
            error=dw_error,
            status=dw_status,
        ),
        mc_sql=ConnectionStatusResponse(
            connected=mc_connected,
            server=mc_server,
            error=mc_error,
            status=mc_status,
        ),
        mobicontrol_api=ConnectionStatusResponse(
            connected=mc_api_connected,
            server=mc_api_server,
            error=mc_api_error,
            status=mc_api_status,
        ),
        llm=ConnectionStatusResponse(
            connected=llm_connected,
            server=llm_server,
            error=llm_error,
            status=llm_status,
        ),
        redis=ConnectionStatusResponse(
            connected=redis_connected,
            server=redis_server,
            error=redis_error,
            status=redis_status,
        ),
        qdrant=ConnectionStatusResponse(
            connected=qdrant_connected,
            server=qdrant_server,
            error=qdrant_error,
            status=qdrant_status,
        ),
        last_checked=datetime.now(timezone.utc),
    )


@router.get("/llm/diagnostics")
def get_llm_diagnostics():
    """Return LLM configuration + connectivity diagnostics for troubleshooting."""
    from device_anomaly.llm.client import (
        AzureOpenAILLMClient,
        DummyLLMClient,
        OpenAICompatibleClient,
        get_default_llm_client,
        get_llm_config_snapshot,
    )

    config = get_llm_config_snapshot()
    base_url_api = config["resolved_base_url_api"]
    models_check = {
        "ok": False,
        "status_code": None,
        "models": [],
        "error": None,
        "models_url": f"{base_url_api}/models",
    }

    try:
        import requests

        resp = requests.get(models_check["models_url"], timeout=5)
        models_check["status_code"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            models_check["models"] = [m.get("id", "") for m in data.get("data", [])]
            models_check["ok"] = bool(models_check["models"])
            if not models_check["ok"]:
                models_check["error"] = "No models returned"
        else:
            models_check["error"] = f"HTTP {resp.status_code}"
    except requests.exceptions.ConnectionError as exc:
        models_check["error"] = f"Connection error: {exc}"
    except requests.exceptions.Timeout as exc:
        models_check["error"] = f"Timeout: {exc}"
    except Exception as exc:
        models_check["error"] = f"Unexpected error: {exc}"

    client = get_default_llm_client()
    client_info = {"type": type(client).__name__}
    if isinstance(client, OpenAICompatibleClient):
        client_info["base_url"] = client.base_url
        client_info["model_name"] = client.model_name
    elif isinstance(client, AzureOpenAILLMClient):
        client_info["azure_endpoint"] = config["resolved_base_url"]
        client_info["deployment_name"] = client.deployment_name
    elif isinstance(client, DummyLLMClient):
        client_info["note"] = "LLM not configured or unreachable"

    return {
        "config": config,
        "models_check": models_check,
        "client": client_info,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _generate_error_signature(failed_connections: list) -> str:
    """Generate a unique signature/hash for similar error patterns.
    
    This creates a normalized signature that groups similar errors together,
    ignoring server-specific details like hostnames/IPs.
    """
    import hashlib
    import json
    import re
    
    # Normalize errors by removing server-specific details
    normalized = []
    for conn in failed_connections:
        error = conn.get("error", "").lower()
        # Remove IP addresses, hostnames, and specific server names
        error = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', error)
        error = re.sub(r'\b[a-z0-9\-]+\.(com|net|org|local|lan)\b', '[HOST]', error)
        error = re.sub(r'\b[A-Z0-9\-]+\b', lambda m: m.group().lower() if len(m.group()) > 10 else m.group(), error)
        
        normalized.append({
            "service": conn.get("service", "").lower(),
            "error_pattern": error,
            "status": conn.get("status", "").lower()
        })
    
    # Sort to ensure consistent hashing regardless of order
    normalized.sort(key=lambda x: x["service"])
    signature_data = json.dumps(normalized, sort_keys=True)
    
    # Generate hash
    return hashlib.sha256(signature_data.encode()).hexdigest()


def _find_similar_cached_advice(
    db: Session,
    error_signature: str,
    tenant_id: str,
    service_type: str = None,
) -> TroubleshootingCache | None:
    """Find cached troubleshooting advice for similar errors."""
    query = db.query(TroubleshootingCache).filter(
        TroubleshootingCache.error_signature == error_signature,
        TroubleshootingCache.tenant_id == tenant_id,
    )
    
    if service_type:
        query = query.filter(TroubleshootingCache.service_type == service_type)
    
    cached = query.first()
    
    if cached:
        # Update usage statistics
        cached.use_count += 1
        cached.last_used = datetime.now(timezone.utc)
        db.commit()
        logger.info(f"Reusing cached troubleshooting advice (signature: {error_signature[:16]}..., uses: {cached.use_count})")
    
    return cached


def _generate_rule_based_advice(failed_connections: list) -> tuple[str, str]:
    """Generate troubleshooting advice using rule-based logic when LLM is unavailable.
    
    Returns (advice, summary) tuple.
    """
    advice_parts = [
        "⚠️ **RULE-BASED TROUBLESHOOTING** ⚠️\n\n"
        "This advice is generated using rule-based logic because the LLM service is not available.\n"
        "For more intelligent, context-aware troubleshooting, ensure your LLM service is running and configured.\n\n"
        "---\n\n"
    ]
    summary_parts = []
    
    for conn in failed_connections:
        service = conn.get("service", "")
        error = conn.get("error", "").lower()
        status = conn.get("status", "").lower()
        
        if "sql" in service.lower():
            if "timeout" in error or "login timeout" in error:
                advice_parts.append(f"**{service}**: Connection timeout detected. This usually means:\n- The database server is unreachable or slow\n- Firewall is blocking the connection\n- Network connectivity issues\n\n**Steps**:\n1. Verify the server is running: Check if the SQL Server service is active\n2. Test network connectivity: Try pinging the server\n3. Check firewall rules: Ensure port 1433 (or your configured port) is open\n4. Verify credentials: Ensure username and password are correct")
                summary_parts.append(f"{service} timeout")
            elif "login failed" in error or "authentication" in error:
                advice_parts.append(f"**{service}**: Authentication failed. This indicates:\n- Invalid username or password\n- Windows authentication not configured properly\n- User account locked or disabled\n\n**Steps**:\n1. Verify credentials in configuration\n2. Check if Windows authentication is required\n3. Ensure the SQL user has proper permissions\n4. Contact database administrator if credentials are correct")
                summary_parts.append(f"{service} authentication error")
            elif "cannot open database" in error or "database" in error and "not found" in error:
                advice_parts.append(f"**{service}**: Database access issue. Possible causes:\n- Database name is incorrect\n- Database doesn't exist\n- User lacks permissions to access the database\n\n**Steps**:\n1. Verify the database name in configuration\n2. Check if the database exists on the server\n3. Ensure the user has access permissions\n4. Contact database administrator")
                summary_parts.append(f"{service} database access error")
            elif "network-related" in error or "connection refused" in error:
                advice_parts.append(f"**{service}**: Network connectivity issue. This suggests:\n- Server is not running\n- Network path is incorrect\n- Firewall is blocking connections\n\n**Steps**:\n1. Verify server is running and accessible\n2. Check network connectivity (ping, telnet)\n3. Review firewall rules\n4. Verify server address and port in configuration")
                summary_parts.append(f"{service} network error")
            else:
                advice_parts.append(f"**{service}**: Connection failed with error: {conn.get('error', 'Unknown error')}\n\n**General SQL Server troubleshooting**:\n1. Verify SQL Server service is running\n2. Check network connectivity\n3. Review firewall settings\n4. Validate credentials and permissions\n5. Check SQL Server error logs for details")
                summary_parts.append(f"{service} connection error")
        
        elif "api" in service.lower():
            if "401" in error or "authentication" in error:
                advice_parts.append(f"**{service}**: Authentication failed (401). This means:\n- Invalid API credentials (client ID/secret)\n- Token expired or invalid\n- Insufficient permissions\n\n**Steps**:\n1. Verify API credentials in configuration\n2. Check if credentials need to be regenerated\n3. Ensure the API client has required permissions\n4. Contact API administrator for credential verification")
                summary_parts.append(f"{service} authentication error")
            elif "403" in error or "forbidden" in error:
                advice_parts.append(f"**{service}**: Access forbidden (403). This indicates:\n- Credentials are valid but lack permissions\n- API endpoint requires different access level\n\n**Steps**:\n1. Verify API client permissions\n2. Check if API access level is sufficient\n3. Contact API administrator to grant permissions")
                summary_parts.append(f"{service} permission error")
            elif "timeout" in error:
                advice_parts.append(f"**{service}**: Request timeout. Possible causes:\n- API server is slow or overloaded\n- Network latency issues\n- Server is not responding\n\n**Steps**:\n1. Check API server status\n2. Verify network connectivity\n3. Try again after a few moments\n4. Contact API administrator if issue persists")
                summary_parts.append(f"{service} timeout")
            elif "connection" in error and ("refused" in error or "error" in error):
                advice_parts.append(f"**{service}**: Cannot connect to API server. This suggests:\n- Server URL is incorrect\n- Server is not running\n- Network/firewall blocking connection\n\n**Steps**:\n1. Verify API server URL in configuration\n2. Check if API server is running\n3. Test network connectivity to the server\n4. Review firewall rules")
                summary_parts.append(f"{service} connection error")
            else:
                advice_parts.append(f"**{service}**: API connection failed: {conn.get('error', 'Unknown error')}\n\n**General API troubleshooting**:\n1. Verify API server URL is correct\n2. Check API server status\n3. Validate credentials\n4. Review network connectivity\n5. Check API documentation for specific error codes")
                summary_parts.append(f"{service} API error")
        
        elif "llm" in service.lower():
            if "cannot connect" in error or "connection" in error:
                advice_parts.append(f"**{service}**: Cannot connect to LLM service. This means:\n- LLM service (e.g., LM Studio) is not running\n- Server URL is incorrect\n- Port is blocked or incorrect\n\n**Steps**:\n1. Start the LLM service (e.g., launch LM Studio)\n2. Verify the service is listening on the configured port\n3. Check LLM_BASE_URL environment variable\n4. Test connection: curl http://localhost:1234/v1/models")
                summary_parts.append("LLM service not running")
            elif "no models loaded" in error:
                advice_parts.append(f"**{service}**: LLM service is running but no models are loaded.\n\n**Steps**:\n1. Open LM Studio (or your LLM service)\n2. Load a compatible model\n3. Ensure the model is fully loaded before using\n4. Verify the model name matches configuration")
                summary_parts.append("LLM model not loaded")
            else:
                advice_parts.append(f"**{service}**: LLM service error: {conn.get('error', 'Unknown error')}\n\n**Steps**:\n1. Verify LLM service is running\n2. Check LLM_BASE_URL configuration\n3. Ensure a model is loaded\n4. Review LLM service logs")
                summary_parts.append("LLM service error")
        
        else:
            advice_parts.append(f"**{service}**: Connection failed: {conn.get('error', 'Unknown error')}\n\n**General troubleshooting**:\n1. Verify service is running\n2. Check network connectivity\n3. Review configuration settings\n4. Check service logs for details")
            summary_parts.append(f"{service} connection error")
    
    summary = "Multiple connection issues detected" if len(summary_parts) > 1 else summary_parts[0] if summary_parts else "Connection issues"
    advice = "\n\n---\n\n".join(advice_parts)
    
    if len(failed_connections) > 1:
        advice = f"**SUMMARY**: {summary}\n\nMultiple services are experiencing connection failures. Address each service individually:\n\n{advice}\n\n**GENERAL RECOMMENDATIONS**:\n- Check if this is a network-wide issue\n- Verify firewall rules allow all required connections\n- Contact IT support if multiple services fail simultaneously"
    
    return advice, summary


def _save_troubleshooting_cache(
    db: Session,
    error_signature: str,
    error_pattern: str,
    advice: str,
    summary: str | None,
    tenant_id: str,
    service_type: str | None = None
) -> TroubleshootingCache:
    """Save troubleshooting advice to cache for future reuse."""
    cached = TroubleshootingCache(
        error_signature=error_signature,
        error_pattern=error_pattern,
        advice=advice,
        summary=summary,
        tenant_id=tenant_id,
        service_type=service_type,
        use_count=1,
        last_used=datetime.now(timezone.utc)
    )
    db.add(cached)
    db.commit()
    db.refresh(cached)
    logger.info(f"Cached new troubleshooting advice (signature: {error_signature[:16]}...)")
    return cached


@router.post("/connections/troubleshoot", response_model=TroubleshootingAdviceResponse)
def get_troubleshooting_advice(
    connection_status: AllConnectionsStatusResponse,
    db: Session = Depends(get_db)
):
    """Get intelligent troubleshooting advice from LLM based on connection errors.
    
    Analyzes all connection failures and provides actionable troubleshooting steps.
    Uses caching to learn from previous similar errors and avoid redundant LLM calls.
    """
    failed_connections = []
    
    if not connection_status.dw_sql.connected and connection_status.dw_sql.error:
        failed_connections.append({
            "service": "XSight Database SQL Server",
            "server": connection_status.dw_sql.server,
            "error": connection_status.dw_sql.error,
            "status": connection_status.dw_sql.status
        })
    
    if not connection_status.mc_sql.connected and connection_status.mc_sql.error:
        failed_connections.append({
            "service": "MobiControl SQL Server",
            "server": connection_status.mc_sql.server,
            "error": connection_status.mc_sql.error,
            "status": connection_status.mc_sql.status
        })
    
    if not connection_status.mobicontrol_api.connected and connection_status.mobicontrol_api.error:
        failed_connections.append({
            "service": "MobiControl API",
            "server": connection_status.mobicontrol_api.server,
            "error": connection_status.mobicontrol_api.error,
            "status": connection_status.mobicontrol_api.status
        })
    
    if not connection_status.llm.connected and connection_status.llm.error:
        failed_connections.append({
            "service": "LLM Service",
            "server": connection_status.llm.server,
            "error": connection_status.llm.error,
            "status": connection_status.llm.status
        })
    
    if not failed_connections:
        return TroubleshootingAdviceResponse(
            advice="All connections are healthy. No troubleshooting needed.",
            summary="All systems operational"
        )
    
    tenant_id = get_tenant_id()
    # Generate error signature for caching
    import json
    error_signature = _generate_error_signature(failed_connections)
    error_pattern_json = json.dumps(failed_connections, indent=2)
    
    # Determine service type (for filtering)
    service_types = set()
    for conn in failed_connections:
        service = conn.get("service", "").lower()
        if "sql" in service:
            service_types.add("sql")
        elif "api" in service:
            service_types.add("api")
        elif "llm" in service:
            service_types.add("llm")
    service_type = list(service_types)[0] if len(service_types) == 1 else None
    
    # Check cache first
    cached = _find_similar_cached_advice(db, error_signature, tenant_id, service_type)
    if cached:
        return TroubleshootingAdviceResponse(
            advice=cached.advice,
            summary=cached.summary
        )
    
    # Cache miss - generate new advice from LLM
    errors_json = json.dumps(failed_connections, indent=2)
    
    prompt = f"""<role>
You are Stella, an AI assistant for SOTI MobiControl administrators. You help troubleshoot connection issues in an enterprise mobile device management system that monitors thousands of mobile devices (Android, iOS, Windows) across warehouses, retail stores, and field operations.
</role>

<output_format>
IMPORTANT: Output ONLY the formatted response below. Do NOT include any internal reasoning, <think> tags, or preamble.

Structure your response EXACTLY as:

SUMMARY: [1-2 sentences: What's broken and business impact]

PRIORITY ACTIONS (in order of importance):
1. [Service Name]: [Specific action - what to check/do first]
2. [Service Name]: [Next action]

QUICK FIXES (things you can try yourself):
- [Actionable step with specific command/location if applicable]
- [Another step]

ESCALATE TO IT IF:
- [Condition requiring IT involvement]
</output_format>

<context>
Services in this system:
- XSight Database SQL Server: Data warehouse storing device telemetry (battery, connectivity, app usage). Connection timeout usually means network/firewall issues.
- MobiControl SQL Server: Core MDM database with device inventory, policies, compliance status. "Host not configured" means missing database_host in config.
- MobiControl API: REST API for device management actions. Requires ENABLE_MOBICONTROL=true environment variable to activate.
- LLM Service: AI service for generating insights. Local service (LM Studio) must be running with a model loaded.
- Redis: Cache layer for performance. Usually auto-recovers.
- Qdrant: Vector database for semantic search. Check if Docker container is running.
- Backend Database: PostgreSQL for application state. Check DATABASE_URL connection string.
</context>

<failed_services>
{errors_json}
</failed_services>

<instructions>
1. Prioritize by business impact: SQL databases > API > LLM > Cache
2. For each failure, identify the SPECIFIC error type (timeout, auth, config, network)
3. Provide concrete steps (file paths, commands, config keys) not generic advice
4. Distinguish between "restart/retry" fixes vs. "need IT/DBA" issues
5. Keep total response under 400 words
6. If multiple services fail with similar errors, note potential common cause (network, firewall, DNS)
</instructions>"""
    
    # Try LLM first, fallback to rule-based if unavailable
    try:
        from device_anomaly.llm.client import get_default_llm_client, DummyLLMClient, strip_thinking_tags
        
        llm_client = get_default_llm_client()
        
        # Check if it's a DummyLLMClient (no real LLM configured)
        if isinstance(llm_client, DummyLLMClient):
            # Use rule-based advice instead of dummy output
            logger.info("LLM not configured or unavailable, using rule-based troubleshooting advice")
            raise ValueError("LLM not configured")
        
        logger.info(f"Using LLM client: {type(llm_client).__name__} to generate troubleshooting advice")
        raw_advice = llm_client.generate(prompt, max_tokens=800, temperature=0.2)
        advice = strip_thinking_tags(raw_advice)

        # Check if we got dummy output (shouldn't happen, but be safe)
        if "DUMMY LLM CLIENT" in advice or "Dummy explanation" in advice or ("Dummy" in advice and len(advice) < 200):
            logger.warning("Received dummy LLM output despite using real client, falling back to rule-based advice")
            raise ValueError("Dummy LLM output received")
        
        logger.info(f"Successfully generated LLM troubleshooting advice from {type(llm_client).__name__} (length: {len(advice)} chars)")
        
        # Extract summary if present
        summary = None
        if "SUMMARY:" in advice:
            summary = advice.split("SUMMARY:")[1].split("\n")[0].strip() if "SUMMARY:" in advice else None
        
        # Save to cache for future use
        try:
            _save_troubleshooting_cache(
                db=db,
                error_signature=error_signature,
                error_pattern=error_pattern_json,
                advice=advice,
                summary=summary,
                tenant_id=tenant_id,
                service_type=service_type
            )
        except Exception as cache_error:
            # Don't fail if caching fails, just log it
            logger.warning(f"Failed to cache troubleshooting advice: {cache_error}")
        
        logger.info(f"Successfully generated LLM troubleshooting advice (length: {len(advice)})")
        return TroubleshootingAdviceResponse(
            advice=advice,
            summary=summary
        )
    except Exception as e:
        # Fallback to rule-based troubleshooting advice
        logger.info(f"Using rule-based troubleshooting (LLM unavailable: {e})")
        
        rule_advice, rule_summary = _generate_rule_based_advice(failed_connections)
        
        # Add note about why rule-based is being used
        if isinstance(e, ValueError) and "LLM not configured" in str(e):
            rule_advice = (
                "⚠️ **LLM SERVICE NOT AVAILABLE** ⚠️\n\n"
                "The LLM service is not configured or cannot be reached. "
                "Using rule-based troubleshooting advice instead.\n\n"
                "To enable LLM-powered troubleshooting:\n"
                "1. Ensure LLM_BASE_URL is set (e.g., http://localhost:1234)\n"
                "2. Ensure LLM_MODEL_NAME is set (e.g., deepseek/deepseek-r1-0528-qwen3-8b)\n"
                "3. Verify your LLM service is running and accessible\n\n"
                "---\n\n" + rule_advice
            )
        
        # Save rule-based advice to cache too
        try:
            _save_troubleshooting_cache(
                db=db,
                error_signature=error_signature,
                error_pattern=error_pattern_json,
                advice=rule_advice,
                summary=rule_summary,
                tenant_id=tenant_id,
                service_type=service_type
            )
        except Exception as cache_error:
            logger.warning(f"Failed to cache rule-based troubleshooting advice: {cache_error}")
        
        return TroubleshootingAdviceResponse(
            advice=rule_advice,
            summary=rule_summary
        )


@router.get("/connections/troubleshoot/cache/stats")
def get_troubleshooting_cache_stats(db: Session = Depends(get_db)):
    """Get statistics about the troubleshooting cache."""
    tenant_id = get_tenant_id()
    total_cached = db.query(TroubleshootingCache).filter(
        TroubleshootingCache.tenant_id == tenant_id
    ).count()
    total_uses = (
        db.query(func.sum(TroubleshootingCache.use_count))
        .filter(TroubleshootingCache.tenant_id == tenant_id)
        .scalar()
        or 0
    )
    most_used = (
        db.query(TroubleshootingCache)
        .filter(TroubleshootingCache.tenant_id == tenant_id)
        .order_by(TroubleshootingCache.use_count.desc())
        .limit(5)
        .all()
    )
    
    return {
        "total_cached_advice": total_cached,
        "total_times_reused": total_uses,
        "average_reuses_per_advice": round(total_uses / total_cached, 2) if total_cached > 0 else 0,
        "most_used_advice": [
            {
                "id": item.id,
                "service_type": item.service_type,
                "summary": item.summary,
                "use_count": item.use_count,
                "last_used": item.last_used.isoformat() if item.last_used else None,
            }
            for item in most_used
        ]
    }


@router.get("/isolation-forest/stats", response_model=IsolationForestStatsResponse)
def get_isolation_forest_stats(
    days: Optional[int] = Query(30, ge=1, le=365),
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Get Isolation Forest model statistics and score distribution.
    
    Returns model configuration, score distribution histogram, and statistics
    for the specified time period (default: last 30 days).
    """
    # Return mock data if Mock Mode is enabled
    if mock_mode:
        mock_data = get_mock_isolation_forest_stats(days)
        return IsolationForestStatsResponse(
            config=IsolationForestConfigResponse(**mock_data["config"]),
            score_distribution=ScoreDistributionResponse(
                bins=[ScoreDistributionBin(**b) for b in mock_data["score_distribution"]["bins"]],
                total_normal=mock_data["score_distribution"]["total_normal"],
                total_anomalies=mock_data["score_distribution"]["total_anomalies"],
                mean_score=mock_data["score_distribution"]["mean_score"],
                median_score=mock_data["score_distribution"]["median_score"],
                min_score=mock_data["score_distribution"]["min_score"],
                max_score=mock_data["score_distribution"]["max_score"],
                std_score=mock_data["score_distribution"]["std_score"],
            ),
            total_predictions=mock_data["total_predictions"],
            anomaly_rate=mock_data["anomaly_rate"],
        )
    
    # Get model configuration from code (default values)
    from device_anomaly.models.anomaly_detector import AnomalyDetectorConfig
    
    default_config = AnomalyDetectorConfig()
    config = IsolationForestConfigResponse(
        n_estimators=default_config.n_estimators,
        contamination=default_config.contamination,
        random_state=default_config.random_state,
        scale_features=default_config.scale_features,
        min_variance=default_config.min_variance,
        model_type="isolation_forest",
    )
    
    # Calculate date range
    now = datetime.now(timezone.utc)
    filter_start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Get all scores (both normal and anomalies) from the time period
    tenant_id = get_tenant_id()
    results = (
        db.query(
            AnomalyResult.anomaly_score,
            AnomalyResult.anomaly_label,
        )
        .filter(AnomalyResult.tenant_id == tenant_id)
        .filter(AnomalyResult.timestamp >= filter_start)
        .all()
    )
    
    if not results:
        # Return empty distribution if no data
        return IsolationForestStatsResponse(
            config=config,
            score_distribution=ScoreDistributionResponse(
                bins=[],
                total_normal=0,
                total_anomalies=0,
                mean_score=0.0,
                median_score=0.0,
                min_score=0.0,
                max_score=0.0,
                std_score=0.0,
            ),
            total_predictions=0,
            anomaly_rate=0.0,
        )
    
    # Extract scores and labels
    scores = [r.anomaly_score for r in results]
    labels = [r.anomaly_label for r in results]
    
    # Calculate statistics
    scores_array = np.array(scores)
    total_predictions = len(scores)
    total_anomalies = sum(1 for l in labels if l == -1)
    total_normal = total_predictions - total_anomalies
    anomaly_rate = total_anomalies / total_predictions if total_predictions > 0 else 0.0
    
    mean_score = float(np.mean(scores_array))
    median_score = float(np.median(scores_array))
    min_score = float(np.min(scores_array))
    max_score = float(np.max(scores_array))
    std_score = float(np.std(scores_array))
    
    # Create histogram bins (30 bins from min to max)
    num_bins = 30
    normal_scores = np.array([score for score, label in zip(scores, labels) if label == 1])
    anomaly_scores = np.array([score for score, label in zip(scores, labels) if label == -1])

    bins = []
    if min_score == max_score:
        epsilon = max(abs(min_score) * 1e-6, 1e-6)
        bin_start = float(min_score - epsilon)
        bin_end = float(max_score + epsilon)
        if normal_scores.size:
            bins.append(ScoreDistributionBin(
                bin_start=bin_start,
                bin_end=bin_end,
                count=int(normal_scores.size),
                is_anomaly=False,
            ))
        if anomaly_scores.size:
            bins.append(ScoreDistributionBin(
                bin_start=bin_start,
                bin_end=bin_end,
                count=int(anomaly_scores.size),
                is_anomaly=True,
            ))
    else:
        bin_edges = np.linspace(min_score, max_score, num_bins + 1)
        normal_hist, _ = np.histogram(normal_scores, bins=bin_edges)
        anomaly_hist, _ = np.histogram(anomaly_scores, bins=bin_edges)

        for i in range(num_bins):
            bin_start = float(bin_edges[i])
            bin_end = float(bin_edges[i + 1])
            normal_count = int(normal_hist[i])
            anomaly_count = int(anomaly_hist[i])

            if normal_count > 0:
                bins.append(ScoreDistributionBin(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    count=normal_count,
                    is_anomaly=False,
                ))

            if anomaly_count > 0:
                bins.append(ScoreDistributionBin(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    count=anomaly_count,
                    is_anomaly=True,
                ))
    
    score_distribution = ScoreDistributionResponse(
        bins=bins,
        total_normal=total_normal,
        total_anomalies=total_anomalies,
        mean_score=mean_score,
        median_score=median_score,
        min_score=min_score,
        max_score=max_score,
        std_score=std_score,
    )
    
    return IsolationForestStatsResponse(
        config=config,
        score_distribution=score_distribution,
        total_predictions=total_predictions,
        anomaly_rate=anomaly_rate,
    )


@router.get("/custom-attributes")
def get_custom_attributes(mock_mode: bool = Depends(get_mock_mode)):
    """Get list of available custom attributes from MobiControl devices.
    
    Fetches a sample of devices from MobiControl and extracts unique custom attribute names.
    Returns a list of custom attribute names that can be used for location grouping.
    """
    # Return mock data if Mock Mode is enabled
    if mock_mode:
        return get_mock_custom_attributes()
    
    try:
        from device_anomaly.data_access.mobicontrol_client import MobiControlClient
        
        settings = get_settings()
        if not settings.mobicontrol.server_url:
            return {"custom_attributes": [], "error": "MobiControl server URL not configured"}
        
        client = MobiControlClient()
        client._ensure_authenticated()
        
        # Fetch a sample of devices to extract custom attributes
        # We'll fetch up to 100 devices to get a good sample
        devices_response = client.get_devices(page=1, page_size=100)
        
        # Extract custom attributes from devices
        # The response structure may vary, so we handle different formats
        if isinstance(devices_response, list):
            devices = devices_response
        elif isinstance(devices_response, dict):
            devices = devices_response.get("data", [])
        else:
            devices = []
        
        logger.info(f"Fetched {len(devices)} devices from MobiControl")
        
        # Log sample device keys for debugging
        if devices and isinstance(devices[0], dict):
            sample_device = devices[0]
            logger.debug(f"Sample device keys: {list(sample_device.keys())[:20]}")
            # Log if we see any custom attribute related fields
            custom_fields = [k for k in sample_device.keys() if 'custom' in k.lower() or 'attribute' in k.lower() or 'Custom' in k]
            if custom_fields:
                logger.info(f"Found custom attribute related fields: {custom_fields}")
        
        custom_attributes = set()
        
        # Try to extract custom attributes from device list
        # Custom attributes might be in CustomAttributes, customAttributes, or CustomData fields
        for device in devices:
            if not isinstance(device, dict):
                continue
            
            # Only extract from CustomAttributes field (not CustomData)
            for field_name in ["CustomAttributes", "customAttributes"]:
                custom_attrs = device.get(field_name)
                if custom_attrs is None:
                    continue
                
                if isinstance(custom_attrs, dict):
                    # If it's a dict, extract the keys (attribute names)
                    custom_attributes.update(custom_attrs.keys())
                elif isinstance(custom_attrs, str):
                    # If it's a string, try to parse as JSON
                    try:
                        import json
                        parsed = json.loads(custom_attrs)
                        if isinstance(parsed, dict):
                            custom_attributes.update(parsed.keys())
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(custom_attrs, list):
                    # If it's a list of attribute objects, extract names
                    for attr in custom_attrs:
                        if isinstance(attr, dict):
                            # Try common keys for attribute name
                            name = attr.get("name") or attr.get("Name") or attr.get("key") or attr.get("Key") or attr.get("attributeName") or attr.get("AttributeName")
                            if name:
                                custom_attributes.add(name)
        
        # If no custom attributes found in list, try fetching individual device details
        # Custom attributes might only be available in full device details
        if not custom_attributes and devices:
            logger.info("No custom attributes found in device list, trying individual device details...")
            try:
                # Sample a few devices to get custom attributes
                sample_size = min(5, len(devices))
                for device in devices[:sample_size]:
                    if not isinstance(device, dict):
                        continue
                    
                    # Get device ID from various possible fields
                    device_id = device.get("DeviceId") or device.get("deviceId") or device.get("id") or device.get("ID")
                    if not device_id:
                        continue
                    
                    try:
                        device_details = client.get_device(str(device_id))
                        if not isinstance(device_details, dict):
                            continue
                        
                        # Only extract from CustomAttributes field (not CustomData)
                        for field_name in ["CustomAttributes", "customAttributes"]:
                            custom_attrs = device_details.get(field_name)
                            if custom_attrs is None:
                                continue
                            
                            if isinstance(custom_attrs, dict):
                                custom_attributes.update(custom_attrs.keys())
                            elif isinstance(custom_attrs, str):
                                try:
                                    import json
                                    parsed = json.loads(custom_attrs)
                                    if isinstance(parsed, dict):
                                        custom_attributes.update(parsed.keys())
                                except (json.JSONDecodeError, TypeError):
                                    pass
                    except Exception as e:
                        logger.debug(f"Failed to fetch details for device {device_id}: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Error fetching individual device details: {e}")
        
        # Convert to sorted list
        custom_attributes_list = sorted(list(custom_attributes))
        
        return {"custom_attributes": custom_attributes_list}
        
    except Exception as e:
        logger.warning(f"Failed to fetch custom attributes from MobiControl: {e}")
        # Return empty list on error, don't fail the request
        return {"custom_attributes": [], "error": str(e)}


@router.get("/location-heatmap", response_model=LocationHeatmapResponse)
def get_location_heatmap(
    attribute_name: Optional[str] = Query(None, description="Custom attribute name (e.g., 'Store', 'Warehouse')"),
    mock_mode: bool = Depends(get_mock_mode),
    db: Session = Depends(get_db),
):
    """Get location heatmap data grouped by custom attribute.
    
    Groups devices by the specified custom attribute (e.g., Store, Warehouse, Location)
    and calculates utilization metrics and baselines for each location.
    
    Note: This is a placeholder implementation. When custom attributes are properly
    synced from MobiControl into the device metadata (extra_data JSON field),
    this endpoint should be updated to query actual data.
    """
    # Return mock data if Mock Mode is enabled
    if mock_mode:
        mock_data = get_mock_location_heatmap(attribute_name)
        return LocationHeatmapResponse(
            locations=[LocationDataResponse(**loc) for loc in mock_data["locations"]],
            attributeName=mock_data["attributeName"],
            totalLocations=mock_data["totalLocations"],
            totalDevices=mock_data["totalDevices"],
        )
    
    # Default attribute name
    attr_name = attribute_name or "Store"
    
    # TODO: Implement actual data aggregation when custom attributes are available
    # For now, return empty data structure that matches the expected response
    # 
    # Future implementation should:
    # 1. Query devices from DeviceMetadata table
    # 2. Extract custom attribute values from extra_data JSON field (or join with MobiControl data)
    # 3. Group devices by attribute value
    # 4. Calculate utilization metrics (active devices / total devices per location)
    # 5. Calculate baselines (historical averages or configured baselines)
    # 6. Count anomalies per location
    
    locations: List[LocationDataResponse] = []
    
    # Placeholder: Return empty list for now
    # When custom attributes are available, this will be populated with actual data
    # Example structure:
    # locations = [
    #     LocationDataResponse(
    #         id="A101",
    #         name="A101",
    #         utilization=75.5,
    #         baseline=80.0,
    #         deviceCount=20,
    #         activeDeviceCount=15,
    #         region="North",
    #         anomalyCount=2,
    #     ),
    #     ...
    # ]
    
    return LocationHeatmapResponse(
        locations=locations,
        attributeName=attr_name,
        totalLocations=len(locations),
        totalDevices=sum(loc.deviceCount for loc in locations),
    )
