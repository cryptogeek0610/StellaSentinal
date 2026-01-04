import json
import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip


class TrainingDataSource(BaseModel):
    """Configuration for a single training data source (customer database)."""
    name: str  # e.g., "BENELUX", "PIBLIC"
    xsight_db: str  # XSight database name
    mc_db: str  # MobiControl database name
    # Optional overrides for host/credentials (defaults to main SQL Server)
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None


class DBSettings(BaseModel):
    """SQL Server connection settings (for XSight/MobiControl source data).

    Both XSight and MobiControl databases typically reside on the same SQL Server.
    Test environment: A0024952\\SQLEXPRESS (a0024952.mobicontrol.cloud:1433)
    """
    host: str
    database: str
    user: str
    password: str
    driver: str = "ODBC Driver 18 for SQL Server"
    port: int = 1433
    # Security: Set to False in production with proper SSL certificates
    # Only use True for local development with self-signed certs
    trust_server_certificate: bool = False
    connect_timeout: int = 15  # 15 seconds for named instances/network latency
    query_timeout: int = 60
    pool_size: int = 5
    pool_recycle: int = 1800  # Recycle connections every 30 minutes


class PostgresSettings(BaseModel):
    """PostgreSQL connection settings (for application backend data)."""
    host: str = "localhost"
    port: int = 5432
    database: str = "anomaly_detection"
    user: str = "postgres"
    password: str = ""

    @property
    def url(self) -> str:
        """Build PostgreSQL connection URL with properly encoded credentials."""
        # URL-encode user and password to handle special characters safely
        encoded_user = quote_plus(self.user)
        encoded_password = quote_plus(self.password)
        return f"postgresql://{encoded_user}:{encoded_password}@{self.host}:{self.port}/{self.database}"


class LLMSettings(BaseModel):
    api_key: str | None = None
    model_name: str | None = None
    base_url: str | None = None
    api_version: str | None = None


class MobiControlSettings(BaseModel):
    server_url: str
    client_id: str | None = None
    client_secret: str | None = None
    username: str | None = None
    password: str | None = None
    tenant_id: str | None = None


class ResultsDBSettings(BaseModel):
    """Settings for the results database (where anomaly results are stored)."""
    url: str | None = os.getenv("RESULTS_DB_URL")
    path: str = os.getenv("RESULTS_DB_PATH", "anomaly_results.db")


class AppSettings(BaseModel):
    env: str = os.getenv("APP_ENV", "local")

    # XSight SQL Database (telemetry source data)
    # Default: host.docker.internal for Docker deployments
    # For production: set DW_DB_HOST to your SQL Server instance
    dw: DBSettings = DBSettings(
        host=os.getenv("DW_DB_HOST", "host.docker.internal"),
        database=os.getenv("DW_DB_NAME", "XSight"),
        user=os.getenv("DW_DB_USER", ""),
        password=os.getenv("DW_DB_PASS", ""),
        port=int(os.getenv("DW_DB_PORT", "1433")),
        driver=os.getenv("DW_DB_DRIVER", "ODBC Driver 18 for SQL Server"),
        trust_server_certificate=os.getenv("DW_TRUST_SERVER_CERT", "true").lower() == "true",
        connect_timeout=int(os.getenv("DW_DB_CONNECT_TIMEOUT", "15")),
        query_timeout=int(os.getenv("DW_DB_QUERY_TIMEOUT", "60")),
        pool_size=int(os.getenv("DW_DB_POOL_SIZE", "5")),
        pool_recycle=int(os.getenv("DW_DB_POOL_RECYCLE", "1800")),
    )

    # Backend database (anomaly detection app state) - PostgreSQL
    # Default: postgres (Docker service name)
    backend_db: PostgresSettings = PostgresSettings(
        host=os.getenv("BACKEND_DB_HOST", "postgres"),
        port=int(os.getenv("BACKEND_DB_PORT", "5432")),
        database=os.getenv("BACKEND_DB_NAME", "anomaly_detection"),
        user=os.getenv("BACKEND_DB_USER", "postgres"),
        password=os.getenv("BACKEND_DB_PASS", "postgres"),
    )

    # MobiControl SQL Database (device inventory, policies - optional enrichment)
    # Default: host.docker.internal for Docker deployments
    # For production: set MC_DB_HOST to your SQL Server instance
    mc: DBSettings = DBSettings(
        host=os.getenv("MC_DB_HOST", "host.docker.internal"),
        database=os.getenv("MC_DB_NAME", "MobiControlDB"),
        user=os.getenv("MC_DB_USER", ""),
        password=os.getenv("MC_DB_PASS", ""),
        port=int(os.getenv("MC_DB_PORT", "1433")),
        driver=os.getenv("MC_DB_DRIVER", "ODBC Driver 18 for SQL Server"),
        trust_server_certificate=os.getenv("MC_TRUST_SERVER_CERT", "true").lower() == "true",
        connect_timeout=int(os.getenv("MC_DB_CONNECT_TIMEOUT", "15")),
        query_timeout=int(os.getenv("MC_DB_QUERY_TIMEOUT", "60")),
        pool_size=int(os.getenv("MC_DB_POOL_SIZE", "5")),
        pool_recycle=int(os.getenv("MC_DB_POOL_RECYCLE", "1800")),
    )

    results_db: ResultsDBSettings = ResultsDBSettings()

    llm: LLMSettings = LLMSettings(
        api_key=os.getenv("LLM_API_KEY"),
        model_name=os.getenv("LLM_MODEL_NAME"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_version=os.getenv("LLM_API_VERSION"),
    )

    mobicontrol: MobiControlSettings = MobiControlSettings(
        server_url=os.getenv("MOBICONTROL_SERVER_URL", ""),
        client_id=os.getenv("MOBICONTROL_CLIENT_ID"),
        client_secret=os.getenv("MOBICONTROL_CLIENT_SECRET"),
        username=os.getenv("MOBICONTROL_USERNAME"),
        password=os.getenv("MOBICONTROL_PASSWORD"),
        tenant_id=os.getenv("MOBICONTROL_TENANT_ID"),
    )

    # Feature flags
    enable_llm: bool = os.getenv("ENABLE_LLM", "false").lower() == "true"
    enable_mobicontrol: bool = os.getenv("ENABLE_MOBICONTROL", "false").lower() == "true"

    # API settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Multi-source training configuration
    # JSON array of training data sources, e.g.:
    # [{"name": "BENELUX", "xsight_db": "XSight_BENELUX", "mc_db": "MobiControl_BENELUX"},
    #  {"name": "PIBLIC", "xsight_db": "XSight_PIBLIC", "mc_db": "MobiControl_PIBLIC"}]
    training_data_sources: List[TrainingDataSource] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        # Parse training data sources from environment
        sources_json = os.getenv("TRAINING_DATA_SOURCES", "")
        if sources_json:
            try:
                sources_list = json.loads(sources_json)
                self.training_data_sources = [
                    TrainingDataSource(**src) for src in sources_list
                ]
            except (json.JSONDecodeError, TypeError) as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to parse TRAINING_DATA_SOURCES: {e}"
                )

    # Vector database (Qdrant)
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))


_settings: AppSettings | None = None


def get_settings(validate_dw: bool = False) -> AppSettings:
    """Get application settings.

    Args:
        validate_dw: If True, validate that DW credentials are set.
                     Set to True when actually connecting to XSight DW.
    """
    global _settings
    if _settings is None:
        _settings = AppSettings()

    # Only validate DW credentials when explicitly requested
    if validate_dw:
        if not _settings.dw.user or not _settings.dw.password:
            raise ValueError("DW_DB_USER and DW_DB_PASS must be set for DW connection.")
        if _settings.dw.password in {"sa", "YourStrong!Passw0rd"}:
            raise ValueError("DW_DB_PASS must not use a default password.")

    return _settings


def reset_settings() -> None:
    """Reset cached settings. Useful for testing."""
    global _settings
    _settings = None
