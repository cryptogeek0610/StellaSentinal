import json
import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, model_validator

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


class LocalLLMSettings(BaseModel):
    """Settings for local LLM services (Ollama, LM Studio, vLLM).

    These settings are used by the LocalLLMClient for enterprise on-prem
    deployments where a local LLM is preferred over cloud APIs.
    """

    provider: str = "ollama"  # ollama, lmstudio, vllm
    base_url: str = "http://ollama:11434"
    model_name: str | None = None  # Auto-discovered if not set
    fallback_model: str = "llama3.2:1b"  # Smaller model for fallback
    enable_caching: bool = True  # Cache responses (15 min TTL)
    fallback_to_rules: bool = True  # Use rule-based analysis if LLM unavailable
    max_retries: int = 3
    timeout_seconds: int = 30


class MobiControlSettings(BaseModel):
    server_url: str
    client_id: str | None = None
    client_secret: str | None = None
    username: str | None = None
    password: str | None = None
    tenant_id: str | None = None

    # Custom Attribute names for battery status sync
    # These map to the Custom Attributes in MobiControl that will display battery info
    attr_battery_health: str = "BatteryHealthPercent"
    attr_battery_status: str = "BatteryStatus"
    attr_replacement_due: str = "BatteryReplacementDue"
    attr_replacement_urgency: str = "BatteryReplacementUrgency"
    attr_replacement_date: str = "BatteryReplacementDate"


class ResultsDBSettings(BaseModel):
    """Settings for the results database (where anomaly results are stored)."""
    url: str | None = os.getenv("RESULTS_DB_URL")
    path: str = os.getenv("RESULTS_DB_PATH", "anomaly_results.db")


class AppSettings(BaseModel):
    """Application settings with dynamic environment variable loading.

    All settings are read from os.environ at instantiation time, not at
    class definition time. This allows hot-reloading of settings.
    """

    env: str = "local"

    # XSight SQL Database (telemetry source data)
    dw: DBSettings = Field(default_factory=lambda: DBSettings(
        host="host.docker.internal",
        database="XSight",
        user="",
        password="",
    ))

    # Backend database (anomaly detection app state) - PostgreSQL
    backend_db: PostgresSettings = Field(default_factory=lambda: PostgresSettings())

    # MobiControl SQL Database (device inventory, policies - optional enrichment)
    mc: DBSettings = Field(default_factory=lambda: DBSettings(
        host="host.docker.internal",
        database="MobiControlDB",
        user="",
        password="",
    ))

    results_db: ResultsDBSettings = Field(default_factory=lambda: ResultsDBSettings())
    llm: LLMSettings = Field(default_factory=lambda: LLMSettings())
    local_llm: LocalLLMSettings = Field(default_factory=lambda: LocalLLMSettings())
    mobicontrol: MobiControlSettings = Field(default_factory=lambda: MobiControlSettings(server_url=""))

    # Feature flags
    enable_llm: bool = False
    enable_mobicontrol: bool = False

    # Extended data ingestion feature flags (ON by default for full data coverage)
    enable_mc_timeseries: bool = True  # MobiControl time-series tables (DeviceStatInt, etc.)
    enable_xsight_hourly: bool = True  # XSight hourly tables (cs_DataUsageByHour, etc.) - HIGH VOLUME
    enable_xsight_extended: bool = True  # Extended XSight tables (cs_WiFiLocation, etc.)
    enable_schema_discovery: bool = True  # Runtime schema discovery caching (safe, metadata only)

    # ML Training expansion flags
    enable_hourly_training: bool = False  # Include hourly granularity tables in ML training
    enable_training_discovery: bool = False  # Auto-discover high-value tables for ML training
    hourly_max_days: int = 7  # Limit hourly data to recent N days (memory management)

    # Table allowlists - if set, only these tables are ingested even when broader flags enabled
    # Comma-separated list of table names
    xsight_table_allowlist: List[str] = Field(default_factory=list)
    mc_table_allowlist: List[str] = Field(default_factory=list)

    # Ingestion configuration
    ingest_lookback_hours: int = 24  # Default lookback for new tables
    ingest_batch_size: int = 50000  # Max rows per batch
    ingest_max_tables_parallel: int = 3  # Max concurrent table loads (weight-based)
    max_backfill_days_hourly: int = 2  # Max days to backfill for hourly tables

    # Watermark configuration
    enable_file_watermark_fallback: bool = False  # File fallback for dev only

    # Canonical event storage (heavy, OFF by default)
    enable_canonical_event_storage: bool = False
    auto_create_canonical_events_tables: bool = False

    # Observability
    enable_ingestion_metrics: bool = True  # Emit per-table metrics
    enable_daily_coverage_report: bool = True  # Generate daily coverage report
    auto_create_metrics_tables: bool = False  # Create metrics tables on startup (safe default: off)

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Multi-source training configuration
    training_data_sources: List[TrainingDataSource] = Field(default_factory=list)

    # Vector database (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ==========================================================================
    # Carl's Requirements: Location and User Assignment Label Configuration
    # ==========================================================================
    # These settings control how device data is enriched with location and user
    # information from MobiControl labels. Enables:
    # - "Warehouse A vs Warehouse B" comparisons
    # - "People with excessive drops" by-user analysis

    # MobiControl label types that represent physical locations
    # Devices with these labels will be mapped to location_metadata for location-based analysis
    location_label_types: List[str] = Field(
        default_factory=lambda: ["Store", "Warehouse", "Site", "Building", "Location", "Branch", "Facility"]
    )

    # MobiControl label types that represent user assignments
    # Devices with these labels will enable by-user analysis (drops, reboots)
    user_assignment_label_types: List[str] = Field(
        default_factory=lambda: ["Owner", "User", "AssignedUser", "Operator", "Employee", "Worker"]
    )

    # Reboot detection patterns for MainLog EventClass search
    reboot_event_patterns: List[str] = Field(
        default_factory=lambda: ["reboot", "restart", "boot", "power cycle", "device started"]
    )

    @model_validator(mode='before')
    @classmethod
    def load_from_environment(cls, data: dict) -> dict:
        """Load all settings from os.environ at instantiation time."""
        # Only load from env if no explicit data is provided
        # This allows explicit values to override env vars

        # Environment
        if 'env' not in data:
            data['env'] = os.getenv("APP_ENV", "local")

        # XSight Database
        if 'dw' not in data:
            data['dw'] = DBSettings(
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

        # Backend Database
        if 'backend_db' not in data:
            data['backend_db'] = PostgresSettings(
                host=os.getenv("BACKEND_DB_HOST", "postgres"),
                port=int(os.getenv("BACKEND_DB_PORT", "5432")),
                database=os.getenv("BACKEND_DB_NAME", "anomaly_detection"),
                user=os.getenv("BACKEND_DB_USER", "postgres"),
                password=os.getenv("BACKEND_DB_PASS", "postgres"),
            )

        # MobiControl Database
        if 'mc' not in data:
            data['mc'] = DBSettings(
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

        # Results Database
        if 'results_db' not in data:
            data['results_db'] = ResultsDBSettings(
                url=os.getenv("RESULTS_DB_URL"),
                path=os.getenv("RESULTS_DB_PATH", "anomaly_results.db"),
            )

        # LLM Settings
        if 'llm' not in data:
            data['llm'] = LLMSettings(
                api_key=os.getenv("LLM_API_KEY"),
                model_name=os.getenv("LLM_MODEL_NAME"),
                base_url=os.getenv("LLM_BASE_URL"),
                api_version=os.getenv("LLM_API_VERSION"),
            )

        # Local LLM Settings (Ollama, LM Studio, vLLM)
        if 'local_llm' not in data:
            data['local_llm'] = LocalLLMSettings(
                provider=os.getenv("LOCAL_LLM_PROVIDER", "ollama"),
                base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://ollama:11434"),
                model_name=os.getenv("LOCAL_LLM_MODEL_NAME"),
                fallback_model=os.getenv("LOCAL_LLM_FALLBACK_MODEL", "llama3.2:1b"),
                enable_caching=os.getenv("LOCAL_LLM_ENABLE_CACHING", "true").lower() == "true",
                fallback_to_rules=os.getenv("LOCAL_LLM_FALLBACK_TO_RULES", "true").lower() == "true",
                max_retries=int(os.getenv("LOCAL_LLM_MAX_RETRIES", "3")),
                timeout_seconds=int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "30")),
            )

        # MobiControl API Settings
        if 'mobicontrol' not in data:
            data['mobicontrol'] = MobiControlSettings(
                server_url=os.getenv("MOBICONTROL_SERVER_URL", ""),
                client_id=os.getenv("MOBICONTROL_CLIENT_ID"),
                client_secret=os.getenv("MOBICONTROL_CLIENT_SECRET"),
                username=os.getenv("MOBICONTROL_USERNAME"),
                password=os.getenv("MOBICONTROL_PASSWORD"),
                tenant_id=os.getenv("MOBICONTROL_TENANT_ID"),
                # Custom Attribute names (can customize to match your MobiControl setup)
                attr_battery_health=os.getenv("MC_ATTR_BATTERY_HEALTH", "BatteryHealthPercent"),
                attr_battery_status=os.getenv("MC_ATTR_BATTERY_STATUS", "BatteryStatus"),
                attr_replacement_due=os.getenv("MC_ATTR_REPLACEMENT_DUE", "BatteryReplacementDue"),
                attr_replacement_urgency=os.getenv("MC_ATTR_REPLACEMENT_URGENCY", "BatteryReplacementUrgency"),
                attr_replacement_date=os.getenv("MC_ATTR_REPLACEMENT_DATE", "BatteryReplacementDate"),
            )

        # Feature flags
        if 'enable_llm' not in data:
            data['enable_llm'] = os.getenv("ENABLE_LLM", "false").lower() == "true"
        if 'enable_mobicontrol' not in data:
            data['enable_mobicontrol'] = os.getenv("ENABLE_MOBICONTROL", "false").lower() == "true"

        # Extended data ingestion feature flags (ON by default for full data coverage)
        if 'enable_mc_timeseries' not in data:
            data['enable_mc_timeseries'] = os.getenv("ENABLE_MC_TIMESERIES", "true").lower() == "true"
        if 'enable_xsight_hourly' not in data:
            data['enable_xsight_hourly'] = os.getenv("ENABLE_XSIGHT_HOURLY", "true").lower() == "true"
        if 'enable_xsight_extended' not in data:
            data['enable_xsight_extended'] = os.getenv("ENABLE_XSIGHT_EXTENDED", "true").lower() == "true"
        if 'enable_schema_discovery' not in data:
            data['enable_schema_discovery'] = os.getenv("ENABLE_SCHEMA_DISCOVERY", "true").lower() == "true"

        # ML Training expansion flags
        if 'enable_hourly_training' not in data:
            data['enable_hourly_training'] = os.getenv("ENABLE_HOURLY_TRAINING", "false").lower() == "true"
        if 'enable_training_discovery' not in data:
            data['enable_training_discovery'] = os.getenv("ENABLE_TRAINING_DISCOVERY", "false").lower() == "true"
        if 'hourly_max_days' not in data:
            data['hourly_max_days'] = int(os.getenv("HOURLY_MAX_DAYS", "7"))

        # Table allowlists (comma-separated)
        if 'xsight_table_allowlist' not in data:
            allowlist = os.getenv("XSIGHT_TABLE_ALLOWLIST", "")
            data['xsight_table_allowlist'] = [t.strip() for t in allowlist.split(",") if t.strip()]
        if 'mc_table_allowlist' not in data:
            allowlist = os.getenv("MC_TABLE_ALLOWLIST", "")
            data['mc_table_allowlist'] = [t.strip() for t in allowlist.split(",") if t.strip()]

        # Ingestion configuration
        if 'ingest_lookback_hours' not in data:
            data['ingest_lookback_hours'] = int(os.getenv("INGEST_LOOKBACK_HOURS", "24"))
        if 'ingest_batch_size' not in data:
            data['ingest_batch_size'] = int(os.getenv("INGEST_BATCH_SIZE", "50000"))
        if 'ingest_max_tables_parallel' not in data:
            data['ingest_max_tables_parallel'] = int(os.getenv("INGEST_MAX_TABLES_PARALLEL", "3"))
        if 'enable_canonical_event_storage' not in data:
            data['enable_canonical_event_storage'] = os.getenv(
                "ENABLE_CANONICAL_EVENT_STORAGE", "false"
            ).lower() == "true"
        if 'auto_create_canonical_events_tables' not in data:
            data['auto_create_canonical_events_tables'] = os.getenv(
                "AUTO_CREATE_CANONICAL_EVENTS_TABLES", "false"
            ).lower() == "true"
        if 'max_backfill_days_hourly' not in data:
            data['max_backfill_days_hourly'] = int(os.getenv("MAX_BACKFILL_DAYS_HOURLY", "2"))

        # Watermark configuration
        if 'enable_file_watermark_fallback' not in data:
            data['enable_file_watermark_fallback'] = os.getenv("ENABLE_FILE_WATERMARK_FALLBACK", "false").lower() == "true"

        # Observability
        if 'enable_ingestion_metrics' not in data:
            data['enable_ingestion_metrics'] = os.getenv("ENABLE_INGESTION_METRICS", "true").lower() == "true"
        if 'enable_daily_coverage_report' not in data:
            data['enable_daily_coverage_report'] = os.getenv("ENABLE_DAILY_COVERAGE_REPORT", "true").lower() == "true"
        if 'auto_create_metrics_tables' not in data:
            data['auto_create_metrics_tables'] = os.getenv("AUTO_CREATE_METRICS_TABLES", "false").lower() == "true"

        # API settings
        if 'api_host' not in data:
            data['api_host'] = os.getenv("API_HOST", "0.0.0.0")
        if 'api_port' not in data:
            data['api_port'] = int(os.getenv("API_PORT", "8000"))

        # Qdrant
        if 'qdrant_host' not in data:
            data['qdrant_host'] = os.getenv("QDRANT_HOST", "localhost")
        if 'qdrant_port' not in data:
            data['qdrant_port'] = int(os.getenv("QDRANT_PORT", "6333"))

        # Carl's Requirements: Location and User Assignment Labels
        if 'location_label_types' not in data:
            labels = os.getenv("LOCATION_LABEL_TYPES", "Store,Warehouse,Site,Building,Location,Branch,Facility")
            data['location_label_types'] = [t.strip() for t in labels.split(",") if t.strip()]
        if 'user_assignment_label_types' not in data:
            labels = os.getenv("USER_ASSIGNMENT_LABEL_TYPES", "Owner,User,AssignedUser,Operator,Employee,Worker")
            data['user_assignment_label_types'] = [t.strip() for t in labels.split(",") if t.strip()]
        if 'reboot_event_patterns' not in data:
            patterns = os.getenv("REBOOT_EVENT_PATTERNS", "reboot,restart,boot,power cycle,device started")
            data['reboot_event_patterns'] = [p.strip() for p in patterns.split(",") if p.strip()]

        # Training data sources
        if 'training_data_sources' not in data:
            sources_json = os.getenv("TRAINING_DATA_SOURCES", "")
            if sources_json:
                try:
                    sources_list = json.loads(sources_json)
                    data['training_data_sources'] = [
                        TrainingDataSource(**src) for src in sources_list
                    ]
                except (json.JSONDecodeError, TypeError):
                    data['training_data_sources'] = []
            else:
                data['training_data_sources'] = []

        return data


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


def reload_settings(env_path: Path | str | None = None) -> AppSettings:
    """
    Hot-reload settings from .env file.

    This function re-reads the .env file and updates os.environ,
    then recreates the settings object with the new values.

    Args:
        env_path: Optional path to .env file. If not provided,
                  searches for it in common locations.

    Returns:
        The newly loaded AppSettings instance.
    """
    global _settings
    import logging

    logger = logging.getLogger(__name__)

    # Find the .env file
    if env_path is None:
        # Try multiple locations
        search_paths = [
            Path("/app/.env"),  # Docker container
            Path(__file__).resolve().parents[3] / ".env",  # From src/device_anomaly/config
            Path.cwd() / ".env",  # Current working directory
        ]
        for path in search_paths:
            if path.exists():
                env_path = path
                break

    if env_path is None or not Path(env_path).exists():
        logger.warning("Could not find .env file for hot-reload")
        return get_settings()

    env_path = Path(env_path)
    logger.info(f"Hot-reloading settings from {env_path}")

    try:
        from dotenv import load_dotenv

        # Reload .env with override=True to update existing env vars
        load_dotenv(env_path, override=True)

        # Clear the cached settings
        _settings = None

        # Create new settings instance (will read from updated os.environ)
        _settings = AppSettings()

        logger.info(
            f"Settings reloaded: ENABLE_LLM={_settings.enable_llm}, "
            f"LLM_BASE_URL={_settings.llm.base_url}"
        )

        return _settings

    except ImportError:
        logger.error("python-dotenv not installed, cannot hot-reload")
        return get_settings()
    except Exception as e:
        logger.exception(f"Failed to reload settings: {e}")
        return get_settings()
