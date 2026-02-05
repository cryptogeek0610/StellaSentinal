from __future__ import annotations

import logging
import time
import urllib.parse
from collections.abc import Callable
from threading import Lock

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from device_anomaly.config.settings import get_settings

logger = logging.getLogger(__name__)


def _with_retry(
    func: Callable[[], Engine],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Engine:
    """Execute engine creation with exponential backoff retry."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except OperationalError as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Database connection attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    str(e)[:100],
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Database connection failed after %d attempts: %s", max_retries, str(e)
                )
    raise last_error


_DW_ENGINE = None
_MC_ENGINE = None
_ENGINE_LOCK = Lock()


def _build_connect_args(settings) -> dict:
    connect_args = {"timeout": settings.connect_timeout}
    try:
        import pyodbc

        if settings.query_timeout:
            connect_args["attrs_before"] = {
                pyodbc.SQL_ATTR_QUERY_TIMEOUT: int(settings.query_timeout),
            }
    except Exception:
        # pyodbc might not expose SQL_ATTR_QUERY_TIMEOUT in some environments.
        pass
    return connect_args


def create_dw_engine() -> Engine:
    """
    Create a SQLAlchemy engine for the DW database.
    Handles both regular hosts and SQL Server named instances (server\\instance).
    """
    settings = get_settings().dw

    user = urllib.parse.quote_plus(settings.user)
    password = urllib.parse.quote_plus(settings.password)
    host = settings.host
    port = settings.port
    database = settings.database
    driver = urllib.parse.quote_plus(settings.driver)
    # Use configurable TrustServerCertificate - set to False in production with proper certs
    trust_cert = "yes" if settings.trust_server_certificate else "no"

    # For named instances (containing backslash), URL-encode the host and omit port
    # SQL Server Browser service will resolve the named instance port
    if "\\" in host:
        # Named instance: URL-encode the backslash as %5C
        host_encoded = host.replace("\\", "%5C")
        conn_str = (
            f"mssql+pyodbc://{user}:{password}@{host_encoded}/{database}"
            f"?driver={driver}"
            f"&TrustServerCertificate={trust_cert}"
        )
    else:
        # Regular host: include port in connection string
        conn_str = (
            f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
            f"?driver={driver}"
            f"&TrustServerCertificate={trust_cert}"
        )

    global _DW_ENGINE
    if _DW_ENGINE is None:
        with _ENGINE_LOCK:
            if _DW_ENGINE is None:
                _DW_ENGINE = create_engine(
                    conn_str,
                    fast_executemany=True,
                    pool_pre_ping=True,
                    pool_size=settings.pool_size,
                    max_overflow=5,
                    pool_recycle=settings.pool_recycle,
                    pool_timeout=settings.connect_timeout,
                    connect_args=_build_connect_args(settings),
                )

                # Log connection events for debugging
                @event.listens_for(_DW_ENGINE, "connect")
                def on_dw_connect(dbapi_conn, connection_record):
                    logger.debug("DW database connection established")

                try:
                    from device_anomaly.observability.db_metrics import (
                        instrument_engine as instrument_db_metrics,
                    )
                    from device_anomaly.observability.sqlalchemy import (
                        instrument_engine as instrument_otel,
                    )

                    instrument_db_metrics(_DW_ENGINE, "dw")
                    instrument_otel(_DW_ENGINE)
                except Exception as e:
                    logger.debug("Failed to instrument DW engine: %s", e)
    return _DW_ENGINE


def create_mc_engine() -> Engine:
    """
    Create a SQLAlchemy engine for the MobiControl database.
    Handles both regular hosts and SQL Server named instances (server\\instance).
    """
    settings = get_settings().mc

    user = urllib.parse.quote_plus(settings.user)
    password = urllib.parse.quote_plus(settings.password)
    host = settings.host
    port = settings.port
    database = settings.database
    driver = urllib.parse.quote_plus(settings.driver)
    # Use configurable TrustServerCertificate - set to False in production with proper certs
    trust_cert = "yes" if settings.trust_server_certificate else "no"

    # For named instances (containing backslash), URL-encode the host and omit port
    # SQL Server Browser service will resolve the named instance port
    if "\\" in host:
        # Named instance: URL-encode the backslash as %5C
        host_encoded = host.replace("\\", "%5C")
        conn_str = (
            f"mssql+pyodbc://{user}:{password}@{host_encoded}/{database}"
            f"?driver={driver}"
            f"&TrustServerCertificate={trust_cert}"
        )
    else:
        # Regular host: include port in connection string
        conn_str = (
            f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
            f"?driver={driver}"
            f"&TrustServerCertificate={trust_cert}"
        )

    global _MC_ENGINE
    if _MC_ENGINE is None:
        with _ENGINE_LOCK:
            if _MC_ENGINE is None:
                _MC_ENGINE = create_engine(
                    conn_str,
                    fast_executemany=True,
                    pool_pre_ping=True,
                    pool_size=settings.pool_size,
                    max_overflow=5,
                    pool_recycle=settings.pool_recycle,
                    pool_timeout=settings.connect_timeout,
                    connect_args=_build_connect_args(settings),
                )

                # Log connection events for debugging
                @event.listens_for(_MC_ENGINE, "connect")
                def on_mc_connect(dbapi_conn, connection_record):
                    logger.debug("MC database connection established")

                try:
                    from device_anomaly.observability.db_metrics import (
                        instrument_engine as instrument_db_metrics,
                    )
                    from device_anomaly.observability.sqlalchemy import (
                        instrument_engine as instrument_otel,
                    )

                    instrument_db_metrics(_MC_ENGINE, "mc")
                    instrument_otel(_MC_ENGINE)
                except Exception as e:
                    logger.debug("Failed to instrument MC engine: %s", e)
    return _MC_ENGINE
