from __future__ import annotations

import urllib.parse

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from device_anomaly.config.settings import get_settings


def create_dw_engine() -> Engine:
    """
    Create a SQLAlchemy engine for the DW database.
    Handles both regular hosts and SQL Server named instances (server\instance).
    """
    settings = get_settings().dw

    user = urllib.parse.quote_plus(settings.user)
    password = urllib.parse.quote_plus(settings.password)
    host = settings.host
    port = settings.port
    database = settings.database
    driver = urllib.parse.quote_plus(settings.driver)

    # For named instances (containing backslash), URL-encode the host and omit port
    # SQL Server Browser service will resolve the named instance port
    if "\\" in host:
        # Named instance: URL-encode the backslash as %5C
        host_encoded = host.replace("\\", "%5C")
        conn_str = (
            f"mssql+pyodbc://{user}:{password}@{host_encoded}/{database}"
            f"?driver={driver}"
            f"&TrustServerCertificate=yes"
        )
    else:
        # Regular host: include port in connection string
        conn_str = (
            f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}"
            f"?driver={driver}"
            f"&TrustServerCertificate=yes"
        )

    engine = create_engine(conn_str, fast_executemany=True)
    return engine
