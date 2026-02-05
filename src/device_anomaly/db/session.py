"""Database session management.

This module provides session management for the backend PostgreSQL database.
It handles connection pooling, session lifecycle, and provides both
context managers and dependency injection patterns for FastAPI.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from device_anomaly.config.settings import get_settings


class DatabaseSession:
    """Manages database connections and sessions.

    This class provides a thread-safe singleton pattern for database connectivity,
    handling connection pooling and session creation.

    Example:
        # Initialize once at app startup
        db = DatabaseSession()

        # Use in a context manager
        with db.session() as session:
            tenant = session.query(Tenant).filter_by(tenant_id='t1').first()

        # Or get a session manually
        session = db.get_session()
        try:
            # work with session
            session.commit()
        finally:
            session.close()
    """

    _instance: DatabaseSession | None = None
    _engine: Engine | None = None
    _session_factory: sessionmaker | None = None
    _lock: Lock = Lock()  # Thread-safe singleton lock

    def __new__(cls) -> DatabaseSession:
        """Thread-safe singleton pattern for database session."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the database session manager."""
        if self._engine is None:
            with self._lock:
                # Double-check to avoid race condition during initialization
                if self._engine is None:
                    self._initialize()

    def _initialize(self) -> None:
        """Initialize the database engine and session factory."""
        settings = get_settings()
        db = settings.backend_db

        # Use PostgreSQL connection URL
        connection_string = db.url
        connect_timeout = int(os.getenv("BACKEND_DB_CONNECT_TIMEOUT", "5"))
        statement_timeout_ms = int(os.getenv("BACKEND_DB_STATEMENT_TIMEOUT_MS", "30000"))
        connect_args = {
            "connect_timeout": connect_timeout,
            "options": f"-c statement_timeout={statement_timeout_ms}",
        }

        self._engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections after 30 minutes
            pool_pre_ping=True,  # Verify connections before use
            echo=settings.env == "local",  # Log SQL in local env
            connect_args=connect_args,
        )
        try:
            from device_anomaly.observability.db_metrics import (
                instrument_engine as instrument_db_metrics,
            )
            from device_anomaly.observability.sqlalchemy import instrument_engine as instrument_otel

            instrument_db_metrics(self._engine, "backend")
            instrument_otel(self._engine)
        except Exception:
            pass

        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        if self._engine is None:
            self._initialize()
        return self._engine

    def get_session(self) -> Session:
        """Get a new database session.

        Returns:
            A new SQLAlchemy Session

        Note:
            Caller is responsible for closing the session.
        """
        if self._session_factory is None:
            self._initialize()
        return self._session_factory()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions.

        Yields:
            A SQLAlchemy Session

        Example:
            with db.session() as session:
                result = session.query(Anomaly).all()
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test if database connection is working.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            with self.session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close all connections and dispose of the engine."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        Primarily useful for testing.
        """
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None


# Convenience function for getting a session
def get_session() -> Generator[Session, None, None]:
    """Dependency injection function for FastAPI.

    Yields:
        A SQLAlchemy Session

    Example:
        @app.get("/anomalies")
        def list_anomalies(session: Session = Depends(get_session)):
            return session.query(Anomaly).all()
    """
    db = DatabaseSession()
    session = db.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
