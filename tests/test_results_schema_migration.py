from sqlalchemy import create_engine, text

from device_anomaly.database.connection import _migrate_results_schema


def _column_names(conn, table: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return {row[1] for row in rows}


def test_migrate_results_schema_adds_device_metadata_columns() -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE device_metadata (device_id INTEGER PRIMARY KEY, tenant_id TEXT)"))
        conn.execute(text("INSERT INTO device_metadata (device_id, tenant_id) VALUES (1, NULL)"))

    _migrate_results_schema(engine)

    with engine.begin() as conn:
        cols = _column_names(conn, "device_metadata")
        tenant_id = conn.execute(text("SELECT tenant_id FROM device_metadata WHERE device_id = 1")).scalar()

    assert "os_version" in cols
    assert "agent_version" in cols
    assert tenant_id == "default"
