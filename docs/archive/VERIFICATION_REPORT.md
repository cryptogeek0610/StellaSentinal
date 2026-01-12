# Verification Report

## Commands run
- `./.venv/bin/python -m pytest -q`
- `./.venv/bin/python scripts/verify.py`
- `python3 -m py_compile $(rg --files -g "*.py")`
- `PYTHONPATH=src ./.venv/bin/python -m device_anomaly.cli.ingestion --dry-run --xsight-table cs_DataUsageByHour --mc-table DeviceStatInt`
- `docker compose build --no-cache`
- `docker compose up -d --force-recreate`
- `docker compose logs -f --tail=200` (terminated after ~20s by command timeout)
- `docker compose ps`
- `curl -sS -m 5 http://127.0.0.1:8000/health`
- `curl -sS -m 5 http://127.0.0.1:8000/health/ready`
- `curl -sS -m 10 http://127.0.0.1:8000/api/dashboard/isolation-forest/stats`
- `docker compose exec -T postgres psql -U postgres -d anomaly_detection -c "\d device_metadata"`

## Failures found
- Postgres logs reported `device_metadata.os_version` missing when the API queried device metadata. Root cause: existing Postgres schema lacked newly added columns, and `create_all()` does not backfill columns on existing tables.
- `device_anomaly.cli.ingestion --dry-run` (host execution) warned that `postgres` hostname was not resolvable. This is expected outside Docker; the run completed in dry-run mode.

## Fixes applied
- Added a safe results DB schema migration for both SQLite and Postgres to add missing `device_metadata.os_version` / `agent_version` columns and ensure tenant defaults for existing tables. Path: `src/device_anomaly/database/connection.py`.
- Added a regression test covering the SQLite migration behavior. Path: `tests/test_results_schema_migration.py`.

## Final status
- Backend tests pass (99 tests).
- `scripts/verify.py` passes (backend + frontend checks).
- `py_compile` passes.
- Docker images rebuilt and containers restarted.
- Health endpoints return `{"status":"healthy"}` and `{"status":"ready", ...}`.
- Dashboard stats endpoint returns a JSON payload successfully.

## Docker status (`docker compose ps`)
```
NAME                       IMAGE                        COMMAND                  SERVICE     CREATED              STATUS                                 PORTS
stellasentinal-app         anomalydetection-app         "uvicorn device_anom…"   app         About a minute ago   Up About a minute                      0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp
stellasentinal-frontend    anomalydetection-frontend    "/docker-entrypoint.…"   frontend    About a minute ago   Up About a minute (healthy)            0.0.0.0:3000->80/tcp, [::]:3000->80/tcp
stellasentinal-ml-worker   anomalydetection-ml-worker   "python -m device_an…"   ml-worker   About a minute ago   Up About a minute
stellasentinal-ollama      ollama/ollama:latest         "/bin/ollama serve"      ollama      About a minute ago   Up About a minute (health: starting)   0.0.0.0:11434->11434/tcp, [::]:11434->11434/tcp
stellasentinal-postgres    postgres:16-alpine           "docker-entrypoint.s…"   postgres    About a minute ago   Up About a minute (healthy)            0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp
stellasentinal-qdrant      qdrant/qdrant:latest         "./entrypoint.sh"        qdrant      About a minute ago   Up About a minute (healthy)            0.0.0.0:6333-6334->6333-6334/tcp, [::]:6333-6334->6333-6334/tcp
stellasentinal-redis       redis:7-alpine               "docker-entrypoint.s…"   redis       About a minute ago   Up About a minute (healthy)            0.0.0.0:6379->6379/tcp, [::]:6379->6379/tcp
stellasentinal-scheduler   anomalydetection-scheduler   "python -m device_an…"   scheduler   About a minute ago   Up About a minute
```

## Notes / warnings observed
- Multiple runtime/test warnings from pandas and pydantic deprecations (no functional failures).
- SQL Server data loader logged missing table `cs_AppUsage` and fell back to basic columns, as designed.
- Ollama container health is still “starting” shortly after restart; it typically stabilizes after it finishes model initialization.

## How to validate
1) `./.venv/bin/python scripts/verify.py`
2) `python3 -m py_compile $(rg --files -g "*.py")`
3) `docker compose build --no-cache`
4) `docker compose up -d --force-recreate`
5) `docker compose ps`
6) `curl -sS http://127.0.0.1:8000/health`
7) `curl -sS http://127.0.0.1:8000/health/ready`
8) `curl -sS http://127.0.0.1:8000/api/dashboard/isolation-forest/stats`
9) `PYTHONPATH=src ./.venv/bin/python -m device_anomaly.cli.ingestion --dry-run --xsight-table cs_DataUsageByHour --mc-table DeviceStatInt`
