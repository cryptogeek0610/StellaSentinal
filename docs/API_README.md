# SOTI Stella Sentinel API

FastAPI backend for the SOTI Stella Sentinel dashboard.

## Running the API

### Development

```bash
# Install dependencies
pip install -e .

# Run with uvicorn
uvicorn device_anomaly.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

API documentation (Swagger UI) will be available at `http://localhost:8000/docs`.

## API Endpoints

### Dashboard

- `GET /api/dashboard/stats` - Get dashboard statistics (KPIs)
- `GET /api/dashboard/trends?days=7` - Get anomaly trends over time

### Anomalies

- `GET /api/anomalies` - List anomalies with filtering and pagination
  - Query parameters:
    - `device_id` (optional): Filter by device ID
    - `start_date` (optional): Filter by start date (ISO format)
    - `end_date` (optional): Filter by end date (ISO format)
    - `status` (optional): Filter by status (open, investigating, resolved, false_positive)
    - `min_score` (optional): Filter by minimum anomaly score
    - `max_score` (optional): Filter by maximum anomaly score
    - `page` (default: 1): Page number
    - `page_size` (default: 50): Items per page

- `GET /api/anomalies/{id}` - Get detailed information about a specific anomaly

- `POST /api/anomalies/{id}/resolve` - Mark an anomaly as resolved
  - Body: `{ "status": "resolved", "notes": "optional notes" }`

- `POST /api/anomalies/{id}/notes` - Add an investigation note
  - Body: `{ "note": "note text", "action_type": "optional" }`

### Devices

- `GET /api/devices/{id}` - Get device details and recent anomalies

## Database

The API uses SQLite by default (for development) to store anomaly results. The database file is created at `anomaly_results.db` in the project root.

To use a different database, set the `RESULTS_DB_URL` environment variable:

```bash
export RESULTS_DB_URL="postgresql://user:password@localhost/anomaly_db"
```

## Persisting Anomaly Results

To persist anomaly detection results to the database, use the `persist_to_db=True` parameter when running experiments:

```python
from device_anomaly.cli.synthetic_experiment import run_synthetic_experiment

run_synthetic_experiment(persist_to_db=True)
```

Or for DW experiments:

```python
from device_anomaly.cli.dw_experiment import run_dw_experiment

run_dw_experiment(
    start_date="2025-01-01",
    end_date="2025-01-31",
    persist_to_db=True
)
```

