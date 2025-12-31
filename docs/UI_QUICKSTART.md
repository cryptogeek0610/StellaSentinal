# UI Quick Start Guide

This guide will help you get the anomaly detection dashboard up and running.

## Prerequisites

- Python 3.10+ with pip
- Node.js 18+ and npm
- (Optional) Docker and Docker Compose for database

## Setup Steps

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -e .

# The API will use SQLite by default (anomaly_results.db)
# No additional database setup needed for development
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

### 3. Run the Application

#### Terminal 1: Start the API Server

```bash
uvicorn device_anomaly.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
API docs: `http://localhost:8000/docs`

#### Terminal 2: Start the Frontend

```bash
cd frontend
npm run dev
```

The UI will be available at `http://localhost:3000`

### 4. Populate Data (Optional)

To see data in the dashboard, you need to run anomaly detection and persist results:

```bash
# Run synthetic experiment and persist results
python -m device_anomaly.cli.synthetic_experiment --persist-to-db

# Or run DW experiment (requires database connection)
python -m device_anomaly.cli.dw_experiment --persist-to-db
```

Note: The CLI scripts need to be updated to accept `--persist-to-db` flag, or you can modify them directly to set `persist_to_db=True`.

## Project Structure

```
AnomalyDetection/
├── src/device_anomaly/
│   ├── api/              # FastAPI backend
│   │   ├── main.py       # FastAPI app
│   │   ├── models.py     # Pydantic models
│   │   └── routes/       # API routes
│   ├── database/         # Database schema and connection
│   ├── data_access/      # Data loading and persistence
│   └── ...
├── frontend/             # React frontend
│   ├── src/
│   │   ├── pages/        # Page components
│   │   ├── api/          # API client
│   │   └── ...
│   └── package.json
└── ...
```

## Features Implemented

### Phase 1: Core UI (MVP) ✅

1. ✅ Database schema for storing anomaly results
2. ✅ FastAPI backend with REST endpoints
3. ✅ Anomaly persistence from detection pipeline
4. ✅ React frontend with TypeScript
5. ✅ Dashboard page with KPI cards and trend chart
6. ✅ Anomaly List page with filters and pagination
7. ✅ Anomaly Detail page with charts and investigation notes

## Next Steps (Future Phases)

- Phase 2: Enhanced features (export, advanced filters, authentication)
- Phase 3: AI integration (LLM explanations)
- Phase 4: Real-time updates (WebSocket), alerting, advanced analytics

## Troubleshooting

### API not responding
- Check that uvicorn is running on port 8000
- Check CORS settings in `api/main.py` if accessing from different origin

### Frontend can't connect to API
- Verify API is running at `http://localhost:8000`
- Check proxy settings in `frontend/vite.config.ts`

### No data in dashboard
- Run anomaly detection experiments with `persist_to_db=True`
- Check that `anomaly_results.db` file exists and has data

### Database errors
- SQLite database is created automatically
- For production, set `RESULTS_DB_URL` environment variable to use PostgreSQL/MySQL

