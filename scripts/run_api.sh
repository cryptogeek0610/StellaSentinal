#!/bin/bash
# Script to run the FastAPI server

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the API server
uvicorn device_anomaly.api.main:app --reload --port 8000 --host 0.0.0.0

