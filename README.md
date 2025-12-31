# 🛡️ StellaSentinal

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)

> **AI-powered anomaly detection for enterprise device fleets** — Detect, explain, and investigate device behavioral anomalies using machine learning and natural language processing.

---

## ✨ Features

- **🔍 Multi-Model Anomaly Detection** — Hybrid approach combining Isolation Forest, statistical baselines, and heuristic rules
- **🤖 LLM-Powered Explanations** — Natural language explanations of detected anomalies using Azure OpenAI or local LLMs
- **📊 Real-time Dashboard** — Interactive React frontend with fleet overview, device drill-downs, and investigation workflows
- **🔄 Pattern Recognition** — Automatic detection of recurring anomaly patterns across device fleets
- **📈 Drift Monitoring** — Track model performance and data drift over time
- **🐳 Docker-Ready** — Full containerized deployment with Docker Compose

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           StellaSentinal                          │
├────────────────┬───────────────────────────────┬────────────────────────┤
│   Data Layer   │       Processing Layer        │   Presentation Layer   │
├────────────────┼───────────────────────────────┼────────────────────────┤
│ • XSight DB    │ • Feature Engineering         │ • React Dashboard      │
│ • MobiControl  │ • Isolation Forest            │ • FastAPI REST API     │
│ • PostgreSQL   │ • Baseline Comparisons        │ • Streamlit Dashboard   │
│                │ • LLM Explainer               │                        │
└────────────────┴───────────────────────────────┴────────────────────────┘
```

### Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Python 3.10+, FastAPI, SQLAlchemy, scikit-learn, pandas |
| **Frontend** | React 18, TypeScript, Tailwind CSS, Vite |
| **Database** | PostgreSQL (app data), SQL Server (XSight/MobiControl) |
| **ML/AI** | Isolation Forest, Azure OpenAI / local LLMs |
| **DevOps** | Docker, Docker Compose, Make |

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose)
- Make (optional, for convenience commands)

### 1. Clone & Configure

```bash
git clone <repository-url>
cd AnomalyDetection

# Copy environment template
cp env.template .env
```

Edit `.env` with your configuration (database credentials, LLM settings).

### 2. Start Services

```bash
# Using Make
make up

# Or using Docker Compose directly
docker-compose up -d
```

### 3. Verify Installation

```bash
# Run smoke tests
make test-synthetic

# Or manually
./scripts/smoke_test.sh
```

### 4. Access the Application

| Service | URL |
|---------|-----|
| **Frontend Dashboard** | http://localhost:3000 (or `FRONTEND_PORT`) |
| **API Documentation** | http://localhost:8000/docs |
| **Streamlit Dashboard** | http://localhost:8501 |

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API_README.md) | REST API endpoints and usage |
| [Architecture Analysis](docs/ARCHITECTURE_ANALYSIS.md) | System design and data flow |
| [Isolation Forest Guide](docs/Isolation_Forest_Guide.md) | ML model details and tuning |
| [SOTI Integration](docs/SOTI_API_Integration_Guide.md) | Connecting to SOTI data sources |
| [Data Points Catalog](docs/SOTI_DataPoints_Catalog.md) | Available telemetry metrics |

---

## 🛠️ Development

### Project Structure

```
├── src/device_anomaly/          # Python backend
│   ├── api/                     # FastAPI routes
│   ├── config/                  # Configuration management
│   ├── data_access/             # Database loaders
│   ├── features/                # Feature engineering
│   ├── llm/                     # LLM integration
│   ├── models/                  # ML models
│   └── ui/                      # Streamlit dashboard
├── frontend/                    # React frontend
│   ├── src/components/          # React components
│   └── src/pages/               # Page components
├── configs/                     # Experiment configurations
├── scripts/                     # Utility scripts
└── tests/                       # Test suite
```

### Available Commands

```bash
make help              # Show all available commands
make up                # Start all services
make down              # Stop all services
make restart           # Restart all services
make logs              # View logs from all services
make shell             # Open shell in app container
make test-synthetic    # Run synthetic data experiment
make test-dw           # Run data warehouse experiment
make init-db           # Initialize database
make clean             # Remove all containers and volumes
```

### Running Experiments

```bash
# Synthetic experiment (no external DB required)
PYTHONPATH=src python -m device_anomaly.cli.synthetic_experiment \
  --config configs/synthetic.yaml

# Data warehouse experiment (requires XSight DB connection)
PYTHONPATH=src python -m device_anomaly.cli.dw_experiment \
  --config configs/dw.yaml
```

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Start the API server
uvicorn device_anomaly.api.main:app --reload

# In another terminal, start the frontend
cd frontend && npm install && npm run dev
```

### Docker Frontend Dev (hot reload)

```bash
docker-compose --profile frontend-dev up -d frontend-dev
```

Open http://localhost:5173 (or `FRONTEND_DEV_PORT`). If port 3000 is already in use, set `FRONTEND_PORT` in `.env` for the nginx frontend container.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=src/device_anomaly --cov-report=html

# Run specific test file
pytest tests/test_anomaly_detection.py -v
```

---

## 🔧 Configuration

### Environment Variables

See [`env.template`](env.template) for all available configuration options:

| Variable | Description | Required |
|----------|-------------|----------|
| `BACKEND_DB_*` | PostgreSQL connection for app data | Yes |
| `DW_DB_*` | XSight SQL Server connection | For DW experiments |
| `MC_DB_*` | MobiControl SQL Server connection | Optional |
| `LLM_*` | LLM provider configuration | For explanations |

### Experiment Configuration

Experiments are configured via YAML files in `configs/`:

```yaml
# configs/synthetic.yaml
data:
  source: synthetic
  n_devices: 1000
  anomaly_fraction: 0.05

model:
  contamination: 0.05
  n_estimators: 100
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on:

- Setting up your development environment
- Coding standards
- Submitting pull requests

---

## 🔒 Security

For security concerns, please review our [Security Policy](docs/SECURITY.md). Do not open public issues for security vulnerabilities.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for Isolation Forest implementation
- [FastAPI](https://fastapi.tiangolo.com/) for the modern API framework
- [Streamlit](https://streamlit.io/) for rapid dashboard prototyping
