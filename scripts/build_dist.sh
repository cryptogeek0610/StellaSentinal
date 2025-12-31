#!/bin/bash
# Build distribution package for easy deployment

set -e

DIST_DIR="dist"
VERSION=$(date +%Y%m%d-%H%M%S)
PACKAGE_NAME="anomaly-detection-${VERSION}"

echo "Building distribution package: ${PACKAGE_NAME}"

# Clean previous build packages (keep README and .gitkeep)
mkdir -p "${DIST_DIR}"
rm -f "${DIST_DIR}"/*.tar.gz
rm -rf "${DIST_DIR}"/anomaly-detection-*
mkdir -p "${DIST_DIR}/${PACKAGE_NAME}"

echo "1. Building frontend..."
cd frontend
npm ci
npm run build
cd ..

echo "2. Copying frontend build and files..."
mkdir -p "${DIST_DIR}/${PACKAGE_NAME}/frontend"
cp -r frontend/dist "${DIST_DIR}/${PACKAGE_NAME}/frontend/"
cp frontend/Dockerfile.prod "${DIST_DIR}/${PACKAGE_NAME}/frontend/Dockerfile"
cp frontend/nginx.conf "${DIST_DIR}/${PACKAGE_NAME}/frontend/"

echo "3. Copying backend source..."
mkdir -p "${DIST_DIR}/${PACKAGE_NAME}/src"
cp -r src "${DIST_DIR}/${PACKAGE_NAME}/"

echo "4. Copying configuration files..."
cp pyproject.toml "${DIST_DIR}/${PACKAGE_NAME}/"
cp env.template "${DIST_DIR}/${PACKAGE_NAME}/"
cp Dockerfile "${DIST_DIR}/${PACKAGE_NAME}/"
cp docker-compose.yml "${DIST_DIR}/${PACKAGE_NAME}/"

echo "5. Copying scripts..."
mkdir -p "${DIST_DIR}/${PACKAGE_NAME}/scripts"
cp scripts/*.sh "${DIST_DIR}/${PACKAGE_NAME}/scripts/" 2>/dev/null || true
cp scripts/*.sql "${DIST_DIR}/${PACKAGE_NAME}/scripts/" 2>/dev/null || true
chmod +x "${DIST_DIR}/${PACKAGE_NAME}/scripts/"*.sh 2>/dev/null || true

echo "6. Copying configs..."
cp -r configs "${DIST_DIR}/${PACKAGE_NAME}/" 2>/dev/null || true

echo "7. Creating deployment README..."
cat > "${DIST_DIR}/${PACKAGE_NAME}/DEPLOYMENT.md" << 'EOF'
# Anomaly Detection Application - Deployment Guide

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for Docker
- Ports available: 3000 (frontend), 8000 (API), 5432 (PostgreSQL), 11434 (Ollama), 6333 (Qdrant)

### Deployment Steps

1. **Extract the package**
   ```bash
   cd /path/to/deployment
   tar -xzf anomaly-detection-*.tar.gz
   cd anomaly-detection-*
   ```

2. **Configure environment**
   ```bash
   cp env.template .env
   # Edit .env with your configuration
   nano .env
   ```

3. **Start the application**
   ```bash
   docker-compose up -d
   ```

4. **Verify services**
   ```bash
   docker-compose ps
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - API Health: http://localhost:8000/health

### Configuration

Edit `.env` file with your settings:

- **Backend Database**: PostgreSQL connection details
- **XSight Database**: SQL Server connection for telemetry data
- **MobiControl API**: OAuth credentials for device management
- **LLM Configuration**: Ollama or other LLM service settings

### Services

The application includes the following services:

- **frontend**: React web application (port 3000)
- **app**: FastAPI backend (port 8000)
- **postgres**: PostgreSQL database (port 5432)
- **ollama**: Local LLM service (port 11434)
- **qdrant**: Vector database for RAG (ports 6333, 6334)

### Troubleshooting

1. **Check service logs**
   ```bash
   docker-compose logs -f [service-name]
   ```

2. **Restart services**
   ```bash
   docker-compose restart
   ```

3. **Rebuild images**
   ```bash
   docker-compose build --no-cache
   docker-compose up -d
   ```

### Updating

To update to a new version:

1. Stop the current deployment
   ```bash
   docker-compose down
   ```

2. Extract new package and follow deployment steps

3. Migrate data if needed (database volumes are preserved)

### Data Persistence

Data is stored in Docker volumes:
- `postgres_data`: Database data
- `ollama_data`: LLM models
- `qdrant_data`: Vector database data

To backup:
```bash
docker run --rm -v anomaly-detection_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .
```

To restore:
```bash
docker run --rm -v anomaly-detection_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /data
```

### Support

For issues or questions, refer to:
- README.md (main documentation)
- docs/API_README.md (API documentation)
- docs/ARCHITECTURE_ANALYSIS.md (system architecture)

EOF

echo "8. Creating quick start script..."
cat > "${DIST_DIR}/${PACKAGE_NAME}/start.sh" << 'EOF'
#!/bin/bash
# Quick start script

if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp env.template .env
    echo "Please edit .env with your configuration before starting."
    exit 1
fi

echo "Starting Anomaly Detection Application..."
docker-compose up -d

echo "Waiting for services to be ready..."
sleep 10

echo ""
echo "Application started!"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
EOF
chmod +x "${DIST_DIR}/${PACKAGE_NAME}/start.sh"

echo "9. Creating stop script..."
cat > "${DIST_DIR}/${PACKAGE_NAME}/stop.sh" << 'EOF'
#!/bin/bash
# Stop script

echo "Stopping Anomaly Detection Application..."
docker-compose down
echo "Application stopped."
EOF
chmod +x "${DIST_DIR}/${PACKAGE_NAME}/stop.sh"

echo "10. Creating package archive..."
cd "${DIST_DIR}"
tar -czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"
cd ..

echo ""
echo "✅ Distribution package created successfully!"
echo "Location: ${DIST_DIR}/${PACKAGE_NAME}.tar.gz"
echo ""
echo "To deploy:"
echo "  1. Copy ${PACKAGE_NAME}.tar.gz to target machine"
echo "  2. Extract: tar -xzf ${PACKAGE_NAME}.tar.gz"
echo "  3. Follow DEPLOYMENT.md instructions"

