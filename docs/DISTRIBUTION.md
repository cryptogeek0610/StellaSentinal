# Distribution Package Guide

This guide explains how to create and deploy distribution packages for easy installation on different machines.

## Building a Distribution Package

To create a distribution package, run:

```bash
make dist
```

Or directly:

```bash
./scripts/build_dist.sh
```

This will:

1. Build the frontend (React application)
2. Package all necessary files into a timestamped archive
3. Create a deployment-ready package in the `dist/` folder

The output will be a file like: `dist/anomaly-detection-YYYYMMDD-HHMMSS.tar.gz`

## Package Contents

Each distribution package includes:

- ✅ **Pre-built frontend** - Optimized React application ready to serve
- ✅ **Backend source code** - Complete Python application
- ✅ **Docker configuration** - docker-compose.yml and all Dockerfiles
- ✅ **Configuration templates** - env.template for easy setup
- ✅ **Deployment scripts** - start.sh and stop.sh for convenience
- ✅ **Documentation** - DEPLOYMENT.md with detailed instructions
- ✅ **Database scripts** - Initialization SQL scripts

## Deploying to Another Machine

### Step 1: Transfer the Package

Copy the `.tar.gz` file to your target machine:

```bash
# On source machine
scp dist/anomaly-detection-*.tar.gz user@target-machine:/path/to/deployment/

# Or use any file transfer method (USB, network share, etc.)
```

### Step 2: Extract and Setup

On the target machine:

```bash
cd /path/to/deployment
tar -xzf anomaly-detection-*.tar.gz
cd anomaly-detection-*
```

### Step 3: Configure Environment

```bash
cp env.template .env
# Edit .env with your settings
nano .env  # or use your preferred editor
```

**Important configuration items:**
- Database connection strings
- API credentials
- LLM service URLs
- Port numbers (if different from defaults)

### Step 4: Start the Application

Use the provided start script:

```bash
./start.sh
```

Or manually:

```bash
docker-compose up -d
```

### Step 5: Verify

Check that all services are running:

```bash
docker-compose ps
```

Access the application:
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Quick Start Scripts

The distribution package includes two convenience scripts:

### start.sh

Starts all services using Docker Compose. Checks for `.env` file and prompts if missing.

### stop.sh

Stops all running services gracefully.

## Updating an Existing Deployment

1. **Stop the current deployment**
   ```bash
   ./stop.sh
   # or
   docker-compose down
   ```

2. **Backup data** (optional but recommended)
   ```bash
   # Backup database volumes
   docker run --rm -v anomaly-detection_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .
   ```

3. **Extract new package** in a new directory

4. **Copy your existing .env** to the new package directory

5. **Start the new version**
   ```bash
   ./start.sh
   ```

## Requirements

The target machine must have:

- **Docker** (version 20.10+)
- **Docker Compose** (version 2.0+)
- **At least 4GB RAM** available for Docker
- **Available ports:**
  - 3000 (frontend)
  - 8000 (API)
  - 5432 (PostgreSQL)
  - 11434 (Ollama)
  - 6333, 6334 (Qdrant)

## Troubleshooting

### Services won't start

Check logs:
```bash
docker-compose logs -f [service-name]
```

### Port conflicts

Edit `docker-compose.yml` or set environment variables to use different ports:
```bash
FRONTEND_PORT=3001 API_PORT=8001 docker-compose up -d
```

### Database connection issues

Verify database credentials in `.env` file and ensure database is accessible from Docker containers.

### Frontend not loading

Check if frontend container is running:
```bash
docker-compose ps frontend
docker-compose logs frontend
```

## Cleanup

To remove old distribution packages:

```bash
make dist-clean
```

Or manually:
```bash
rm -rf dist/anomaly-detection-*
rm -f dist/*.tar.gz
```

## Notes

- Distribution packages are **git-ignored** and not committed to the repository
- Each package is **self-contained** and ready for deployment
- Database volumes are **managed by Docker** on the target machine
- The package includes production-ready frontend builds (optimized and minified)

