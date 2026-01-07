#!/bin/bash
# SQL Server Setup Script
# Starts SQL Server container and restores customer databases from BAK files
#
# Usage: ./scripts/setup-sqlserver.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SQL Server Setup for Multi-Tenant Training ===${NC}"
echo ""

# Load environment variables if .env exists
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Set defaults
SA_PASSWORD="${SQLSERVER_SA_PASSWORD:?SQLSERVER_SA_PASSWORD must be set in .env}"
SQL_PORT="${SQLSERVER_PORT:-1433}"

echo -e "${YELLOW}Step 1: Starting SQL Server container...${NC}"
cd "$PROJECT_DIR"
docker-compose up -d sqlserver

echo ""
echo -e "${YELLOW}Step 2: Waiting for SQL Server to be healthy...${NC}"
echo "This may take 30-60 seconds..."

# Wait for SQL Server to be ready
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if docker exec stellasentinal-sqlserver /opt/mssql-tools18/bin/sqlcmd \
        -S localhost -U sa -P "$SA_PASSWORD" -C -Q "SELECT 1" &>/dev/null; then
        echo -e "${GREEN}SQL Server is ready!${NC}"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "  Waiting... (attempt $ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${RED}Error: SQL Server did not become ready in time${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 3: Restoring databases from BAK files...${NC}"
echo "This may take several minutes for large databases..."
echo ""

# Run the restore script
docker exec stellasentinal-sqlserver /opt/mssql-tools18/bin/sqlcmd \
    -S localhost -U sa -P "$SA_PASSWORD" -C \
    -i /var/opt/mssql/scripts/restore-databases.sql

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Available databases:"
echo "  - XSight_DW      (XSight telemetry data)"
echo ""
echo "Note: MobiControlDB requires SQL Server 2025 (not available on ARM/Apple Silicon)"
echo ""
echo "Connection details:"
echo "  Host: localhost (or sqlserver from Docker network)"
echo "  Port: $SQL_PORT"
echo "  User: sa"
echo "  Pass: (see SQLSERVER_SA_PASSWORD in .env)"
echo ""
echo "To configure runtime, set in .env:"
echo "  DW_DB_HOST=sqlserver"
echo "  DW_DB_NAME=XSight_DW"
echo "  DW_DB_USER=sa"
echo "  DW_DB_PASS=<your_password>"
echo ""
