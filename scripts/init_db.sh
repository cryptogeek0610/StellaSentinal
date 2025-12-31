#!/bin/bash
# Database initialization helper script
# This script helps create the database and provides a template for schema setup

set -e

if [ -z "${DW_DB_USER:-}" ] && [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

DB_NAME="${DW_DB_NAME:-SOTI_XSight_dw}"
DB_USER="${DW_DB_USER:-}"
DB_PASS="${DW_DB_PASS:-}"

if [ -z "$DB_USER" ] || [ -z "$DB_PASS" ]; then
    echo "Error: DW_DB_USER and DW_DB_PASS must be set in the environment."
    exit 1
fi

echo "=========================================="
echo "Database Initialization Helper"
echo "=========================================="
echo ""
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo ""

# Check if database already exists
echo "Checking if database exists..."
if docker-compose exec -T sqlserver /opt/mssql-tools/bin/sqlcmd \
    -S localhost -U "$DB_USER" -P "$DB_PASS" \
    -Q "SELECT name FROM sys.databases WHERE name = '$DB_NAME'" \
    -h -1 | grep -q "$DB_NAME"; then
    echo "✓ Database '$DB_NAME' already exists"
else
    echo "Creating database '$DB_NAME'..."
    docker-compose exec -T sqlserver /opt/mssql-tools/bin/sqlcmd \
        -S localhost -U "$DB_USER" -P "$DB_PASS" \
        -Q "CREATE DATABASE [$DB_NAME];"
    echo "✓ Database created"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Connect to the database:"
echo "   make shell-db"
echo "   # Then: USE SOTI_XSight_dw; GO"
echo ""
echo "2. Create your tables based on the schema expected by:"
echo "   - src/device_anomaly/data_access/dw_loader.py"
echo "   - src/device_anomaly/data_access/persistence.py"
echo ""
echo "3. Expected DW tables (from dw_loader.py):"
echo "   - Device"
echo "   - Model"
echo "   - cs_BatteryStat"
echo "   - cs_AppUsage"
echo "   - cs_DataUsage"
echo "   - cs_BatteryAppDrain"
echo "   - cs_Heatmap"
echo ""
echo "4. Expected anomaly result tables (from persistence.py):"
echo "   - dbo.ml_AnomalyResults"
echo "   - dbo.ml_AnomalyEvents"
echo "   - dbo.ml_DeviceAnomalyPatterns"
echo ""
echo "5. Seed with sample data to test the pipeline"
echo ""
echo "=========================================="
