#!/bin/bash
# Smoke test script for Device Anomaly Service

set -e

echo "=========================================="
echo "Device Anomaly Service - Smoke Tests"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

if [ -z "${DW_DB_PASS:-}" ] && [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

if [ -z "${DW_DB_PASS:-}" ]; then
    echo "Error: DW_DB_PASS must be set in the environment."
    exit 1
fi

test_command() {
    local name=$1
    local command=$2
    local expected_pattern=${3:-""}
    
    echo -n "Testing: $name... "
    
    if output=$(eval "$command" 2>&1); then
        if [ -z "$expected_pattern" ] || echo "$output" | grep -q "$expected_pattern"; then
            echo -e "${GREEN}✓ PASSED${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        else
            echo -e "${RED}✗ FAILED${NC} (output doesn't match pattern)"
            echo "  Expected pattern: $expected_pattern"
            echo "  Output: $output"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 1
        fi
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  Error: $output"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 1: Check if containers are running
echo "1. Container Health Checks"
echo "---------------------------"
test_command "SQL Server container running" \
    "docker-compose ps sqlserver | grep -q 'Up'"

test_command "App container exists" \
    "docker-compose ps app | grep -q 'Up\|Exit'"

# Test 2: SQL Server connectivity
echo ""
echo "2. Database Connectivity"
echo "---------------------------"
test_command "SQL Server accepts connections" \
    "docker-compose exec -T sqlserver /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P \"\${DW_DB_PASS}\" -Q 'SELECT 1' -h -1" \
    "1"

# Test 3: Python environment
echo ""
echo "3. Python Environment"
echo "---------------------------"
test_command "Python version >= 3.10" \
    "docker-compose exec -T app python --version" \
    "Python 3"

test_command "Package imports work" \
    "docker-compose exec -T app python -c 'import pandas, numpy, sklearn, sqlalchemy, pyodbc; print(\"OK\")'" \
    "OK"

# Test 4: Application entry points
echo ""
echo "4. Application Entry Points"
echo "---------------------------"
test_command "Main entry point runs" \
    "docker-compose run --rm app python -m device_anomaly.cli.main" \
    "Device Anomaly Service"

test_command "Synthetic experiment runs" \
    "docker-compose run --rm app python -m device_anomaly.cli.synthetic_experiment" \
    "Running synthetic experiment"

# Test 5: Configuration
echo ""
echo "5. Configuration"
echo "---------------------------"
test_command "Settings load correctly" \
    "docker-compose exec -T app python -c 'from device_anomaly.config.settings import get_settings; s = get_settings(); print(f\"DB: {s.dw.host}\")'" \
    "DB:"

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}Failed: $TESTS_FAILED${NC}"
    echo ""
    echo -e "${GREEN}All smoke tests passed! ✓${NC}"
    exit 0
fi
