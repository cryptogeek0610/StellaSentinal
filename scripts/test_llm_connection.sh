#!/bin/bash
# Test script to verify local LLM connection from Docker container

set -e

LLM_BASE_URL="${LLM_BASE_URL:-http://host.docker.internal:1234}"

echo "=========================================="
echo "Testing Local LLM Connection"
echo "=========================================="
echo ""
echo "LLM Base URL: $LLM_BASE_URL"
echo ""

# Test 1: Check if container can reach host
echo "1. Testing host connectivity..."
if docker-compose exec -T app ping -c 1 host.docker.internal &>/dev/null 2>&1 || \
   docker-compose exec -T app ping -c 1 192.168.3.82 &>/dev/null 2>&1; then
    echo "   ✓ Host is reachable"
else
    echo "   ✗ Cannot reach host (this is normal if ping is disabled)"
fi

# Test 2: Check LLM server endpoint
echo ""
echo "2. Testing LLM server endpoint..."
if response=$(docker-compose exec -T app curl -s -f --max-time 5 "$LLM_BASE_URL/health" 2>&1); then
    echo "   ✓ LLM server is reachable"
    echo "   Response: $response"
elif response=$(docker-compose exec -T app curl -s -f --max-time 5 "$LLM_BASE_URL/v1/models" 2>&1); then
    echo "   ✓ LLM server is reachable (via /v1/models)"
    echo "   Response: $response"
else
    echo "   ✗ Cannot reach LLM server at $LLM_BASE_URL"
    echo "   Error: $response"
    echo ""
    echo "   Troubleshooting:"
    echo "   1. Verify LLM server is running on host: curl http://localhost:1234/health"
    echo "   2. Check LLM_BASE_URL in .env file"
    echo "   3. For Linux, try using host IP: LLM_BASE_URL=http://192.168.3.82:1234"
    exit 1
fi

# Test 3: Test from Python (if LLM client is implemented)
echo ""
echo "3. Testing Python LLM client..."
if docker-compose exec -T app python -c "
import os
from device_anomaly.config.settings import get_settings
settings = get_settings()
print(f'LLM Base URL: {settings.llm.base_url}')
print(f'LLM Model: {settings.llm.model_name}')
if settings.llm.base_url:
    print('✓ LLM settings loaded correctly')
else:
    print('⚠ LLM settings not configured')
" 2>&1; then
    echo "   ✓ Python can access LLM configuration"
else
    echo "   ⚠ Could not test Python LLM client"
fi

echo ""
echo "=========================================="
echo "LLM Connection Test Complete"
echo "=========================================="

