#!/bin/bash

# Lumina Project - Configuration Validation Script
# Este script verifica que toda la configuración de Docker está correcta

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
WARN=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Lumina Project Configuration Validator${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to print test result
test_result() {
    local test_name=$1
    local result=$2
    local error_msg=$3
    
    if [ $result -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((PASS++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        if [ -n "$error_msg" ]; then
            echo -e "  ${RED}→ $error_msg${NC}"
        fi
        ((FAIL++))
    fi
}

warn_result() {
    local test_name=$1
    echo -e "${YELLOW}⚠ WARN${NC}: $test_name"
    ((WARN++))
}

# ==========================================
# Check 1: .env file exists
# ==========================================
echo -e "${BLUE}[1/10]${NC} Checking .env file..."
if [ -f ./backend/.env ]; then
    test_result ".env file exists" 0
else
    test_result ".env file exists" 1 ".env not found - run: cp .env.example .env"
fi
echo ""

# ==========================================
# Check 2: Docker is running
# ==========================================
echo -e "${BLUE}[2/10]${NC} Checking Docker..."
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        test_result "Docker daemon running" 0
    else
        test_result "Docker daemon running" 1 "Docker daemon not accessible"
    fi
else
    test_result "Docker installed" 1 "Docker not found in PATH"
fi
echo ""

# ==========================================
# Check 3: Docker Compose is available
# ==========================================
echo -e "${BLUE}[3/10]${NC} Checking Docker Compose..."
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    test_result "Docker Compose available" 0
else
    test_result "Docker Compose available" 1 "docker-compose or docker compose not found"
fi
echo ""

# ==========================================
# Check 4: Required files exist
# ==========================================
echo -e "${BLUE}[4/10]${NC} Checking required files..."
REQUIRED_FILES=(
    "docker/docker-compose.yml"
    "docker/Dockerfile.api"
    "docker/Dockerfile.mlflow"
    "backend/config/settings.py"
    "backend/workers/celery_app.py"
    "backend/db/00_create_mlflow_db.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (MISSING)"
        ((FAIL++))
    fi
done
echo ""

# ==========================================
# Check 5: Docker-compose syntax
# ==========================================
echo -e "${BLUE}[5/10]${NC} Validating docker-compose.yml..."
if docker-compose config &> /dev/null || docker compose config &> /dev/null; then
    test_result "docker-compose.yml syntax" 0
else
    test_result "docker-compose.yml syntax" 1 "Invalid YAML syntax"
fi
echo ""

# ==========================================
# Check 6: MLflow configuration in compose
# ==========================================
echo -e "${BLUE}[6/10]${NC} Checking MLflow configuration..."
if grep -q "MLFLOW_TRACKING_URI: http://mlflow:5000" docker/docker-compose.yml; then
    test_result "MLflow URI in api service" 0
else
    test_result "MLflow URI in api service" 1 "Missing or incorrect MLFLOW_TRACKING_URI"
fi

if grep -q "MLFLOW_TRACKING_URI: http://mlflow:5000" docker/docker-compose.yml | grep -A 20 "celery-worker:"; then
    test_result "MLflow URI in celery-worker service" 0
else
    # Try to find if MLFLOW_TRACKING_URI is in celery-worker
    if docker-compose config 2>/dev/null | grep -A 50 "celery-worker:" | grep -q "MLFLOW_TRACKING_URI"; then
        test_result "MLflow URI in celery-worker service" 0
    else
        echo -e "${YELLOW}⚠ WARN${NC}: Could not fully verify celery-worker MLFLOW_TRACKING_URI"
        ((WARN++))
    fi
fi
echo ""

# ==========================================
# Check 7: Settings.py defaults
# ==========================================
echo -e "${BLUE}[7/10]${NC} Checking settings.py defaults..."
if grep -q 'MLFLOW_TRACKING_URI: str = "http://mlflow:5000' backend/config/settings.py; then
    test_result "MLFLOW_TRACKING_URI default correct" 0
else
    test_result "MLFLOW_TRACKING_URI default correct" 1 "Still using localhost?"
fi

if grep -q 'CELERY_BROKER_URL: str = Field(default="redis://redis:6379' backend/config/settings.py; then
    test_result "CELERY_BROKER_URL default correct" 0
else
    test_result "CELERY_BROKER_URL default correct" 1 "Still using localhost?"
fi

if grep -q 'REDIS_URL: str = Field(default="redis://redis:6379' backend/config/settings.py; then
    test_result "REDIS_URL default correct" 0
else
    test_result "REDIS_URL default correct" 1 "Still using localhost?"
fi
echo ""

# ==========================================
# Check 8: MLflow DB script
# ==========================================
echo -e "${BLUE}[8/10]${NC} Checking MLflow DB initialization script..."
if [ -x "backend/db/00_create_mlflow_db.sh" ]; then
    test_result "MLflow script executable" 0
else
    warn_result "MLflow script not executable (will work in Docker, but consider: chmod +x backend/db/00_create_mlflow_db.sh)"
fi

if grep -q "lumina_mlflow" backend/db/00_create_mlflow_db.sh; then
    test_result "MLflow script creates database" 0
else
    test_result "MLflow script creates database" 1 "Script doesn't create lumina_mlflow database"
fi
echo ""

# ==========================================
# Check 9: Network configuration
# ==========================================
echo -e "${BLUE}[9/10]${NC} Checking Docker network configuration..."
if grep -q "lumina-network:" docker/docker-compose.yml; then
    test_result "Docker network defined" 0
else
    test_result "Docker network defined" 1 "lumina-network not defined"
fi
echo ""

# ==========================================
# Check 10: Celery configuration
# ==========================================
echo -e "${BLUE}[10/10]${NC} Checking Celery configuration..."
if grep -q "settings.CELERY_BROKER_URL" backend/workers/celery_app.py; then
    test_result "Celery reads from settings" 0
else
    test_result "Celery reads from settings" 1 "Celery not using settings for broker URL"
fi
echo ""

# ==========================================
# Summary
# ==========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"
echo -e "${YELLOW}Warnings: $WARN${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Ensure .env is configured with correct values"
    echo "2. Run: docker-compose down -v && docker-compose up --build"
    echo "3. Monitor logs: docker-compose logs -f"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some critical checks failed. Please review above.${NC}"
    echo ""
    exit 1
fi
