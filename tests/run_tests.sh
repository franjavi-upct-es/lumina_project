#!/bin/bash

# =============================================================================
# LUMINA PROJECT - Test Runner Script
# =============================================================================
# Usage:
#   ./tests/run_tests.sh              # Run all tests (except slow)
#   ./tests/run_tests.sh --all        # Run all tests including slow
#   ./tests/run_tests.sh --services   # Test only Docker services
#   ./tests/run_tests.sh --api        # Test only API endpoints
#   ./tests/run_tests.sh --celery     # Test only Celery tasks
#   ./tests/run_tests.sh --quick      # Quick smoke tests
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} LUMINA QUANT LAB - Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker services are running
check_docker_services() {
    echo -e "${YELLOW}Checking Docker services...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not installed!${NC}"
        return 1
    fi
    
    # Check specific containers
    local services=("lumina-timescaledb" "lumina-redis" "lumina-api")
    local all_running=true
    
    for service in "${services[@]}"; do
        if docker ps --format '{{.Names}}' | grep -q "^${service}$"; then
            echo -e "  ${GREEN}✓${NC} $service is running"
        else
            echo -e "  ${RED}✗${NC} $service is NOT running"
            all_running=false
        fi
    done
    
    if [ "$all_running" = false ]; then
        echo ""
        echo -e "${YELLOW}Start services with:${NC}"
        echo "  cd docker && docker-compose up -d"
        return 1
    fi
    
    return 0
}

# Run quick smoke tests
run_quick_tests() {
    echo -e "\n${YELLOW}Running quick smoke tests...${NC}\n"
    
    python -m pytest "$TESTS_DIR/test_docker_services.py::TestRedisService::test_redis_connection" \
                     "$TESTS_DIR/test_docker_services.py::TestTimescaleDBService::test_postgres_connection" \
                     "$TESTS_DIR/test_docker_services.py::TestFastAPIService::test_api_health_endpoint" \
                     -v --tb=short 2>&1 || true
}

# Run Docker services tests
run_services_tests() {
    echo -e "\n${YELLOW}Running Docker services tests...${NC}\n"
    
    python -m pytest "$TESTS_DIR/test_docker_services.py" -v --tb=short 2>&1
}

# Run API endpoint tests
run_api_tests() {
    echo -e "\n${YELLOW}Running API endpoint tests...${NC}\n"
    
    python -m pytest "$TESTS_DIR/test_api_endpoints.py" -v --tb=short 2>&1
}

# Run Celery task tests
run_celery_tests() {
    echo -e "\n${YELLOW}Running Celery task tests...${NC}\n"
    
    python -m pytest "$TESTS_DIR/test_celery_tasks.py" -v --tb=short 2>&1
}

# Run data collection tests
run_data_tests() {
    echo -e "\n${YELLOW}Running data collection tests...${NC}\n"
    
    python -m pytest "$TESTS_DIR/test_data_collection.py" -v --tb=short 2>&1
}

# Run all tests
run_all_tests() {
    local include_slow=$1
    
    echo -e "\n${YELLOW}Running all tests...${NC}\n"
    
    if [ "$include_slow" = "true" ]; then
        python -m pytest "$TESTS_DIR" -v --tb=short -m "slow or not slow" 2>&1
    else
        python -m pytest "$TESTS_DIR" -v --tb=short 2>&1
    fi
}

# Run integration tests (slow)
run_integration_tests() {
    echo -e "\n${YELLOW}Running integration tests (slow)...${NC}\n"
    
    python -m pytest "$TESTS_DIR/test_integration_full.py" -v -s --tb=short -m slow 2>&1
}

# Print usage
print_usage() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  --all         Run all tests including slow integration tests"
    echo "  --quick       Quick smoke tests (services health)"
    echo "  --services    Test Docker services only"
    echo "  --api         Test API endpoints only"
    echo "  --celery      Test Celery tasks only"
    echo "  --data        Test data collection only"
    echo "  --integration Run full integration tests (slow)"
    echo "  --help        Show this help message"
    echo ""
    echo "Without options, runs all tests except slow integration tests."
}

# Main
main() {
    cd "$PROJECT_DIR"
    
    # Parse arguments
    case "${1:-}" in
        --help|-h)
            print_usage
            exit 0
            ;;
        --quick)
            check_docker_services || exit 1
            run_quick_tests
            ;;
        --services)
            check_docker_services || exit 1
            run_services_tests
            ;;
        --api)
            check_docker_services || exit 1
            run_api_tests
            ;;
        --celery)
            check_docker_services || exit 1
            run_celery_tests
            ;;
        --data)
            run_data_tests
            ;;
        --integration)
            check_docker_services || exit 1
            run_integration_tests
            ;;
        --all)
            check_docker_services || exit 1
            run_all_tests "true"
            ;;
        "")
            check_docker_services || exit 1
            run_all_tests "false"
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} Tests completed!${NC}"
    echo -e "${GREEN}========================================${NC}"
}

main "$@"
