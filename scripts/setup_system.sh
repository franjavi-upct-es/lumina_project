#!/bin/bash
# scripts/setup_system.sh
# Automated setup script for Lumina Quant Lab v2
# Run with: chmod +x scripts/setup_system.sh && ./scripts/setup_system.sh

set -e # Exit on error

# Get project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_command() {
    if command -v $1 &>/dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Main setup
main() {
    print_header "🚀 Lumina Quant Lab v2 - Automated Setup"

    # Step 1: Check prerequisites
    print_header "Step 1: Checking Prerequisites"

    MISSING_DEPS=0

    if ! check_command "docker"; then
        print_error "Please install Docker: https://docs.docker.com/get-docker/"
        MISSING_DEPS=1
    fi

    if ! check_command "docker-compose"; then
        print_error "Please install Docker Compose: https://docs.docker.com/compose/install/"
        MISSING_DEPS=1
    fi

    if ! check_command "python3"; then
        print_error "Please install Python 3.11+"
        MISSING_DEPS=1
    fi

    check_command "git"
    check_command "curl"
    check_command "jq"

    if [ $MISSING_DEPS -eq 1 ]; then
        print_error "Missing required dependencies. Please install them first."
        exit 1
    fi

    print_success "All prerequisites met!"

    # Step 2: Create directory structure
    print_header "Step 2: Creating Directory Structure"

    mkdir -p models
    mkdir -p data/{parquet,features,raw}
    mkdir -p logs
    mkdir -p notebooks
    mkdir -p tests

    print_success "Directory structure created"

    # Step 3: Create .env file if doesn't exist
    print_header "Step 3: Setting Up Environment Variables"

    if [ ! -f $PROJECT_ROOT/backend/.env ]; then
        print_info "Creating .env file..."

        cat >$PROJECT_ROOT/backend/.env <<'EOF'
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database
POSTGRES_USER=lumina
POSTGRES_PASSWORD=lumina_password
POSTGRES_DB=lumina_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5435
DATABASE_URL=postgresql://lumina:lumina_password@localhost:5435/lumina_db

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Security
SECRET_KEY=change-me-use-a-long-random-secret
ALLOWED_ORIGINS='["http://localhost:3000","http://localhost:8501","http://localhost:8000","http://localhost:8888"]'
API_KEYS=[]
API_KEY_HASHES=[]
REQUIRE_API_KEY=false

# API Keys (add your own)
ALPHA_VANTAGE_API_KEY=
FRED_API_KEY=
NEWS_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Paths
MODEL_STORAGE_PATH=./models
FEATURE_STORE_PATH=./data/features
PARQUET_STORAGE_PATH=./data/parquet
LOG_FILE_PATH=./logs/lumina.log
EOF

        print_success ".env file created"
        print_warning "Please edit .env and add your API keys if needed"
    else
        print_info ".env file already exists, skipping..."
    fi

    # Cargar variables de entorno de backend/.env de forma segura
    if [ -f "$PROJECT_ROOT/backend/.env" ]; then
        print_info "Cargando variables de entorno de backend/.env..."
        set -a
        source "$PROJECT_ROOT/backend/.env"
        set +a
    fi

    # Step 4: Create Python virtual environment
    print_header "Step 4: Setting Up Python Environment"

    # Ask user for package manager preference
    echo ""
    echo -e "${BLUE}Choose package manager:${NC}"
    echo "  1) pip (traditional)"
    echo "  2) uv (fast, modern)"
    echo ""
    read -p "Select option (1 or 2, default: 1): " -n 1 -r PKG_MANAGER
    echo ""

    if [[ $PKG_MANAGER == "2" ]]; then
        if ! check_command "uv"; then
            print_warning "uv is not installed. Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
            
            if ! check_command "uv"; then
                print_error "Failed to install uv. Falling back to pip."
                PKG_MANAGER="1"
            else
                print_success "uv installed successfully"
            fi
        fi
    fi

    if [ ! -d ".venv" ]; then
        print_info "Creating virtual environment..."
        if [[ $PKG_MANAGER == "2" ]]; then
            uv venv .venv
        else
            python3 -m venv .venv
        fi
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi

    print_info "Activating virtual environment..."
    source .venv/bin/activate
    
    # Add project to PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    if [[ $PKG_MANAGER == "2" ]]; then
        print_info "Installing dependencies with uv..."
        
        if [ -f "pyproject.toml" ]; then
            print_info "Installing from pyproject.toml..."
            uv sync --all-groups
            print_success "Dependencies installed"
        elif [ -f "requirements/base.txt" ]; then
            uv pip install -r requirements/base.txt
            print_success "Base dependencies installed"
            
            if [ -f "requirements/dev.txt" ]; then
                uv pip install -r requirements/dev.txt
                print_success "Dev dependencies installed"
            fi
        fi
    else
        print_info "Upgrading pip..."
        pip install --upgrade pip setuptools wheel --quiet

        print_info "Installing dependencies with pip..."
        if [ -f "requirements/base.txt" ]; then
            pip install -r requirements/base.txt --quiet
            print_success "Base dependencies installed"
        fi

        if [ -f "requirements/dev.txt" ]; then
            pip install -r requirements/dev.txt --quiet
            print_success "Dev dependencies installed"
        fi
    fi

    # Step 5: Start Docker services
    print_header "Step 5: Starting Docker Services"

    cd "$PROJECT_ROOT/docker"

    # Clean up any stale networks/containers
    print_info "Cleaning up any existing services..."
    docker-compose down 2>/dev/null || true

    print_info "Starting TimescaleDB and Redis..."
    docker-compose up -d timescaledb redis

    print_info "Waiting for services to be ready (15 seconds)..."
    sleep 15

    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_success "Docker services started"
    else
        print_error "Failed to start Docker services"
        docker-compose logs
        exit 1
    fi

    # Step 6: Initialize database
    print_header "Step 6: Initializing Database"

    print_info "Waiting for TimescaleDB to be ready..."
    for i in {1..30}; do
        if docker exec lumina-timescaledb pg_isready -U lumina >/dev/null 2>&1; then
            print_success "TimescaleDB is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "TimescaleDB failed to start"
            exit 1
        fi
        sleep 1
    done

    print_info "Running database initialization script..."
    if docker exec -i lumina-timescaledb psql -U lumina -d lumina_db <"$PROJECT_ROOT/backend/db/timescale_setup.sql" >/dev/null 2>&1; then
        print_success "Database initialized"
    else
        print_warning "Database initialization had warnings (this may be normal if tables exist)"
    fi

    cd "$PROJECT_ROOT"

    # Step 7: Verify installations
    print_header "Step 7: Verifying Installation"

    print_info "Testing database connection..."
    if docker exec lumina-timescaledb psql -U lumina -d lumina_db -c "SELECT version();" >/dev/null 2>&1; then
        print_success "Database connection: OK"
    else
        print_error "Database connection: FAILED"
    fi

    print_info "Testing Redis connection..."
    if docker exec lumina-redis redis-cli ping >/dev/null 2>&1; then
        print_success "Redis connection: OK"
    else
        print_error "Redis connection: FAILED"
    fi

    print_info "Testing Python imports..."
    if python3 -c "from backend.data_engine.collectors.yfinance_collector import YFinanceCollector" 2>/dev/null; then
        print_success "Python imports: OK"
    else
        print_error "Python imports: FAILED"
    fi

    # Step 8: Optional - Seed database
    print_header "Step 8: Database Seeding (Optional)"

    read -p "Do you want to seed the database with sample data? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Seeding database with sample data..."
        print_warning "This will take 5-10 minutes..."

        python3 scripts/seed_data.py --tickers AAPL MSFT GOOGL TSLA NVDA --price-days 365 --feature-days 90

        if [ $? -eq 0 ]; then
            print_success "Database seeded successfully"
        else
            print_error "Database seeding failed"
        fi
    else
        print_info "Skipping database seeding"
    fi

    # Step 9: Start remaining services
    print_header "Step 9: Starting Additional Services"

    read -p "Do you want to start all services now? (API, Celery, MLflow, Streamlit) (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_ROOT/docker"

        print_info "Recreating network and services..."
        docker-compose down 2>/dev/null || true

        print_info "Starting all services..."
        docker-compose up -d

        print_info "Waiting for services to start (20 seconds)..."
        sleep 20

        print_success "All services started"

        cd "$PROJECT_ROOT"
    else
        print_info "Skipping additional services"
    fi

    # Step 10: Summary and next steps
    print_header "🎉 Setup Complete!"

    echo ""
    echo -e "${GREEN}✅ Lumina Quant Lab v2 is ready!${NC}"
    echo ""
    echo -e "${BLUE}📊 Available Services:${NC}"
    echo "  • TimescaleDB:  Running on port 5435"
    echo "  • Redis:        Running on port 6379"

    if docker-compose -f docker/docker-compose.yml ps api 2>/dev/null | grep -q "Up"; then
        echo "  • API:          http://localhost:8000"
        echo "  • API Docs:     http://localhost:8000/docs"
    else
        echo "  • API:          Not started (run: docker-compose up -d api)"
    fi

    if docker-compose -f docker/docker-compose.yml ps streamlit 2>/dev/null | grep -q "Up"; then
        echo "  • Streamlit:    http://localhost:8501"
    else
        echo "  • Streamlit:    Not started (run: docker-compose up -d streamlit)"
    fi

    if docker-compose -f docker/docker-compose.yml ps mlflow 2>/dev/null | grep -q "Up"; then
        echo "  • MLflow:       http://localhost:5000"
    else
        echo "  • MLflow:       Not started (run: docker-compose up -d mlflow)"
    fi

    if docker-compose -f docker/docker-compose.yml ps flower 2>/dev/null | grep -q "Up"; then
        echo "  • Flower:       http://localhost:5555"
    else
        echo "  • Flower:       Not started (run: docker-compose up -d flower)"
    fi

    if docker-compose -f docker/docker-compose.yml ps jupyter 2>/dev/null | grep -q "Up"; then
        echo "  • Jupyter:      http://localhost:8888"
    else
        echo "  • Jupyter:      Not started (run: docker-compose up -d jupyter)"
    fi

    echo ""
    echo -e "${BLUE}📝 Next Steps:${NC}"
    echo "  1. Activate Python environment:  source .venv/bin/activate"
    echo "  2. Start API:                    cd backend && uvicorn api.main:app --reload"
    echo "  3. Run tests:                    pytest tests/ -v"
    echo "  4. Start all services:           cd docker && docker-compose up -d"
    echo "  5. Open Streamlit:               http://localhost:8501"
    echo ""
    echo -e "${BLUE}📚 Documentation:${NC}"
    echo "  • Testing Guide:     TESTING_GUIDE.md"
    echo "  • Implementation:    IMPLEMENTATION_ROADMAP.md"
    echo "  • API Docs:          http://localhost:8000/docs"
    echo ""
    echo -e "${BLUE}🔍 Useful Commands:${NC}"
    echo "  • View logs:         docker-compose logs -f [service]"
    echo "  • Stop services:     docker-compose down"
    echo "  • Restart service:   docker-compose restart [service]"
    echo "  • Run tests:         pytest tests/ -v"
    echo ""

    print_success "Happy Trading! 🚀"
}

# Run main function
main "$@"
