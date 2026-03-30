#!/bin/bash

# Lumina Project - Quick Start Script for Docker Setup
# This script automates the setup and validation of the Docker environment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Lumina Project - Docker Setup Wizard  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"

# Step 1: Check .env
echo -e "${YELLOW}[Step 1/4]${NC} Setting up environment variables..."
if [ ! -f ./backend/.env ]; then
    if [ -f .env.example ]; then
        echo -e "${YELLOW}→${NC} .env not found, creating from .env.example..."
        cp .env.example .env
        echo -e "${GREEN}✓${NC} .env created. Please edit with your values:"
        echo "  - POSTGRES_PASSWORD (change from default)"
        echo "  - SECRET_KEY (generate new)"
        echo "  - API Keys (if needed)"
        echo ""
        
        # Prompt user to edit
        read -p "Press Enter after editing .env, or Ctrl+C to abort: "
    else
        echo -e "${RED}✗${NC} .env.example not found!"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} .env already exists"
fi
echo ""

# Step 2: Validate configuration
echo -e "${YELLOW}[Step 2/4]${NC} Validating configuration..."
if [ -x "scripts/validate_docker_config.sh" ]; then
    bash scripts/validate_docker_config.sh
    VALIDATION_RESULT=$?
    if [ $VALIDATION_RESULT -ne 0 ]; then
        echo -e "${RED}✗${NC} Configuration validation failed. Please fix issues and retry."
        exit 1
    fi
else
    echo -e "${YELLOW}⚠${NC} Validation script not executable, skipping..."
fi
echo ""

# Step 3: Docker setup
echo -e "${YELLOW}[Step 3/4]${NC} Preparing Docker environment..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗${NC} Docker is not installed. Please install Docker first."
    exit 1
fi

echo -e "${YELLOW}→${NC} Stopping any running containers..."
docker-compose down 2>/dev/null || true
echo -e "${GREEN}✓${NC} Containers stopped"

echo -e "${YELLOW}→${NC} Removing old volumes (WARNING: This deletes data)..."
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose down -v
    echo -e "${GREEN}✓${NC} Volumes removed"
else
    echo -e "${YELLOW}⚠${NC} Keeping existing volumes"
fi
echo ""

# Step 4: Build and start
echo -e "${YELLOW}[Step 4/4]${NC} Building and starting services..."
echo -e "${YELLOW}→${NC} Building images (this may take a few minutes)..."
docker-compose up --build -d

echo -e "${GREEN}✓${NC} Services started!"
echo ""
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Setup Complete! Next Steps:            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"

echo "1. Monitor service startup:"
echo -e "   ${YELLOW}docker-compose logs -f${NC}"
echo ""

echo "2. Wait for all services to be healthy (check timestamps):"
echo -e "   ${YELLOW}docker-compose ps${NC}"
echo ""

echo "3. Test connectivity:"
echo -e "   ${YELLOW}docker-compose exec redis redis-cli ping${NC}"
echo -e "   ${YELLOW}curl http://localhost:5000/health${NC}"
echo -e "   ${YELLOW}curl http://localhost:8000/health${NC}"
echo ""

echo "4. View your services:"
echo -e "   - API: ${GREEN}http://localhost:8000${NC}"
echo -e "   - MLflow: ${GREEN}http://localhost:5000${NC}"
echo -e "   - Frontend: ${GREEN}http://localhost:8501${NC}"
echo ""

echo "5. For more information, see:"
echo -e "   - ${YELLOW}DOCKER_CONFIGURATION_FIX.md${NC}"
echo -e "   - ${YELLOW}README.md${NC}"
echo ""

echo -e "${GREEN}Setup complete! Your Lumina project is ready.${NC}\n"
