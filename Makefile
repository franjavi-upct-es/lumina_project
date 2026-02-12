DOCKER_COMPOSE = docker-compose --env-file ../backend/.env

.PHONY: help build up up-gpu down restart logs clean build-ml test lint format setup-gpu check-gpu

help:
	@echo "Lumina Quant Platform - Make Commands"
	@echo "======================================"
	@echo "build          - Build all containers"
	@echo "build-ml       - Build only ML service"
	@echo "up             - Start all services (CPU only)"
	@echo "up-gpu         - Start all services with GPU support"
	@echo "down           - Stop all services"
	@echo "restart        - Restart all services"
	@echo "logs           - Show logs"
	@echo "logs-ml        - Show ML worker logs"
	@echo "clean          - Remove all containers and volumes"
	@echo "test           - Run tests"
	@echo "lint           - Run linter"
	@echo "format         - Format code"
	@echo "setup-gpu      - Install NVIDIA Container Toolkit for Docker GPU"
	@echo "check-gpu      - Check GPU availability (host + Docker)"

build:
	cd docker && $(DOCKER_COMPOSE) build

build-ml:
	cd docker && $(DOCKER_COMPOSE) build ml-worker

up:
	cd docker && $(DOCKER_COMPOSE) up -d

up-gpu:
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		cd docker && $(DOCKER_COMPOSE) --profile gpu up -d; \
    else \
        echo "Warning: NVIDIA GPU not detected, falling back to CPU mode"; \
        make up; \
    fi	

down:
	cd docker && $(DOCKER_COMPOSE) down

restart:
	cd docker && $(DOCKER_COMPOSE) restart

logs:
	cd docker && $(DOCKER_COMPOSE) logs -f

logs-ml:
	cd docker && $(DOCKER_COMPOSE) logs -f ml-worker

clean:
	cd docker && $(DOCKER_COMPOSE) down -v
	docker system prune -f

test:
	pytest tests/ -v

lint:
	cd backend && ruff check .

format:
	cd backend && ruff format .

# GPU setup and checks
setup-gpu:
	@bash scripts/setup_nvidia_docker.sh

check-gpu:
	@echo "Checking NVIDIA GPU..."
	@nvidia-smi || echo "No NVIDIA GPU found"
	@echo ""
	@echo "Checking Docker NVIDIA runtime..."
	@docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi || echo "Docker NVIDIA runtime not configured. Run: make setup-gpu"