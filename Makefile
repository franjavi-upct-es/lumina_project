.PHONY: help build up down restart logs clean build-ml test lint format

help:
	@echo "Lumina Quant Platform - Make Commands"
	@echo "======================================"
	@echo "build          - Build all containers"
	@echo "build-ml       - Build only ML service"
	@echo "up             - Start all services"
	@echo "down           - Stop all services"
	@echo "restart        - Restart all services"
	@echo "logs           - Show logs"
	@echo "clean          - Remove all containers and volumes"
	@echo "test           - Run tests"
	@echo "lint           - Run linter"
	@echo "format         - Format code"

build:
	cd docker && docker-compose build

build-ml:
	cd docker && docker-compose build ml_service

up:
	cd docker && docker-compose up -d

down:
	cd docker && docker-compose down

restart:
	cd docker && docker-compose restart

logs:
	cd docker && docker-compose logs -f

logs-ml:
	cd docker && docker-compose logs -f ml_service

clean:
	cd docker && docker-compose down -v
	docker system prune -f

test:
	pytest tests/ -v

lint:
	cd backend && ruff check .

format:
	cd backend && ruff format .

# Check GPU availability
check-gpu:
	@echo "Checking NVIDIA GPU..."
	@nvidia-smi || echo "No NVIDIA GPU found"
	@echo "\nChecking Docker NVIDIA runtime..."
	@docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi || echo "Docker NVIDIA runtime not configured"