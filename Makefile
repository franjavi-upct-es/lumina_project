# Makefile
# Lumina V3 — common development tasks.

UV ?= uv
UV_RUN := $(UV) run
LOCAL_REDIS_URL ?= redis://localhost:6379/0
LOCAL_TIMESCALE_URL ?= postgresql://lumina:lumina@localhost:5432/lumina

.PHONY: help install dev test test-unit test-integration format lint type openapi \
        migrate up up-8gb up-blackwell down restart logs ps clean \
        docker-build docker-build-api docker-build-data \
        docker-build-perception docker-build-brain docker-build-brain-blackwell \
        docker-build-frontend backfill-yfinance backfill-polygon run-arena deploy

help:
	@echo "Lumina V3 — available targets:"
	@echo ""
	@echo "  Local development:"
	@echo "    install              uv sync --all-extras"
	@echo "    dev                  uv run uvicorn with hot reload on :8000"
	@echo "    test                 uv run pytest full suite"
	@echo "    test-unit            uv run pytest fast tests only"
	@echo "    test-integration     uv run pytest integration tests"
	@echo "    format               uv run ruff format ."
	@echo "    lint                 uv run ruff check + uv run ruff format --check"
	@echo "    type                 uv run mypy backend"
	@echo "    migrate              uv run alembic upgrade head"
	@echo "    deploy               build all images, migrate, and start services"
	@echo ""
	@echo "  Docker — build:"
	@echo "    docker-build                     all service images (including frontend)"
	@echo "    docker-build-api                 API image only"
	@echo "    docker-build-data                data-engine image only"
	@echo "    docker-build-perception          perception image only"
	@echo "    docker-build-brain               brain image (CUDA 12.4)"
	@echo "    docker-build-brain-blackwell     brain image (CUDA 12.8 for Blackwell)"
	@echo "    docker-build-frontend            frontend dashboard image only"
	@echo ""
	@echo "  Docker — run:"
	@echo "    up                   full stack (default)"
	@echo "    up-8gb               full stack with semantic on CPU (8GB GPUs)"
	@echo "    up-blackwell         full stack with Blackwell brain image"
	@echo "    down                 stop services (preserve volumes)"
	@echo "    logs                 tail logs across all services"
	@echo "    ps                   show service status"
	@echo ""
	@echo "  Data + Simulation:"
	@echo "    backfill-yfinance    daily bars (free)"
	@echo "    backfill-polygon     1-min bars (paid)"
	@echo "    run-arena            execute a Spartan Arena simulation run"

install:
	$(UV) sync --all-extras

dev:
	REDIS_URL=$(LOCAL_REDIS_URL) \
	TIMESCALE_URL=$(LOCAL_TIMESCALE_URL) \
	$(UV_RUN) uvicorn backend.api.main:app --reload --port 8000

test:
	$(UV_RUN) pytest -v

test-unit:
	$(UV_RUN) pytest -v -m "not integration"

test-integration:
	$(UV_RUN) pytest -v -m "integration"

format:
	$(UV_RUN) ruff format .

lint:
	$(UV_RUN) ruff check backend tests scripts
	$(UV_RUN) ruff format --check .

type:
	$(UV_RUN) mypy backend

# Regenerate the TypeScript API client from the backend OpenAPI schema.
# Run after any change to backend/api/schemas.py or route signatures so the
# frontend types in frontend/src/types/api.generated.ts stay in sync.
openapi:
	$(UV_RUN) python scripts/dump_openapi.py
	cd frontend && npm run gen:api

migrate:
	REDIS_URL=$(LOCAL_REDIS_URL) \
	TIMESCALE_URL=$(LOCAL_TIMESCALE_URL) \
	$(UV_RUN) alembic upgrade head

deploy:
	bash scripts/deploy.sh

# ----- Docker — build -----------------------------------------------------
docker-build: docker-build-api docker-build-data docker-build-perception docker-build-brain docker-build-frontend

docker-build-api:
	docker build -f docker/Dockerfile.api  -t lumina/api:latest         .

docker-build-data:
	docker build -f docker/Dockerfile.data -t lumina/data:latest        .

docker-build-perception:
	docker build -f docker/Dockerfile.perception -t lumina/perception:latest .

docker-build-brain:
	docker build -f docker/Dockerfile.brain -t lumina/brain:latest      .

docker-build-brain-blackwell:
	docker build -f docker/Dockerfile.brain.blackwell \
	             -t lumina/brain:blackwell .

docker-build-frontend:
	docker build -f docker/Dockerfile.frontend -t lumina/frontend:latest .

# ----- Docker — run -------------------------------------------------------
up:
	docker compose up -d

up-8gb:
	docker compose -f docker-compose.yml -f docker-compose.gpu-8gb.yml up -d

up-blackwell: docker-build-brain-blackwell
	BRAIN_IMAGE=lumina/brain:blackwell docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

ps:
	docker compose ps

backfill-yfinance:
	REDIS_URL=$(LOCAL_REDIS_URL) \
	TIMESCALE_URL=$(LOCAL_TIMESCALE_URL) \
	$(UV_RUN) python -m scripts.backfill_historical --source yfinance \
	    --start 1900-01-01 --end 2025-12-31

backfill-polygon:
	REDIS_URL=$(LOCAL_REDIS_URL) \
	TIMESCALE_URL=$(LOCAL_TIMESCALE_URL) \
	$(UV_RUN) python -m scripts.backfill_historical --source polygon \
	    --start 1900-01-01 --end 2025-12-31

run-arena:
	$(UV_RUN) python scripts/run_arena.py \
	    --ticker AAPL --start 1900-01-01 --end 2024-01-01 \
	    --n-trajectories 10 --n-steps 200 --output-dir ./artifacts/arena
