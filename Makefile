# Makefile
# Lumina V3 — common development tasks.

UV ?= uv
UV_RUN := $(UV) run

.PHONY: help install dev test test-unit test-integration lint type \
        migrate up up-8gb up-blackwell down restart logs ps clean \
        docker-build docker-build-api docker-build-data \
        docker-build-perception docker-build-brain docker-build-brain-blackwell \
        backfill-yfinance backfill-polygon run-arena

help:
	@echo "Lumina V3 — available targets:"
	@echo ""
	@echo "  Local development:"
	@echo "    install              uv sync --all-extras"
	@echo "    dev                  uv run uvicorn with hot reload on :8000"
	@echo "    test                 uv run pytest full suite"
	@echo "    test-unit            uv run pytest fast tests only"
	@echo "    test-integration     uv run pytest integration tests"
	@echo "    lint                 uv run ruff check + uv run ruff format --check"
	@echo "    type                 uv run mypy backend"
	@echo "    migrate              uv run alembic upgrade head"
	@echo ""
	@echo "  Docker — build:"
	@echo "    docker-build                     all four service images"
	@echo "    docker-build-api                 API image only"
	@echo "    docker-build-data                data-engine image only"
	@echo "    docker-build-perception          perception image only"
	@echo "    docker-build-brain               brain image (CUDA 12.4)"
	@echo "    docker-build-brain-blackwell     brain image (CUDA 12.8 for Blackwell)"
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
	REDIS_URL=redis://localhost:6379/0 \
	TIMESCALE_URL=postgresql://lumina:lumina@localhost:5432/lumina \
	$(UV_RUN) uvicorn backend.api.main:app --reload --port 8000

test:
	$(UV_RUN) pytest -v

test-unit:
	$(UV_RUN) pytest -v -m "not integration"

test-integration:
	$(UV_RUN) pytest -v -m "integration"

lint:
	$(UV_RUN) ruff check backend tests scripts
	$(UV_RUN) ruff format --check .

type:
	$(UV_RUN) mypy backend

migrate:
	$(UV_RUN) alembic upgrade head

# ----- Docker — build -----------------------------------------------------
docker-build: docker-build-api docker-build-data docker-build-perception docker-build-brain

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
	$(UV_RUN) python -m scripts.backfill_historical --source yfinance \
	    --start 1980-01-01 --end 2024-12-31

backfill-polygon:
	$(UV_RUN) python -m scripts.backfill_historical --source polygon \
	    --start 1980-01-01 --end 2024-12-31

run-arena:
	$(UV_RUN) python scripts/run_arena.py \
	    --ticker AAPL --start 1980-01-01 --end 2024-01-01 \
	    --n-trajectories 10 --n-steps 200 --output-dir ./artifacts/arena
