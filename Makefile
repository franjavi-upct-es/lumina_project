.PHONY: build build-api build-ml build-jupyter build-streamlit up down clean prune logs

DOCKER_COMPOSE := docker-compose -f docker/docker-compose.yml

build:
	$(DOCKER_COMPOSE) build --parallel

build-api:
	docker build -t lumina-api:latest -f docker/Dockerfile.api .

build-ml:
	docker build -t lumina-ml:latest -f docker/Dockerfile.ml .

build-jupyter:
	docker build -t lumina-jupyter:latest -f docker/Dockerfile.jupyter .

build-streamlit:
	docker build -t lumina-streamlit:latest -f docker/Dockerfile.streamlit ./frontend/streamlit-app

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

restart:
	$(DOCKER_COMPOSE) restart

logs:
	$(DOCKER_COMPOSE) logs -f

logs-api:
	$(DOCKER_COMPOSE) logs -f api

logs-worker:
	$(DOCKER_COMPOSE) logs -f celery_worker

clean:
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

prune:
	docker system prune -af --volumes
	docker builder prune -af

size:
	@echo "Image sizes:"
	@docker images | grep lumina

ps:
	$(DOCKER_COMPOSE) ps

shell-api:
	$(DOCKER_COMPOSE) exec api /bin/bash

shell-db:
	$(DOCKER_COMPOSE) exec timescaledb psql -U lumina_user -d lumina_quant

rebuild: down build up

rebuild-api: down build-api
	$(DOCKER_COMPOSE) up -d api celery_worker celery_beat flower

rebuild-ml: down build-ml
	$(DOCKER_COMPOSE) up -d ml_service

health:
	@echo "Checking services health..."
	@curl -f http://localhost:8000/health || echo "API: DOWN"
	@curl -f http://localhost:8501/_stcore/health || echo "Streamlit: DOWN"
	@curl -f http://localhost:5000/health || echo "MLflow: DOWN"