#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Starting Lumina V3 Deployment"
echo "=========================================================="

echo "[1/4] Building all Docker images..."
make docker-build

echo "[2/4] Starting database and cache services..."
docker compose up -d redis timescale
echo "Waiting for services to become healthy..."
sleep 5

echo "[3/4] Running database migrations..."
make migrate

echo "[4/4] Starting the rest of the stack (API, Data, Perception, Brain, Dashboard)..."
make up

echo "=========================================================="
echo " Deployment successful!"
echo " Use 'make logs' to monitor the system."
echo "=========================================================="
