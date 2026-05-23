#!/usr/bin/env bash
# docker/docker-entrypoint.sh
# Service-mode dispatcher for the Lumina V3 multipurpose containers.
#
# A single Docker image (data / perception / brain) is built once and
# can be launched in several roles selected at container-start time via
# the LUMINA_SERVICE_MODE environment variable. This avoids the
# explosion of one-image-per-microservice while keeping each container
# focused on a single Python module.
#
# Supported modes (per image, the README lists which are valid where):
#
#   api                    → uvicorn backend.api.main:app
#   migrate                → alembic upgrade head
#   ingestion              → backend.data_engine.pipelines.ingestion
#   tft_inference          → backend.perception.temporal.inference
#   semantic_inference     → backend.perception.semantic.inference
#   graph_inference        → backend.perception.structural.inference
#   state_assembler        → backend.fusion.state_assembler
#   paper_trading          → backend.paper_trading.runner
#   train_agent            → scripts.train_agent
#   sleep                  → infinite sleep (debug)
#
# Anything else aborts with a clear error message.

set -euo pipefail

MODE="${LUMINA_SERVICE_MODE:-}"

if [[ -z "${MODE}" ]]; then
    echo "[lumina-entrypoint] LUMINA_SERVICE_MODE is not set." >&2
    echo "[lumina-entrypoint] Set one of: api, migrate, ingestion," >&2
    echo "[lumina-entrypoint]            tft_inference, semantic_inference," >&2
    echo "[lumina-entrypoint]            graph_inference, state_assembler," >&2
    echo "[lumina-entrypoint]            paper_trading, train_agent, sleep." >&2
    exit 64  # EX_USAGE
fi

echo "[lumina-entrypoint] Launching service mode: ${MODE}"

case "${MODE}" in
    api)
        exec uvicorn backend.api.main:app \
             --host 0.0.0.0 --port 8000 \
             --workers "${UVICORN_WORKERS:-1}" \
             --log-level "${UVICORN_LOG_LEVEL:-info}"
        ;;
    migrate)
        # The migrate one-shot is idempotent — `alembic upgrade head` on a
        # fully-up-to-date database is a no-op that exits 0. docker-compose
        # uses this to gate the api / data services on a successful schema.
        exec alembic upgrade head
        ;;
    ingestion)
        exec python -m backend.data_engine.pipelines.ingestion
        ;;
    tft_inference)
        exec python -m backend.perception.temporal.inference
        ;;
    semantic_inference)
        exec python -m backend.perception.semantic.inference
        ;;
    graph_inference)
        exec python -m backend.perception.structural.inference
        ;;
    state_assembler)
        exec python -m backend.fusion.state_assembler
        ;;
    paper_trading)
        exec python -m backend.paper_trading.runner
        ;;
    train_agent)
        exec python -m scripts.train_agent
        ;;
    sleep)
        echo "[lumina-entrypoint] Sleeping forever for debugging." >&2
        exec sleep infinity
        ;;
    *)
        echo "[lumina-entrypoint] Unknown service mode: ${MODE}" >&2
        exit 64
        ;;
esac
