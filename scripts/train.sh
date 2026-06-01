#!/usr/bin/env bash
set -e

echo "Starting full Spartan Curriculum Training..."
uv run python -m scripts.train_agent "$@"
