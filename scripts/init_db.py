#!/usr/bin/env python
"""
Initialization script: creates hypertables in TimescaleDB, configures indexes, loads initial historical data.
As implemented in Lumina V3, this delegates schema creation to Alembic.
"""
import subprocess
import sys
from loguru import logger

def main():
    logger.info("Initializing TimescaleDB schemas via Alembic migrations...")
    result = subprocess.run(["uv", "run", "alembic", "upgrade", "head"])
    if result.returncode != 0:
        logger.error("Database initialization failed.")
        sys.exit(result.returncode)
    
    logger.info("Database schemas and hypertables initialized successfully.")
    logger.info("Run `make backfill-yfinance` to load initial historical data.")

if __name__ == "__main__":
    main()
