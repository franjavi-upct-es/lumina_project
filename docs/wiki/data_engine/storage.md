# Storage Backends: TimescaleDB and Redis

??? note "Relevant source files"

    - [gh:.dockerignore]
    - [gh:alembic/versions/003_add_portfolio_and_backtest.py]
    - [gh:backend/data_engine/storage/__init__.py]
    - [gh:backend/data_engine/storage/timescale.py]
    - [gh:backend/fusion/state_assembler.py]
    - [gh:tests/data_engine/storage/test_timescale.py]

Lumina V3 utilizes a dual-storage strategy to balance the high-throughput,
low-latency requirements of real-time trading with the complex, time-series
analytical needs of historical backtesting and feature engineering.

The system partitions data between **TimescaleDB** (the "Cold Store" and
Analytical Engine) and **Redis** (the "Hot Store" and IPC Backbone).
