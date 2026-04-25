"""Pydantic BaseSettings loading from .env. Singleton via lru_cache."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import loguru
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- Environment ----------
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # ---------- Data sources ----------
    POLYGON_API_KEY: str = ""
    POLYGON_WS_URL: str = "wss://socket.polygon.io/stocks"
    POLYGON_REST_URL: str = "https://api.polygon.io"
    NEWSAPI_KEY: str = ""
    NEWS_POLL_INTERVAL_SECONDS: int = 60

    # ---------- Broker ----------
    BROKER_MODE: Literal["alpaca", "paper"] = "paper"
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_PAPER: bool = True
    INITIAL_CAPITAL: float = 100_000.0

    # ---------- Storage ----------
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50
    TIMESCALE_URL: str = "postgresql://lumina:lumina@localhost:5432/lumina"
    TIMESCALE_POOL_MIN: int = 5
    TIMESCALE_POOL_MAX: int = 20

    # ---------- Ingestion ----------
    INGESTION_BATCH_SIZE: int = 100
    INGESTION_FLUSH_INTERVAL_S: float = 5.0
    INGESTION_BACKPRESSURE_FACTOR: float = 2.0
    INGESTION_MAX_RETRIES: int = 3

    # ---------- Collectors ----------
    COLLECTOR_MAX_BACKOFF_S: float = 60.0
    COLLLECTOR_HEARTBEAT_S: float = 30.0

    # ---------- Feature store ----------
    FEATURE_STORE_BATCH_SIZE: int = 64
    FEATURE_STORE_NUM_WORKERS: int = 4

    # ---------- Safety ----------
    UNCERTAINTY_THRESHOLD: float = 0.85
    MAX_DRAWDOWN_LIMIT: float = 0.20

    # ---------- API ----------
    API_KEY: str = ""
    CORS_ORIGIN: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


@lru_cache()
def get_settings() -> Settings:
    return Settings()
