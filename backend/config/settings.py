# backend/config/settings.py
"""Pydantic BaseSettings loading from .env. Singleton via lru_cache."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ArenaSettings(BaseSettings):
    """Runtime configuration for the Spartan Arena subsystem."""

    model_config = SettingsConfigDict(
        env_prefix="ARENA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    artifact_dir: Path = Path("./artifacts/arena")
    """Path to write structured per-step JSONL artifacts. Used in addition to TimescaleDB."""

    enable_run_summarizer_llm: bool = False
    """If True, run_summarizer may load a small local LLM (Phi-3-mini class) for
    end-of-run narratives. If False, fall back to a pure template summary."""

    run_summarizer_model_path: Path | None = None
    """Path to the local SLM weights directory. Read only if enable_run_summarizer_llm."""

    default_playback_multiplier: float = 1.0
    """Default playback multiplier exposed to the frontend. 1.0 = run as fast as
    the pipeline allows. >1.0 = sleep extra after each step (for human watching)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- Environment ----------
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LIVE_TICKERS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "QQQ"]
    )
    ALLOW_RANDOM_MODELS: bool = False
    GRAPH_INFERENCE_INTERVAL_SECONDS: float = 24 * 3600
    AGENT_TICK_INTERVAL_SECONDS: float = 1.0

    # ---------- Synthetic feed ----------
    SYNTHETIC_FEED_BOOTSTRAP_DAYS: int = 90
    SYNTHETIC_FEED_TICK_INTERVAL_SECONDS: float = 1.0
    SYNTHETIC_FEED_NEWS_INTERVAL_SECONDS: float = 30.0
    SYNTHETIC_FEED_SEED: int = 7

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
    INGESTION_BATCH_SIZE: int = 1000
    INGESTION_FLUSH_INTERVAL_S: float = 5.0
    INGESTION_BACKPRESSURE_FACTOR: float = 2.0
    INGESTION_MAX_RETRIES: int = 3

    # ---------- Collectors ----------
    COLLECTOR_MAX_BACKOFF_S: float = 60.0
    COLLECTOR_HEARTBEAT_S: float = 30.0

    # ---------- Feature store ----------
    FEATURE_STORE_BATCH_SIZE: int = 64
    FEATURE_STORE_NUM_WORKERS: int = 4

    # ---------- MLflow ----------
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MLFLOW_EXPERIMENT_NAME: str = "lumina_v3_arena"

    # ---------- Safety ----------
    UNCERTAINTY_THRESHOLD: float = 0.85
    MAX_DRAWDOWN_LIMIT: float = 0.20

    # ---------- API ----------
    API_KEY: str = ""
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    @field_validator("LIVE_TICKERS", mode="before")
    @classmethod
    def parse_live_tickers(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [str(t).strip().upper() for t in value if str(t).strip()]
        return [ticker.strip().upper() for ticker in value.split(",") if ticker.strip()]

    # Nested settings group for the Spartan Arena subsystem. Uses its own
    # env_prefix ("ARENA_") so it can be tuned independently in deployment.
    arena: ArenaSettings = Field(default_factory=ArenaSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
