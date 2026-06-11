# backend/config/settings.py
"""Pydantic BaseSettings loading from .env. Singleton via lru_cache."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
<<<<<<< HEAD
from typing import Annotated, Literal
=======
from urllib.parse import unquote, urlparse
>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e

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
    MLFLOW_BACKTEST_EXPERIMENT_NAME: str = "lumina_v3_backtests"

    # ---------- Dashboard simulation workers ----------
    BACKTEST_WORKER_POLL_SECONDS: float = 1.0
    BACKTEST_SYNTHETIC_STEPS: int = 252
    ARENA_WORKER_POLL_SECONDS: float = 1.0
    ARENA_SYNTHETIC_STEPS: int = 240
    ALLOW_SYNTHETIC_SIMULATION_FALLBACK: bool = False

    # ---------- Safety ----------
    UNCERTAINTY_THRESHOLD: float = 0.85
    MAX_DRAWDOWN_LIMIT: float = 0.20

    # ---------- API ----------
    API_KEY: str = ""
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

<<<<<<< HEAD
    @field_validator("CORS_ORIGINS", mode="before")
=======
    # Backtesting
    DEFAULT_INITIAL_CAPITAL: float = 100000.0
    DEFAULT_COMMISSION: float = 0.001  # 0.1%
    DEFAULT_SLIPPAGE: float = 0.0005  # 0.05%

    # Risk Management
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    MAX_PORTFOLIO_LEVERAGE: float = 1.0  # No leverage by default
    VAR_CONFIDENCE_LEVEL: float = 0.95

    # Monitoring
    SENTRY_DNS: str | None = None
    PROMETHEUS_PORT: int = 9090

    # Logging
    LOG_FORMAT: str = Field(default="text", pattern="^(text|json)$")
    LOG_FILE_PATH: str | None = None
    LOG_MAX_BYTES: int = 10485760
    LOG_BACKUP_COUNT: int = 5

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_value(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            normalized = v.strip().strip("\"'").lower()
            if normalized in {"1", "true", "yes", "on", "debug", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "production"}:
                return False
        return v

    @model_validator(mode="before")
    @classmethod
    def populate_postgres_user(cls, values):
        if not isinstance(values, dict) or values.get("POSTGRES_USER"):
            return values

        database_url = values.get("DATABASE_URL")
        if not database_url:
            return values

        username = urlparse(str(database_url)).username
        if username:
            return {**values, "POSTGRES_USER": unquote(username)}

        return values

    @field_validator("API_KEYS", mode="after")
>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e
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

<<<<<<< HEAD
    # Nested settings group for the Spartan Arena subsystem. Uses its own
    # env_prefix ("ARENA_") so it can be tuned independently in deployment.
    arena: ArenaSettings = Field(default_factory=ArenaSettings)
=======
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v):
        if not str(v).startswith("postgresql"):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection string")
        return v

    @field_validator("ALLOWED_ORIGINS", "API_KEYS", "API_KEY_HASHES", mode="before")
    @classmethod
    def parse_list_field(cls, v):
        """Parse list fields from environment variables.

        Supports both JSON arrays and comma-separated values:
        - '["http://localhost:3000", "http://localhost:8501"]'  -> JSON
        - 'http://localhost:3000,http://localhost:8501'          -> CSV
        """
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
                v = v[1:-1].strip()
            if not v or v == "[]":
                return []
            # Try JSON first
            if v.startswith("["):
                import json

                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if item]
                except json.JSONDecodeError:
                    pass
            # Fall back to comma-separated
            return [item.strip() for item in v.split(",") if item.strip()]
        return list(v)

    @model_validator(mode="after")
    def validate_security_settings(self):
        if self.is_production:
            if "*" in self.ALLOWED_ORIGINS:
                raise ValueError("ALLOWED_ORIGINS cannot include '*' in production")
            if not (self.API_KEYS or self.API_KEY_HASHES):
                raise ValueError("API_KEYS or API_KEY_HASHES must be set in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    def get_database_url(self, async_driver: bool = False) -> str:
        """
        Get database URL with optional async driver
        """
        url = str(self.DATABASE_URL)
        if async_driver:
            url = url.replace("postgresql://", "postgresql+asyncpg://")
        else:
            url = url.replace("postgresql://", "postgresql+psycopg2://")
        return url
>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e


@lru_cache
def get_settings() -> Settings:
<<<<<<< HEAD
    return Settings()
=======
    """
    Get cached settings instance
    """
    return Settings()  # type: ignore


# Create a convenience instance
settings = get_settings()
>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e
