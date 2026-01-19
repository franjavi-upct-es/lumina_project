# backend/config/settings.py
"""
Configuration settings using Pydantic for type safety and validation
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

# Load backend-specific env file (stable across local and container paths)
BACKEND_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = BACKEND_ROOT / ".env"


class Settings(BaseSettings):
    """
    Application settings with validation
    """

    model_config = ConfigDict(
        extra="ignore",
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Application
    APP_NAME: str = "Lumina Quant Lab"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)")

    # API
    API_V2_PREFIX: str = "/api/v2"
    SECRET_KEY: str = Field(min_length=32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # CORS
    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8501",  # Streamlit
            "http://localhost:8888",  # Jupyter
        ],
        validation_alias=AliasChoices("ALLOWED_ORIGINS", "CORS_ORIGINS"),
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)

    # API Keys
    API_KEYS: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("API_KEYS", "API_KEY"),
    )
    API_KEY_HASHES: list[str] = Field(default_factory=list)
    REQUIRE_API_KEY: bool = Field(default=False)

    # Rate limiting
    RATE_LIMIT_MAX_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # Database
    DATABASE_URL: str = Field(default="postgresql://localhost:5432/lumina_quant")
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_ECHO: bool = False

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_CACHE_TTL: int = 3600  # 1 hour

    # Celery
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")
    CELERY_TASK_TIME_LIMIT: int = 3600  # 1 hour
    CELERY_TASK_SOFT_TIME_LIMIT: int = 3300  # 55 hour

    # Data Sources
    YFINANCE_RATE_LIMIT: int = 2000  # requests per hour
    ALPHA_VANTAGE_API_KEY: str | None = None
    FRED_API_KEY: str | None = None
    NEWS_API_KEY: str | None = None
    REDDIT_CLIENT_ID: str | None = None
    REDDIT_CLIENT_SECRET: str | None = None
    TWITTER_BEARER_TOKEN: str | None = None

    # ML/AI
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MODEL_STORAGE_PATH: str = "/app/models"
    FEATURE_STORE_PATH: str = "/app/data/features"

    # PyTorch/ML Settings
    CUDA_VISIBLE_DEVICES: str = "0"
    PYTORCH_ENABLE_MPS_FALLBACK: bool = True  # For Apple Silicon

    # Data Storage
    PARQUET_STORAGE_PATH: str = "/app/data/parquet"
    MAX_PARQUET_FILE_SIZE_MB: int = 100

    # Feature Engineering
    MAX_FEATURES: int = 200
    FEATURE_CACHE_HOURS: int = 24

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

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_enviroment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v):
        if not str(v).startswith("postgresql"):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection string")
        return v

    @field_validator("ALLOWED_ORIGINS", "API_KEYS", "API_KEY_HASHES", mode="before")
    @classmethod
    def split_csv_values(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
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


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance
    """
    return Settings()


# Create a convenience instance
settings = get_settings()
