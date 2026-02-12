# backend/db/models.py
"""
SQLAlchemy models for Lumina Quant Lab
Uses async SQLAlchemy with asyncpg for PostgreSQL/TimescaleDB
"""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
from loguru import logger
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB, TIMESTAMP, UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool

from backend.config.settings import get_settings

settings = get_settings()


# Base class for all models
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""

    pass


# ============================================================================
# TIME SERIES DATA MODELS (Hypertables)
# ============================================================================


class PriceData(Base):
    """
    Historical price data (OHLCV)
    TimescaleDB hypertable partitioned by time
    """

    __tablename__ = "price_data"

    time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), primary_key=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(10), primary_key=True, nullable=False)
    open: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    high: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    low: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    close: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    adjusted_close: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    dividends: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    stock_splits: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)

    __table_args__ = (
        Index("idx_price_ticker", "ticker", "time", postgresql_using="btree"),
        {"timescaledb_hypertable": {"time_column_name": "time"}},
    )

    def __repr__(self) -> str:
        return f"<PriceData(ticker={self.ticker}, time={self.time}, close={self.close})>"


class Feature(Base):
    """
    Engineered features for ML models
    TimescaleDB hypertable for efficient time-series storage
    """

    __tablename__ = "features"

    time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), primary_key=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(10), primary_key=True, nullable=False)
    feature_name: Mapped[str] = mapped_column(String(100), primary_key=True, nullable=False)
    feature_value: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    feature_category: Mapped[str | None] = mapped_column(
        String(50)
    )  # 'technical', 'fundamental', 'sentiment', 'macro'

    __table_args__ = (
        Index(
            "idx_features_ticker",
            "ticker",
            "feature_name",
            "time",
            postgresql_using="btree",
        ),
        Index(
            "idx_features_category",
            "feature_category",
            "time",
            postgresql_using="btree",
        ),
        {"timescaledb_hypertable": {"time_column_name": "time"}},
    )

    def __repr__(self) -> str:
        return (
            f"<Feature(ticker={self.ticker}, name={self.feature_name}, value={self.feature_value})>"
        )


class SentimentData(Base):
    """
    Sentiment data from various sources (news, social media, etc.)
    TimescaleDB hypertable
    """

    __tablename__ = "sentiment_data"

    time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), primary_key=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(10), primary_key=True, nullable=False)
    source: Mapped[str] = mapped_column(
        String(50), primary_key=True, nullable=False
    )  # 'news', 'reddit', 'twitter', 'finbert'
    sentiment_score: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    confidence: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    volume: Mapped[int | None] = mapped_column(Integer)  # Number of mentions/articles
    text_snippet: Mapped[str | None] = mapped_column(Text)
    meta_data: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (
        Index("idx_sentiment_ticker", "ticker", "time", postgresql_using="btree"),
        Index("idx_sentiment_source", "source", "time", postgresql_using="btree"),
        {"timescaledb_hypertable": {"time_column_name": "time"}},
    )

    def __repr__(self) -> str:
        return f"<SentimentData(ticker={self.ticker}, source={self.source}, score={self.sentiment_score})>"


class Prediction(Base):
    """
    Model predictions with confidence intervals
    TimescaleDB hypertable
    """

    __tablename__ = "predictions"

    prediction_time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), primary_key=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(10), primary_key=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), primary_key=True, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(20))
    predicted_price: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    confidence_lower: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    confidence_upper: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    uncertainty: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    actual_price: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)  # NULL until realized
    error: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)  # Computed when actual available

    __table_args__ = (
        Index(
            "idx_predictions_ticker",
            "ticker",
            "prediction_time",
            postgresql_using="btree",
        ),
        Index(
            "idx_predictions_model",
            "model_name",
            "prediction_time",
            postgresql_using="btree",
        ),
        {"timescaledb_hypertable": {"time_column_name": "prediction_time"}},
    )

    def __repr__(self) -> str:
        return f"<Prediction(ticker={self.ticker}, model={self.model_name}, price={self.predicted_price})>"


class MarketRegime(Base):
    """
    Market regime detection (bull/bear/sideways)
    TimescaleDB hypertable
    """

    __tablename__ = "market_regimes"

    time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), primary_key=True, nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(10), primary_key=True, nullable=False)
    regime_type: Mapped[str | None] = mapped_column(String(20))  # 'bull', 'bear', 'sideways'
    probability: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    volatility: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    expected_duration_days: Mapped[int | None] = mapped_column(Integer)
    meta_data: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (
        Index("idx_regimes_ticker", "ticker", "time", postgresql_using="btree"),
        {"timescaledb_hypertable": {"time_column_name": "time"}},
    )

    def __repr__(self) -> str:
        return f"<MarketRegime(ticker={self.ticker}, regime={self.regime_type}, prob={self.probability})>"


# ============================================================================
# BACKTESTING MODELS
# ============================================================================


class BacktestResult(Base):
    """
    Results from strategy backtests
    Regular table (not hypertable)
    """

    __tablename__ = "backtest_results"

    backtest_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    start_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    end_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    tickers: Mapped[list[str]] = mapped_column(ARRAY(String(50)))
    initial_capital: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    final_capital: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    total_return: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    annualized_return: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    volatility: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    sharpe_ratio: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    sortino_ratio: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    calmar_ratio: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    max_drawdown: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    win_rate: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    profit_factor: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    num_trades: Mapped[int] = mapped_column(Integer)
    avg_trade: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    config: Mapped[dict | None] = mapped_column(JSONB)  # Store backtest configuration
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    # Relationship to trades
    trades: Mapped[list["BacktestTrade"]] = relationship(
        "BacktestTrade", back_populates="backtest", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "idx_backtest_strategy",
            "strategy_name",
            "created_at",
            postgresql_using="btree",
        ),
        Index("idx_backtest_dates", "start_date", "end_date", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        return f"<BacktestResult(id={self.backtest_id}, strategy={self.strategy_name}, return={self.total_return:.2%})>"


class BacktestTrade(Base):
    """
    Individual trades from backtests
    """

    __tablename__ = "backtest_trades"

    trade_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    backtest_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("backtest_results.backtest_id", ondelete="CASCADE"),
        nullable=False,
    )
    entry_time: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    exit_time: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    direction: Mapped[str] = mapped_column(String(10))  # 'long', 'short'
    entry_price: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    exit_price: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    quantity: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    pnl: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    pnl_percent: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    commission: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    slippage: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    meta_data: Mapped[dict | None] = mapped_column(JSONB)

    # Relationship to backtest
    backtest: Mapped["BacktestResult"] = relationship("BacktestResult", back_populates="trades")

    __table_args__ = (
        Index("idx_trades_backtest", "backtest_id", "entry_time", postgresql_using="btree"),
        CheckConstraint("direction IN ('long', 'short')", name="check_direction"),
    )

    def __repr__(self) -> str:
        return f"<BacktestTrade(ticker={self.ticker}, direction={self.direction}, pnl={self.pnl})>"


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================


class Model(Base):
    """
    Metadata for trained ML models
    Integrates with MLflow for experiment tracking
    """

    __tablename__ = "models"

    model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_type: Mapped[str] = mapped_column(
        String(50)
    )  # 'lstm', 'transformer', 'xgboost', 'ensemble'
    version: Mapped[str] = mapped_column(String(20))
    trained_on: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    ticker: Mapped[str | None] = mapped_column(String(10))
    training_samples: Mapped[int] = mapped_column(Integer)
    validation_samples: Mapped[int] = mapped_column(Integer)

    # Performance metrics
    mae: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    rmse: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    r2_score: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)
    sharpe_backtest: Mapped[float | None] = mapped_column(DOUBLE_PRECISION)

    # Configuration
    hyperparameters: Mapped[dict | None] = mapped_column(JSONB)
    feature_importance: Mapped[dict | None] = mapped_column(JSONB)

    # MLflow integration
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100))

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_models_name", "model_name", "version", postgresql_using="btree"),
        Index("idx_models_ticker", "ticker", "trained_on", postgresql_using="btree"),
        Index("idx_models_active", "is_active", "created_at", postgresql_using="btree"),
        CheckConstraint(
            "model_type IN ('lstm', 'transformer', 'xgboost', 'ensemble')",
            name="check_model_type",
        ),
    )

    def __repr__(self) -> str:
        return f"<Model(name={self.model_name}, type={self.model_type}, ticker={self.ticker})>"


# ============================================================================
# PORTFOLIO MANAGEMENT
# ============================================================================


class PortfolioPosition(Base):
    """
    Portfolio positions and transactions (paper trading)
    """

    __tablename__ = "portfolio_positions"

    position_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(50), nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(10))  # 'buy', 'sell'
    quantity: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    price: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    commission: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    total_amount: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    balance_after: Mapped[float] = mapped_column(DOUBLE_PRECISION)
    executed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_transactions_user", "user_id", "executed_at", postgresql_using="btree"),
        Index("idx_transactions_ticker", "ticker", "executed_at", postgresql_using="btree"),
        CheckConstraint("transaction_type IN ('buy', 'sell')", name="check_transaction_type"),
    )

    def __repr__(self) -> str:
        return f"<PortfolioPosition(user={self.user_id}, ticker={self.ticker}, type={self.transaction_type})>"


class PortfolioBalance(Base):
    """
    Current portfolio balance for users
    """

    __tablename__ = "portfolio_balance"

    user_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    cash: Mapped[float] = mapped_column(DOUBLE_PRECISION, default=100000.0)
    equity: Mapped[float] = mapped_column(DOUBLE_PRECISION, default=0.0)
    total_value: Mapped[float] = mapped_column(DOUBLE_PRECISION, default=100000.0)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_balance_user", "user_id", postgresql_using="btree"),
        CheckConstraint("cash >= 0", name="check_positive_cash"),
        CheckConstraint("total_value >= 0", name="check_positive_total"),
    )

    def __repr__(self) -> str:
        return f"<PortfolioBalance(user={self.user_id}, total={self.total_value})>"


# ============================================================================
# MATERIALIZED VIEWS (for reference - created by TimescaleDB)
# ============================================================================

# Note: These are created by TimescaleDB continuous aggregates
# They are not ORM models but can be queried directly


# ============================================================================
# DATABASE ENGINE AND SESSION MANAGEMENT
# ============================================================================

# Global engine and session factory
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_async_engine() -> AsyncEngine:
    """
    Get or create async SQLAlchemy engine

    Returns:
        AsyncEngine instance
    """
    global _engine

    if _engine is None:
        # Get database URL and ensure it uses asyncpg
        database_url = settings.get_database_url(async_driver=True)

        # Create async engine with SSL disabled for localhost
        _engine = create_async_engine(
            database_url,
            echo=settings.DB_ECHO,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT,
            pool_recycle=settings.DB_POOL_RECYCLE,
            pool_pre_ping=True,  # Verify connections before using
            poolclass=NullPool if settings.is_production else None,
            connect_args={"server_settings": {"jit": "off"}, "ssl": False}
            if "localhost" in database_url
            else {},
        )

        logger.info(f"Created async database engine: {database_url.split('@')[1]}")

    return _engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create async session factory

    Returns:
        async_sessionmaker instance
    """
    global _async_session_factory

    if _async_session_factory is None:
        engine = get_async_engine()

        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

        logger.info("Created async session factory")

    return _async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI routes to get async database session

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_async_session)):
            ...

    Yields:
        AsyncSession instance
    """
    session_factory = get_async_session_factory()

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize database - create all tables

    Note: Hypertables and continuous aggregates are created by SQL scripts
    This only creates regular tables
    """
    try:
        engine = get_async_engine()

        logger.info("Initializing database tables...")

        # Create all tables
        async with engine.begin() as conn:
            # Only create regular tables (not hypertables)
            # Hypertables are created by timescale_setup.sql
            await conn.run_sync(Base.metadata.create_all)

        logger.success("Database tables initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db():
    """
    Close database connections (for cleanup)
    """
    global _engine, _async_session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("Database connections closed")


def reset_db_engine():
    """
    Synchronously reset the global engine and session factory.

    Must be called before creating a new event loop in Celery workers
    to avoid 'Future attached to a different loop' errors. The old engine's
    connections are bound to a previous (now closed) event loop and cannot
    be reused.
    """
    global _engine, _async_session_factory
    _engine = None
    _async_session_factory = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


async def check_db_connection() -> bool:
    """
    Check if database connection is working

    Returns:
        True if connection successful, False otherwise
    """
    try:
        engine = get_async_engine()

        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()

        logger.info("Database connection check: OK")
        return True

    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def execute_raw_sql(sql: str) -> list:
    """
    Execute raw SQL query

    Args:
        sql: SQL query string

    Returns:
        List of rows

    Example:
        rows = await execute_raw_sql("SELECT * FROM price_data LIMIT 10")
    """
    try:
        engine = get_async_engine()

        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            if result.returns_rows:
                return result.fetchall()
            return []

    except Exception as e:
        logger.error(f"Failed to execute SQL: {e}")
        raise


# ============================================================================
# BULK INSERT HELPERS (for data ingestion)
# ============================================================================


async def bulk_insert_price_data(data: list[dict]) -> int:
    """
    Bulk insert price data for performance

    Args:
        data: List of dictionaries with price data

    Returns:
        Number of rows inserted

    Example:
        data = [
            {"time": datetime(...), "ticker": "AAPL", "close": 150.0, ...},
            ...
        ]
        count = await bulk_insert_price_data(data)
    """
    if not data:
        return 0

    try:
        # Normalize timestamps to tz-aware UTC for TIMESTAMP(timezone=True)

        import pandas as pd

        for row in data:
            t = row.get("time")
            if t is None:
                continue
            if isinstance(t, pd.Timestamp):
                row["time"] = (
                    t.tz_localize("UTC").to_pydatetime()
                    if t.tzinfo is None
                    else t.tz_convert("UTC").to_pydatetime()
                )
            elif isinstance(t, datetime):
                row["time"] = t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)

        session_factory = get_async_session_factory()

        async with session_factory() as session:
            # Use bulk_insert_mappings for better performance
            await session.execute(
                PriceData.__table__.insert(),
                data,
            )
            await session.commit()

        logger.info(f"Bulk inserted {len(data)} price data rows")
        return len(data)

    except Exception as e:
        logger.error(f"Bulk insert failed: {e}")
        raise


async def bulk_insert_features(data: list[dict]) -> int:
    """
    Bulk insert features for performance with UPSERT to handle duplicates

    Args:
        data: List of dictionaries with feature data

    Returns:
        Number of rows inserted
    """
    if not data:
        return 0

    try:
        # Normalize timestamps to tz-aware UTC for TIMESTAMP(timezone=True)

        from sqlalchemy.dialects.postgresql import insert

        for row in data:
            t = row.get("time")
            if t is None:
                continue
            if isinstance(t, pd.Timestamp):
                row["time"] = (
                    t.tz_localize("UTC").to_pydatetime()
                    if t.tzinfo is None
                    else t.tz_convert("UTC").to_pydatetime()
                )
            elif isinstance(t, datetime):
                row["time"] = t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)

        session_factory = get_async_session_factory()

        async with session_factory() as session:
            stmt = insert(Feature).values(data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "time", "feature_name"],
                set_={
                    "feature_value": stmt.excluded.feature_value,
                    "feature_category": stmt.excluded.feature_category,
                },
            )
            await session.execute(stmt)
            await session.commit()

        logger.info(f"Bulk inserted/updated {len(data)} feature rows")
        return len(data)

    except Exception as e:
        logger.error(f"Bulk insert features failed: {e}")
        raise


# ============================================================================
# QUERY HELPERS
# ============================================================================


async def get_latest_price(ticker: str) -> PriceData | None:
    """
    Get most recent price data for a ticker

    Args:
        ticker: Stock ticker symbol

    Returns:
        PriceData instance or None
    """
    from sqlalchemy import desc, select

    session_factory = get_async_session_factory()

    async with session_factory() as session:
        query = (
            select(PriceData)
            .where(PriceData.ticker == ticker)
            .order_by(desc(PriceData.time))
            .limit(1)
        )

        result = await session.execute(query)
        return result.scalar_one_or_none()


async def get_price_history(
    ticker: str, start_date: datetime, end_date: datetime
) -> list[PriceData]:
    """
    Get price history for a ticker in date range

    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date

    Returns:
        List of PriceData instances
    """
    from sqlalchemy import and_, select

    session_factory = get_async_session_factory()

    async with session_factory() as session:
        query = (
            select(PriceData)
            .where(
                and_(
                    PriceData.ticker == ticker,
                    PriceData.time >= start_date,
                    PriceData.time <= end_date,
                )
            )
            .order_by(PriceData.time)
        )

        result = await session.execute(query)
        return list(result.scalars().all())


async def get_active_models(ticker: str | None = None) -> list[Model]:
    """
    Get all active models, optionally filtered by ticker

    Args:
        ticker: Optional ticker to filter by

    Returns:
        List of Model instances
    """
    from sqlalchemy import select

    session_factory = get_async_session_factory()

    async with session_factory() as session:
        query = select(Model).where(Model.is_active.is_(True))

        if ticker:
            query = query.where(Model.ticker == ticker)

        query = query.order_by(Model.trained_on.desc())

        result = await session.execute(query)
        return list(result.scalars().all())


async def delete_features_by_ticker(ticker: str, start_date: datetime, end_date: datetime) -> int:
    """
    Delete features for a ticker in date range

    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date

    Returns:
        Number of rows deleted
    """
    from sqlalchemy import and_, delete

    session_factory = get_async_session_factory()

    async with session_factory() as session:
        stmt = delete(Feature).where(
            and_(Feature.ticker == ticker, Feature.time >= start_date, Feature.time <= end_date)
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Base
    "Base",
    # Time series models
    "PriceData",
    "Feature",
    "SentimentData",
    "Prediction",
    "MarketRegime",
    # Backtesting models
    "BacktestResult",
    "BacktestTrade",
    # Model management
    "Model",
    # Portfolio models
    "PortfolioPosition",
    "PortfolioBalance",
    # Engine and session
    "get_async_engine",
    "get_async_session_factory",
    "get_async_session",
    "init_db",
    "close_db",
    # Utilities
    "check_db_connection",
    "execute_raw_sql",
    # Bulk operations
    "bulk_insert_price_data",
    "bulk_insert_features",
    # Query helpers
    "delete_features_by_ticker",
    "get_latest_price",
    "get_price_history",
    "get_active_models",
]
