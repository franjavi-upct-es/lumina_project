# backend/api/routes/data.py
"""
Data Collection and Feature Engineering Endpoints

This module provides endpoints for:
- Market data collection (stocks, crypto, forex)
- Technical indicator calculation
- Feature engineering
- Data quality checks
- Feature store access

Supports multiple data sources:
- YFinance: Historical stock data
- Alpha Vantage: Real-time and fundamental data
- FRED: Macroeconomic indicators
- NewsAPI: Financial news
- Social media: Sentiment data
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import (
    check_rate_limit,
    get_async_db,
    get_feature_engineer,
    get_redis,
    get_yfinance_collector,
    verify_api_key,
)
from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.data_engine.transformers.feature_engineering import FeatureEngineer

router = APIRouter(dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
settings = get_settings()


# ============================================================================
# Request/Response Models
# ============================================================================


class MarketDataRequest(BaseModel):
    """Request for market data"""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'APPL', 'TSLA')")
    start_date: datetime
    end_date: datetime = Field(default_factory=datetime.utcnow())
    interval: str = Field("1d", pattern="^(1m|5m|15m|30m|1h|1d|1wk|1mo)$")


class MarketDataResponse(BaseModel):
    """Market data response"""

    ticker: str
    interval: str
    start_date: datetime
    end_date: datetime
    data_points: int
    data: list[dict]


class TechnicalIndicatorsRequest(BaseModel):
    """Request for technical indicators"""

    ticker: str
    indicators: list[str] = Field(
        ..., description="List of indicators: 'sma', 'ema', 'rsi', 'macd', 'bbands', etc."
    )
    start_date: datetime
    end_date: datetime = Field(default_factory=datetime.utcnow())
    interval: str = Field("1d")

    # Indicators parameters
    sma_period: int = Field(20, ge=5, le=200)
    ema_period: int = Field(20, ge=5, le=200)
    rsi_period: int = Field(14, ge=5, le=50)
    macd_fast: int = Field(12)
    macd_slow: int = Field(26)
    macd_signal: int = Field(9)


class FeaturesResponse(BaseModel):
    """Engineering features response"""

    ticker: str
    timestamp: datetime
    feature_count: int
    features: dict


# ============================================================================
# Market Data Endpoints
# ============================================================================


@router.get("/market-data/{ticker}", response_model=MarketDataResponse)
async def get_market_data(
    ticker: str,
    start_date: datetime = Query(..., description="Start date for data retrieval"),
    end_date: datetime = Query(default_factory=datetime.utcnow()),
    interval: str = Query("1d", pattern="^(1m|5m|15m|30m|1h|1d|1wk|1mo)$"),
    collector: YFinanceCollector = Depends(get_yfinance_collector),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get historical market data for a ticker.

    Retrieves OHLCV data from YFinance with support for multiple intervals.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        interval: Data interval
        collector: YFinance collector
        db: Database session

    Returns:
        MarketDataResponse: Market data with OHLCV
    """
    logger.info(
        f"Fetching market data for {ticker}: {start_date} to {end_date}, interval={interval}"
    )

    try:
        # Collect data
        data = await collector.collect_stock_data(
            ticker=ticker, start_date=start_date, end_date=end_date, interval=interval
        )

        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

        # Convert to list of dicts
        data_list = data.reset_index().to_dict("records")

        return MarketDataResponse(
            ticker=ticker,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_points=len(data_list),
            data=data_list,
        )

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")


@router.get("/market-data/{ticker}/latest")
async def get_latest_price(
    ticker: str,
    collector: YFinanceCollector = Depends(get_yfinance_collector),
):
    """
    Get latest price for a ticker.

    Returns real-time or latest available price.

    Args:
        ticker: Stock ticker symbol
        collector: YFinance collector

    Returns:
        dict: Latest price information
    """
    logger.info(f"Fetching latest price for {ticker}")

    try:
        # Get latest day's data
        data = await collector.collect_stock_data(
            ticker=ticker,
            start_date=datetime.utcnow() - timedelta(days=5),
            end_date=datetime.utcnow(),
            interval="1d",
        )

        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")

        # Get latest row
        latest = data.iloc[-1]

        return {
            "ticker": ticker,
            "timestamp": data.index[-1].isoformat(),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"]),
            "volume": int(latest["Volume"]),
        }

    except Exception as e:
        logger.error(f"Error fetching latest price: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest price: {str(e)}")


@router.post("/market-data/batch")
async def get_batch_market_data(
    tickers: list[str] = Field(..., min_length=1, max_length=50),
    start_date: datetime = Query(...),
    end_date: datetime = Query(default_factory=datetime.utcnow),
    interval: str = Query("1d"),
    collector: YFinanceCollector = Depends(get_yfinance_collector),
):
    """
    Get market data for multiple tickers:

    Batch endpoint for efficient multi-ticker data retrieval.

    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        interval: Data interval
        collector: YFinance collector

    Returns:
        dict: Market data for all tickers
    """
    logger.info(f"Fetching batch market data for {len(tickers)} tickers")

    results = {}
    errors = {}

    for ticker in tickers:
        try:
            data = await collect.collect_stock_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )

            if data is not None and not data.empty:
                results[ticker] = data.reset_index().to_dict("records")
            else:
                errors[ticker] = "No data available"

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            errors[ticker] = str(e)

    return {
        "success": results,
        "errors": errors,
        "total_requested": len(tickers),
        "successful": len(results),
        "failed": len(errors),
    }


# ============================================================================
# Technical Indicators Endpoints
# ============================================================================


@router.post("/indicators/calculate", response_model=FeaturesResponse)
async def calculate_technical_indicators(
    request: TechnicalIndicatorsRequest,
    collector: YFinanceCollector = Depends(get_yfinance_collector),
    engineer: FeatureEngineer = Depends(get_feature_engineer),
):
    """
    Calculate technical indicators for a ticker.

    Computes requested technical indicators using native implementations.

    Supported Indicators:
    - SMA: Simple Moving Average
    - EMA: Exponential Moving Average
    - RSI: Relative Strength Index
    - MACD: Moving Average Convergence Divergence
    - ATR: Average True Range
    - OBV: On-Balance Volume
    - Stochastic Oscillator

    Args:
        request: Indicator calculation request
        collector: YFinance collector
        engineer: Feature engineer

    Returns:
        FeaturesResponse: Calculated indicators
    """
    logger.info(f"Calculating indicators for {request.ticker}: {request.indicators}")

    try:
        # Fetch market data
        data = await collector.collect_stock_data(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
        )

        if data is None or data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for ticker {request.ticker}"
            )

        # Calculate indicators
        features = {}

        if "sma" in request.indicators:
            from backend.ml_engine.features.technical_indicators import calculate_sma

            features["sma"] = calculate_sma(data["Close"], window=request.sma_period).tolist()

        if "ema" in request.indicators:
            from backend.ml_engine.features.technical_indicators import calculate_ema

            features["ema"] = calculate_ema(data["Close"], span=request.ema_period).tolist()

        if "rsi" in request.indicators:
            from backend.ml_engine.features.technical_indicators import calculate_rsi

            features["rsi"] = calculate_rsi(data["Close"], window=request.rsi_period).tolist()

        if "macd" in request.indicators:
            from backend.ml_engine.features.technical_indicators import calculate_macd

            macd_line, signal_line, histogram = calculate_macd(
                data["Close"],
                fast=request.macd_fast,
                slow=request.macd_slow,
                signal=request.macd_signal,
            )
            features["macd"] = {
                "macd_line": macd_line.tolist(),
                "signal_line": signal_line.tolist(),
                "histogram": histogram.tolist(),
            }

        return FeaturesResponse(
            ticker=request.ticker,
            timestamp=datetime.utcnow(),
            feature_count=len(features),
            features=features,
        )

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate indicators: {str(e)}")


# ============================================================================
# Feature Store Endpoints
# ============================================================================


@router.get("/features/{ticker}/embeddings")
async def get_feature_embeddings(
    ticker: str,
    embedding_type: str = Query(
        ...,
        pattern="^(temporal|semantic|structural|fused)$",
        description="Type of embedding to retrieve",
    ),
    timestamp: datetime | None = None,
    redis: Redis = Depends(get_redis),
):
    """
    Get pre-computed feature embeddings from feature store.

    Retrieves cached embeddings from Redis feature store for fast access.

    Embedding Types:
    - temporal: TFT time-series embeddings (128-dim)
    - semantic: BERT sentiment embeddings (64-dim)
    - structural: GNN market graph embeddings (32-dim)
    - fused: Combined multi-modal embeddings (224-dim)

    Args:
        ticker: Stock ticker symbol
        embedding_type: Type of embedding
        timestamp: Optional timestamp (default: latest)
        redis: Redis connection

    Returns:
        dict: Embedding vector and metadata
    """
    logger.info(f"Fetching {embedding_type} embedding for {ticker}")

    try:
        # Build Redis key
        if timestamp:
            key = f"embedding:{embedding_type}:{ticker}:{timestamp.isoformat()}"
        else:
            key = f"embedding:{embedding_type}:{ticker}:latest"

        # Get from Redis
        embedding = redis.get(key)

        if embedding is None:
            raise HTTPException(status_code=404, detail=f"Embedding not found for {ticker}")

        # Parse embedding (assuming JSON-encoded)
        import json

        embedding_data = json.loads(embedding)

        return {
            "ticker": ticker,
            "embedding_type": embedding_type,
            "timestamp": embedding_data.get("timestamp"),
            "dimension": len(embedding_data.get("vector", [])),
            "vector": embedding_data.get("vector"),
            "metadata": embedding_data.get("metadata", {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch embedding: {str(e)}")


@router.get("/features/{ticker}/freshness")
async def check_feature_freshness(
    ticker: str,
    redis: Redis = Depends(get_redis),
):
    """
    Check freshness of cached features.

    Returns age of cached embeddings to determine if refresh is needed.

    Args:
        ticker: Stock ticker symbol
        redis: Redis connection

    Returns:
        dict: Freshness information for all embedding types
    """
    logger.info(f"Checking feature freshness for {ticker}")

    freshness = {}

    for embedding_type in ["temporal", "semantic", "structural", "fused"]:
        key = f"embedding:{embedding_type}:{ticker}:latest"

        # Check if key exists
        if redis.exists(key):
            # Get TTL
            ttl = redis.ttl(key)

            # Get timestamp from value
            try:
                import json

                data = json.loads(redis.get(key))
                created_at = datetime.fromisoformat(data.get("timestamp"))
                age_seconds = (datetime.utcnow() - created_at).total_seconds()
            except Exception:
                age_seconds = None

            freshness[embedding_type] = {
                "exists": True,
                "ttl_seconds": ttl,
                "age_seconds": age_seconds,
                "fresh": ttl > 0,
            }
        else:
            freshness[embedding_type] = {
                "exists": False,
                "ttl_seconds": None,
                "age_seconds": None,
                "fresh": False,
            }

    return {
        "ticker": ticker,
        "checked_at": datetime.utcnow().isoformat(),
        "embeddings": freshness,
    }


# ============================================================================
# Data Quality Endpoints
# ============================================================================


@router.get("/quality-check/{ticker}")
async def check_data_quality(
    ticker: str,
    start_date: datetime = Query(default=datetime.utcnow() - timedelta(days=30)),
    end_date: datetime = Query(default_factory=datetime.utcnow),
    collector: YFinanceCollector = Depends(get_yfinance_collector),
):
    """
    Check data quality for a ticker.

    Analyzes market data for:
    - Missing values
    - Outliers
    - Gaps in time series
    - Volume anomalies

    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        collector: YFinance collector

    Returns:
        dict: Data quality report
    """
    logger.info(f"Checking data quality for {ticker}")

    try:
        data = await collector.collect_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
        )

        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")

        # Check for missing values
        missing_values = data.isnull().sum().to_dict()

        # Check for zero volume days
        zero_volume_days = int((data["Volume"] == 0).sum())

        # Check for price gaps (>10% change)
        price_changes = data["Close"].pct_change()
        large_gaps = int((abs(price_changes) > 0.10).sum())

        return {
            "ticker": ticker,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_rows": len(data),
            "missing_values": missing_values,
            "zero_volume_days": zero_volume_days,
            "large_price_gaps": large_gaps,
            "quality_score": 100
            - (
                (sum(missing_values.values()) / len(data) * 50)
                + (zero_volume_days / len(data) * 25)
                + (large_gaps / len(data) * 25)
            ),
        }

    except Exception as e:
        logger.error(f"Error checking data quality: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check data quality: {str(e)}")
