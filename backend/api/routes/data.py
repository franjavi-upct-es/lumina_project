# backend/api/routes/data.py
"""
Data endpoints for price data, features, and market information
"""

from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from loguru import logger

from data_engine.collectors.yfinance_collector import YFinanceCollector
from data_engine.transformers.feature_engineering import FeatureEngineer
from config.settings import get_settings

router = APIRouter()
settings = get_settings()


# Request/Response Models
class PriceDataResponse(BaseModel):
    ticker: str
    start_date: datetime
    end_date: datetime
    interval: str
    data_points: int
    data: List[dict]


class FeatureResponse(BaseModel):
    ticker: str
    features: List[str]
    categories: dict
    data_points: int
    data: Optional[List[dict]] = None


class CompanyInfoResponse(BaseModel):
    ticker: str
    name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    description: Optional[str]


class BatchDataRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=1, max_items=100)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    interval: str = "1d"


# Dependency for data collector
async def get_collector():
    return YFinanceCollector(rate_limit=settings.YFINANCE_RATE_LIMIT)


@router.get("/health", tags=["Health"])
async def data_health_check():
    """
    Check data collection services health
    """
    collector = YFinanceCollector()
    health = await collector.health_check()
    return health


@router.get("/{ticker}/prices", response_model=PriceDataResponse)
async def get_historical_prices(
    ticker: str,
    start_date: Optional[datetime] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[datetime] = Query(None, description="End date (YYYY-MM-DD)"),
    interval: str = Query("1d", regex="^(1d|1h|5m|15m|30m|1wk|1mo)$"),
    collector: YFinanceCollector = Depends(get_collector),
):
    """
    Get historical price data for a ticker

    **Intervals:**
    - 1d: Daily
    - 1h: Hourly
    - 5m, 15m, 30m: Intraday
    - 1wk: Weekly
    - 1mo: Monthly
    """
    try:
        # Set defaults
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")

        # Collect data
        data = await collector.collect_with_retry(
            ticker=ticker, start_date=start_date, end_date=end_date, interval=interval
        )

        if data is None or data.height == 0:
            raise HTTPException(
                status_code=404, detail=f"No data found for ticker {ticker}"
            )

        # Convert to dict for response
        data_dict = data.to_dicts()

        return PriceDataResponse(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            data_points=len(data_dict),
            data=data_dict,
        )

    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/prices")
async def get_batch_prices(
    request: BatchDataRequest, collector: YFinanceCollector = Depends(get_collector)
):
    """
    Get historical prices for multiple tickers in parallel

    **Limitations:**
    - Maximum 100 tickers per request
    - Rate limited to protect APIs
    """
    try:
        logger.info(f"Batch request for {len(request.tickers)} tickers")

        # Collect data for all tickers
        results = await collector.collect_batch(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            max_concurrent=5,
            interval=request.interval,
        )

        # Format response
        response = {
            "requested": len(request.tickers),
            "successful": len(results),
            "failed": len(request.tickers) - len(results),
            "data": {
                ticker: {"data_points": data.height, "data": data.to_dicts()}
                for ticker, data in results.items()
            },
        }

        return response

    except Exception as e:
        logger.error(f"Error in batch collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/features", response_model=FeatureResponse)
async def get_features(
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    categories: Optional[str] = Query(
        None,
        description="Comma-separated categories: price,volume,volatility,momentum,trend,statistical",
    ),
    include_data: bool = Query(False, description="Include feature values in response"),
    collector: YFinanceCollector = Depends(get_collector),
):
    """
    Get computed features for a ticker

    **Feature Categories:**
    - price: Returns, gaps, ranges
    - volume: Volume indicators, OBV, VWAP
    - volatility: ATR, Bollinger Bands, historical volatility
    - momentum: RSI, Stochastic, MACD
    - trend: Moving averages, ADX, Parabolic SAR
    - statistical: Skewness, kurtosis, z-scores
    """
    try:
        # Collect raw data
        data = await collector.collect_with_retry(
            ticker=ticker, start_date=start_date, end_date=end_date
        )

        if data is None:
            raise HTTPException(
                status_code=404, detail=f"No data found for ticker {ticker}"
            )

        # Engineer features
        fe = FeatureEngineer()
        enriched_data = fe.create_all_features(data, add_lags=True, add_rolling=True)

        # Filter by categories if specified
        if categories:
            requested_categories = [c.strip() for c in categories.split(",")]
            feature_list = []
            for category in requested_categories:
                feature_list.extend(fe.get_feature_names_by_category(category))
        else:
            feature_list = fe.get_all_feature_names()

        # Build response
        response = FeatureResponse(
            ticker=ticker,
            features=feature_list,
            categories={
                cat: fe.get_feature_names_by_category(cat)
                for cat in [
                    "price",
                    "volume",
                    "volatility",
                    "momentum",
                    "trend",
                    "statistical",
                ]
            },
            data_points=enriched_data.height,
            data=enriched_data.to_dicts() if include_data else None,
        )

        return response

    except Exception as e:
        logger.error(f"Error computing features for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/info", response_model=CompanyInfoResponse)
async def get_company_info(
    ticker: str, collector: YFinanceCollector = Depends(get_collector)
):
    """
    Get company information and fundamental metrics
    """
    try:
        info = await collector.get_company_info(ticker)

        if info is None:
            raise HTTPException(
                status_code=404, detail=f"Company information not found for {ticker}"
            )

        return CompanyInfoResponse(**info)

    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/options")
async def get_options_chain(
    ticker: str,
    expiration_date: Optional[str] = Query(
        None, description="Expiration date (YYYY-MM-DD)"
    ),
    collector: YFinanceCollector = Depends(get_collector),
):
    """
    Get options chain data for a ticker
    """
    try:
        options_data = await collector.get_options_data(ticker, expiration_date)

        if options_data is None:
            raise HTTPException(
                status_code=404, detail=f"Options data not found for {ticker}"
            )

        return {
            "ticker": ticker,
            "expiration": options_data["expiration"],
            "calls": options_data["calls"].to_dicts(),
            "puts": options_data["puts"].to_dicts(),
        }

    except Exception as e:
        logger.error(f"Error fetching options for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/institutional-holders")
async def get_institutional_holders(
    ticker: str, collector: YFinanceCollector = Depends(get_collector)
):
    """
    Get institutional holders information
    """
    try:
        holders = await collector.get_institutional_holders(ticker)

        if holders is None:
            raise HTTPException(
                status_code=404,
                detail=f"Institutional holders data not found for {ticker}",
            )

        return {"ticker": ticker, "holders": holders.to_dicts()}

    except Exception as e:
        logger.error(f"Error fetching institutional holders for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/earnings")
async def get_earnings_history(
    ticker: str, collector: YFinanceCollector = Depends(get_collector)
):
    """
    Get historical earnings data
    """
    try:
        earnings = await collector.get_earnings_history(ticker)

        if earnings is None:
            raise HTTPException(
                status_code=404, detail=f"Earnings data not found for {ticker}"
            )

        return {"ticker": ticker, "earnings": earnings.to_dicts()}

    except Exception as e:
        logger.error(f"Error fetching earnings for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/status")
async def get_market_status():
    """
    Get current market status
    """
    try:
        now = datetime.now()

        # Simple market hours check (NYSE hours in UTC)
        # 9:30 AM - 4:00 PM EST = 14:30 - 21:00 UTC
        is_weekday = now.weekday() < 5
        hour_utc = now.hour
        is_market_hours = 14 <= hour_utc < 21

        status = {
            "is_open": is_weekday and is_market_hours,
            "current_time": now.isoformat(),
            "timezone": "UTC",
            "next_open": "Market hours: 9:30 AM - 4:00 PM EST (Mon-Fri)",
            "message": "Market is open"
            if (is_weekday and is_market_hours)
            else "Market is closed",
        }

        return status

    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
