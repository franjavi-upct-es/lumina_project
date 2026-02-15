# backend/data_engine/collectors/__init__.py
"""
Data Collectors Module for Lumina V3
====================================

Provides various data collection interfaces for different data sources.

Market Data (Maintained from V2):
- YFinanceCollector: Free historical stock data from Yahoo Finance
- AlphaVantageCollector: Premium market data with fundamentals

Economic Data (Maintained from V2):
- FREDCollector: Federal Reserve Economic Data

News & Sentiment (Maintained from V2):
- NewsCollector: Financial news aggregation
- RedditCollector: Reddit social sentiment

New V3 Collectors:
- PriceStreamCollector: WebSocket real-time price streaming
- NewsStreamCollector: WebSocket news streaming
- SocialScrapperCollector: Unified Twitter/Reddit scraping
- ChainScrapperCollector: On-chain blockchain data

All collectors inherit from BaseDataCollector and implement:
- async collect(): Main data collection method
- async validate_data(): Data quality validation
- async collect_with_retry(): Retry mechanism with rate limiting

Usage:
    from backend.data_engine.collectors import YFinanceCollector

    collector = YFinanceCollector(rate_limit=100)
    data = await collector.collect_with_retry(
        ticker="AAPL",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1)
    )

Version: 3.0.0
"""

from backend.data_engine.collectors.alpha_vantage import AlphaVantageCollector
from backend.data_engine.collectors.base_collector import BaseDataCollector
from backend.data_engine.collectors.fred_collector import FredCollector as FREDCollector
from backend.data_engine.collectors.news_collector import NewsCollector
from backend.data_engine.collectors.reddit_collector import RedditCollector
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector

__all__ = [
    "BaseDataCollector",
    "YFinanceCollector",
    "AlphaVantageCollector",
    "FREDCollector",
    "NewsCollector",
    "RedditCollector",
]
