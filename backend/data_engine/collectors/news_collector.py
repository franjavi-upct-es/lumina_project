# backend/data_engine/collectors/news_collector.py
"""
News collector for financial news from multiple sources
Aggregates news for sentiment analysis and event detection
"""

from typing import Optional, Dict, List
from datetime import datetime, timedelta
import polars as pl
import requests
from loguru import logger
import asyncio
import re

from backend.data_engine.collectors.base_collector import BaseDataCollector
from backend.config.settings import get_settings

settings = get_settings()


class NewsCollector(BaseDataCollector):
    """
    Collector for financial news from various sources

    Sources:
    - NewsAPI: General news aggregator
    - Finnhub: Financial news
    - Alpha Vantage News Sentiment
    - RSS feeds from financial sites
    """

    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 100):
        """
        Initialize News collector

        Args:
            api_key: NewsAPI key (defaults to settings)
            rate_limit: Max requests per minute
        """
        super().__init__(name="NewsAPI", rate_limit=rate_limit)

        self.api_key = api_key or settings.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2"

        if not self.api_key:
            logger.warning("NewsAPI key not configured")

        # Financial news sources
        self.financial_sources = [
            "bloomberg",
            "reuters",
            "the-wall-street-journal",
            "financial-times",
            "cnbc",
            "business-insider",
            "fortune",
            "the-economist",
        ]

    async def collect(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[pl.DataFrame]:
        """
        Collect news articles about a ticker/company

        Args:
            ticker: Stock ticker or company name
            start_date: Start date for news
            end_date: End date for news
            **kwargs: Additional parameters
                - language: News language (default: 'en')
                - sort_by: 'relevancy', 'popularity', or 'publishedAt'
                - page_size: Number of articles per page (max 100)

        Returns:
            DataFrame with news articles
        """
        if not self.api_key:
            logger.error("NewsAPI key not configured")
            return None

        try:
            # Set defaults
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=30)  # Last 30 days

            # Build search query
            # Include ticker and common company name variations
            query = f'"{ticker}" OR ${ticker}'

            # NewsAPI parameters
            params = {
                "q": query,
                "apiKey": self.api_key,
                "language": kwargs.get("language", "en"),
                "sortBy": kwargs.get("sort_by", "relevancy"),
                "pageSize": min(kwargs.get("page_size", 100), 100),
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
            }

            # Add sources if not specified
            if "sources" not in kwargs:
                # Use financial sources
                sources = ",".join(self.financial_sources[:20])  # API limit
                params["sources"] = sources

            # Make request
            url = f"{self.base_url}/everything"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, params=params, timeout=30)
            )

            if response.status_code != 200:
                logger.error(f"NewsAPI error: {response.status_code}")
                return None

            data = response.json()

            # Check status
            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return None

            articles = data.get("articles", [])

            if not articles:
                logger.warning(f"No articles found for {ticker}")
                return None

            # Parse articles
            records = []
            for article in articles:
                # Parse published date
                published_at = article.get("publishedAt")
                if published_at:
                    try:
                        published_dt = datetime.fromisoformat(
                            published_at.replace("Z", "+00:00")
                        )
                    except:
                        published_dt = None
                else:
                    published_dt = None

                records.append(
                    {
                        "time": published_dt,
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "content": article.get("content"),
                        "url": article.get("url"),
                        "source": article.get("source", {}).get("name"),
                        "author": article.get("author"),
                        "url_to_image": article.get("urlToImage"),
                    }
                )

            if not records:
                return None

            # Create DataFrame
            df = pl.DataFrame(records)

            # Filter out None times
            df = df.filter(pl.col("time").is_not_null())

            # Sort by time
            df = df.sort("time", descending=True)

            # Add metadata
            df = self._add_metadata(df, ticker, "newsapi")

            logger.success(f"Collected {df.height} articles for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error collecting news for {ticker}: {e}")
            return None

    async def validate_data(self, data: Optional[pl.DataFrame]) -> bool:
        """
        Validate news data
        """
        if data is None or data.height == 0:
            return False

        required_columns = ["time", "title"]
        missing = [col for col in required_columns if col not in data.columns]

        if missing:
            logger.error(f"Missing columns: {missing}")
            return False

        return True

    async def get_top_headlines(
        self,
        category: str = "business",
        country: str = "us",
        page_size: int = 20,
    ) -> Optional[pl.DataFrame]:
        """
        Get top headlines for a category

        Args:
            category: News category (business, technology, etc.)
            country: Country code (us, gb, etc.)
            page_size: Number of articles

        Returns:
            DataFrame with top headlines
        """
        if not self.api_key:
            return None

        try:
            params = {
                "category": category,
                "country": country,
                "pageSize": min(page_size, 100),
                "apiKey": self.api_key,
            }

            url = f"{self.base_url}/top-headlines"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            if data.get("status") != "ok":
                return None

            articles = data.get("articles", [])

            if not articles:
                return None

            # Parse articles
            records = []
            for article in articles:
                published_at = article.get("publishedAt")
                if published_at:
                    try:
                        published_dt = datetime.fromisoformat(
                            published_at.replace("Z", "+00:00")
                        )
                    except:
                        continue
                else:
                    continue

                records.append(
                    {
                        "time": published_dt,
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "source": article.get("source", {}).get("name"),
                        "url": article.get("url"),
                    }
                )

            if not records:
                return None

            df = pl.DataFrame(records)
            df = df.sort("time", descending=True)

            logger.success(f"Collected {df.height} top headlines")
            return df

        except Exception as e:
            logger.error(f"Error fetching top headlines: {e}")
            return None

    async def search_news(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sort_by: str = "relevancy",
        page_size: int = 50,
    ) -> Optional[pl.DataFrame]:
        """
        Search for news with custom query

        Args:
            query: Search query
            start_date: Start date
            end_date: End date
            sort_by: Sort method
            page_size: Results per page

        Returns:
            DataFrame with search results
        """
        if not self.api_key:
            return None

        try:
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=30)

            params = {
                "q": query,
                "apiKey": self.api_key,
                "sortBy": sort_by,
                "pageSize": min(page_size, 100),
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
            }

            url = f"{self.base_url}/everything"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            if data.get("status") != "ok":
                return None

            articles = data.get("articles", [])

            if not articles:
                return None

            records = []
            for article in articles:
                published_at = article.get("publishedAt")
                if published_at:
                    try:
                        published_dt = datetime.fromisoformat(
                            published_at.replace("Z", "+00:00")
                        )
                    except:
                        continue
                else:
                    continue

                records.append(
                    {
                        "time": published_dt,
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "content": article.get("content"),
                        "source": article.get("source", {}).get("name"),
                        "url": article.get("url"),
                    }
                )

            if not records:
                return None

            df = pl.DataFrame(records)
            df = df.sort("time", descending=True)

            logger.success(f"Found {df.height} articles for '{query}'")
            return df

        except Exception as e:
            logger.error(f"Error searching news: {e}")
            return None

    async def get_market_news(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Get general market news

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with market news
        """
        # Search for general market terms
        market_keywords = [
            "stock market",
            "S&P 500",
            "Dow Jones",
            "NASDAQ",
            "Federal Reserve",
            "interest rates",
        ]

        query = " OR ".join([f'"{kw}"' for kw in market_keywords])

        return await self.search_news(
            query=query,
            start_date=start_date,
            end_date=end_date,
            sort_by="publishedAt",
            page_size=100,
        )

    async def get_sector_news(
        self,
        sector: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Get news for a specific sector

        Args:
            sector: Sector name (technology, healthcare, finance, etc.)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with sector news
        """
        query = f"{sector} sector stocks"

        return await self.search_news(
            query=query,
            start_date=start_date,
            end_date=end_date,
            sort_by="relevancy",
            page_size=50,
        )

    def extract_tickers_from_text(self, text: str) -> List[str]:
        """
        Extract potential stock tickers from text

        Args:
            text: Text to search

        Returns:
            List of potential tickers
        """
        if not text:
            return []

        # Pattern for stock tickers: $SYMBOL or (SYMBOL)
        patterns = [
            r"\$([A-Z]{1,5})\b",  # $AAPL
            r"\(([A-Z]{1,5})\)",  # (AAPL)
            r"\b([A-Z]{2,5})\b",  # AAPL (more general)
        ]

        tickers = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            tickers.update(matches)

        # Filter out common words that might match
        common_words = {
            "US",
            "UK",
            "CEO",
            "CFO",
            "IPO",
            "ETF",
            "SEC",
            "FDA",
            "AI",
            "IT",
            "PR",
            "HR",
            "TV",
            "PC",
            "NY",
            "LA",
        }

        tickers = tickers - common_words

        return list(tickers)

    async def get_trending_tickers(self, days: int = 1) -> Optional[Dict[str, int]]:
        """
        Get trending tickers from recent news

        Args:
            days: Number of days to look back

        Returns:
            Dictionary mapping tickers to mention count
        """
        try:
            # Get recent market news
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            news = await self.get_market_news(
                start_date=start_date,
                end_date=end_date,
            )

            if news is None or news.height == 0:
                return None

            # Extract tickers from all articles
            ticker_counts = {}

            for row in news.iter_rows(named=True):
                text = f"{row.get('title', '')} {row.get('description', '')}"
                tickers = self.extract_tickers_from_text(text)

                for ticker in tickers:
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

            # Sort by count
            sorted_tickers = dict(
                sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
            )

            logger.success(f"Found {len(sorted_tickers)} trending tickers")
            return sorted_tickers

        except Exception as e:
            logger.error(f"Error getting trending tickers: {e}")
            return None

    async def get_earnings_news(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Optional[pl.DataFrame]:
        """
        Get earnings-related news

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with earnings news
        """
        query = (
            "earnings report OR quarterly earnings OR earnings beat OR earnings miss"
        )

        return await self.search_news(
            query=query,
            start_date=start_date,
            end_date=end_date,
            sort_by="publishedAt",
            page_size=100,
        )
