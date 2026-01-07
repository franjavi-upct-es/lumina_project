# backend/data_engine/collectors/reddit_collector.py
"""
Reddit collector for social sentiment from financial subreddits
Tracks mentions, sentiment, and discussions about stocks
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import polars as pl
import praw
from loguru import logger
import asyncio
import re

from backend.data_engine.collectors.base_collector import BaseDataCollector
from backend.config.settings import get_settings

settings = get_settings()


class RedditCollector(BaseDataCollector):
    """
    Collector for Reddit data from financial subreddits

    Popular subreddits:
    - r/wallstreetbets: Retail trading discussions
    - r/stocks: General stock market
    - r/investing: Long-term investing
    - r/options: Options trading
    - r/SecurityAnalysis: Fundamental analysis
    - r/StockMarket: Market news and discussion
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "LuminaQuantLab/2.0",
        rate_limit: int = 60,
    ):
        """
        Initialize Reddit collector

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string
            rate_limit: Max requests per minute
        """
        super().__init__(name="Reddit", rate_limit=rate_limit)

        self.client_id = client_id or settings.REDDIT_CLIENT_ID
        self.client_secret = client_secret or settings.REDDIT_CLIENT_SECRET
        self.user_agent = user_agent

        self.reddit = None

        if self.client_id and self.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
                # Test connection
                self.reddit.user.me()
                logger.success("Reddit API authenticated successfully")
            except Exception as e:
                logger.warning(f"Reddit authentication failed: {e}")
                self.reddit = None
        else:
            logger.warning("Reddit API credentials not configured")

        # Popular financial subreddits
        self.financial_subreddits = [
            "wallstreetbets",
            "stocks",
            "investing",
            "options",
            "SecurityAnalysis",
            "StockMarket",
            "pennystocks",
            "Daytrading",
            "Forex",
            "CryptoCurrency",
        ]

    async def collect(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[pl.DataFrame]:
        """
        Collect Reddit posts and comments mentioning a ticker

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (not strictly enforced by Reddit API)
            end_date: End date
            **kwargs: Additional parameters
                - subreddits: List of subreddits to search (default: financial_subreddits)
                - limit: Max posts per subreddit (default: 100)
                - sort: 'hot', 'new', 'top', 'rising' (default: 'hot')
                - time_filter: 'hour', 'day', 'week', 'month', 'year', 'all' (for 'top')

        Returns:
            DataFrame with Reddit posts/comments
        """
        if not self.reddit:
            logger.error("Reddit API not configured")
            return None

        try:
            subreddits = kwargs.get("subreddits", self.financial_subreddits)
            limit = kwargs.get("limit", 100)
            sort = kwargs.get("sort", "hot")
            time_filter = kwargs.get("time_filter", "week")

            # Search query - Reddit doesn't support cashtags directly
            query = f"${ticker} OR {ticker}"

            logger.info(f"Searching Reddit for {ticker} in {len(subreddits)} subreddits")

            # Collect posts from all subreddits
            all_posts = []

            loop = asyncio.get_event_loop()

            for subreddit_name in subreddits:
                try:
                    posts = await loop.run_in_executor(
                        None,
                        lambda: self._search_subreddit(
                            subreddit_name, query, limit, sort, time_filter
                        ),
                    )
                    all_posts.extend(posts)
                except Exception as e:
                    logger.warning(f"Error searching r/{subreddit_name}: {e}")
                    continue

            if not all_posts:
                logger.warning(f"No Reddit posts found for {ticker}")
                return None

            # Create DataFrame
            df = pl.DataFrame(all_posts)

            # Filter by date if provided
            if start_date:
                df = df.filter(pl.col("time") >= start_date)
            if end_date:
                df = df.filter(pl.col("time") <= end_date)

            # Sort by time
            df = df.sort("time", descending=True)

            # Add metadata
            df = self._add_metadata(df, ticker, "reddit")

            logger.success(f"Collected {df.height} Reddit posts for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error collecting Reddit data for {ticker}: {e}")
            return None

    def _search_subreddit(
        self,
        subreddit_name: str,
        query: str,
        limit: int,
        sort: str,
        time_filter: str,
    ) -> List[Dict[str, Any]]:
        """
        Search a specific subreddit (synchronous)
        """
        posts = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Get posts based on sort method
            if sort == "top":
                submissions = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort == "new":
                submissions = subreddit.new(limit=limit)
            elif sort == "rising":
                submissions = subreddit.rising(limit=limit)
            else:  # hot
                submissions = subreddit.hot(limit=limit)

            # Search through submissions
            for submission in submissions:
                # Check if query matches title or selftext
                if (
                    query.lower() in submission.title.lower()
                    or query.lower() in submission.selftext.lower()
                ):
                    posts.append(
                        {
                            "time": datetime.fromtimestamp(submission.created_utc),
                            "subreddit": subreddit_name,
                            "post_id": submission.id,
                            "title": submission.title,
                            "text": submission.selftext,
                            "author": str(submission.author) if submission.author else "[deleted]",
                            "score": submission.score,
                            "upvote_ratio": submission.upvote_ratio,
                            "num_comments": submission.num_comments,
                            "url": f"https://reddit.com{submission.permalink}",
                            "flair": submission.link_flair_text,
                            "is_self": submission.is_self,
                            "type": "post",
                        }
                    )

        except Exception as e:
            logger.error(f"Error searching r/{subreddit_name}: {e}")

        return posts

    async def validate_data(self, data: Optional[pl.DataFrame]) -> bool:
        """
        Validate Reddit data
        """
        if data is None or data.height == 0:
            return False

        required_columns = ["time", "title", "subreddit"]
        missing = [col for col in required_columns if col not in data.columns]

        if missing:
            logger.error(f"Missing columns: {missing}")
            return False

        return True

    async def get_trending_tickers(
        self,
        subreddit: str = "wallstreetbets",
        limit: int = 100,
        time_filter: str = "day",
    ) -> Optional[Dict[str, int]]:
        """
        Get trending tickers from a subreddit

        Args:
            subreddit: Subreddit name
            limit: Number of posts to analyze
            time_filter: Time period

        Returns:
            Dictionary mapping tickers to mention count
        """
        if not self.reddit:
            return None

        try:
            logger.info(f"Finding trending tickers in r/{subreddit}")

            loop = asyncio.get_event_loop()
            posts = await loop.run_in_executor(
                None, lambda: self._get_top_posts(subreddit, limit, time_filter)
            )

            if not posts:
                return None

            # Extract tickers from all posts
            ticker_counts = {}

            for post in posts:
                text = f"{post['title']} {post['text']}"
                tickers = self._extract_tickers(text)

                for ticker in tickers:
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

            # Sort by count
            sorted_tickers = dict(sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True))

            logger.success(f"Found {len(sorted_tickers)} trending tickers")
            return sorted_tickers

        except Exception as e:
            logger.error(f"Error getting trending tickers: {e}")
            return None

    def _get_top_posts(
        self, subreddit_name: str, limit: int, time_filter: str
    ) -> List[Dict[str, Any]]:
        """
        Get top posts from subreddit (synchronous)
        """
        posts = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            submissions = subreddit.top(time_filter=time_filter, limit=limit)

            for submission in submissions:
                posts.append(
                    {
                        "time": datetime.fromtimestamp(submission.created_utc),
                        "title": submission.title,
                        "text": submission.selftext,
                        "score": submission.score,
                    }
                )

        except Exception as e:
            logger.error(f"Error getting top posts: {e}")

        return posts

    def _extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text

        Args:
            text: Text to search

        Returns:
            List of tickers found
        """
        if not text:
            return []

        # Patterns for tickers
        patterns = [
            r"\$([A-Z]{1,5})\b",  # $AAPL
            r"\b([A-Z]{2,5})\b",  # AAPL
        ]

        tickers = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            tickers.update(matches)

        # Filter out common words and invalid tickers
        blacklist = {
            "THE",
            "AND",
            "FOR",
            "ARE",
            "BUT",
            "NOT",
            "YOU",
            "ALL",
            "CAN",
            "HER",
            "WAS",
            "ONE",
            "OUR",
            "OUT",
            "DAY",
            "GET",
            "HAS",
            "HIM",
            "HIS",
            "HOW",
            "ITS",
            "MAY",
            "NEW",
            "NOW",
            "OLD",
            "SEE",
            "TWO",
            "WAY",
            "WHO",
            "BOY",
            "DID",
            "ITS",
            "LET",
            "PUT",
            "SAY",
            "SHE",
            "TOO",
            "USE",
            "CEO",
            "CFO",
            "COO",
            "CTO",
            "ETF",
            "IPO",
            "SEC",
            "FDA",
            "FBI",
            "CIA",
            "IRS",
            "USA",
            "API",
            "ATH",
            "ATL",
            "GDP",
            "CPI",
            "FAQ",
            "IMO",
            "TBH",
            "FYI",
            "IMHO",
            "LOL",
            "YOLO",
            "DD",
            "TA",
            "WSB",
            "GME",
            "AMC",  # Remove if you want to track these
        }

        tickers = tickers - blacklist

        # Filter by length (valid tickers are 1-5 chars)
        tickers = {t for t in tickers if 1 <= len(t) <= 5}

        return list(tickers)

    async def get_subreddit_sentiment(
        self,
        subreddit: str,
        limit: int = 100,
        time_filter: str = "day",
    ) -> Optional[pl.DataFrame]:
        """
        Get posts from a subreddit for sentiment analysis

        Args:
            subreddit: Subreddit name
            limit: Number of posts
            time_filter: Time period

        Returns:
            DataFrame with posts
        """
        if not self.reddit:
            return None

        try:
            loop = asyncio.get_event_loop()
            posts = await loop.run_in_executor(
                None, lambda: self._get_subreddit_posts(subreddit, limit, time_filter)
            )

            if not posts:
                return None

            df = pl.DataFrame(posts)
            df = df.sort("time", descending=True)

            logger.success(f"Collected {df.height} posts from r/{subreddit}")
            return df

        except Exception as e:
            logger.error(f"Error collecting subreddit sentiment: {e}")
            return None

    def _get_subreddit_posts(
        self, subreddit_name: str, limit: int, time_filter: str
    ) -> List[Dict[str, Any]]:
        """
        Get posts from subreddit (synchronous)
        """
        posts = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            submissions = subreddit.top(time_filter=time_filter, limit=limit)

            for submission in submissions:
                posts.append(
                    {
                        "time": datetime.fromtimestamp(submission.created_utc),
                        "subreddit": subreddit_name,
                        "title": submission.title,
                        "text": submission.selftext,
                        "score": submission.score,
                        "upvote_ratio": submission.upvote_ratio,
                        "num_comments": submission.num_comments,
                        "author": str(submission.author) if submission.author else "[deleted]",
                    }
                )

        except Exception as e:
            logger.error(f"Error getting subreddit posts: {e}")

        return posts

    async def get_ticker_sentiment_summary(
        self,
        ticker: str,
        days: int = 7,
    ) -> Optional[Dict[str, Any]]:
        """
        Get sentiment summary for a ticker

        Args:
            ticker: Stock ticker
            days: Number of days to look back

        Returns:
            Summary statistics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Collect Reddit data
            data = await self.collect_with_retry(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                limit=200,
                sort="new",
            )

            if data is None or data.height == 0:
                return None

            # Calculate summary stats
            total_posts = data.height
            total_score = data["score"].sum()
            avg_score = data["score"].mean()
            avg_comments = data["num_comments"].mean()

            # Count mentions per day
            mentions_per_day = (
                data.group_by_dynamic("time", every="1d").agg(pl.count()).sort("time")
            )

            # Get top subreddits
            top_subreddits = (
                data.group_by("subreddit")
                .agg(pl.count().alias("count"))
                .sort("count", descending=True)
                .head(5)
            )

            summary = {
                "ticker": ticker,
                "period_days": days,
                "total_mentions": total_posts,
                "total_score": float(total_score),
                "avg_score": float(avg_score),
                "avg_comments": float(avg_comments),
                "mentions_per_day": mentions_per_day.to_dicts(),
                "top_subreddits": top_subreddits.to_dicts(),
            }

            logger.success(f"Generated sentiment summary for {ticker}")
            return summary

        except Exception as e:
            logger.error(f"Error generating sentiment summary: {e}")
            return None
