# backend/nlp_engine/twitter_scraper.py
"""
Twitter/X Scraper for Financial Sentiment Analysis

This module provides functionality to collect and analyze tweets
related to financial markets, stocks, and crypto for sentiment analysis.
Uses Tweepy for Twitter API v2 integration.

Key Features:
- Collect tweets mentioning specific tickers (cashtags)
- Track financial influencers and accounts
- Real-time stream support for live data
- Rate limiting and pagination handling
- Async support for high-throughput collection

Note: Requires Twitter API v2 access (Basic or higher tier)
"""

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import polars as pl
import tweepy
from loguru import logger

from backend.config.settings import get_settings

settings = get_settings()


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================


class TweetType(Enum):
    """Types of tweets"""

    ORIGINAL = "original"
    REPLY = "reply"
    RETWEET = "retweet"
    QUOTE = "quote"


class SearchMode(Enum):
    """Twitter search modes"""

    RECENT = "recent"  # Last 7 days (Basic tier)
    ALL = "all"  # Full archive (Academic/Enterprise tier)


@dataclass
class Tweet:
    """Data class representing a Tweet"""

    id: str
    text: str
    author_id: str
    author_username: str
    author_name: str
    author_followers: int
    author_verified: bool
    created_at: datetime
    retweet_count: int
    like_count: int
    reply_count: int
    quote_count: int
    impression_count: int
    tweet_type: TweetType
    language: str
    source: str
    conversation_id: str | None
    in_reply_to_user_id: str | None
    referenced_tweets: list[dict[str, str]]
    tickers_mentioned: list[str]
    hashtags: list[str]
    urls: list[str]
    context_annotations: list[dict[str, Any]]


@dataclass
class TwitterUser:
    """Data class representing a Twitter user"""

    id: str
    username: str
    name: str
    description: str
    followers_count: int
    following_count: int
    tweet_count: int
    verified: bool
    created_at: datetime
    profile_image_url: str | None
    url: str | None


@dataclass
class StreamRule:
    """Data class for Twitter stream rules"""

    value: str
    tag: str
    id: str | None = None


# ============================================================================
# FINANCIAL TWITTER ACCOUNTS
# ============================================================================


# Influential financial Twitter accounts to track
FINANCIAL_ACCOUNTS: dict[str, str] = {
    # News Organizations
    "WSJmarkets": "Wall Street Journal Markets",
    "markets": "Bloomberg Markets",
    "ReutersBiz": "Reuters Business",
    "CNBC": "CNBC",
    "FT": "Financial Times",
    "TheEconomist": "The Economist",
    "YahooFinance": "Yahoo Finance",
    # Analysts & Investors
    "jimcramer": "Jim Cramer",
    "Carl_C_Icahn": "Carl Icahn",
    "RessCapital": "Resource Capital",
    "zaborprime": "Zack Morris",
    # Fed & Government
    "federalreserve": "Federal Reserve",
    "SECGov": "SEC",
    "ABORFIELD": "SEC Chair",
    "USTreasury": "US Treasury",
    # Crypto
    "caborfield": "Crypto News",
    "whale_alert": "Whale Alert",
    "CoinDesk": "CoinDesk",
    "Cointelegraph": "Cointelegraph",
}


# Words to exclude when extracting tickers
TICKER_BLACKLIST: set[str] = {
    "THE",
    "AND",
    "FOR",
    "ARE",
    "BUT",
    "NOT",
    "YOU",
    "ALL",
    "CAN",
    "CEO",
    "CFO",
    "COO",
    "CTO",
    "IPO",
    "SEC",
    "FDA",
    "FBI",
    "CIA",
    "USA",
    "UK",
    "EU",
    "API",
    "ATH",
    "ATL",
    "GDP",
    "CPI",
    "NFT",
    "ETF",
    "ETN",
    "SPAC",
    "REIT",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "BTC",
    "ETH",
    "SOL",
    "ADA",
    "DOT",  # Keep or remove crypto based on needs
    "DM",
    "RT",
    "PM",
    "AM",
    "IMO",
    "FYI",
    "TBH",
    "LOL",
    "NEWS",
    "LIVE",
    "NOW",
    "NEW",
    "HOT",
    "TOP",
    "BIG",
    "LOW",
}


# ============================================================================
# TWITTER SCRAPER CLASS
# ============================================================================


class TwitterScraper:
    """
    Scraper for collecting and analyzing financial tweets

    This class provides methods to:
    - Search for tweets mentioning specific tickers
    - Track influential financial accounts
    - Stream real-time tweets
    - Extract sentiment signals from tweet metrics

    Example:
        ```python
        scraper = TwitterScraper()

        # Search for ticker mentions
        tweets = await scraper.search_ticker(
            ticker="AAPL",
            max_results=100
        )

        # Get tweets from financial influencers
        influencer_tweets = await scraper.get_influencer_tweets(
            usernames=["jimcramer", "WSJmarkets"],
            max_results=50
        )
        ```

    Note: Requires Twitter API v2 credentials with appropriate access level.
    """

    def __init__(
        self,
        bearer_token: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        access_token: str | None = None,
        access_secret: str | None = None,
        rate_limit_delay: float = 1.0,
    ):
        """
        Initialize Twitter scraper

        Args:
            bearer_token: Twitter API v2 bearer token (from environment if not provided)
            api_key: Twitter API key (for user context)
            api_secret: Twitter API secret
            access_token: User access token
            access_secret: User access secret
            rate_limit_delay: Delay between API requests in seconds
        """
        # Get credentials from settings or parameters
        self.bearer_token = bearer_token or getattr(settings, "TWITTER_BEARER_TOKEN", None)
        self.api_key = api_key or getattr(settings, "TWITTER_API_KEY", None)
        self.api_secret = api_secret or getattr(settings, "TWITTER_API_SECRET", None)
        self.access_token = access_token or getattr(settings, "TWITTER_ACCESS_TOKEN", None)
        self.access_secret = access_secret or getattr(settings, "TWITTER_ACCESS_SECRET", None)

        self.rate_limit_delay = rate_limit_delay

        self.client: tweepy.Client | None = None
        self._initialized = False

        # Initialize connection
        self._initialize_connection()

    def _initialize_connection(self) -> bool:
        """
        Initialize connection to Twitter API

        Returns:
            True if connection successful, False otherwise
        """
        if not self.bearer_token:
            logger.warning(
                "Twitter API credentials not configured. "
                "Set TWITTER_BEARER_TOKEN environment variable."
            )
            return False

        try:
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_secret,
                wait_on_rate_limit=True,
            )

            # Test connection by getting own user (if authenticated)
            # For app-only auth, we'll just verify the client was created
            self._initialized = True
            logger.success("Twitter API connection initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Twitter API connection: {e}")
            self.client = None
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if scraper is properly initialized"""
        return self._initialized and self.client is not None

    # ========================================================================
    # TICKER EXTRACTION
    # ========================================================================

    def extract_tickers(self, text: str) -> list[str]:
        """
        Extract stock ticker symbols (cashtags) from tweet text

        Args:
            text: Tweet text to search

        Returns:
            List of unique ticker symbols found
        """
        if not text:
            return []

        tickers: set[str] = set()

        # Pattern: Cashtag format ($AAPL)
        cashtag_pattern = r"\$([A-Z]{1,5})\b"
        matches = re.findall(cashtag_pattern, text.upper())
        tickers.update(matches)

        # Remove blacklisted words
        tickers -= TICKER_BLACKLIST

        # Filter by valid length
        tickers = {t for t in tickers if 1 <= len(t) <= 5}

        return sorted(list(tickers))

    def extract_hashtags(self, text: str) -> list[str]:
        """
        Extract hashtags from tweet text

        Args:
            text: Tweet text

        Returns:
            List of hashtags (without #)
        """
        if not text:
            return []

        pattern = r"#(\w+)"
        matches = re.findall(pattern, text)
        return matches

    # ========================================================================
    # TWEET SEARCH
    # ========================================================================

    async def search_ticker(
        self,
        ticker: str,
        max_results: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        search_mode: SearchMode = SearchMode.RECENT,
        language: str = "en",
    ) -> list[Tweet]:
        """
        Search for tweets mentioning a specific ticker

        Args:
            ticker: Stock ticker symbol (with or without $)
            max_results: Maximum number of tweets to return
            start_time: Start of search window (defaults to 7 days ago)
            end_time: End of search window (defaults to now)
            search_mode: Search recent tweets or full archive
            language: Tweet language filter

        Returns:
            List of Tweet objects
        """
        if not self.is_initialized:
            logger.error("Twitter scraper not initialized")
            return []

        # Build query - search for cashtag
        ticker_clean = ticker.upper().lstrip("$")
        query = f"${ticker_clean} lang:{language} -is:retweet"

        # Set time window (default: last 7 days for Basic tier)
        if end_time is None:
            from datetime import UTC

            end_time = datetime.now(tz=UTC)
        if start_time is None:
            start_time = end_time - timedelta(days=7)

        logger.info(f"Searching Twitter for ${ticker_clean}")

        tweets = await self._search_tweets(
            query=query,
            max_results=max_results,
            start_time=start_time,
            end_time=end_time,
        )

        return tweets

    async def search_multiple_tickers(
        self,
        tickers: list[str],
        max_results_per_ticker: int = 50,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, list[Tweet]]:
        """
        Search for tweets mentioning multiple tickers

        Args:
            tickers: List of ticker symbols
            max_results_per_ticker: Max tweets per ticker
            start_time: Start of search window
            end_time: End of search window

        Returns:
            Dictionary mapping tickers to their tweets
        """
        results: dict[str, list[Tweet]] = {}

        for ticker in tickers:
            try:
                tweets = await self.search_ticker(
                    ticker=ticker,
                    max_results=max_results_per_ticker,
                    start_time=start_time,
                    end_time=end_time,
                )
                results[ticker] = tweets

                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.warning(f"Error searching for ${ticker}: {e}")
                results[ticker] = []

        return results

    async def search_query(
        self,
        query: str,
        max_results: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Tweet]:
        """
        Search for tweets with a custom query

        Args:
            query: Twitter search query (supports operators)
            max_results: Maximum tweets to return
            start_time: Start of search window
            end_time: End of search window

        Returns:
            List of Tweet objects
        """
        if not self.is_initialized:
            logger.error("Twitter scraper not initialized")
            return []

        if end_time is None:
            end_time = datetime.now(tz=timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(days=7)

        return await self._search_tweets(
            query=query,
            max_results=max_results,
            start_time=start_time,
            end_time=end_time,
        )

    async def _search_tweets(
        self,
        query: str,
        max_results: int,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Tweet]:
        """
        Internal method to search tweets with pagination
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._search_tweets_sync(query, max_results, start_time, end_time)
        )

    def _search_tweets_sync(
        self,
        query: str,
        max_results: int,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Tweet]:
        """
        Synchronous tweet search with pagination
        """
        tweets: list[Tweet] = []

        try:
            # Tweet fields to request
            tweet_fields = [
                "id",
                "text",
                "author_id",
                "created_at",
                "public_metrics",
                "referenced_tweets",
                "conversation_id",
                "in_reply_to_user_id",
                "lang",
                "source",
                "context_annotations",
                "entities",
            ]

            # User fields to request
            user_fields = [
                "id",
                "username",
                "name",
                "verified",
                "public_metrics",
                "description",
                "created_at",
                "profile_image_url",
            ]

            # Expansions
            expansions = ["author_id", "referenced_tweets.id"]

            # Paginate through results
            collected = 0
            next_token = None

            while collected < max_results:
                # Calculate how many to fetch this iteration
                batch_size = min(100, max_results - collected)

                response = self.client.search_recent_tweets(
                    query=query,
                    max_results=batch_size,
                    start_time=start_time,
                    end_time=end_time,
                    tweet_fields=tweet_fields,
                    user_fields=user_fields,
                    expansions=expansions,
                    next_token=next_token,
                )

                if not response.data:
                    break

                # Build user lookup
                users_lookup = {}
                if response.includes and "users" in response.includes:
                    for user in response.includes["users"]:
                        users_lookup[user.id] = user

                # Parse tweets
                for tweet_data in response.data:
                    tweet = self._parse_tweet(tweet_data, users_lookup)
                    tweets.append(tweet)
                    collected += 1

                # Check for more pages
                if response.meta and "next_token" in response.meta:
                    next_token = response.meta["next_token"]
                else:
                    break

        except tweepy.errors.TooManyRequests:
            logger.warning("Twitter rate limit reached, waiting...")
            # tweepy handles rate limiting with wait_on_rate_limit=True
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")

        logger.info(f"Collected {len(tweets)} tweets")
        return tweets

    def _parse_tweet(
        self,
        tweet_data: tweepy.Tweet,
        users_lookup: dict[str, Any],
    ) -> Tweet:
        """
        Parse tweepy Tweet object into our Tweet dataclass
        """
        # Get author info
        author = users_lookup.get(tweet_data.author_id)
        author_username = author.username if author else "unknown"
        author_name = author.name if author else "Unknown"
        author_followers = (
            author.public_metrics.get("followers_count", 0)
            if author and hasattr(author, "public_metrics")
            else 0
        )
        author_verified = author.verified if author else False

        # Get metrics
        metrics = tweet_data.public_metrics or {}

        # Determine tweet type
        tweet_type = TweetType.ORIGINAL
        referenced_tweets = []
        if tweet_data.referenced_tweets:
            for ref in tweet_data.referenced_tweets:
                referenced_tweets.append(
                    {
                        "type": ref.type,
                        "id": ref.id,
                    }
                )
                if ref.type == "replied_to":
                    tweet_type = TweetType.REPLY
                elif ref.type == "retweeted":
                    tweet_type = TweetType.RETWEET
                elif ref.type == "quoted":
                    tweet_type = TweetType.QUOTE

        # Extract tickers and hashtags
        tickers = self.extract_tickers(tweet_data.text)
        hashtags = self.extract_hashtags(tweet_data.text)

        # Extract URLs from entities
        urls = []
        if hasattr(tweet_data, "entities") and tweet_data.entities:
            if "urls" in tweet_data.entities:
                urls = [
                    u.get("expanded_url", u.get("url", "")) for u in tweet_data.entities["urls"]
                ]

        # Context annotations
        context_annotations = []
        if hasattr(tweet_data, "context_annotations") and tweet_data.context_annotations:
            context_annotations = tweet_data.context_annotations

        return Tweet(
            id=str(tweet_data.id),
            text=tweet_data.text,
            author_id=str(tweet_data.author_id),
            author_username=author_username,
            author_name=author_name,
            author_followers=author_followers,
            author_verified=author_verified,
            created_at=tweet_data.created_at,
            retweet_count=metrics.get("retweet_count", 0),
            like_count=metrics.get("like_count", 0),
            reply_count=metrics.get("reply_count", 0),
            quote_count=metrics.get("quote_count", 0),
            impression_count=metrics.get("impression_count", 0),
            tweet_type=tweet_type,
            language=tweet_data.lang or "en",
            source=tweet_data.source or "unknown",
            conversation_id=str(tweet_data.conversation_id) if tweet_data.conversation_id else None,
            in_reply_to_user_id=str(tweet_data.in_reply_to_user_id)
            if tweet_data.in_reply_to_user_id
            else None,
            referenced_tweets=referenced_tweets,
            tickers_mentioned=tickers,
            hashtags=hashtags,
            urls=urls,
            context_annotations=context_annotations,
        )

    # ========================================================================
    # USER TIMELINE
    # ========================================================================

    async def get_user_tweets(
        self,
        username: str,
        max_results: int = 100,
        exclude_replies: bool = True,
        exclude_retweets: bool = True,
    ) -> list[Tweet]:
        """
        Get recent tweets from a specific user

        Args:
            username: Twitter username (without @)
            max_results: Maximum tweets to return
            exclude_replies: Whether to exclude replies
            exclude_retweets: Whether to exclude retweets

        Returns:
            List of Tweet objects
        """
        if not self.is_initialized:
            logger.error("Twitter scraper not initialized")
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._get_user_tweets_sync(
                username, max_results, exclude_replies, exclude_retweets
            ),
        )

    def _get_user_tweets_sync(
        self,
        username: str,
        max_results: int,
        exclude_replies: bool,
        exclude_retweets: bool,
    ) -> list[Tweet]:
        """
        Get user tweets synchronously
        """
        tweets: list[Tweet] = []

        try:
            # Get user ID from username
            user_response = self.client.get_user(
                username=username,
                user_fields=["id", "username", "name", "verified", "public_metrics"],
            )

            if not user_response.data:
                logger.warning(f"User @{username} not found")
                return []

            user_id = user_response.data.id

            # Build exclude list
            exclude = []
            if exclude_replies:
                exclude.append("replies")
            if exclude_retweets:
                exclude.append("retweets")

            # Get tweets
            tweet_fields = [
                "id",
                "text",
                "author_id",
                "created_at",
                "public_metrics",
                "referenced_tweets",
                "conversation_id",
                "lang",
                "source",
            ]

            response = self.client.get_users_tweets(
                id=user_id,
                max_results=min(max_results, 100),
                tweet_fields=tweet_fields,
                exclude=exclude if exclude else None,
            )

            if not response.data:
                return []

            # Build user lookup with single user
            users_lookup = {str(user_response.data.id): user_response.data}

            for tweet_data in response.data:
                tweet = self._parse_tweet(tweet_data, users_lookup)
                tweets.append(tweet)

        except Exception as e:
            logger.error(f"Error getting tweets for @{username}: {e}")

        return tweets

    async def get_influencer_tweets(
        self,
        usernames: list[str] | None = None,
        max_results_per_user: int = 20,
    ) -> dict[str, list[Tweet]]:
        """
        Get tweets from financial influencers

        Args:
            usernames: List of usernames (defaults to FINANCIAL_ACCOUNTS)
            max_results_per_user: Max tweets per user

        Returns:
            Dictionary mapping usernames to their tweets
        """
        if usernames is None:
            usernames = list(FINANCIAL_ACCOUNTS.keys())

        results: dict[str, list[Tweet]] = {}

        for username in usernames:
            try:
                tweets = await self.get_user_tweets(
                    username=username,
                    max_results=max_results_per_user,
                )
                results[username] = tweets

                await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.warning(f"Error getting tweets from @{username}: {e}")
                results[username] = []

        return results

    # ========================================================================
    # USER LOOKUP
    # ========================================================================

    async def get_user(self, username: str) -> TwitterUser | None:
        """
        Get user information

        Args:
            username: Twitter username (without @)

        Returns:
            TwitterUser object or None if not found
        """
        if not self.is_initialized:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._get_user_sync(username))

    def _get_user_sync(self, username: str) -> TwitterUser | None:
        """
        Get user synchronously
        """
        try:
            response = self.client.get_user(
                username=username,
                user_fields=[
                    "id",
                    "username",
                    "name",
                    "description",
                    "verified",
                    "public_metrics",
                    "created_at",
                    "profile_image_url",
                    "url",
                ],
            )

            if not response.data:
                return None

            user = response.data
            metrics = user.public_metrics or {}

            return TwitterUser(
                id=str(user.id),
                username=user.username,
                name=user.name,
                description=user.description or "",
                followers_count=metrics.get("followers_count", 0),
                following_count=metrics.get("following_count", 0),
                tweet_count=metrics.get("tweet_count", 0),
                verified=user.verified or False,
                created_at=user.created_at,
                profile_image_url=user.profile_image_url,
                url=user.url,
            )

        except Exception as e:
            logger.error(f"Error getting user @{username}: {e}")
            return None

    # ========================================================================
    # TRENDING ANALYSIS
    # ========================================================================

    async def get_trending_tickers(
        self,
        base_query: str = "stock OR stocks OR trading -is:retweet lang:en",
        max_tweets: int = 500,
        min_mentions: int = 3,
        top_n: int = 20,
    ) -> dict[str, dict[str, Any]]:
        """
        Analyze tweets to find trending tickers

        Args:
            base_query: Base search query
            max_tweets: Tweets to analyze
            min_mentions: Minimum mentions to include
            top_n: Top N tickers to return

        Returns:
            Dictionary of trending tickers with metrics
        """
        if not self.is_initialized:
            return {}

        # Search for financial tweets
        tweets = await self.search_query(
            query=base_query,
            max_results=max_tweets,
        )

        if not tweets:
            return {}

        # Aggregate ticker mentions
        ticker_data: dict[str, dict[str, Any]] = {}

        for tweet in tweets:
            for ticker in tweet.tickers_mentioned:
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        "mentions": 0,
                        "total_likes": 0,
                        "total_retweets": 0,
                        "total_impressions": 0,
                        "verified_mentions": 0,
                        "tweets": [],
                    }

                ticker_data[ticker]["mentions"] += 1
                ticker_data[ticker]["total_likes"] += tweet.like_count
                ticker_data[ticker]["total_retweets"] += tweet.retweet_count
                ticker_data[ticker]["total_impressions"] += tweet.impression_count

                if tweet.author_verified:
                    ticker_data[ticker]["verified_mentions"] += 1

                ticker_data[ticker]["tweets"].append(tweet)

        # Filter and sort
        filtered = {
            ticker: data for ticker, data in ticker_data.items() if data["mentions"] >= min_mentions
        }

        sorted_tickers = sorted(
            filtered.items(),
            key=lambda x: (x[1]["mentions"], x[1]["total_likes"]),
            reverse=True,
        )[:top_n]

        # Format output
        result = {}
        for ticker, data in sorted_tickers:
            result[ticker] = {
                "mentions": data["mentions"],
                "total_likes": data["total_likes"],
                "total_retweets": data["total_retweets"],
                "total_impressions": data["total_impressions"],
                "verified_mentions": data["verified_mentions"],
                "avg_likes": data["total_likes"] / data["mentions"],
                "engagement_score": (data["total_likes"] + data["total_retweets"] * 2)
                / data["mentions"],
                "top_tweets": [
                    {
                        "text": t.text[:280],
                        "author": f"@{t.author_username}",
                        "likes": t.like_count,
                        "retweets": t.retweet_count,
                        "verified": t.author_verified,
                    }
                    for t in sorted(data["tweets"], key=lambda x: x.like_count, reverse=True)[:5]
                ],
            }

        logger.info(f"Found {len(result)} trending tickers")
        return result

    # ========================================================================
    # DATA EXPORT
    # ========================================================================

    def tweets_to_dataframe(self, tweets: list[Tweet]) -> pl.DataFrame:
        """
        Convert list of Tweet objects to Polars DataFrame

        Args:
            tweets: List of Tweet objects

        Returns:
            Polars DataFrame with tweet data
        """
        if not tweets:
            return pl.DataFrame()

        data = []
        for tweet in tweets:
            data.append(
                {
                    "id": tweet.id,
                    "text": tweet.text,
                    "author_id": tweet.author_id,
                    "author_username": tweet.author_username,
                    "author_name": tweet.author_name,
                    "author_followers": tweet.author_followers,
                    "author_verified": tweet.author_verified,
                    "created_at": tweet.created_at,
                    "retweet_count": tweet.retweet_count,
                    "like_count": tweet.like_count,
                    "reply_count": tweet.reply_count,
                    "quote_count": tweet.quote_count,
                    "impression_count": tweet.impression_count,
                    "tweet_type": tweet.tweet_type.value,
                    "language": tweet.language,
                    "tickers_mentioned": tweet.tickers_mentioned,
                    "hashtags": tweet.hashtags,
                }
            )

        df = pl.DataFrame(data)
        return df.sort("created_at", descending=True)

    async def collect_to_dataframe(
        self,
        ticker: str,
        max_results: int = 100,
    ) -> pl.DataFrame:
        """
        Collect tweets for a ticker and return as DataFrame

        Args:
            ticker: Stock ticker symbol
            max_results: Maximum tweets

        Returns:
            Polars DataFrame with tweets
        """
        tweets = await self.search_ticker(
            ticker=ticker,
            max_results=max_results,
        )

        return self.tweets_to_dataframe(tweets)


# ============================================================================
# STREAMING CLIENT (for real-time data)
# ============================================================================


class TwitterStreamClient(tweepy.StreamingClient):
    """
    Streaming client for real-time Twitter data

    Use this for real-time monitoring of financial tweets.
    Requires Elevated access or higher.
    """

    def __init__(
        self,
        bearer_token: str,
        callback: Callable | None = None,
    ):
        """
        Initialize streaming client

        Args:
            bearer_token: Twitter API bearer token
            callback: Function to call with each tweet
        """
        super().__init__(bearer_token)
        self.callback = callback
        self.tweet_count = 0

    def on_tweet(self, tweet: tweepy.Tweet):
        """
        Called when a tweet is received
        """
        self.tweet_count += 1

        if self.callback:
            self.callback(tweet)
        else:
            logger.info(f"Tweet received: {tweet.text[:100]}...")

    def on_error(self, status_code: int):
        """
        Called on stream error
        """
        logger.error(f"Stream error: {status_code}")

        if status_code == 420:
            # Rate limit - disconnect
            return False

        return True

    def on_connection_error(self):
        """
        Called on connection error
        """
        logger.warning("Stream connection error, reconnecting...")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def search_ticker_tweets(
    ticker: str,
    max_results: int = 100,
) -> pl.DataFrame:
    """
    Quick function to search for ticker tweets

    Args:
        ticker: Stock ticker symbol
        max_results: Maximum tweets

    Returns:
        DataFrame with tweets
    """
    scraper = TwitterScraper()
    return await scraper.collect_to_dataframe(
        ticker=ticker,
        max_results=max_results,
    )


async def get_financial_news_tweets(max_results: int = 50) -> dict[str, list[Tweet]]:
    """
    Get tweets from financial news accounts

    Args:
        max_results: Max tweets per account

    Returns:
        Dictionary of account -> tweets
    """
    scraper = TwitterScraper()
    news_accounts = ["WSJmarkets", "markets", "ReutersBiz", "CNBC", "YahooFinance"]

    return await scraper.get_influencer_tweets(
        usernames=news_accounts,
        max_results_per_user=max_results,
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    """Example usage of TwitterScraper"""

    async def main():
        scraper = TwitterScraper()

        if not scraper.is_initialized:
            print("Scraper not initialized. Please configure Twitter API credentials.")
            return

        # Example 1: Search for ticker
        print("\n=== AAPL Tweets ===")
        aapl_tweets = await scraper.search_ticker(
            ticker="AAPL",
            max_results=10,
        )

        for tweet in aapl_tweets[:5]:
            print(f"  @{tweet.author_username}: {tweet.text[:80]}...")
            print(f"    Likes: {tweet.like_count}, RTs: {tweet.retweet_count}")

        # Example 2: Get trending tickers
        print("\n=== Trending Tickers ===")
        trending = await scraper.get_trending_tickers(top_n=10)

        for ticker, data in list(trending.items())[:10]:
            print(f"${ticker}: {data['mentions']} mentions, {data['total_likes']} likes")

        # Example 3: Get influencer tweets
        print("\n=== Financial News ===")
        news_tweets = await scraper.get_influencer_tweets(
            usernames=["WSJmarkets", "CNBC"],
            max_results_per_user=5,
        )

        for username, tweets in news_tweets.items():
            print(f"\n@{username}:")
            for tweet in tweets[:2]:
                print(f"  - {tweet.text[:60]}...")

    asyncio.run(main())
