# backend/nlp_engine/reddit_scraper.py
"""
Reddit Scraper for Financial Sentiment Analysis

This module provides functionality to scrape and analyze Reddit posts
from financial subreddits for sentiment analysis and trend detection.
Uses PRAW (Python Reddit API Wrapper) for data collection.

Key Features:
- Scrape posts from multiple financial subreddits
- Extract ticker mentions from posts and comments
- Track trending stocks and sentiment shifts
- Rate limiting and error handling
- Async support for high-throughput scraping
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import polars as pl
import praw
from loguru import logger
from praw.models import Comment, Submission

from backend.config.settings import get_settings

settings = get_settings()


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================


class SubredditCategory(Enum):
    """Categories of financial subreddits"""

    RETAIL_TRADING = "retail_trading"
    INVESTING = "investing"
    OPTIONS = "options"
    CRYPTO = "crypto"
    ANALYSIS = "analysis"
    NEWS = "news"


class SortMethod(Enum):
    """Reddit sort methods for posts"""

    HOT = "hot"
    NEW = "new"
    TOP = "top"
    RISING = "rising"
    CONTROVERSIAL = "controversial"


class TimeFilter(Enum):
    """Time filters for Reddit top/controversial posts"""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ALL = "all"


@dataclass
class RedditPost:
    """Data class representing a Reddit post"""

    id: str
    subreddit: str
    title: str
    text: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    url: str
    permalink: str
    flair: str | None
    is_self: bool
    awards_count: int
    tickers_mentioned: list[str]


@dataclass
class RedditComment:
    """Data class representing a Reddit comment"""

    id: str
    post_id: str
    subreddit: str
    text: str
    author: str
    score: int
    created_utc: datetime
    parent_id: str
    is_top_level: bool
    tickers_mentioned: list[str]


# ============================================================================
# SUBREDDIT CONFIGURATION
# ============================================================================


FINANCIAL_SUBREDDITS: dict[str, SubredditCategory] = {
    # Retail Trading
    "wallstreetbets": SubredditCategory.RETAIL_TRADING,
    "pennystocks": SubredditCategory.RETAIL_TRADING,
    "smallstreetbets": SubredditCategory.RETAIL_TRADING,
    "Daytrading": SubredditCategory.RETAIL_TRADING,
    "swingtrading": SubredditCategory.RETAIL_TRADING,
    # Investing
    "stocks": SubredditCategory.INVESTING,
    "investing": SubredditCategory.INVESTING,
    "StockMarket": SubredditCategory.INVESTING,
    "ValueInvesting": SubredditCategory.INVESTING,
    "dividends": SubredditCategory.INVESTING,
    "Bogleheads": SubredditCategory.INVESTING,
    # Options Trading
    "options": SubredditCategory.OPTIONS,
    "thetagang": SubredditCategory.OPTIONS,
    "wallstreetbetsOGs": SubredditCategory.OPTIONS,
    # Crypto
    "CryptoCurrency": SubredditCategory.CRYPTO,
    "Bitcoin": SubredditCategory.CRYPTO,
    "ethereum": SubredditCategory.CRYPTO,
    "CryptoMarkets": SubredditCategory.CRYPTO,
    # Analysis
    "SecurityAnalysis": SubredditCategory.ANALYSIS,
    "UndervaluedStonks": SubredditCategory.ANALYSIS,
    "FluentInFinance": SubredditCategory.ANALYSIS,
    # News
    "finance": SubredditCategory.NEWS,
    "Economics": SubredditCategory.NEWS,
    "business": SubredditCategory.NEWS,
}


# Words to exclude when extracting tickers (common words that look like tickers)
TICKER_BLACKLIST: set[str] = {
    # Common words
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
    "LET",
    "PUT",
    "SAY",
    "SHE",
    "TOO",
    "USE",
    "DAD",
    "MOM",
    "SON",
    "YES",
    "BIG",
    "TOP",
    "LOW",
    "RUN",
    "TRY",
    "ASK",
    "OWN",
    "WHY",
    "FUN",
    "BAD",
    "RED",
    "EAT",
    "BUY",
    "OTC",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CAD",
    # Acronyms
    "CEO",
    "CFO",
    "COO",
    "CTO",
    "IPO",
    "SEC",
    "FDA",
    "FBI",
    "CIA",
    "IRS",
    "USA",
    "UK",
    "EU",
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
    "EPS",
    "PE",
    "PB",
    "PS",
    "ROE",
    "ROI",
    "YOY",
    "QOQ",
    "EOD",
    "EOW",
    "EOM",
    "HODL",
    "FOMO",
    "FUD",
    "DCA",
    "BTFD",
    "ATM",
    "ITM",
    "OTM",
    "IV",
    "DTE",
    "LEAP",
    "FD",
    "PDT",
    "REPO",
    "QE",
    "ETF",
    "ETN",
    "SPAC",
    "REIT",
    "EDIT",
    "POST",
    "LINK",
    "HELP",
    "INFO",
    "NEWS",
    "TLDR",
    "THIS",
    "THAT",
    "WHAT",
    "WHEN",
    "THEN",
    "THAN",
    "HERE",
    "BEEN",
    "HAVE",
    "WILL",
    "WERE",
    "SOME",
    "THEY",
    "THEM",
    "FROM",
    "JUST",
    "MUCH",
    "MORE",
    "MOST",
    "ONLY",
    "OVER",
    "SUCH",
    "VERY",
    "SAME",
    "ALSO",
}


# ============================================================================
# REDDIT SCRAPER CLASS
# ============================================================================


class RedditScraper:
    """
    Scraper for collecting and analyzing Reddit financial data

    This class provides methods to:
    - Collect posts from financial subreddits
    - Extract and count ticker mentions
    - Track trending stocks
    - Analyze sentiment patterns

    Example:
        ```python
        scraper = RedditScraper()

        # Get trending tickers from WSB
        trending = await scraper.get_trending_tickers(
            subreddits=["wallstreetbets"],
            time_filter=TimeFilter.DAY
        )

        # Search for specific ticker mentions
        posts = await scraper.search_ticker(
            ticker="AAPL",
            subreddits=["stocks", "investing"],
            limit=100
        )
        ```
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str = "LuminaQuantLab/2.0",
        rate_limit_delay: float = 1.0,
    ):
        """
        Initialize Reddit scraper

        Args:
            client_id: Reddit API client ID (from environment if not provided)
            client_secret: Reddit API client secret (from environment if not provided)
            user_agent: User agent string for API requests
            rate_limit_delay: Delay between API requests in seconds
        """
        self.client_id = client_id or getattr(settings, "REDDIT_CLIENT_ID", None)
        self.client_secret = client_secret or getattr(settings, "REDDIT_CLIENT_SECRET", None)
        self.user_agent = user_agent
        self.rate_limit_delay = rate_limit_delay

        self.reddit: praw.Reddit | None = None
        self._initialized = False

        # Initialize connection
        self._initialize_connection()

    def _initialize_connection(self) -> bool:
        """
        Initialize connection to Reddit API

        Returns:
            True if connection successful, False otherwise
        """
        if not self.client_id or not self.client_secret:
            logger.warning(
                "Reddit API credentials not configured. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
            )
            return False

        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )

            # Test connection (read-only mode doesn't need authentication)
            _ = self.reddit.read_only

            self._initialized = True
            logger.success("Reddit API connection initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Reddit API connection: {e}")
            self.reddit = None
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if scraper is properly initialized"""
        return self._initialized and self.reddit is not None

    # ========================================================================
    # TICKER EXTRACTION
    # ========================================================================

    def extract_tickers(self, text: str) -> list[str]:
        """
        Extract stock ticker symbols from text

        Uses multiple patterns to identify tickers:
        - Cashtag format: $AAPL
        - Uppercase words: AAPL
        - With filters for common false positives

        Args:
            text: Text to search for tickers

        Returns:
            List of unique ticker symbols found
        """
        if not text:
            return []

        tickers: set[str] = set()

        # Pattern 1: Cashtag format ($AAPL) - high confidence
        cashtag_pattern = r"\$([A-Z]{1,5})\b"
        cashtag_matches = re.findall(cashtag_pattern, text.upper())
        tickers.update(cashtag_matches)

        # Pattern 2: Standalone uppercase (1-5 chars) - lower confidence
        # Only match if surrounded by whitespace or punctuation
        uppercase_pattern = r"(?<![A-Za-z])([A-Z]{2,5})(?![A-Za-z])"
        uppercase_matches = re.findall(uppercase_pattern, text)

        # Filter uppercase matches more strictly
        for match in uppercase_matches:
            # Only include if it looks like a valid ticker
            if self._is_likely_ticker(match, text):
                tickers.add(match)

        # Remove blacklisted words
        tickers -= TICKER_BLACKLIST

        # Filter by valid length (1-5 characters)
        tickers = {t for t in tickers if 1 <= len(t) <= 5}

        return sorted(list(tickers))

    def _is_likely_ticker(self, candidate: str, context: str) -> bool:
        """
        Determine if a candidate string is likely a ticker symbol

        Args:
            candidate: Potential ticker symbol
            context: Full text for context analysis

        Returns:
            True if likely a ticker, False otherwise
        """
        # Skip if in blacklist
        if candidate in TICKER_BLACKLIST:
            return False

        # Skip single character (too many false positives)
        if len(candidate) < 2:
            return False

        # Higher confidence if preceded by $ or keywords
        confidence_patterns = [
            rf"\${candidate}\b",  # Cashtag
            rf"\b{candidate}\s+stock",  # "AAPL stock"
            rf"\b{candidate}\s+calls?",  # "AAPL calls"
            rf"\b{candidate}\s+puts?",  # "AAPL puts"
            rf"bought?\s+{candidate}",  # "bought AAPL"
            rf"sold?\s+{candidate}",  # "sold AAPL"
            rf"long\s+{candidate}",  # "long AAPL"
            rf"short\s+{candidate}",  # "short AAPL"
        ]

        for pattern in confidence_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True

        # If no confidence patterns match, only include longer tickers
        # to reduce false positives
        return len(candidate) >= 3

    # ========================================================================
    # POST COLLECTION
    # ========================================================================

    async def collect_posts(
        self,
        subreddits: list[str] | None = None,
        sort: SortMethod = SortMethod.HOT,
        time_filter: TimeFilter = TimeFilter.DAY,
        limit: int = 100,
        include_comments: bool = False,
        max_comments_per_post: int = 10,
    ) -> list[RedditPost]:
        """
        Collect posts from specified subreddits

        Args:
            subreddits: List of subreddit names (defaults to all financial subreddits)
            sort: Sort method for posts
            time_filter: Time filter for top/controversial posts
            limit: Maximum posts per subreddit
            include_comments: Whether to fetch comments
            max_comments_per_post: Max comments to fetch per post

        Returns:
            List of RedditPost objects
        """
        if not self.is_initialized:
            logger.error("Reddit scraper not initialized")
            return []

        subreddits = subreddits or list(FINANCIAL_SUBREDDITS.keys())
        all_posts: list[RedditPost] = []

        for subreddit_name in subreddits:
            try:
                posts = await self._collect_subreddit_posts(
                    subreddit_name=subreddit_name,
                    sort=sort,
                    time_filter=time_filter,
                    limit=limit,
                )
                all_posts.extend(posts)

                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.warning(f"Error collecting from r/{subreddit_name}: {e}")
                continue

        logger.info(f"Collected {len(all_posts)} posts from {len(subreddits)} subreddits")
        return all_posts

    async def _collect_subreddit_posts(
        self,
        subreddit_name: str,
        sort: SortMethod,
        time_filter: TimeFilter,
        limit: int,
    ) -> list[RedditPost]:
        """
        Collect posts from a single subreddit (async wrapper)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._collect_subreddit_posts_sync(subreddit_name, sort, time_filter, limit),
        )

    def _collect_subreddit_posts_sync(
        self,
        subreddit_name: str,
        sort: SortMethod,
        time_filter: TimeFilter,
        limit: int,
    ) -> list[RedditPost]:
        """
        Collect posts from a single subreddit (synchronous)
        """
        posts: list[RedditPost] = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Get posts based on sort method
            if sort == SortMethod.HOT:
                submissions = subreddit.hot(limit=limit)
            elif sort == SortMethod.NEW:
                submissions = subreddit.new(limit=limit)
            elif sort == SortMethod.TOP:
                submissions = subreddit.top(time_filter=time_filter.value, limit=limit)
            elif sort == SortMethod.RISING:
                submissions = subreddit.rising(limit=limit)
            elif sort == SortMethod.CONTROVERSIAL:
                submissions = subreddit.controversial(time_filter=time_filter.value, limit=limit)
            else:
                submissions = subreddit.hot(limit=limit)

            for submission in submissions:
                post = self._parse_submission(submission)
                posts.append(post)

        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")

        return posts

    def _parse_submission(self, submission: Submission) -> RedditPost:
        """
        Parse a PRAW Submission object into RedditPost
        """
        full_text = f"{submission.title} {submission.selftext}"
        tickers = self.extract_tickers(full_text)

        return RedditPost(
            id=submission.id,
            subreddit=submission.subreddit.display_name,
            title=submission.title,
            text=submission.selftext or "",
            author=str(submission.author) if submission.author else "[deleted]",
            score=submission.score,
            upvote_ratio=submission.upvote_ratio,
            num_comments=submission.num_comments,
            created_utc=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
            url=submission.url,
            permalink=f"https://reddit.com{submission.permalink}",
            flair=submission.link_flair_text,
            is_self=submission.is_self,
            awards_count=submission.total_awards_received,
            tickers_mentioned=tickers,
        )

    # ========================================================================
    # TICKER SEARCH
    # ========================================================================

    async def search_ticker(
        self,
        ticker: str,
        subreddits: list[str] | None = None,
        sort: SortMethod = SortMethod.NEW,
        time_filter: TimeFilter = TimeFilter.WEEK,
        limit: int = 100,
    ) -> list[RedditPost]:
        """
        Search for posts mentioning a specific ticker

        Args:
            ticker: Stock ticker symbol to search for
            subreddits: Subreddits to search (defaults to all financial)
            sort: Sort method for results
            time_filter: Time filter
            limit: Maximum results

        Returns:
            List of posts mentioning the ticker
        """
        if not self.is_initialized:
            logger.error("Reddit scraper not initialized")
            return []

        subreddits = subreddits or list(FINANCIAL_SUBREDDITS.keys())

        # Build search query
        query = f"${ticker} OR {ticker}"

        all_posts: list[RedditPost] = []

        for subreddit_name in subreddits:
            try:
                posts = await self._search_subreddit(
                    subreddit_name=subreddit_name,
                    query=query,
                    sort=sort,
                    time_filter=time_filter,
                    limit=limit // len(subreddits) + 1,
                )
                all_posts.extend(posts)

                await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.warning(f"Error searching r/{subreddit_name} for {ticker}: {e}")
                continue

        # Sort by score and limit
        all_posts.sort(key=lambda p: p.score, reverse=True)
        return all_posts[:limit]

    async def _search_subreddit(
        self,
        subreddit_name: str,
        query: str,
        sort: SortMethod,
        time_filter: TimeFilter,
        limit: int,
    ) -> list[RedditPost]:
        """
        Search within a subreddit (async wrapper)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._search_subreddit_sync(subreddit_name, query, sort, time_filter, limit),
        )

    def _search_subreddit_sync(
        self,
        subreddit_name: str,
        query: str,
        sort: SortMethod,
        time_filter: TimeFilter,
        limit: int,
    ) -> list[RedditPost]:
        """
        Search within a subreddit (synchronous)
        """
        posts: list[RedditPost] = []

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            submissions = subreddit.search(
                query=query,
                sort=sort.value,
                time_filter=time_filter.value,
                limit=limit,
            )

            for submission in submissions:
                post = self._parse_submission(submission)
                posts.append(post)

        except Exception as e:
            logger.error(f"Error searching r/{subreddit_name}: {e}")

        return posts

    # ========================================================================
    # TRENDING ANALYSIS
    # ========================================================================

    async def get_trending_tickers(
        self,
        subreddits: list[str] | None = None,
        sort: SortMethod = SortMethod.HOT,
        time_filter: TimeFilter = TimeFilter.DAY,
        limit_per_subreddit: int = 100,
        min_mentions: int = 2,
        top_n: int = 20,
    ) -> dict[str, dict[str, Any]]:
        """
        Get trending tickers across subreddits

        Args:
            subreddits: Subreddits to analyze
            sort: Sort method for posts
            time_filter: Time filter
            limit_per_subreddit: Posts to analyze per subreddit
            min_mentions: Minimum mentions to be included
            top_n: Number of top tickers to return

        Returns:
            Dictionary mapping tickers to analysis data:
            {
                "AAPL": {
                    "mentions": 150,
                    "total_score": 45000,
                    "avg_score": 300,
                    "subreddits": ["wallstreetbets", "stocks"],
                    "sentiment_posts": [...],
                }
            }
        """
        if not self.is_initialized:
            logger.error("Reddit scraper not initialized")
            return {}

        # Collect posts
        posts = await self.collect_posts(
            subreddits=subreddits,
            sort=sort,
            time_filter=time_filter,
            limit=limit_per_subreddit,
        )

        if not posts:
            return {}

        # Aggregate ticker mentions
        ticker_data: dict[str, dict[str, Any]] = {}

        for post in posts:
            for ticker in post.tickers_mentioned:
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        "mentions": 0,
                        "total_score": 0,
                        "posts": [],
                        "subreddits": set(),
                    }

                ticker_data[ticker]["mentions"] += 1
                ticker_data[ticker]["total_score"] += post.score
                ticker_data[ticker]["posts"].append(post)
                ticker_data[ticker]["subreddits"].add(post.subreddit)

        # Filter and sort
        filtered_tickers = {
            ticker: data for ticker, data in ticker_data.items() if data["mentions"] >= min_mentions
        }

        # Sort by mentions (could also sort by score)
        sorted_tickers = sorted(
            filtered_tickers.items(),
            key=lambda x: x[1]["mentions"],
            reverse=True,
        )[:top_n]

        # Format output
        result = {}
        for ticker, data in sorted_tickers:
            result[ticker] = {
                "mentions": data["mentions"],
                "total_score": data["total_score"],
                "avg_score": data["total_score"] / data["mentions"],
                "subreddits": list(data["subreddits"]),
                "top_posts": [
                    {
                        "title": p.title,
                        "score": p.score,
                        "subreddit": p.subreddit,
                        "url": p.permalink,
                    }
                    for p in sorted(data["posts"], key=lambda x: x.score, reverse=True)[:5]
                ],
            }

        logger.info(f"Found {len(result)} trending tickers")
        return result

    # ========================================================================
    # COMMENT COLLECTION
    # ========================================================================

    async def collect_comments(
        self,
        post_id: str,
        limit: int = 100,
        depth: int = 2,
    ) -> list[RedditComment]:
        """
        Collect comments from a specific post

        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments
            depth: Maximum depth of comment tree to traverse

        Returns:
            List of RedditComment objects
        """
        if not self.is_initialized:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._collect_comments_sync(post_id, limit, depth)
        )

    def _collect_comments_sync(
        self,
        post_id: str,
        limit: int,
        depth: int,
    ) -> list[RedditComment]:
        """
        Collect comments synchronously
        """
        comments: list[RedditComment] = []

        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=depth)

            for comment in submission.comments.list()[:limit]:
                if isinstance(comment, Comment):
                    tickers = self.extract_tickers(comment.body)

                    comments.append(
                        RedditComment(
                            id=comment.id,
                            post_id=post_id,
                            subreddit=comment.subreddit.display_name,
                            text=comment.body,
                            author=str(comment.author) if comment.author else "[deleted]",
                            score=comment.score,
                            created_utc=datetime.fromtimestamp(
                                comment.created_utc, tz=timezone.utc
                            ),
                            parent_id=comment.parent_id,
                            is_top_level=comment.parent_id.startswith("t3_"),
                            tickers_mentioned=tickers,
                        )
                    )

        except Exception as e:
            logger.error(f"Error collecting comments for post {post_id}: {e}")

        return comments

    # ========================================================================
    # DATA EXPORT
    # ========================================================================

    def posts_to_dataframe(self, posts: list[RedditPost]) -> pl.DataFrame:
        """
        Convert list of RedditPost objects to Polars DataFrame

        Args:
            posts: List of RedditPost objects

        Returns:
            Polars DataFrame with post data
        """
        if not posts:
            return pl.DataFrame()

        data = []
        for post in posts:
            data.append(
                {
                    "id": post.id,
                    "subreddit": post.subreddit,
                    "title": post.title,
                    "text": post.text,
                    "author": post.author,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc,
                    "url": post.url,
                    "permalink": post.permalink,
                    "flair": post.flair,
                    "is_self": post.is_self,
                    "awards_count": post.awards_count,
                    "tickers_mentioned": post.tickers_mentioned,
                }
            )

        df = pl.DataFrame(data)
        return df.sort("created_utc", descending=True)

    def comments_to_dataframe(self, comments: list[RedditComment]) -> pl.DataFrame:
        """
        Convert list of RedditComment objects to Polars DataFrame

        Args:
            comments: List of RedditComment objects

        Returns:
            Polars DataFrame with comment data
        """
        if not comments:
            return pl.DataFrame()

        data = []
        for comment in comments:
            data.append(
                {
                    "id": comment.id,
                    "post_id": comment.post_id,
                    "subreddit": comment.subreddit,
                    "text": comment.text,
                    "author": comment.author,
                    "score": comment.score,
                    "created_utc": comment.created_utc,
                    "parent_id": comment.parent_id,
                    "is_top_level": comment.is_top_level,
                    "tickers_mentioned": comment.tickers_mentioned,
                }
            )

        df = pl.DataFrame(data)
        return df.sort("score", descending=True)

    async def collect_to_dataframe(
        self,
        subreddits: list[str] | None = None,
        sort: SortMethod = SortMethod.HOT,
        time_filter: TimeFilter = TimeFilter.DAY,
        limit: int = 100,
    ) -> pl.DataFrame:
        """
        Collect posts and return as DataFrame

        Args:
            subreddits: Subreddits to collect from
            sort: Sort method
            time_filter: Time filter
            limit: Maximum posts per subreddit

        Returns:
            Polars DataFrame with collected posts
        """
        posts = await self.collect_posts(
            subreddits=subreddits,
            sort=sort,
            time_filter=time_filter,
            limit=limit,
        )

        return self.posts_to_dataframe(posts)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def get_wsb_trending(top_n: int = 10) -> dict[str, dict[str, Any]]:
    """
    Quick function to get trending tickers from r/wallstreetbets

    Args:
        top_n: Number of top tickers to return

    Returns:
        Dictionary of trending tickers with analysis
    """
    scraper = RedditScraper()
    return await scraper.get_trending_tickers(
        subreddits=["wallstreetbets"],
        time_filter=TimeFilter.DAY,
        top_n=top_n,
    )


async def search_ticker_mentions(
    ticker: str,
    days_back: int = 7,
) -> pl.DataFrame:
    """
    Quick function to search for ticker mentions

    Args:
        ticker: Stock ticker to search
        days_back: Number of days to look back

    Returns:
        DataFrame with posts mentioning the ticker
    """
    scraper = RedditScraper()

    time_filter = TimeFilter.WEEK if days_back <= 7 else TimeFilter.MONTH

    posts = await scraper.search_ticker(
        ticker=ticker,
        time_filter=time_filter,
        limit=100,
    )

    return scraper.posts_to_dataframe(posts)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    """Example usage of RedditScraper"""

    async def main():
        scraper = RedditScraper()

        if not scraper.is_initialized:
            print("Scraper not initialized. Please configure Reddit API credentials.")
            return

        # Example 1: Get trending tickers
        print("\n=== Trending Tickers (WSB) ===")
        trending = await scraper.get_trending_tickers(
            subreddits=["wallstreetbets"],
            time_filter=TimeFilter.DAY,
            top_n=10,
        )

        for ticker, data in trending.items():
            print(f"{ticker}: {data['mentions']} mentions, avg score: {data['avg_score']:.0f}")

        # Example 2: Search for specific ticker
        print("\n=== AAPL Mentions ===")
        aapl_posts = await scraper.search_ticker(
            ticker="AAPL",
            subreddits=["stocks", "investing"],
            limit=10,
        )

        for post in aapl_posts[:5]:
            print(f"  [{post.subreddit}] {post.title[:60]}... (score: {post.score})")

    asyncio.run(main())
