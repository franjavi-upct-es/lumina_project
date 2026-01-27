# backend/nlp_engine/__init__.py
"""
NLP Engine Module for Lumina 2.0
================================
Natural language processing engine for financial sentiment analysis

Modules:
- reddit_scraper: Reddit data extraction from financial subreddits
- twitter_scraper: Twitter/X data extraction for financial sentiment
- finbert_analyzer: FinBERT-based sentiment analysis
- news_scraper: Financial news extraction
- sentiment_aggregator: Multi-source sentiment aggregation

Author: Lumina Quant Lab
Version: 2.0.0
"""

from backend.nlp_engine.finbert_analyzer import FinBERTAnalyzer
from backend.nlp_engine.news_scraper import NewsScraper
from backend.nlp_engine.reddit_scraper import (
    RedditComment,
    RedditPost,
    RedditScraper,
    SortMethod,
    SubredditCategory,
    TimeFilter,
    get_wsb_trending,
    search_ticker_mentions,
)
from backend.nlp_engine.sentiment_aggregator import SentimentAggregator
from backend.nlp_engine.twitter_scraper import (
    StreamRule,
    Tweet,
    TwitterScraper,
    TwitterStreamClient,
    TwitterUser,
    get_financial_news_tweets,
    search_ticker_tweets,
)

__all__ = [
    # Reddit components
    "RedditScraper",
    "RedditPost",
    "RedditComment",
    "SubredditCategory",
    "SortMethod",
    "TimeFilter",
    "get_wsb_trending",
    "search_ticker_mentions",
    # Twitter components
    "TwitterScraper",
    "TwitterStreamClient",
    "Tweet",
    "TwitterUser",
    "StreamRule",
    "search_ticker_tweets",
    "get_financial_news_tweets",
    # FinBERT components
    "FinBERTAnalyzer",
    # NewsScraper components
    "NewsScraper",
    # SentimentAggregator components
    "SentimentAggregator",
]
