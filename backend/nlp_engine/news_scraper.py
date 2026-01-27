# backend/nlp_engine/news_scraper.py
"""
Web scraper for financial news from multiple sources
"""

import asyncio
from datetime import datetime

import aiohttp
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger


class NewsScraper:
    """
    Scrapes financial news from various sources
    """

    def __init__(self, user_agent: str | None = None):
        """Initialize scraper"""
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        logger.info("Initialized NewsScraper")

    async def scrape_url(self, url: str, extract_text: bool = True) -> dict | None:
        """
        Scrape a single URL

        Args:
            url: URL to scrape
            extract_text: Extract article text

        Returns:
            Dictionary with scraped data
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers={"User-Agent": self.user_agent}, timeout=30
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch {url}: {response.status}")
                        return None

                    html = await response.text()

            soup = BeautifulSoup(html, "html.parser")

            # Extract metadata
            result = {"url": url, "title": self._extract_title(soup), "scraped_at": datetime.now()}

            # Extract text if requested
            if extract_text:
                result["text"] = self._extract_text(soup)
                result["summary"] = self._extract_summary(soup)

            return result

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    async def scrape_urls_batch(self, urls: list[str], max_concurrent: int = 5) -> list[dict]:
        """
        Scrape multiple URLs concurrently

        Args:
            urls: List of URLs
            max_concurrent: Max concurrent requests

        Returns:
            List of scraped data
        """
        logger.info(f"Scraping {len(urls)} URLs")

        sempahore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_limit(url):
            async with sempahore:
                return await self.scrape_url(url)

        tasks = [scrape_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures
        valid_results = [r for r in results if isinstance(r, dict) and r is not None]

        logger.success(f"Successfully scraped {len(valid_results)}/{len(urls)} URLs")
        return valid_results

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        # Try various title selectors
        title = None

        # OpenGraph title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]

        # Standard title tag
        if not title:
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text()

        # H1 tag
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text()

        return title.strip() if title else ""

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main article text"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()

        # Try to find article body
        article = soup.find("article")
        if not article:
            article = soup.find("div", class_=["article", "post", "content"])
        if not article:
            article = soup.find("main")

        if article:
            text = article.get_text(separator=" ", strip=True)
        else:
            # Fallback to all paragraphs
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text(strip=True) for p in paragraphs])

        # Clean text
        text = " ".join(text.split())
        return text[:5000]  # Limit length

    def _extract_summary(self, soup: BeautifulSoup) -> str:
        """Extract article summary/description"""
        # Try OpenGraph description
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return og_desc["content"].strip()

        # Try meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"].strip()

        # Fallback to first paragraph
        first_p = soup.find("p")
        if first_p:
            return first_p.get_text(strip=True)[:500]

        return ""

    def scrape_to_dataframe(self, urls: list[str]) -> pd.DataFrame:
        """
        Scrape URLs and return as DataFrame

        Args:
            urls: List of URLs to scrape

        Returns:
            DataFrame with scraped data
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        results = loop.run_until_complete(self.scrape_urls_batch(urls))
        loop.close()

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)
