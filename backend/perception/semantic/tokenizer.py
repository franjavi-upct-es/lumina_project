# backend/perception/semantic/tokenizer.py
"""
Financial Tokenizer

Domain-specific tokenization for financial text with specialized vocabulary
and preprocessing for news, earnings calls, and social media.

Key Features:
- Financial entity recognition (ticker symbols, currencies)
- Number normalization (percentages, prices, ratios)
- Domain-specific vocabulary (earnings, guidance, EBITDA, etc.)
- Social media handling (hashtags, mentions, emojis)

Based on: transformers.AutoTokenizer with custom preprocessing
"""

import re
from dataclasses import dataclass

from loguru import logger


@dataclass
class TokenizerConfig:
    """
    Configuration for financial tokenizer.

    Attributes:
        max_length: Maximum sequence length
        padding: Padding strategy ('max_length', 'longest', 'do_not_pad')
        truncation: Enable truncation
        normalize_numbers: Normalize numeric values
        preserve_tickers: Preserve ticker symbols
        lowercase: Convert to lowercase (except tickers)
    """

    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    normalize_numbers: bool = True
    preserve_tickers: bool = True
    lowercase: bool = True


class FinancialTokenizer:
    """
    Tokenizer specialized for financial text.

    Handles:
    - Ticker symbols (AAPL, MSFT, etc.)
    - Financial numbers ($123.45, 25%, 3.2B)
    - Financial terms (earnings, guidance, EBITDA)
    - Social media text (hashtags, mentions)

    Example:
        >>> tokenizer = FinancialTokenizer()
        >>> text = "AAPL beat earnings by 15%, +3.2% after-hours"
        >>> tokens = tokenizer.tokenize(text)
        >>> # Preserves 'AAPL', normalizes '15%' and '3.2%'
    """

    def __init__(
        self, config: TokenizerConfig | None = None, model_name: str = "distilroberta-base"
    ):
        """
        Initialize financial tokenizer.

        Args:
            config: Tokenizer configuration
            model_name: Base tokenizer model
        """
        self.config = config or TokenizerConfig()
        self.model_name = model_name

        # Financial vocabulary
        self.financial_terms = {
            "earnings",
            "revenue",
            "profit",
            "loss",
            "guidance",
            "beat",
            "miss",
            "eps",
            "ebitda",
            "margin",
            "fcf",
            "capex",
            "opex",
            "debt",
            "equity",
            "ipo",
            "merger",
            "acquisition",
            "buyback",
            "dividend",
            "split",
            "rally",
            "selloff",
            "breakout",
            "support",
            "resistance",
            "bullish",
            "bearish",
            "hawkish",
            "dovish",
        }

        # Ticker pattern (1-5 uppercase letters)
        self.ticker_pattern = re.compile(r"\b[A-Z]{1,5}\b")

        # Number patterns
        self.percent_pattern = re.compile(r"[-+]?\d+\.?\d*%")
        self.currency_pattern = re.compile(r"\$\d+\.?\d*[KMB]?")
        self.number_pattern = re.compile(r"[-+]?\d+\.?\d*")

        # Initialize base tokenizer (lazy loading in production)
        self._base_tokenizer = None

        logger.debug(f"FinancialTokenizer initialized: {model_name}")

    def preprocess(self, text: str) -> str:
        """
        Preprocess financial text.

        Args:
            text: Raw text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Preserve ticker symbols (mark for special handling)
        if self.config.preserve_tickers:
            tickers = self.ticker_pattern.findall(text)
            for ticker in tickers:
                # Mark tickers to prevent lowercasing
                text = text.replace(ticker, f"<TICKER>{ticker}</TICKER>")

        # Normalize numbers if configured
        if self.config.normalize_numbers:
            text = self._normalize_numbers(text)

        # Lowercase (except marked tickers)
        if self.config.lowercase:
            # Lowercase everything
            text = text.lower()
            # Restore ticker case
            text = re.sub(r"<ticker>([a-z]+)</ticker>", lambda m: m.group(1).upper(), text)

        # Clean up ticker markers
        text = text.replace("<TICKER>", "").replace("</TICKER>", "")

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def _normalize_numbers(self, text: str) -> str:
        """
        Normalize financial numbers.

        Examples:
            "$123.45M" → "PRICE_MID"
            "+15.3%" → "PERCENT_POS"
            "-3.2%" → "PERCENT_NEG"
        """
        # Percentages
        text = re.sub(r"\+\d+\.?\d*%", "PERCENT_POS", text)
        text = re.sub(r"-\d+\.?\d*%", "PERCENT_NEG", text)
        text = re.sub(r"\d+\.?\d*%", "PERCENT", text)

        # Currency with scale
        text = re.sub(r"\$\d+\.?\d*B", "PRICE_LARGE", text)  # Billions
        text = re.sub(r"\$\d+\.?\d*M", "PRICE_MID", text)  # Millions
        text = re.sub(r"\$\d+\.?\d*K", "PRICE_SMALL", text)  # Thousands
        text = re.sub(r"\$\d+\.?\d*", "PRICE", text)

        return text

    def tokenize(self, text: str, return_tensors: str | None = None) -> dict:
        """
        Tokenize text.

        Args:
            text: Input text
            return_tensors: 'pt' for PyTorch, 'np' for NumPy

        Returns:
            dictionary with input_ids, attention_mask, etc.
        """
        # Preprocess
        processed_text = self.preprocess(text)

        # For now, return simple tokenization
        # In production, use transformers tokenizer
        tokens = {
            "input_ids": self._simple_tokenize(processed_text),
            "attention_mask": None,
            "text": processed_text,
        }

        return tokens

    def _simple_tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization (placeholder)."""
        return text.split()

    def batch_encode(self, texts: list[str], **kwargs) -> dict:
        """
        Batch tokenize multiple texts.

        Args:
            texts: List of input texts
            **kwargs: Additional arguments

        Returns:
            Batch of tokenized inputs
        """
        return [self.tokenize(text, **kwargs) for text in texts]

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """
        Extract financial entities from text.

        Args:
            text: Input text

        Returns:
            Dictionary of entity types to values
        """
        entities = {"tickers": [], "percentages": [], "currencies": [], "financial_terms": []}

        # Extract tickers
        entities["tickers"] = self.ticker_pattern.findall(text)

        # Extract percentages
        entities["percentages"] = self.percent_pattern.findall(text)

        # Extract currency amounts
        entities["currencies"] = self.currency_pattern.findall(text)

        # Extract financial terms
        words = text.lower().split()
        entities["financial_terms"] = [w for w in words if w in self.financial_terms]

        return entities


def create_financial_tokenizer(
    model_name: str = "distilroberta-base", max_length: int = 512
) -> FinancialTokenizer:
    """
    Factory function to create financial tokenizer.

    Args:
        model_name: Base model name
        max_length: Maximum sequence length

    Returns:
        Configured FinancialTokenizer
    """
    config = TokenizerConfig(max_length=max_length)
    return FinancialTokenizer(config=config, model_name=model_name)
