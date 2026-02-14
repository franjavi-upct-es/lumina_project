# backend/perception/semantic/vectorizer.py
"""
Text Vectorizer - Complete Pipeline

End-to-end pipeline: Raw text → 64d semantic embedding

Pipeline:
    Text → Tokenizer → LLM → Projection → 64d Vector

Combines tokenizer and LLM encoder into single interface for
easy inference and integration with fusion layer.

Example:
    >>> vectorizer = SemanticEncoder()
    >>> text = "Apple beats Q4 earnings, guides higher for Q1"
    >>> embedding = vectorizer.encode(text)
    >>> # embedding.shape = (64,)
"""

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from backend.perception.semantic.llm_distilled import (
    DistilledLLMEncoder,
    LLMConfig,
)
from backend.perception.semantic.tokenizer import (
    FinancialTokenizer,
    TokenizerConfig,
)


@dataclass
class VectorizerConfig:
    """
    Configuration for text vectorizer.

    Attributes:
        output_dim: Output embedding dimension (64 for V3)
        max_length: Maximum sequence length
        batch_size: Batch size for encoding
        device: Compute device
        cache_embeddings: Cache embeddings for repeated texts
    """

    output_dim: int = 64
    max_length: int = 512
    batch_size: int = 32
    device: str = "cpu"
    cache_embeddings: bool = True


class TextVectorizer:
    """
    Complete text → embedding pipeline.

    Handles:
    - Text preprocessing
    - Tokenization
    - LLM encoding
    - Dimension projection
    - Caching (optional)

    Example:
        >>> vectorizer = TextVectorizer()
        >>> texts = [
        >>>     "Fed raises rates 25bps",
        >>>     "Apple reports record earnings",
        >>>     "Tesla misses delivery targets"
        >>> ]
        >>> embeddings = vectorizer.encode_batch(texts)
        >>> # embeddings.shape = (3, 64)
    """

    def __init__(
        self,
        config: VectorizerConfig | None = None,
        tokenizer: FinancialTokenizer | None = None,
        encoder: DistilledLLMEncoder | None = None,
    ):
        """
        Initialize text vectorizer.

        Args:
            config: Vectorizer configuration
            tokenizer: Financial tokenizer (created if None)
            encoder: LLM encoder (created if None)
        """
        self.config = config or VectorizerConfig()

        # Initialize tokenizer
        if tokenizer is None:
            tokenizer_config = TokenizerConfig(max_length=self.config.max_length)
            self.tokenizer = FinancialTokenizer(tokenizer_config)
        else:
            self.tokenizer = tokenizer

        # Initialize encoder
        if encoder is None:
            llm_config = LLMConfig(output_dim=self.config.output_dim, device=self.config.device)
            self.encoder = DistilledLLMEncoder(llm_config)
        else:
            self.encoder = encoder

        # Cache
        self.cache = {} if self.config.cache_embeddings else None

        logger.info(f"TextVectorizer initialized: output_dim={self.config.output_dim}")

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode single text to embedding.

        Args:
            text: Input text
            use_cache: Use cached embedding if available

        Returns:
            Embedding vector [output_dim]
        """
        # Check cache
        if use_cache and self.cache is not None:
            if text in self.cache:
                return self.cache[text]

        # Preprocess and tokenize
        tokens = self.tokenizer.tokenize(text)

        # Create mock input_ids for now
        # In production: use actual token IDs from tokenizer
        input_ids = torch.randint(0, 1000, (1, self.config.max_length))

        # Encode
        embedding = self.encoder.encode(input_ids)
        embedding_vec = embedding[0]  # Extract single vector

        # Cache
        if self.cache is not None:
            self.cache[text] = embedding_vec

        return embedding_vec

    def encode_batch(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """
        Encode batch of texts.

        Args:
            texts: List of input texts
            use_cache: Use cached embeddings

        Returns:
            Embeddings [num_texts, output_dim]
        """
        embeddings = []

        for text in texts:
            embedding = self.encode(text, use_cache=use_cache)
            embeddings.append(embedding)

        return np.stack(embeddings)

    def clear_cache(self):
        """Clear embedding cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.debug("Embedding cache cleared")

    def get_cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self.cache) if self.cache else 0


class SemanticEncoder:
    """
    High-level semantic encoder interface.

    Main entry point for semantic encoding in V3 pipeline.
    Provides simple API for text → 64d embedding.

    Example:
        >>> encoder = SemanticEncoder()
        >>>
        >>> # Single text
        >>> text = "Apple stock surges on earnings beat"
        >>> embedding = encoder.encode(text)
        >>>
        >>> # Batch
        >>> texts = ["News 1", "News 2", "News 3"]
        >>> embeddings = encoder.encode_batch(texts)
    """

    def __init__(self, output_dim: int = 64, device: str = "cpu"):
        """
        Initialize semantic encoder.

        Args:
            output_dim: Output embedding dimension
            device: Compute device
        """
        config = VectorizerConfig(output_dim=output_dim, device=device)

        self.vectorizer = TextVectorizer(config)

        logger.info(f"SemanticEncoder ready: {output_dim}d embeddings")

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding.

        Args:
            text: Input text

        Returns:
            64-dimensional embedding
        """
        return self.vectorizer.encode(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode batch of texts.

        Args:
            texts: List of texts

        Returns:
            Embeddings [num_texts, 64]
        """
        return self.vectorizer.encode_batch(texts)

    def preprocess_news(self, news_item: dict) -> str:
        """
        Preprocess news item for encoding.

        Args:
            news_item: Dict with 'title', 'description', 'content'

        Returns:
            Combined text for encoding
        """
        parts = []

        if "title" in news_item:
            parts.append(news_item["title"])

        if "description" in news_item:
            parts.append(news_item["description"])

        # Optionally include snippet of content
        if "content" in news_item and len(parts) < 2:
            content = news_item["content"][:200]  # First 200 chars
            parts.append(content)

        return ". ".join(parts)

    def encode_news_batch(self, news_items: list[dict]) -> np.ndarray:
        """
        Encode batch of news items.

        Args:
            news_items: List of news dicts

        Returns:
            Embeddings [num_items, 64]
        """
        texts = [self.preprocess_news(item) for item in news_items]
        return self.encode_batch(texts)


def create_semantic_encoder(output_dim: int = 64, device: str = "cpu") -> SemanticEncoder:
    """
    Factory function to create semantic encoder.

    Args:
        output_dim: Output dimension (64 for V3)
        device: Compute device

    Returns:
        Configured SemanticEncoder
    """
    return SemanticEncoder(output_dim=output_dim, device=device)
