# backend/perception/temporal/inference.py
"""
Temporal Inference Engine

High-level interface for real-time temporal encoding.
Combines preprocessor and encoder for production inference.

Pipeline:
    OHLCV DataFrame → Preprocessor → Features → TFT → 128d Embedding

Optimized for:
- Real-time encoding (<10ms latency)
- Batch processing
- Caching preprocessed features
- Redis integration ready

Example:
    >>> from backend.perception.temporal import TemporalInference
    >>>
    >>> inference = TemporalInference()
    >>>
    >>> # Real-time encoding
    >>> df = get_latest_ohlcv('AAPL', bars=60)
    >>> embedding = inference.encode(df)  # 128d vector
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from backend.perception.temporal.encoder import (
    TemporalEncoder,
    TFTConfig,
)
from backend.perception.temporal.preprocessor import (
    PreprocessorConfig,
    TemporalPreprocessor,
)


@dataclass
class InferenceConfig:
    """
    Configuration for temporal inference.

    Attributes:
        lookback_window: Sequence length
        output_dim: Embedding dimension (128)
        input_dim: Number of features (auto-detected)
        use_cache: Cache preprocessed features
        device: Compute device
    """

    lookback_window: int = 60
    output_dim: int = 128
    input_dim: int | None = None
    use_cache: bool = True
    device: str = "cpu"


class TemporalInference:
    """
    Complete inference pipeline for temporal encoding.

    Handles:
    - Feature preprocessing
    - Normalization
    - Model inference
    - Caching

    Example:
        >>> inference = TemporalInference()
        >>>
        >>> # Single encoding
        >>> df = load_ohlcv('AAPL')
        >>> emb = inference.encode(df)
        >>>
        >>> # Batch encoding
        >>> dfs = {'AAPL': df1, 'MSFT': df2, 'NVDA': df3}
        >>> embeddings = inference.encode_batch(dfs)
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        preprocessor: TemporalPreprocessor | None = None,
        encoder: TemporalEncoder | None = None,
    ):
        """
        Initialize inference engine.

        Args:
            config: Inference configuration
            preprocessor: Preprocessor instance
            encoder: Encoder instance
        """
        self.config = config or InferenceConfig()

        # Initialize preprocessor
        if preprocessor is None:
            preproc_config = PreprocessorConfig(lookback_window=self.config.lookback_window)
            self.preprocessor = TemporalPreprocessor(preproc_config)
        else:
            self.preprocessor = preprocessor

        # Initialize encoder (lazy if input_dim not known)
        self.encoder = encoder

        # Cache
        self.cache = {} if self.config.use_cache else None

        logger.info("TemporalInference initialized")

    def _ensure_encoder(self, input_dim: int):
        """Lazily initialize encoder with correct input dimension."""
        if self.encoder is None:
            tft_config = TFTConfig(
                input_dim=input_dim,
                output_dim=self.config.output_dim,
                lookback_window=self.config.lookback_window,
            )
            self.encoder = TemporalEncoder(tft_config)
            logger.debug(f"Initialized encoder: input_dim={input_dim}")

    def encode(self, df: pd.DataFrame, use_cache: bool = True) -> np.ndarray:
        """
        Encode OHLCV dataframe to embedding.

        Args:
            df: OHLCV dataframe
            use_cache: Use cached embedding if available

        Returns:
            128-dimensional embedding
        """
        # Create cache key
        cache_key = None
        if use_cache and self.cache is not None:
            # Simple cache key (in production: hash of df)
            cache_key = f"{len(df)}_{df['close'].iloc[-1]:.2f}"

            if cache_key in self.cache:
                return self.cache[cache_key]

        # Preprocess
        features, static = self.preprocessor.preprocess(df)

        # Ensure encoder initialized
        input_dim = features.shape[1]
        self._ensure_encoder(input_dim)

        # Encode
        embedding = self.encoder.encode(features)

        # Cache
        if cache_key and self.cache is not None:
            self.cache[cache_key] = embedding

        return embedding

    def encode_batch(self, dataframes: dict) -> dict:
        """
        Encode batch of dataframes.

        Args:
            dataframes: Dict of {symbol: dataframe}

        Returns:
            Dict of {symbol: embedding}
        """
        embeddings = {}

        for symbol, df in dataframes.items():
            try:
                embedding = self.encode(df)
                embeddings[symbol] = embedding
            except Exception as e:
                logger.warning(f"Failed to encode {symbol}: {e}")
                continue

        return embeddings

    def clear_cache(self):
        """Clear embedding cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.debug("Cache cleared")


def encode_real_time(df: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    """
    Quick function for real-time encoding.

    Args:
        df: OHLCV dataframe
        lookback: Sequence length

    Returns:
        128d temporal embedding
    """
    config = InferenceConfig(lookback_window=lookback)
    inference = TemporalInference(config)

    return inference.encode(df)
