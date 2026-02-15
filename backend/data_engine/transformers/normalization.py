# backend/data_engine/transformers/normalization.py
"""
Feature Normalization for V3
============================

Normalizes features for neural network inputs.
Supports multiple normalization strategies.

Version: 3.0.0
"""

from typing import Literal

import polars as pl
from loguru import logger

NormalizationMethod = Literal["zscore", "minmax", "robust", "log", "none"]


class FeatureNormalizer:
    """
    Feature normalization for neural networks

    Supports multiple normalization methods optimized for different feature types.
    """

    def __init__(self):
        """Initialize feature normalizer"""
        self.stats = {}
        logger.info("FeatureNormalizer initialized")

    def fit_transform(
        self,
        data: pl.DataFrame,
        features: list[str],
        method: NormalizationMethod = "zscore",
    ) -> pl.DataFrame:
        """
        Fit normalizer and transform features

        Args:
            data: Input DataFrame
            features: Features to normalize
            method: Normalization method

        Returns:
            DataFrame with normalized features
        """
        df = data.clone()

        for feature in features:
            if feature not in df.columns:
                continue

            try:
                # Calculate statistics
                if method == "zscore":
                    mean = df[feature].mean()
                    std = df[feature].std()
                    self.stats[feature] = {"mean": mean, "std": std}

                    df = df.with_columns([((pl.col(feature) - mean) / std).alias(feature)])

                elif method == "minmax":
                    min_val = df[feature].min()
                    max_val = df[feature].max()
                    self.stats[feature] = {"min": min_val, "max": max_val}

                    df = df.with_columns(
                        [((pl.col(feature) - min_val) / (max_val - min_val)).alias(feature)]
                    )

                elif method == "robust":
                    q25 = df[feature].quantile(0.25)
                    q75 = df[feature].quantile(0.75)
                    iqr = q75 - q25
                    median = df[feature].median()
                    self.stats[feature] = {"median": median, "iqr": iqr}

                    df = df.with_columns([((pl.col(feature) - median) / iqr).alias(feature)])

                elif method == "log":
                    # Log transform (for skewed distributions)
                    df = df.with_columns([pl.col(feature).log().alias(feature)])

            except Exception as e:
                logger.error(f"Error normalizing {feature}: {e}")

        logger.success(f"Normalized {len(features)} features using {method}")
        return df

    def transform(
        self,
        data: pl.DataFrame,
        features: list[str],
    ) -> pl.DataFrame:
        """
        Transform features using fitted statistics

        Args:
            data: Input DataFrame
            features: Features to transform

        Returns:
            DataFrame with transformed features
        """
        df = data.clone()

        for feature in features:
            if feature not in df.columns or feature not in self.stats:
                continue

            try:
                stats = self.stats[feature]

                if "mean" in stats and "std" in stats:
                    # Z-score
                    df = df.with_columns(
                        [((pl.col(feature) - stats["mean"]) / stats["std"]).alias(feature)]
                    )

                elif "min" in stats and "max" in stats:
                    # Min-max
                    df = df.with_columns(
                        [
                            (
                                (pl.col(feature) - stats["min"]) / (stats["max"] - stats["min"])
                            ).alias(feature)
                        ]
                    )

                elif "median" in stats and "iqr" in stats:
                    # Robust
                    df = df.with_columns(
                        [((pl.col(feature) - stats["median"]) / stats["iqr"]).alias(feature)]
                    )

            except Exception as e:
                logger.error(f"Error transforming {feature}: {e}")

        return df


# Global instance
_normalizer: FeatureNormalizer | None = None


def get_feature_normalizer() -> FeatureNormalizer:
    """Get global feature normalizer"""
    global _normalizer
    if _normalizer is None:
        _normalizer = FeatureNormalizer()
    return _normalizer
