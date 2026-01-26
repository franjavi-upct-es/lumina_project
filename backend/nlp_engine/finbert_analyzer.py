# backend/nlp_engine/finbert_analyzer.py
"""
FinBERT sentiment analyzer for financial text
Uses pre-trained FinBERT model for domain-specific sentiment analysis
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinBERTAnalyzer:
    """
    Financial sentiment anlyzer using FinBERT

    FinBERT is BERT fine-tuned on financial text for sentiment analysis.
    Returns sentiment scores: positive, negative, neutral
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        Initialize FinBERT analyzer

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        logger.info(f"Initializing FinBERT analyzer with model: {model_name}")

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        self.batch_size = batch_size
        self.max_length = max_length

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Label mapping
            self.labels = ["negative", "neutral", "positive"]

            logger.success("FinBERT model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise

    def analyze(self, texts: list[str], return_all_scores: bool = False) -> list[dict[str, Any]]:
        """
        Analyze sentiment of texts

        Args:
            texts: List of texts to analyze
            return_all_scores: Return scores for all labels

        Returns:
            List of sentiment results
        """
        if not texts:
            return []

        logger.info(f"Analyzing sentiment for {len(texts)} texts")

        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_results = self._analyze_batch(batch_texts, return_all_scores)
            results.extend(batch_results)

        logger.success(f"Analyzed {len(texts)} texts")
        return results

    def _analyze_batch(
        self,
        texts: list[str],
        return_all_scores: bool,
    ) -> list[dict[str, Any]]:
        """
        Analyze a batch of texts

        Returns:
            List of sentiment dictionaries
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)

            # Move to CPU and convert to numpy
            probabilities = probabilities.cpu().numpy()

            # Build results
            results = []
            for probs in probabilities:
                # Find dominant sentiment
                dominant_idx = np.argmax(probs)
                dominant_label = self.labels[dominant_idx]
                dominant_score = float(probs[dominant_idx])

                result = {
                    "sentiment": dominant_label,
                    "score": dominant_score,
                    "confidence": dominant_score,  # Alias
                }

                # Include all scores if requested
                if return_all_scores:
                    result["scores"] = {
                        label: float(score) for label, score in zip(self.labels, probs)
                    }

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error analyzing batch: {e}")
            # Return neutral sentiment as fallback
            return [{"sentiment": "neutral", "score": 0.33, "confidence": 0.33} for _ in texts]

    def analyze_dataframe(
        self, df: pd.DataFrame, text_column: str = "text", return_all_scores: bool = False
    ) -> pd.DataFrame:
        """
        Analyze sentiment for texts in a DataFrame

        Args:
            df: Input DataFrame
            text_column: Column containing text
            return_all_scores: Return all sentiment scores

        Returns:
            DataFrame with added sentiment columns
        """
        if text_column not in df.columns:
            logger.error(f"Column {text_column} not found in DataFrame")
            return df

        logger.info(f"Analyzing sentiment for {len(df)} rows")

        # Get texts
        texts = df[text_column].fillna("").tolist()

        # Analyze
        results = self.analyze(texts, return_all_scores)

        # Add results to DataFrame
        df["sentiment"] = [r["sentiment"] for r in results]
        df["sentiment_score"] = [r["score"] for r in results]
        df["sentiment_confidence"] = [r["confidence"] for r in results]

        if return_all_scores:
            df["sentiment_positive"] = [r["scores"]["positive"] for r in results]
            df["sentiment_negative"] = [r["scores"]["negative"] for r in results]
            df["sentiment_neutral"] = [r["scores"]["neutral"] for r in results]

        logger.success("Sentiment analysis complete")
        return df

    def analyze_polars(
        self, df: pl.DataFrame, text_column: str = "text", return_all_scores: bool = False
    ) -> pl.DataFrame:
        """
        Analyze sentiment for Polars DataFrame

        Args:
            df: Input Polars DataFrame
            text_column: Column containing text
            return_all_scores: Return all sentiment scores

        Returns:
            DataFrame with added sentiment columns
        """
        # Convert to pandas
        df_pd = df.to_pandas()

        # Analyze
        df_pd = self.analyze_dataframe(df_pd, text_column, return_all_scores)

        # Convert back to polars
        return pl.from_pandas(df_pd)

    def aggregate_sentiment(
        self,
        texts: list[str],
        weights: list[float] | None = None,
    ) -> dict[str, float]:
        """
        Aggregate sentiment across multiple texts

        Args:
            texts: List of texts
            weights: Optional weights for each text

        Returns:
            Dictionary with aggregate sentiment
        """
        if not texts:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
            }

        # Analyze all texts
        results = self.analyze(texts, return_all_scores=True)

        # Extract scores
        positive_scores = [r["scores"]["positive"] for r in results]
        negative_scores = [r["scores"]["negative"] for r in results]
        neutral_scores = [r["scores"]["neutral"] for r in results]

        # Apply weights
        if weights is None:
            weights = np.ones(len(texts)) / len(texts)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()

        # Weighted average
        avg_positive = float(np.average(positive_scores, weights=weights))
        avg_negative = float(np.average(negative_scores, weights=weights))
        avg_neutral = float(np.average(neutral_scores, weights=weights))

        # Determine overall sentiment
        scores = {
            "positive": avg_positive,
            "negative": avg_negative,
            "neutral": avg_neutral,
        }

        dominant_sentiment = max(scores.items(), key=lambda x: x[1])

        return {
            "sentiment": dominant_sentiment[0],
            "score": dominant_sentiment[1],
            **scores,
            "num_texts": len(texts),
        }

    def sentiment_time_series(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        time_column: str = "time",
        resample_rule: str = "1D",
    ) -> pd.DataFrame:
        """
        Create sentiment time series

        Args:
            df: DataFrame with texts and timestamps
            text_column: Text column name
            time_column: Time column name
            resample_rule: Pandas resample rule (e.g., '1D', '1H')

        Returns:
            DataFrame with sentiment time series
        """
        logger.info(f"Creating sentiment time series with rule: {resample_rule}")

        # Analyze sentiment
        df = self.analyze_dataframe(df, text_column, return_all_scores=True)

        # Set time as index
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)

        # Resample and aggregate
        sentiment_ts = df.resample(resample_rule).agg(
            {
                "sentiment_positive": "mean",
                "sentiment_negative": "mean",
                "sentiment_neutral": "mean",
                "sentiment_score": "mean",
                text_column: "count",  # Number of texts per period
            }
        )

        # Resume count column
        sentiment_ts = sentiment_ts.rename(columns={text_column: "num_texts"})

        # Calculate net sentiment
        sentiment_ts["net_sentiment"] = (
            sentiment_ts["sentiment_positive"] - sentiment_ts["sentiment_negative"]
        )

        logger.success(f"Created sentiment time series with {len(sentiment_ts)} periods")
        return sentiment_ts

    def batch_process_file(
        self,
        input_file: str,
        output_file: str,
        text_column: str = "text",
        chunk_size: int = 1000,
    ):
        """
        Process large file in chunks

        Args:
            input_file: Input CSV/parquet file
            output_file: Output file path
            text_column: Text column name
            chunk_size: Chunk size for processing
        """
        logger.info(f"Processing file: {input_file}")

        # Detect file type
        input_path = Path(input_file)
        file_type = input_path.suffix.lower()

        if file_type == ".csv":
            reader = pd.read_csv(input_file, chunksize=chunk_size)
        elif file_type == ".parquet":
            # Read parquet in chunks
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(input_file)
            reader = parquet_file.iter_batches(batch_size=chunk_size)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Process chunks
        first_chunk = True
        total_processed = 0

        for chunk in reader:
            if file_type == ".parquet":
                chunk = chunk.to_pandas()

            # Analyze sentiment
            chunk = self.analyze_dataframe(chunk, text_column, return_all_scores=True)

            # Write to output
            if first_chunk:
                chunk.to_csv(output_file, index=False, mode="w")
                first_chunk = False
            else:
                chunk.to_csv(output_file, index=False, mode="a", header=False)

            total_processed += len(chunk)
            logger.info(f"Processed {total_processed} rows")

        logger.success(f"File processing complete: {output_file}")

    def compare_sentiments(
        self,
        texts1: list[str],
        texts2: list[str],
        labels: tuple[str, str] = ("Group 1", "Group 2"),
    ) -> dict[str, Any]:
        """
        Compare sentiment between two groups of texts

        Args:
            texts1: First group of texts
            texts2: Second group of texts
            labels: Labels for the groups

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Companing sentiment: {labels[0]} vs {labels[1]}")

        # Aggregate sentiment for each group
        agg1 = self.aggregate_sentiment(texts1)
        agg2 = self.aggregate_sentiment(texts2)

        # Calculate differences
        comparison = {
            "group1": {
                "label": labels[0],
                "sentiment": agg1,
            },
            "group2": {"label": labels[1], "sentiment": agg2},
            "difference": {
                "positive": agg1["positive"] - agg2["positive"],
                "negative": agg1["negative"] - agg2["negative"],
                "neutral": agg1["neutral"] - agg2["neutral"],
                "score": agg1["score"] - agg2["score"],
            },
        }

        # Determine which group is more positive
        if agg1["score"] > agg2["score"]:
            comparison["more_positive"] = labels[0]
        elif agg2["score"] > agg1["score"]:
            comparison["more_positive"] = labels[1]
        else:
            comparison["more_positive"] = "equal"

        return comparison
