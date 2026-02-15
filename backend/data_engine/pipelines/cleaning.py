# backend/data_engine/pipelines/cleaning.py
"""
Data Cleaning Pipeline for V3
=============================

Handles data quality, outlier detection, and missing value imputation.
Ensures clean data for perception encoders.

Version: 3.0.0
"""

import polars as pl
from loguru import logger


class CleaningPipeline:
    """
    Data cleaning and quality assurance pipeline

    Removes outliers, imputes missing values, and validates data quality.
    """

    def __init__(
        self,
        outlier_std: float = 5.0,
        max_missing_ratio: float = 0.1,
    ):
        """
        Initialize cleaning pipeline

        Args:
            outlier_std: Standard deviations for outlier detection
            max_missing_ratio: Maximum allowed missing value ratio
        """
        self.outlier_std = outlier_std
        self.max_missing_ratio = max_missing_ratio

        logger.info("CleaningPipeline initialized")

    def clean(
        self,
        data: pl.DataFrame,
        remove_outliers: bool = True,
        impute_missing: bool = True,
    ) -> pl.DataFrame:
        """
        Clean dataset

        Args:
            data: Input DataFrame
            remove_outliers: Remove outliers
            impute_missing: Impute missing values

        Returns:
            Cleaned DataFrame
        """
        df = data.clone()

        try:
            # Check data quality
            quality = self.check_quality(df)
            logger.info(f"Data quality: {quality['missing_ratio']:.2%} missing")

            # Remove duplicates
            df = df.unique(subset=["time"], keep="last")

            # Remove outliers
            if remove_outliers:
                df = self._remove_outliers(df)

            # Impute missing values
            if impute_missing:
                df = self._impute_missing(df)

            logger.success("Data cleaning complete")
            return df

        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return data

    def check_quality(self, data: pl.DataFrame) -> dict:
        """
        Check data quality metrics

        Returns:
            Dict with quality metrics
        """
        try:
            total_values = len(data) * len(data.columns)
            missing_count = data.null_count().sum().item() if len(data) > 0 else 0
            missing_ratio = missing_count / total_values if total_values > 0 else 0.0

            duplicates = len(data) - len(data.unique(subset=["time"]))

            return {
                "rows": len(data),
                "columns": len(data.columns),
                "missing_count": missing_count,
                "missing_ratio": missing_ratio,
                "duplicates": duplicates,
                "quality_ok": missing_ratio <= self.max_missing_ratio,
            }

        except Exception as e:
            logger.error(f"Error checking quality: {e}")
            return {"quality_ok": False, "error": str(e)}

    def _remove_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove outliers using z-score method"""
        try:
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]

            for col in numeric_cols:
                if col == "time":
                    continue

                mean = df[col].mean()
                std = df[col].std()

                if std > 0:
                    df = df.with_columns(
                        [
                            pl.when((pl.col(col) - mean).abs() > (self.outlier_std * std))
                            .then(None)
                            .otherwise(pl.col(col))
                            .alias(col)
                        ]
                    )

            logger.debug("Removed outliers")
            return df

        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return df

    def _impute_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Impute missing values"""
        try:
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]

            for col in numeric_cols:
                if col == "time":
                    continue

                # Forward fill then backward fill
                df = df.with_columns([pl.col(col).forward_fill().backward_fill().alias(col)])

            logger.debug("Imputed missing values")
            return df

        except Exception as e:
            logger.error(f"Error imputing missing: {e}")
            return df


# Global instance
_pipeline: CleaningPipeline | None = None


def get_cleaning_pipeline() -> CleaningPipeline:
    """Get global cleaning pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = CleaningPipeline()
    return _pipeline
