# backend/ml_engine/evaluation/error_analysis.py
"""
Comprehensive error analysis for ML models
Identifies patterns in predictions errors and model weaknesses
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import confusion_matrix


class ErrorAnalyzer:
    """
    Analyze prediction errors to identify model weaknesses

    Features:
    - Error distribution analysis
    - Error patterns by feature ranges
    - Temporal error analysis
    - Residual diagnostics
    - Outlier detection
    """

    def __init__(self):
        self.errors: np.ndarray | None = None
        self.predictions: np.ndarray | None = None
        self.actuals: np.ndarray | None = None
        self.features: pd.DataFrame | None = None

    def set_data(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        features: pd.DataFrame | None = None,
    ):
        """
        Set prediction data for analysis

        Args:
            predictions: Model predictions
            actuals: Actual values
            features: Feature values used for predictions
        """
        self.predictions = predictions.flatten()
        self.actuals = actuals.flatten()
        self.errors = self.predictions - self.actuals
        self.features = features

        logger.info(f"Error analyzer initialized with {len(self.errors)} samples")

    def analyze_error_distribution(self) -> dict[str, Any]:
        """
        Analyze the distribution of errors

        Returns:
            dictionary with distribution statistics
        """
        if self.errors is None:
            raise ValueError("No data set. Call set_data() first.")

        # Basic statistics
        mean_error = float(np.mean(self.errors))
        median_error = float(np.median(self.errors))
        std_error = float(np.std(self.errors))

        # Percentiles
        percentiles = {
            "5th": float(np.percentile(self.errors, 5)),
            "25th": float(np.percentile(self.errors, 25)),
            "75th": float(np.percentile(self.errors, 75)),
            "95th": float(np.percentile(self.errors, 95)),
        }

        # Skewness and kurtosis
        skewness = float(stats.skew(self.errors))
        kurtosis = float(stats.kurtosis(self.errors))

        # Test for normality
        _, normality_pvalue = stats.normaltest(self.errors)
        is_normal = normality_pvalue > 0.05

        # Bias analysis
        positive_errors = self.errors[self.errors > 0]
        negative_errors = self.errors[self.errors < 0]

        bias_ratio = len(positive_errors) / len(self.errors)

        return {
            "mean_error": mean_error,
            "median_error": median_error,
            "std_error": std_error,
            "percentiles": percentiles,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "is_normal": bool(is_normal),
            "normality_pvalue": float(normality_pvalue),
            "positive_error_ratio": float(bias_ratio),
            "avg_positive_error": float(np.mean(positive_errors))
            if len(positive_errors) > 0
            else 0.0,
            "avg_negative_error": float(np.mean(negative_errors))
            if len(negative_errors) > 0
            else 0.0,
        }

    def analyze_error_by_magnitude(self, bins: int = 10) -> pd.DataFrame:
        """
        Analyze how vary with prediction magnitude

        Args:
            bins: Number of bins to divide predictions into

        Returns:
            DataFrame with error statistics by magnitude bin
        """
        if self.predictions is None:
            raise ValueError("No data set")

        # Create bins based on prediction magnitude
        pred_bins = pd.qcut(self.predictions, q=bins, duplicates="drop")

        # Group errors by bins
        error_df = pd.DataFrame(
            {
                "prediction": self.predictions,
                "actual": self.actuals,
                "error": self.errors,
                "abs_error": np.abs(self.errors),
                "squared_error": self.errors**2,
                "bin": pred_bins,
            }
        )

        # Aggregate by bin
        grouped = error_df.groupby("bin").agg(
            {
                "prediction": ["mean", "min", "max", "count"],
                "error": ["mean", "std"],
                "abs_error": ["mean", "max"],
                "squared_error": "mean",
            }
        )

        grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]

        return grouped.reset_index()

    def analyze_error_by_feature(self, feature_name: str, bins: int = 10) -> pd.DataFrame:
        """
        Analyze how errors vary with a specific feature

        Args:
            feature_name: Name of feature to analyze
            bins: Number of bins

        Returns:
            DataFrame with error statistics by feature range
        """
        if self.features is None or feature_name not in self.features.columns:
            raise ValueError(f"Feature {feature_name} not available")

        feature_values = self.features[feature_name].values

        # Create bins
        try:
            feature_bins = pd.qcut(feature_values, q=bins, duplicates="drop")
        except ValueError:
            # If qcut fails, use cut instead
            feature_bins = pd.cut(feature_values, bins=bins)

        # Group errors
        error_df = pd.DataFrame(
            {
                "feature": feature_values,
                "error": self.errors,
                "abs_error": np.abs(self.errors),
                "bin": feature_bins,
            }
        )

        grouped = error_df.groupby("bin").agg(
            {
                "feature": ["mean", "min", "max", "count"],
                "error": ["mean", "std"],
                "abs_error": ["mean", "max"],
            }
        )

        grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]

        return grouped.reset_index()

    def analyze_temporal_errors(self, timestamps: pd.DatetimeIndex | None = None) -> dict[str, Any]:
        """
        Analyze how errors change over time

        Args:
            timestamps: Timestamps for each prediction

        Returns:
            dictionary with temporal analysis
        """
        if timestamps is None and self.features is not None:
            if "time" in self.features.columns:
                timestamps = pd.to_datetime(self.features["time"])
            else:
                raise ValueError("No timestamps provided")

        if timestamps is None:
            raise ValueError("Timestamps required for temporal analysis")

        # Create time series of errors
        error_series = pd.Series(self.errors, index=timestamps)

        # Rolling statistics
        rolling_mean = error_series.rolling(window=20).mean()
        rolling_std = error_series.rolling(window=20).std()

        # Test for trend
        time_numeric = np.arange(len(error_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, self.errors)

        has_trend = p_value < 0.05

        # Autocorrelation
        from statsmodels.tsa.stattools import acf

        autocorr = acf(self.errors, nlags=10, fft=True)

        return {
            "has_trend": bool(has_trend),
            "trend_slope": float(slope),
            "trend_pvalue": autocorr.tolist(),
            "mean_error_drift": float(rolling_mean.iloc[-1] - rolling_mean.iloc[0]),
            "volatility_drift": float(rolling_std.iloc[-1] - rolling_std.iloc[0]),
        }

    def detect_outliers(
        self,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Detect outlier predictions

        Args:
            method: Method to use ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            Tuple of (outlier_mask. outlier_info)
        """
        if method == "iqr":
            q1 = (np.percentile(self.errors, 25),)
            q3 = (np.percentile(self.errors, 75),)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outlier_mask = (self.errors < lower_bound) | (self.errors > upper_bound)

            info = {
                "method": "iqr",
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "num_outliers": int(outlier_mask.sum()),
                "outlier_ratio": float(outlier_mask.mean()),
            }

        elif method == "zscore":
            z_scores = np.abs(stats.zscore(self.errors))
            outlier_mask = z_scores > threshold

            info = {
                "method": "zscore",
                "threshold": threshold,
                "num_outliers": int(outlier_mask.sum()),
                "outlier_ratio": float(outlier_mask.mean()),
                "max_zscore": float(z_scores.max()),
            }

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_pred = iso_forest.fit_predict(self.errors.reshape(-1, 1))
            outlier_mask = outlier_pred == -1

            info = {
                "method": "isolation_forest",
                "num_outliers": int(outlier_mask.sum()),
                "outlier_ratio": float(outlier_mask.mean()),
            }
        else:
            raise ValueError(f"Unknown method: {method}")

        return outlier_mask, info

    def residual_diagnostics(self) -> dict[str, Any]:
        """
        Perform residual diagnostics

        Returns:
            dictionary with diagnostics results
        """
        # Residuals vs fitted
        residuals = self.errors
        fitted = self.predictions

        # Test for heteroscedasticity (Breusch-Pagan test)
        from scipy import stats as sp_stats

        # Simple heteroscedasticity test: correlation between |residuals| and fitted
        abs_residuals = np.abs(residuals)
        hetero_corr, hetero_pval = sp_stats.pearsonr(fitted, abs_residuals)

        has_heteroscedasticity = hetero_pval < 0.05

        # Test for autocorrelation (Durbin-Watson)
        from statsmodels.stats.stattools import durbin_watson

        dw_stat = durbin_watson(residuals)

        # DW stat around 2 indicates no autocorrelation
        # <1 or >3 indicates autocorrelation
        has_autocorrelation = dw_stat < 1.5 or dw_stat > 2.5

        # Q-Q plot data for normality check
        theoretical_quantiles = stats.probplot(residuals, dist="norm")[0][0]
        sample_quantiles = stats.probplot(residuals, dist="norm")[0][1]

        # Correlation with theoretical normal
        qq_correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]

        return {
            "heteroscedasticity": {
                "present": bool(has_heteroscedasticity),
                "correlation": float(hetero_corr),
                "pvalue": float(hetero_pval),
            },
            "autocorrelation": {
                "present": bool(has_autocorrelation),
                "durbin_watson": float(dw_stat),
            },
            "normality": {
                "qq_correlation": float(qq_correlation),
                "is_normal": qq_correlation > 0.99,
            },
        }

    def error_breakdown_by_direction(self) -> dict[str, Any]:
        """
        Analyze errors separately for over and under predictions

        Returns:
            dictionary with directional error analysis
        """
        over_predictions = self.errors > 0
        under_predictions = self.errors < 0

        over_errors = self.errors[over_predictions]
        under_errors = self.errors[under_predictions]

        return {
            "over_predictions": {
                "count": int(over_predictions.sum()),
                "ratio": float(over_predictions.mean()),
                "mean_error": float(np.mean(over_errors)) if len(over_errors) > 0 else 0.0,
                "median_error": float(np.median(over_errors)) if len(over_errors) > 0 else 0.0,
                "max_error": float(np.max(over_errors)) if len(over_errors) > 0 else 0.0,
            },
            "under_predictions": {
                "count": int(under_predictions.sum()),
                "ratio": float(under_predictions.mean()),
                "mean_error": float(np.mean(under_errors)) if len(under_errors) > 0 else 0.0,
                "median_error": float(np.median(under_errors)) if len(under_errors) > 0 else 0.0,
                "min_error": float(np.min(under_errors)) if len(under_errors) > 0 else 0.0,
            },
        }

    def generate_error_report(self) -> dict[str, Any]:
        """
        Generate comprehensive error analysis report

        Returns:
            Complete error analysis report
        """
        logger.info("Generating comprehensive error report")

        report = {
            "summary": {
                "total_samples": len(self.errors),
                "mean_absolute_error": float(np.mean(np.abs(self.errors))),
                "root_mean_squared_error": float(np.sqrt(np.mean(self.errors**2))),
                "mean_error": float(np.mean(self.errors)),
                "median_absolute_error": float(np.median(np.abs(self.errors))),
            },
            "distribution": self.analyze_error_distribution(),
            "directional": self.error_breakdown_by_direction(),
            "diagnostics": self.residual_diagnostics(),
        }

        # Add magnitude analysis
        try:
            report["by_magnitude"] = self.analyze_error_by_magnitude().to_dict("records")
        except Exception as e:
            logger.warning(f"Could not analyze by magnitude: {e}")

        # Add outlier detection
        try:
            outlier_mask, outlier_info = self.detect_outliers(method="iqr")
            report["outliers"] = outlier_info
        except Exception as e:
            logger.warning(f"Could not detect outliers: {e}")

        logger.success("Error report generated")

        return report


class ClassificationErrorAnalyzer:
    """
    Error analyzer specifically for classification models
    """

    def __init__(self):
        self.predictions: np.ndarray | None = None
        self.actuals: np.ndarray | None = None
        self.probabilities: np.ndarray | None = None

    def set_data(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: np.ndarray | None = None,
    ):
        """Set prediction data"""
        self.predictions = predictions
        self.actuals = actuals
        self.probabilities = probabilities

    def confusion_matrix_analysis(self) -> dict[str, Any]:
        """
        Analyze confusion matrix

        Returns:
            Confusion matrix metrics
        """
        cm = confusion_matrix(self.actuals, self.predictions)

        # Calculate per-class metrics
        num_classes = cm.shape[0]
        per_class_metrics = {}

        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class_metrics[f"class_{i}"] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": int(cm[i, :].sum()),
            }

        return {
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class_metrics,
            "overall_accuracy": float(cm.diagonal().sum() / cm.sum()),
        }

    def calibration_analysis(self) -> dict[str, Any]:
        """
        Analyze probability calibration

        Returns:
            Calibration metrics
        """
        if self.probabilities is None:
            raise ValueError("Probabilities required for calibration analysis")

        from sklearn.calibration import calibration_curve

        # Binary classification calibration
        if self.probabilities.shape[1] == 2:
            prob_true, prob_pred = calibration_curve(
                self.actuals, self.probabilities[:, 1], n_bins=10
            )

            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(prob_true - prob_pred))

            return {
                "expected_calibration_error": float(ece),
                "calibration_curve": {
                    "prob_true": prob_true.tolist(),
                    "prob_pred": prob_pred.tolist(),
                },
            }

        return {"message": "Multi-class calibration not yet implemented"}


def compare_error_distributions(
    errors1: np.ndarray, errors2: np.ndarray, names: tuple[str, str] = ("Model 1", "Model 2")
) -> dict[str, Any]:
    """
    Compare error distributions between two models

    Args:
        errors1: Errors from first model
        errors2: Errors from second model
        names: Names for the models

    Returns:
        Comparison results
    """
    # Statistical test for difference
    t_stat, p_value = stats.ttest_ind(errors1, errors2)

    # Compare MAE
    mae1 = np.mean(np.abs(errors1))
    mae2 = np.mean(np.abs(errors2))
    mae_improvement = (mae1 - mae2) / mae1 * 100

    # Compare RMSE
    rmse1 = np.sqrt(np.mean(errors1**2))
    rmse2 = np.sqrt(np.mean(errors2**2))
    rmse_improvement = (rmse1 - rmse2) / rmse1 * 100

    return {
        "statistical_test": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significantly_different": p_value < 0.05,
        },
        "mae_comparison": {
            names[0]: float(mae1),
            names[1]: float(mae2),
            "improvement_pct": float(mae_improvement),
        },
        "rmse_comparison": {
            names[0]: float(rmse1),
            names[1]: float(rmse2),
            "improvement_pct": float(rmse_improvement),
        },
    }
