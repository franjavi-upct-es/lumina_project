# backend/ml_engine/evaluation/metrics.py
"""
Comprehensive metrics for evaluating financial ML models
Includes standard metrics plus finance-specific measures
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


class FinancialMetrics:
    """
    Calculate financial-specific metrics for model evaluation

    Includes:
    - Standard regression metrics (MAE, RMSE, R², MAPE)
    - Directional accuracy
    - Hit ratio
    - Information coefficient
    - Capture ratios
    """

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns_true: np.ndarray | None = None,
        returns_pred: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Calculate all available metrics

        Args:
            y_true: True values
            y_pred: Predicted values
            returns_true: True returns (optional)
            returns_pred: Predicted returns (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Standard regression metrics
        metrics.update(FinancialMetrics.regression_metrics(y_true, y_pred))

        # Directional metrics
        metrics.update(FinancialMetrics.directional_metrics(y_true, y_pred))

        # Financial metrics
        if returns_true is not None and returns_pred is not None:
            metrics.update(FinancialMetrics.returns_metrics(returns_true, returns_pred))

        return metrics

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Calculate standard regression metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        # Flattern arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            logger.warning("No valid samples for metrics calculation")
            return {}

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # MAPE (handle zero values)
        mask_nonzero = y_true != 0
        if mask_nonzero.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask_nonzero], y_pred[mask_nonzero]) * 100
        else:
            mape = np.inf

        # Additional metrics
        errors = y_pred - y_true
        me = np.mean(errors)  # Mean Error (bias)

        # Median absolute error
        medae = np.median(np.abs(errors))

        # Explained variance
        explained_var = 1 - (np.var(errors) / np.var(y_true))

        # Max error
        max_error = np.max(np.abs(errors))

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "mape": float(mape),
            "mean_error": float(me),
            "median_absolute_error": float(medae),
            "explained_variance": float(explained_var),
            "max_error": float(max_error),
        }

    @staticmethod
    def directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Calculate directional accuracy metrics
        Important for trading strategies

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of directional metrics
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        if len(y_true) < 2:
            return {}

        # Calculate changes (direction)
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)

        # Directional accuracy (same sign)
        correct_direction = np.sign(true_changes) == np.sign(pred_changes)
        directional_accuracy = np.mean(correct_direction)

        # Up/Down capture
        up_moves = true_changes > 0
        down_moves = true_changes < 0

        up_accuracy = np.mean(correct_direction[up_moves]) if up_moves.sum() > 0 else 0
        down_accuracy = np.mean(correct_direction[down_moves]) if down_moves.sum() > 0 else 0

        # Hit ratio (different from directional accuracy)
        # Measures if prediction magnitude is correct
        prediction_correct = np.abs(y_pred - y_true) < np.std(y_true) * 0.5
        hit_ratio = np.mean(prediction_correct)

        return {
            "directional_accuracy": float(directional_accuracy),
            "up_move_accuracy": float(up_accuracy),
            "down_move_accuracy": float(down_accuracy),
            "hit_ratio": float(hit_ratio),
        }

    @staticmethod
    def returns_metrics(
        returns_true: np.ndarray,
        returns_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate metrics based on returns

        Args:
            returns_true: True returns
            returns_pred: Predicted returns

        Returns:
            Dictionary of return-based metrics
        """
        returns_true = returns_true.flatten()
        returns_pred = returns_pred.flatten()

        # Information Coefficient (IC)
        # Correlation between predicted and actual returns
        ic, ic_pvalue = stats.spearmanr(returns_pred, returns_true)

        # Rank Information Coefficient
        rank_ic, rank_ic_pvalue = stats.spearmanr(
            stats.rankdata(returns_pred), stats.rankdata(returns_true)
        )

        # Capture ratios
        up_periods = returns_true > 0
        down_periods = returns_true < 0

        if up_periods.sum() > 0:
            up_capture = np.mean(returns_pred[up_periods]) / np.mean(returns_true[up_periods])
        else:
            up_capture = 0.0

        if down_periods.sum() > 0:
            down_capture = np.mean(returns_pred[down_periods]) / np.mean(returns_true[down_periods])
        else:
            down_capture = 0.0

        return {
            "information_coefficient": float(ic),
            "ic_pvalue": float(ic_value),
            "rank_ic": float(rank_ic),
            "rank_ic_pvalue": float(rank_ic_pvalue),
            "up_capture_ratio": float(up_capture),
            "down_capture_ratio": float(down_capture),
        }

    @staticmethod
    def prediction_interval_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        confidence_level: float = 0.95,
    ) -> dict[str, float]:
        """
        Evaluate prediction intervals

        Args:
            y_true: True values
            y_pred: Predicted values
            lower_bound: Lower prediction bound
            upper_bound: Upper prediction bound
            confidence_level: Confidence level of intervals

        Returns:
            Interval quality metrics
        """
        # Coverage (percentage of actuals withing intervals)
        within_interval = (y_true >= lower_bound) / (y_true <= upper_bound)
        coverage = np.mean(within_interval)

        # Average interval width
        avg_width = np.mean(upper_bound - lower_bound)

        # Calibration (coverage should match confidence level)
        calibration_error = abs(coverage - confidence_level)

        # Sharpness (narrower intervals are better, given good coverage)
        sharpness = avg_width / np.std(y_true)

        return {
            "coverage": float(coverage),
            "avg_interval_width": float(avg_width),
            "calibration_error": float(calibration_error),
            "sharpness": float(sharpness),
            "is_well_calibrated": calibration_error < 0.05,
        }

    @staticmethod
    def time_series_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int | None = None
    ) -> dict[str, float]:
        """
        Time series specific metrics

        Args:
            y_true: True values
            y_pred: Predicted values
            seasonal_period: Period of seasonality (if applicable)

        Returns:
            Time series metrics
        """
        # Theil's U statistic
        # Compares forecast to naive forecast
        naive_pred = np.roll(y_true, 1)[1:]
        y_true_shift = y_true[1:]
        y_pred_shift = y_pred[1:]

        mse_model = np.mean((y_true_shift - y_pred_shift) ** 2)
        mse_naive = np.mean((y_true_shift - naive_pred) ** 2)

        theil_u = np.sqrt(mse_model) / np.sqrt(mse_naive) if mse_naive > 0 else np.inf

        # Forecast bias
        forecast_bias = np.mean(y_pred - y_true)

        # Mean absolute scaled error (MASE)
        if seasonal_period and len(y_true) > seasonal_period:
            naive_seasonal = y_true[:-seasonal_period]
            y_true_seasonal = y_true[seasonal_period:]

            mae_naive = np.mean(np.abs(y_true_seasonal - naive_seasonal))
            mae_model = np.mean(np.abs(y_true - y_pred))

            mase = mae_model / mae_naive if mae_naive > 0 else np.inf
        else:
            mase = np.nan

        return {
            "theil_u": float(theil_u),
            "forecast_bias": float(forecast_bias),
            "mase": float(mase) if not np.isnan(mase) else None,
        }


class BacktestMetrics:
    """
    Metrics for evaluating model performance in backtesting context
    """

    @staticmethod
    def trading_metrics(
        predictions: np.ndarray,
        actuals: np.ndarray,
        prices: np.ndarray,
        transaction_cost: float = 0.001,
    ) -> dict[str, float]:
        """
        Calculate trading-specific metrics

        Args:
            predictions: Model predictions
            actuals: Actual values
            prices: Price series
            transaction_cost: Transaction cost rate

        Returns:
            Trading metrics
        """
        # Generate signals from predictions
        pred_changes = np.diff(predictions)
        signals = np.sign(pred_changes)

        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]

        # Strategy returns ( assuming we follow signals)
        strategy_returns = signals * price_returns[1:]

        # Apply transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
        costs = position_changes * transaction_cost
        strategy_returns -= costs[:-1]

        # Calculate metrics
        total_return = np.prod(1 + strategy_returns) - 1

        # Annualized metrics
        num_periods = len(strategy_returns)
        periods_per_year = 252  # Assuming daily data
        years = num_periods / periods_per_year

        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = np.std(strategy_returns) * np.sqrt(periods_per_year)

        # Sharpe ratio
        risk_free_rate = 0.05
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Win rate
        winning_trades = strategy_returns > 0
        win_rate = np.mean(winning_trades)

        # Maximum drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "win_rate": float(win_rate),
            "max_drawdown": float(max_drawdown),
            "num_trades": int(position_changes.sum()),
        }


def compare_models(
    models_result: dict[str, dict[str, float]], metric: str = "rmse"
) -> pd.DataFrame:
    """
    Compare multiple models on a specific metric

    Args:
        models_result: Dictionary of model_name -> metrics
        metric: Metric to compare on

    Returns:
        DataFrame with comparison
    """
    comparison = []

    for model_name, metrics in models_result.items():
        if metric in metrics:
            comparison.append(
                {
                    "model": model_name,
                    metric: metrics[metric],
                }
            )

    df = pd.DataFrame(comparison)

    # Sort by metric (lower is better for most metrics)
    if metric in ["mae", "mse", "rmse", "mape", "max_error"]:
        df = df.sort_values(metric, ascending=True)
    else:
        df = df.sort_values(metric, ascending=True)

    # Add rank
    df["rank"] = range(1, len(df) + 1)

    return df


def statistical_significance_test(
    errors1: np.ndarray, errors2: np.ndarray, test: str = "paired_t"
) -> dict[str, float]:
    """
    Test if difference between two models is statistically significant

    Args:
        errors1: Errors from model 1
        errors2: Errors from model 2
        test: Type of test ('paired_t', 'wilcoxon', 'sign')

    Returns:
        Test result
    """
    if test == "paired_t":
        # Paired t-test
        statistic, pvalue = stats.ttest_rel(errors1, errors2)
    elif test == "wilcoxon":
        # Wilcoxon signed-rank test (non-parametric)
        statistic, pvalue = stats.wilcoxon(errors1, errors2)
    elif test == "sign":
        # Sign test
        differences = errors1 - errors2
        statistic = np.sum(differences > 0)
        n = len(differences)
        pvalue = 2 * min(
            stats.binom.cdf(statistic, n, 0.5), 1 - stats.binom.cdf(statistic - 1, n, 0.5)
        )
    else:
        raise ValueError(f"Unknown test: {test}")

    is_significant = pvalue < 0.05

    return {
        "test": test,
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "is_significant": bool(is_significant),
        "better_model": 1 if np.mean(np.abs(errors1)) < np.mean(np.abs(errors2)) else 2,
    }


def diebold_mariano_test(
    errors1: np.ndarray, errors2: np.ndarray, horizon: int = 1
) -> dict[str, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy

    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        horizon: Forecast horizon

    Returns:
        Test results
    """
    # Loss differential
    loss1 = errors1**2
    loss2 = errors2**2
    d = loss1 - loss2

    # Mean loss differential
    mean_d = np.mean(d)

    # Variance with HAC adjustment for autocorrelation
    n = len(d)

    # Simple variance (can be improved with HAC)
    var_d = np.var(d, ddof=1) / n

    # DM statistic
    dm_stat = mean_d / np.sqrt(var_d) if var_d > 0 else 0

    # P-value (two-tailed)
    pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "pvalue": float(pvalue),
        "is_significant": pvalue < 0.05,
        "better_model": 1 if mean_d < 0 else 2,
    }
