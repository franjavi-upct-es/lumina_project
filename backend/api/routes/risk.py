# backend/api/routes/risk.py
"""
Risk Analysis endpoints for portfolio and individual securities
Comprehensive risk metrics including VaR, CVaR, stress testing, and more
"""

from datetime import datetime, timedelta
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field
from scipy import stats

from backend.api.dependencies import check_rate_limit, verify_api_key
from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector

router = APIRouter(dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
settings = get_settings()


# ============================================================================
# REQUEST MODELS
# ============================================================================


class VaRCalculationRequest(BaseModel):
    """Request for VaR calculation"""

    tickers: list[str] = Field(..., min_items=1, max_items=50)
    weights: dict[str, float] | None = None
    start_date: datetime
    end_date: datetime
    confidence_levels: list[float] = Field([0.95, 0.99], description="Confidence levels for VaR")
    method: str = Field("historical", regex="^(historical|parametric|monte_carlo)$")
    holding_period: int = Field(1, ge=1, le=30, description="Holding period in days")


class StressTestRequest(BaseModel):
    """Request for stress testing"""

    tickers: list[str] = Field(..., min_items=1)
    weights: dict[str, float]
    scenarios: dict[str, float] | None = None  # Custom scenarios
    include_historical: bool = True


class DrawdownAnalysisRequest(BaseModel):
    """Request for drawdown analysis"""

    tickers: list[str] = Field(..., min_items=1)
    weights: dict[str, float] | None = None
    start_date: datetime
    end_date: datetime
    top_n_drawdowns: int = Field(10, ge=1, le=50)


class CorrelationBreakdownRequest(BaseModel):
    """Request for correlation breakdown analysis"""

    tickers: list[str] = Field(..., min_items=2)
    start_date: datetime
    end_date: datetime
    rolling_window: int | None = Field(None, ge=20, le=252)


class TailRiskRequest(BaseModel):
    """Request for tail risk analysis"""

    tickers: list[str]
    weights: dict[str, float]
    start_date: datetime
    end_date: datetime
    threshold_percentile: float = Field(5.0, ge=1.0, le=10.0)


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class VaRResponse(BaseModel):
    """VaR calculation response"""

    method: str
    holding_period: int
    var_metrics: dict[str, dict[str, float]]  # {confidence_level: {var, cvar, etc}}
    portfolio_value: float | None = None
    var_amount: dict[str, float]  # Dollar amounts
    summary: str


class StressTestResponse(BaseModel):
    """Stress test response"""

    scenarios: dict[str, dict[str, Any]]
    worst_case: dict[str, float]
    best_case: dict[str, float]
    current_exposure: dict[str, float]
    recommendations: list[str]


class DrawdownResponse(BaseModel):
    """Drawdown analysis response"""

    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    current_drawdown: float
    recovery_time: int | None
    top_drawdowns: list[dict[str, Any]]
    drawdown_series: list[dict[str, Any]]


class CorrelationResponse(BaseModel):
    """Correlation analysis response"""

    correlation_matrix: dict[str, dict[str, float]]
    average_correlation: float
    rolling_correlations: list[dict[str, Any]] | None = None
    correlation_breakdown: dict[str, Any]


class TailRiskResponse(BaseModel):
    """Tail risk analysis response"""

    left_tail_mean: float
    right_tail_mean: float
    tail_ratio: float
    expected_shortfall: float
    tail_events: list[dict[str, Any]]
    tail_statistics: dict[str, float]


class RiskContributionResponse(BaseModel):
    """Risk contribution analysis"""

    total_risk: float
    marginal_risk: dict[str, float]
    component_risk: dict[str, float]
    percentage_contribution: dict[str, float]
    diversification_ratio: float


# ============================================================================
# VALUE AT RISK (VaR) ENDPOINTS
# ============================================================================


@router.post("/var", response_model=VaRResponse)
async def calculate_var(request: VaRCalculationRequest):
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR)

    **Methods:**
    - historical: Historical simulation
    - parametric: Assumes normal distribution
    - monte_carlo: Monte Carlo simulation

    **Returns:**
    VaR at specified confidence levels with CVaR
    """
    try:
        logger.info(f"Calculating VaR using {request.method} method")

        # Collect historical data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in request.tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=request.start_date, end_date=request.end_date
            )
            if data and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        if not returns_data:
            raise HTTPException(status_code=400, detail="No data available for any ticker")

        # Calculate returns
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Portfolio weights
        if request.weights:
            weights = np.array([request.weights.get(t, 0.0) for t in returns_df.columns])
            weights = weights / weights.sum()  # Normalize
        else:
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Scale for holding period
        if request.holding_period > 1:
            portfolio_returns = portfolio_returns * np.sqrt(request.holding_period)

        # Calculate VaR based on method
        var_metrics = {}

        for confidence_level in request.confidence_levels:
            if request.method == "historical":
                var, cvar = _calculate_historical_var(portfolio_returns, confidence_level)
            elif request.method == "parametric":
                var, cvar = _calculate_parametric_var(portfolio_returns, confidence_level)
            elif request.method == "monte_carlo":
                var, cvar = _calculate_monte_carlo_var(portfolio_returns, confidence_level)
            else:
                var, cvar = _calculate_historical_var(portfolio_returns, confidence_level)

            var_metrics[f"{confidence_level:.0%}"] = {
                "var": float(var),
                "cvar": float(cvar),
                "var_percent": float(var * 100),
                "cvar_percent": float(cvar * 100),
            }

        # Calculate dollar amounts if portfolio value provided
        portfolio_value = 100000.0  # Default
        var_amount = {
            level: {
                "var_amount": portfolio_value * metrics["var"],
                "cvar_amount": portfolio_value * metrics["cvar"],
            }
            for level, metrics in var_metrics.items()
        }

        # Summary
        var_95 = var_metrics.get("95%", {}).get("var", 0)
        summary = (
            f"At 95% confidence, maximum expected loss is "
            f"${abs(var_95 * portfolio_value):,.2f} ({abs(var_95 * 100):.2f}%) "
            f"over {request.holding_period} day(s)"
        )

        return VaRResponse(
            method=request.method,
            holding_period=request.holding_period,
            var_metrics=var_metrics,
            portfolio_value=portfolio_value,
            var_amount=var_amount,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _calculate_historical_var(returns: pd.Series, confidence_level: float) -> tuple:
    """Calculate VaR using historical simulation"""
    var = -np.percentile(returns, (1 - confidence_level) * 100)
    cvar = -returns[returns <= -var].mean()
    return var, cvar


def _calculate_parametric_var(returns: pd.Series, confidence_level: float) -> tuple:
    """Calculate VaR assuming normal distribution"""
    mean = returns.mean()
    std = returns.std()

    z_score = stats.norm.ppf(1 - confidence_level)
    var = -(mean + z_score * std)

    # CVaR for normal distribution
    pdf_at_var = stats.norm.pdf(z_score)
    cvar = -(mean - std * pdf_at_var / (1 - confidence_level))

    return var, cvar


def _calculate_monte_carlo_var(
    returns: pd.Series, confidence_level: float, n_simulations: int = 10000
) -> tuple:
    """Calculate VaR using Monte Carlo simulation"""
    mean = returns.mean()
    std = returns.std()

    # Generate simulations
    simulated_returns = np.random.normal(mean, std, n_simulations)

    var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
    cvar = -simulated_returns[simulated_returns <= -var].mean()

    return var, cvar


# ============================================================================
# STRESS TESTING
# ============================================================================


@router.post("/stress-test", response_model=StressTestResponse)
async def stress_test(request: StressTestRequest):
    """
    Perform stress testing on portfolio

    Tests portfolio performance under extreme scenarios:
    - Historical crises (2008, 2020, 1987)
    - Custom scenarios
    - Factor shocks
    """
    try:
        logger.info(f"Running stress test for {len(request.tickers)} assets")

        # Collect data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in request.tickers:
            data = await collector.collect_with_retry(
                ticker=ticker,
                start_date=datetime.now() - timedelta(days=365),
                end_date=datetime.now(),
            )
            if data and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Portfolio weights
        weights = np.array([request.weights.get(t, 0.0) for t in returns_df.columns])
        weights = weights / weights.sum()

        # Define stress scenarios
        scenarios = {}

        if request.include_historical:
            # Historical scenarios
            scenarios.update(
                {
                    "2008_financial_crisis": {
                        "market_shock": -0.35,
                        "description": "Global financial crisis scenario",
                        "volatility_multiplier": 2.5,
                    },
                    "2020_covid_crash": {
                        "market_shock": -0.28,
                        "description": "COVID-19 pandemic crash",
                        "volatility_multiplier": 3.0,
                    },
                    "1987_black_monday": {
                        "market_shock": -0.22,
                        "description": "1987 stock market crash",
                        "volatility_multiplier": 2.0,
                    },
                    "2000_dotcom_bubble": {
                        "market_shock": -0.40,
                        "description": "Dot-com bubble burst",
                        "volatility_multiplier": 1.8,
                    },
                }
            )

        # Add custom scenarios
        if request.scenarios:
            for name, shock in request.scenarios.items():
                scenarios[name] = {
                    "market_shock": shock,
                    "description": f"Custom scenario: {name}",
                    "volatility_multiplier": 1.5,
                }

        # Calculate impact for each scenario
        results = {}
        for scenario_name, scenario_data in scenarios.items():
            shock = scenario_data["market_shock"]
            vol_mult = scenario_data["volatility_multiplier"]

            # Apply shock to portfolio
            portfolio_return = (returns_df.mean() * weights).sum()
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov(), weights)))

            stressed_return = portfolio_return + shock
            stressed_vol = portfolio_vol * vol_mult

            # Calculate impact
            results[scenario_name] = {
                "shock": float(shock),
                "expected_loss_pct": float(stressed_return * 100),
                "stressed_volatility": float(stressed_vol * 100),
                "description": scenario_data["description"],
                "portfolio_impact": float(stressed_return),
            }

        # Find worst and best cases
        worst_case = min(results.items(), key=lambda x: x[1]["portfolio_impact"])
        best_case = max(results.items(), key=lambda x: x[1]["portfolio_impact"])

        # Current exposure
        current_vol = float(
            np.sqrt(np.dot(weights.T, np.dot(returns_df.cov(), weights))) * np.sqrt(252)
        )

        # Recommendations
        recommendations = []
        if current_vol > 0.20:
            recommendations.append("Consider reducing volatility exposure")
        if abs(worst_case[1]["expected_loss_pct"]) > 30:
            recommendations.append("Portfolio shows high sensitivity to extreme events")
        recommendations.append("Consider hedging strategies for tail risk protection")

        return StressTestResponse(
            scenarios=results,
            worst_case={"scenario": worst_case[0], **worst_case[1]},
            best_case={"scenario": best_case[0], **best_case[1]},
            current_exposure={
                "annual_volatility": current_vol,
                "worst_expected_loss": abs(worst_case[1]["expected_loss_pct"]),
            },
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Error in stress testing: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# DRAWDOWN ANALYSIS
# ============================================================================


@router.post("/drawdown", response_model=DrawdownResponse)
async def analyze_drawdown(request: DrawdownAnalysisRequest):
    """
    Comprehensive drawdown analysis

    Analyzes historical drawdowns including:
    - Maximum drawdown
    - Average drawdown
    - Drawdown duration
    - Recovery periods
    """
    try:
        logger.info("Performing drawdown analysis")

        # Collect data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in request.tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=request.start_date, end_date=request.end_date
            )
            if data and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Portfolio weights
        if request.weights:
            weights = np.array([request.weights.get(t, 0.0) for t in returns_df.columns])
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Calculate cumulative returns
        cumulative = (1 + portfolio_returns).cumprod()

        # Calculate running maximum
        running_max = cumulative.expanding().max()

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        # Maximum drawdown
        max_dd = float(drawdown.min())

        # Current drawdown
        current_dd = float(drawdown.iloc[-1])

        # Average drawdown
        avg_dd = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        peak_value = None

        for i, (dd, _cum_val) in enumerate(zip(drawdown, cumulative, strict=True)):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                peak_value = running_max.iloc[i]
            elif dd >= -0.0001 and in_drawdown:  # Small threshold for recovery
                # End of drawdown
                in_drawdown = False
                trough_value = cumulative.iloc[i - 1]

                drawdown_periods.append(
                    {
                        "start_date": (
                            returns_df.index[start_idx].isoformat()
                            if hasattr(returns_df.index[start_idx], "isoformat")
                            else str(start_idx)
                        ),
                        "end_date": (
                            returns_df.index[i].isoformat()
                            if hasattr(returns_df.index[i], "isoformat")
                            else str(i)
                        ),
                        "duration_days": i - start_idx,
                        "drawdown": float(drawdown.iloc[start_idx:i].min()),
                        "peak_value": float(peak_value),
                        "trough_value": float(trough_value),
                    }
                )

        # Sort by severity
        drawdown_periods.sort(key=lambda x: x["drawdown"])
        top_drawdowns = drawdown_periods[: request.top_n_drawdowns]

        # Maximum drawdown duration
        max_dd_duration = (
            max([p["duration_days"] for p in drawdown_periods]) if drawdown_periods else 0
        )

        # Recovery time (if currently in drawdown)
        recovery_time = None
        if current_dd < -0.01:  # In drawdown
            # Estimate based on historical recovery times
            if drawdown_periods:
                avg_recovery = np.mean([p["duration_days"] for p in drawdown_periods])
                recovery_time = int(avg_recovery)

        # Drawdown series for visualization
        drawdown_series = [
            {
                "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                "drawdown": float(dd),
            }
            for idx, dd in drawdown.items()
        ]

        return DrawdownResponse(
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_dd,
            current_drawdown=current_dd,
            recovery_time=recovery_time,
            top_drawdowns=top_drawdowns,
            drawdown_series=drawdown_series[-100:],  # Last 100 points
        )

    except Exception as e:
        logger.error(f"Error in drawdown analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================


@router.post("/correlation", response_model=CorrelationResponse)
async def analyze_correlation(request: CorrelationBreakdownRequest):
    """
    Detailed correlation analysis

    Analyzes correlations including:
    - Static correlation matrix
    - Rolling correlations
    - Correlation breakdown
    """
    try:
        logger.info("Performing correlation analysis")

        # Collect data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in request.tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=request.start_date, end_date=request.end_date
            )
            if data and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Average correlation (excluding diagonal)
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = float(corr_matrix.where(mask).mean().mean())

        # Rolling correlations
        rolling_correlations = None
        if request.rolling_window:
            rolling_corr_data = []
            for i in range(request.rolling_window, len(returns_df)):
                window_data = returns_df.iloc[i - request.rolling_window : i]
                window_corr = window_data.corr()

                avg_window_corr = float(window_corr.where(mask).mean().mean())
                rolling_corr_data.append(
                    {
                        "date": (
                            returns_df.index[i].isoformat()
                            if hasattr(returns_df.index[i], "isoformat")
                            else str(i)
                        ),
                        "avg_correlation": avg_window_corr,
                    }
                )

            rolling_correlations = rolling_corr_data

        # Correlation breakdown
        breakdown = {
            "high_correlation_pairs": [],
            "low_correlation_pairs": [],
            "negative_correlation_pairs": [],
        }

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = float(corr_matrix.iloc[i, j])
                pair = {
                    "asset1": corr_matrix.columns[i],
                    "asset2": corr_matrix.columns[j],
                    "correlation": corr_value,
                }

                if corr_value > 0.7:
                    breakdown["high_correlation_pairs"].append(pair)
                elif corr_value < 0.3:
                    breakdown["low_correlation_pairs"].append(pair)
                if corr_value < 0:
                    breakdown["negative_correlation_pairs"].append(pair)

        # Sort pairs
        breakdown["high_correlation_pairs"].sort(key=lambda x: x["correlation"], reverse=True)
        breakdown["low_correlation_pairs"].sort(key=lambda x: x["correlation"])
        breakdown["negative_correlation_pairs"].sort(key=lambda x: x["correlation"])

        return CorrelationResponse(
            correlation_matrix=corr_matrix.to_dict(),
            average_correlation=avg_corr,
            rolling_correlations=rolling_correlations,
            correlation_breakdown=breakdown,
        )

    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# TAIL RISK ANALYSIS
# ============================================================================


@router.post("/tail-risk", response_model=TailRiskResponse)
async def analyze_tail_risk(request: TailRiskRequest):
    """
    Analyze tail risk and extreme events

    Focuses on extreme negative returns (left tail)
    """
    try:
        logger.info("Analyzing tail risk")

        # Collect data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in request.tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=request.start_date, end_date=request.end_date
            )
            if data and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Portfolio weights
        weights = np.array([request.weights.get(t, 0.0) for t in returns_df.columns])
        weights = weights / weights.sum()

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Define tail threshold
        left_threshold = np.percentile(portfolio_returns, request.threshold_percentile)
        right_threshold = np.percentile(portfolio_returns, 100 - request.threshold_percentile)

        # Left tail (losses)
        left_tail = portfolio_returns[portfolio_returns <= left_threshold]
        left_tail_mean = float(left_tail.mean())

        # Right tail (gains)
        right_tail = portfolio_returns[portfolio_returns >= right_threshold]
        right_tail_mean = float(right_tail.mean())

        # Tail ratio
        tail_ratio = float(abs(right_tail_mean / left_tail_mean)) if left_tail_mean != 0 else 0

        # Expected shortfall (CVaR at threshold)
        expected_shortfall = float(-left_tail.mean())

        # Find tail events
        tail_events = []
        for idx, ret in portfolio_returns.items():
            if ret <= left_threshold:
                tail_events.append(
                    {
                        "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                        "return": float(ret),
                        "return_pct": float(ret * 100),
                        "severity": "extreme" if ret < left_threshold * 1.5 else "moderate",
                    }
                )

        # Sort by severity
        tail_events.sort(key=lambda x: x["return"])

        # Tail statistics
        tail_stats = {
            "skewness": float(portfolio_returns.skew()),
            "kurtosis": float(portfolio_returns.kurt()),
            "left_tail_frequency": float(len(left_tail) / len(portfolio_returns)),
            "max_loss": float(portfolio_returns.min()),
            "max_gain": float(portfolio_returns.max()),
        }

        return TailRiskResponse(
            left_tail_mean=left_tail_mean,
            right_tail_mean=right_tail_mean,
            tail_ratio=tail_ratio,
            expected_shortfall=expected_shortfall,
            tail_events=tail_events[:20],  # Top 20 worst events
            tail_statistics=tail_stats,
        )

    except Exception as e:
        logger.error(f"Error in tail risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# RISK CONTRIBUTION
# ============================================================================


@router.post("/risk-contribution", response_model=RiskContributionResponse)
async def analyze_risk_contribution(
    tickers: Annotated[list[str], Query()],
    weights: Annotated[dict[str, float], Query()],
    start_date: Annotated[datetime, Query()],
    end_date: Annotated[datetime, Query()],
):
    """
    Analyze risk contribution of each asset to portfolio risk

    Calculates:
    - Marginal risk contribution
    - Component risk contribution
    - Percentage risk contribution
    """
    try:
        logger.info("Analyzing risk contribution")

        # Collect data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=start_date, end_date=end_date
            )
            if data and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Weights array
        weights_array = np.array([weights.get(t, 0.0) for t in returns_df.columns])
        weights_array = weights_array / weights_array.sum()

        # Covariance matrix
        cov_matrix = returns_df.cov().values * 252  # Annualized

        # Portfolio variance
        portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
        portfolio_vol = np.sqrt(portfolio_variance)

        # Marginal risk contribution
        marginal_contrib = np.dot(cov_matrix, weights_array) / portfolio_vol

        # Component risk contribution
        component_contrib = weights_array * marginal_contrib

        # Percentage contribution
        pct_contrib = component_contrib / portfolio_vol

        # Diversification ratio
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.dot(weights_array, individual_vols)
        diversification_ratio = float(weighted_vol / portfolio_vol)

        # Format results
        marginal_risk = {
            ticker: float(contrib)
            for ticker, contrib in zip(returns_df.columns, marginal_contrib, strict=True)
        }

        component_risk = {
            ticker: float(contrib)
            for ticker, contrib in zip(returns_df.columns, component_contrib, strict=True)
        }

        percentage_contribution = {
            ticker: float(contrib * 100)
            for ticker, contrib in zip(returns_df.columns, pct_contrib, strict=True)
        }

        return RiskContributionResponse(
            total_risk=float(portfolio_vol),
            marginal_risk=marginal_risk,
            component_risk=component_risk,
            percentage_contribution=percentage_contribution,
            diversification_ratio=diversification_ratio,
        )

    except Exception as e:
        logger.error(f"Error in risk contribution analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# LIQUIDITY RISK
# ============================================================================


@router.get("/liquidity-risk/{ticker}")
async def analyze_liquidity_risk(
    ticker: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    """
    Analyze liquidity risk for a specific security

    Metrics:
    - Average daily volume
    - Bid-ask spread
    - Volume volatility
    - Liquidity score
    """
    try:
        logger.info(f"Analyzing liquidity risk for {ticker}")

        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

        # Collect data
        collector = YFinanceCollector()
        data = await collector.collect_with_retry(
            ticker=ticker, start_date=start_date, end_date=end_date
        )

        if not data or data.height == 0:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        data_pd = data.to_pandas()

        # Calculate metrics
        avg_volume = float(data_pd["volume"].mean())
        volume_volatility = float(data_pd["volume"].std() / avg_volume)

        # Estimate bid-ask spread (simplified using high-low spread)
        avg_spread = float(((data_pd["high"] - data_pd["low"]) / data_pd["close"]).mean())

        # Liquidity score (0-100, higher is better)
        # Based on volume and spread
        volume_score = min(avg_volume / 1000000, 100)  # Normalize to millions
        spread_score = max(0, 100 - (avg_spread * 10000))
        liquidity_score = float((volume_score + spread_score) / 2)

        # Days to liquidate large position (simplified)
        position_size = 100000  # Example: $100k position
        avg_price = float(data_pd["close"].mean())
        shares = position_size / avg_price
        days_to_liquidate = int(shares / (avg_volume * 0.1))  # 10% of daily volume

        return {
            "ticker": ticker,
            "avg_daily_volume": avg_volume,
            "volume_volatility": volume_volatility,
            "avg_spread_pct": avg_spread * 100,
            "liquidity_score": liquidity_score,
            "days_to_liquidate_100k": days_to_liquidate,
            "liquidity_rating": (
                "High" if liquidity_score > 70 else "Medium" if liquidity_score > 40 else "Low"
            ),
            "warnings": (
                ["Low liquidity - may face slippage"]
                if liquidity_score < 40
                else ["Consider market impact"]
                if liquidity_score < 70
                else []
            ),
        }

    except Exception as e:
        logger.error(f"Error analyzing liquidity risk: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# SCENARIO ANALYSIS
# ============================================================================


@router.post("/scenario-analysis")
async def scenario_analysis(
    tickers: Annotated[list[str], Query()],
    weights: Annotated[dict[str, float], Query()],
    market_change: Annotated[float, Query(description="Market change in %")],
    volatility_change: Annotated[float, Query(description="Volatility change multiplier")] = 0.0,
    correlation_change: Annotated[float, Query(description="Correlation change")] = 0.0,
):
    """
    Analyze portfolio under custom scenario

    Allows users to define custom market conditions
    """
    try:
        logger.info("Running custom scenario analysis")

        # Collect historical data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in tickers:
            data = await collector.collect_with_retry(
                ticker=ticker,
                start_date=datetime.now() - timedelta(days=365),
                end_date=datetime.now(),
            )
            if data and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Current portfolio metrics
        weights_array = np.array([weights.get(t, 0.0) for t in returns_df.columns])
        weights_array = weights_array / weights_array.sum()

        current_return = float((returns_df.mean() * weights_array).sum() * 252)
        current_vol = float(
            np.sqrt(np.dot(weights_array.T, np.dot(returns_df.cov() * 252, weights_array)))
        )

        # Apply scenario
        scenario_return = current_return + (market_change / 100)
        scenario_vol = current_vol * (1 + volatility_change)

        # Calculate impact
        portfolio_value = 100000.0
        current_value = portfolio_value * (1 + current_return)
        scenario_value = portfolio_value * (1 + scenario_return)

        impact = scenario_value - current_value
        impact_pct = (impact / portfolio_value) * 100

        return {
            "scenario_parameters": {
                "market_change_pct": market_change,
                "volatility_multiplier": 1 + volatility_change,
                "correlation_adjustment": correlation_change,
            },
            "current_metrics": {
                "expected_return": current_return * 100,
                "volatility": current_vol * 100,
                "portfolio_value": portfolio_value,
            },
            "scenario_metrics": {
                "expected_return": scenario_return * 100,
                "volatility": scenario_vol * 100,
                "expected_value": scenario_value,
            },
            "impact": {
                "absolute_change": impact,
                "percentage_change": impact_pct,
                "interpretation": (
                    f"Under this scenario, portfolio would {'gain' if impact > 0 else 'lose'} "
                    f"${abs(impact):,.2f} ({abs(impact_pct):.2f}%)"
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error in scenario analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# RISK SUMMARY DASHBOARD
# ============================================================================


@router.get("/dashboard")
async def risk_dashboard(
    tickers: Annotated[list[str], Query()],
    weights: Annotated[dict[str, float], Query()],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    """
    Comprehensive risk dashboard with all key metrics

    Single endpoint to get overview of all risk metrics
    """
    try:
        logger.info("Generating risk dashboard")

        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

        # Collect data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=start_date, end_date=end_date
            )
            if data and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Portfolio setup
        weights_array = np.array([weights.get(t, 0.0) for t in returns_df.columns])
        weights_array = weights_array / weights_array.sum()
        portfolio_returns = (returns_df * weights_array).sum(axis=1)

        # Calculate all metrics
        var_95 = float(-np.percentile(portfolio_returns, 5))
        cvar_95 = float(-portfolio_returns[portfolio_returns <= -var_95].mean())

        # Volatility
        annual_vol = float(portfolio_returns.std() * np.sqrt(252))

        # Sharpe ratio
        risk_free_rate = 0.05
        sharpe = float(((portfolio_returns.mean() * 252) - risk_free_rate) / annual_vol)

        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min())
        current_drawdown = float(drawdown.iloc[-1])

        # Correlation
        corr_matrix = returns_df.corr()
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_correlation = float(corr_matrix.where(mask).mean().mean())

        # Tail risk
        left_tail = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]
        tail_mean = float(left_tail.mean())

        # Beta vs market
        try:
            spy_data = await collector.collect_with_retry(
                ticker="SPY", start_date=start_date, end_date=end_date
            )
            if spy_data and spy_data.height > 0:
                spy_returns = spy_data.select("close").to_series().pct_change().drop_nulls()
                common_len = min(len(portfolio_returns), len(spy_returns))
                covariance = np.cov(portfolio_returns[-common_len:], spy_returns[-common_len:])[
                    0, 1
                ]
                market_variance = np.var(spy_returns[-common_len:])
                beta = float(covariance / market_variance) if market_variance > 0 else 1.0
            else:
                beta = 1.0
        except Exception:
            beta = 1.0

        return {
            "summary": {
                "risk_score": float(min(100, max(0, 100 - (annual_vol * 200)))),  # 0-100 scale
                "risk_level": (
                    "Low" if annual_vol < 0.15 else "Medium" if annual_vol < 0.25 else "High"
                ),
            },
            "value_at_risk": {
                "var_95_pct": var_95 * 100,
                "cvar_95_pct": cvar_95 * 100,
                "var_95_amount": var_95 * 100000,
                "cvar_95_amount": cvar_95 * 100000,
            },
            "volatility_metrics": {
                "annual_volatility": annual_vol * 100,
                "sharpe_ratio": sharpe,
                "beta": beta,
            },
            "drawdown_metrics": {
                "max_drawdown": max_drawdown * 100,
                "current_drawdown": current_drawdown * 100,
                "in_drawdown": current_drawdown < -0.01,
            },
            "correlation_metrics": {
                "average_correlation": avg_correlation,
                "diversification_benefit": float(1 - avg_correlation),
            },
            "tail_risk": {
                "left_tail_mean": tail_mean * 100,
                "skewness": float(portfolio_returns.skew()),
                "kurtosis": float(portfolio_returns.kurt()),
            },
            "risk_warnings": _generate_risk_warnings(
                annual_vol, max_drawdown, sharpe, avg_correlation
            ),
        }

    except Exception as e:
        logger.error(f"Error generating risk dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _generate_risk_warnings(
    volatility: float, max_dd: float, sharpe: float, correlation: float
) -> list[str]:
    """Generate risk warnings based on metrics"""
    warnings = []

    if volatility > 0.30:
        warnings.append("⚠️ High volatility detected - consider risk reduction")

    if abs(max_dd) > 0.25:
        warnings.append("⚠️ Large historical drawdowns - review risk tolerance")

    if sharpe < 0.5:
        warnings.append("⚠️ Low risk-adjusted returns - review strategy")

    if correlation > 0.8:
        warnings.append("⚠️ High correlation among assets - limited diversification")

    if not warnings:
        warnings.append("✅ Portfolio risk metrics within normal ranges")

    return warnings
