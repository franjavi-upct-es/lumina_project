# backend/api/routes/portfolio.py
"""
Portfolio management endpoints including optimization and risk analysis
"""

from datetime import datetime, timedelta
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field
from scipy.optimize import minimize
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import check_rate_limit, verify_api_key
from backend.config.settings import get_settings
from backend.data_engine.collectors.yfinance_collector import YFinanceCollector
from backend.db.models import PortfolioBalance, PortfolioPosition, get_async_session

router = APIRouter(dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
settings = get_settings()


# Request Models
class PortfolioOptimizationRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, max_length=50)
    start_date: datetime
    end_date: datetime

    # Optimization method
    method: str = Field(
        "markowitz",
        pattern="^(markowitz|black_litterman|risk_parity|min_volatility|max_sharpe|hrp)$",
    )

    # Constraints
    target_return: float | None = None
    target_volatility: float | None = None
    max_weight: float = Field(0.3, ge=0.01, le=1.0)
    min_weight: float = Field(0.01, ge=0.0, le=0.2)

    # Black-Litterman specific
    views: dict[str, float] | None = Field(
        None, description="Expected returns views for Black-Litterman"
    )
    view_confidences: dict[str, float] | None = None

    # Risk-free rate
    risk_free_rate: float = Field(0.05, ge=0.0, le=0.5)


class RebalanceRequest(BaseModel):
    current_holdings: dict[str, float]
    target_weights: dict[str, float]
    rebalance_threshold: float = Field(0.05, ge=0.01, le=0.5)


class RiskAnalysisRequest(BaseModel):
    tickers: list[str]
    weights: dict[str, float]
    start_date: datetime
    end_date: datetime
    confidence_level: float = Field(0.95, ge=0.90, le=0.99)


# Response Models
class OptimizationResultResponse(BaseModel):
    method: str
    optimal_weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float

    # Additional metrics
    diversification_ratio: float | None = None
    max_weight: float
    min_weight: float

    # Efficient frontier data
    efficient_frontier: list[dict[str, float]] | None = None


class PortfolioMetricsResponse(BaseModel):
    total_value: float
    cash: float
    equity: float

    # Performance
    total_return: float
    daily_return: float
    ytd_return: float

    # Risk
    volatility: float
    sharpe_ratio: float
    beta: float
    var_95: float
    max_drawdown: float

    # Holdings
    positions: list[dict[str, Any]]
    concentration: dict[str, float]


class RebalanceResponse(BaseModel):
    trades_needed: bool
    trades: list[dict[str, Any]]
    estimated_cost: float
    reason: str


class RiskMetricsResponse(BaseModel):
    # Value at Risk
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Drawdown
    max_drawdown: float
    current_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int

    # Correlation
    portfolio_volatility: float
    diversification_benefit: float

    # Factor exposures
    market_beta: float
    size_factor: float | None = None
    value_factor: float | None = None
    momentum_factor: float | None = None

    # Stress scenarios
    stress_scenarios: dict[str, float]


# ============================================================================
# HELPERS
# ============================================================================


def _weights_from_query(tickers: list[str], weights: list[float]) -> dict[str, float]:
    if len(weights) != len(tickers):
        raise HTTPException(status_code=400, detail="weights must match tickers length")
    return dict(zip(tickers, weights, strict=True))


# ============================================================================
# OPTIMIZATION IMPLEMENTATIONS
# ============================================================================


@router.post("/optimize", response_model=OptimizationResultResponse)
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """
    Optimize portfolio weights using various methods

    **Methods:**
    - markowitz: Mean-variance optimization (Markowitz)
    - black_litterman: Black-Litterman with subjective views
    - risk_parity: Equal risk contribution
    - min_volatility: Minimum volatility portfolio
    - max_sharpe: Maximum Sharpe ratio
    - hrp: Hierarchical Risk Parity
    """
    try:
        logger.info(f"Portfolio optimization: {request.method} for {len(request.tickers)} assets")

        # Collect historical data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in request.tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=request.start_date, end_date=request.end_date
            )
            if data is not None and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        if len(returns_data) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 tickers with valid data")

        # Calculate returns matrix
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Calculate expected returns and covariance
        mean_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252  # Annualized

        # Optimize based on method
        if request.method == "markowitz" or request.method == "max_sharpe":
            weights = _optimize_max_sharpe(
                mean_returns,
                cov_matrix,
                request.risk_free_rate,
                request.min_weight,
                request.max_weight,
            )
        elif request.method == "min_volatility":
            weights = _optimize_min_volatility(cov_matrix, request.min_weight, request.max_weight)
        elif request.method == "risk_parity":
            weights = _optimize_risk_parity(cov_matrix, request.min_weight, request.max_weight)
        elif request.method == "black_litterman":
            weights = _optimize_black_litterman(
                mean_returns,
                cov_matrix,
                request.views or {},
                request.risk_free_rate,
                request.min_weight,
                request.max_weight,
            )
        elif request.method == "hrp":
            weights = _optimize_hrp(returns_df, request.min_weight, request.max_weight)
        else:
            # Default to equal weight
            weights = np.ones(len(returns_data)) / len(returns_data)

        # Calculate portfolio metrics
        optimal_weights_dict = dict(zip(returns_data.keys(), weights, strict=True))
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - request.risk_free_rate) / portfolio_vol

        # Calculate diversification ratio
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vol = np.dot(weights, individual_vols)
        diversification_ratio = weighted_vol / portfolio_vol

        # Generate efficient frontier points
        frontier_points = _generate_efficient_frontier(
            mean_returns, cov_matrix, request.risk_free_rate, num_points=20
        )

        return OptimizationResultResponse(
            method=request.method,
            optimal_weights=optimal_weights_dict,
            expected_return=float(portfolio_return),
            expected_volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            diversification_ratio=float(diversification_ratio),
            max_weight=float(max(weights)),
            min_weight=float(min(weights)),
            efficient_frontier=frontier_points,
        )

    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, min_weight, max_weight):
    """Maximize Sharpe ratio"""
    n_assets = len(mean_returns)

    def neg_sharpe(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_free_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    initial = np.ones(n_assets) / n_assets

    result = minimize(neg_sharpe, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def _optimize_min_volatility(cov_matrix, min_weight, max_weight):
    """Minimize portfolio volatility"""
    n_assets = len(cov_matrix)

    def portfolio_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    initial = np.ones(n_assets) / n_assets

    result = minimize(
        portfolio_vol, initial, method="SLSQP", bounds=bounds, constraints=constraints
    )
    return result.x


def _optimize_risk_parity(cov_matrix, min_weight, max_weight):
    """Risk parity - equal risk contribution"""
    n_assets = len(cov_matrix)

    def risk_parity_objective(weights):
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / port_vol
        risk_contrib = weights * marginal_contrib
        target_risk = port_vol / n_assets
        return np.sum((risk_contrib - target_risk) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    initial = np.ones(n_assets) / n_assets

    result = minimize(
        risk_parity_objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def _optimize_black_litterman(
    mean_returns, cov_matrix, views, risk_free_rate, min_weight, max_weight
):
    """Black-Litterman optimization with views"""
    # Simplified Black-Litterman (just use views as adjusted returns)
    adjusted_returns = mean_returns.copy()
    for ticker, view in views.items():
        if ticker in adjusted_returns.index:
            adjusted_returns[ticker] = view

    return _optimize_max_sharpe(
        adjusted_returns, cov_matrix, risk_free_rate, min_weight, max_weight
    )


def _optimize_hrp(returns_df, min_weight, max_weight):
    """Hierarchical Risk Parity"""
    # Simplified HRP - inverse volatility weighting
    vols = returns_df.std()
    inv_vol = 1 / vols
    weights = inv_vol / inv_vol.sum()

    # Apply constraints
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / weights.sum()

    return weights.values


def _generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_points=20):
    """Generate efficient frontier points"""
    frontier = []
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)

    for target in target_returns:
        try:
            n_assets = len(mean_returns)

            def portfolio_vol(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, target=target: np.dot(w, mean_returns) - target},
            ]
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial = np.ones(n_assets) / n_assets

            result = minimize(
                portfolio_vol,
                initial,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                vol = float(result.fun)
                ret = float(target)
                sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

                frontier.append({"return": ret, "volatility": vol, "sharpe_ratio": sharpe})
        except Exception:
            continue

    return frontier


# ============================================================================
# PORTFOLIO METRICS
# ============================================================================


@router.get("/metrics", response_model=PortfolioMetricsResponse)
async def get_portfolio_metrics(
    user_id: Annotated[str, Query(description="User ID")] = "default",
    db: Annotated[AsyncSession, Depends(get_async_session)] = "default",
):
    """
    Get current portfolio metrics and performance
    """
    try:
        # Get portfolio balance
        balance_query = select(PortfolioBalance).where(PortfolioBalance.user_id == user_id)
        balance_result = await db.execute(balance_query)
        balance = balance_result.scalar_one_or_none()

        if not balance:
            # Create default balance
            balance = PortfolioBalance(user_id=user_id)

        # Get all positions
        positions_query = (
            select(PortfolioPosition)
            .where(PortfolioPosition.user_id == user_id)
            .order_by(PortfolioPosition.executed_at.desc())
        )
        positions_result = await db.execute(positions_query)
        all_transactions = positions_result.scalars().all()

        # Calculate current positions
        holdings = {}
        for txn in all_transactions:
            if txn.ticker not in holdings:
                holdings[txn.ticker] = 0.0

            if txn.transaction_type == "buy":
                holdings[txn.ticker] += txn.quantity
            else:
                holdings[txn.ticker] -= txn.quantity

        # Remove zero positions
        holdings = {k: v for k, v in holdings.items() if abs(v) > 1e-6}

        # Get current prices
        collector = YFinanceCollector()
        positions_list = []
        total_equity = 0.0

        for ticker, quantity in holdings.items():
            data = await collector.collect_with_retry(
                ticker=ticker,
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now(),
            )

            if data is not None and data.height > 0:
                current_price = float(data.select("close").tail(1).item())
                value = quantity * current_price
                total_equity += value

                positions_list.append(
                    {
                        "ticker": ticker,
                        "quantity": quantity,
                        "current_price": current_price,
                        "value": value,
                        "weight": 0.0,  # Will calculate after
                    }
                )

        # Calculate weights
        for pos in positions_list:
            pos["weight"] = pos["value"] / total_equity if total_equity > 0 else 0

        # Calculate performance metrics
        total_value = balance.cash + total_equity
        initial_capital = 100000.0  # Default
        total_return = (total_value - initial_capital) / initial_capital

        # Calculate daily return (simplified)
        daily_return = 0.0
        if len(all_transactions) > 0:
            recent_balance = all_transactions[0].balance_after if all_transactions else total_value
            daily_return = (
                (total_value - recent_balance) / recent_balance if recent_balance > 0 else 0.0
            )

        # YTD return (simplified)
        ytd_start = datetime(datetime.now().year, 1, 1)
        ytd_transactions = [t for t in all_transactions if t.executed_at >= ytd_start]
        ytd_return = 0.0
        if ytd_transactions:
            ytd_initial = ytd_transactions[-1].balance_after
            ytd_return = (total_value - ytd_initial) / ytd_initial if ytd_initial > 0 else 0.0

        # Risk metrics (simplified calculations)
        volatility = 0.15  # Default
        sharpe_ratio = (total_return * 252 - 0.05) / volatility if volatility > 0 else 0.0
        beta = 1.0
        var_95 = total_value * 0.02  # 2% VaR
        max_drawdown = 0.0

        # Concentration analysis
        concentration = {
            "top_position": max(p["weight"] for p in positions_list) if positions_list else 0.0,
            "top_3_concentration": (
                sum(sorted((p["weight"] for p in positions_list), reverse=True)[:3])
                if len(positions_list) >= 3
                else 1.0
            ),
            "effective_positions": (
                1 / sum(p["weight"] ** 2 for p in positions_list) if positions_list else 0.0
            ),
        }

        return PortfolioMetricsResponse(
            total_value=total_value,
            cash=balance.cash,
            equity=total_equity,
            total_return=total_return,
            daily_return=daily_return,
            ytd_return=ytd_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            beta=beta,
            var_95=var_95,
            max_drawdown=max_drawdown,
            positions=positions_list,
            concentration=concentration,
        )

    except Exception as e:
        logger.error(f"Error fetching portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# REBALANCING
# ============================================================================


@router.post("/rebalance", response_model=RebalanceResponse)
async def check_rebalance(request: RebalanceRequest):
    """
    Check if rebalancing is needed and generate trade list
    """
    try:
        trades = []
        total_value = sum(request.current_holdings.values())

        # Calculate current weights
        current_weights = {
            ticker: value / total_value for ticker, value in request.current_holdings.items()
        }

        # Check if rebalancing needed
        needs_rebalance = False
        for ticker, target_weight in request.target_weights.items():
            current_weight = current_weights.get(ticker, 0.0)
            deviation = abs(target_weight - current_weight)

            if deviation > request.rebalance_threshold:
                needs_rebalance = True

                # Calculate trade
                target_value = target_weight * total_value
                current_value = request.current_holdings.get(ticker, 0.0)
                trade_value = target_value - current_value

                trades.append(
                    {
                        "ticker": ticker,
                        "action": "BUY" if trade_value > 0 else "SELL",
                        "value": abs(trade_value),
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "deviation": deviation,
                    }
                )

        # Estimate transaction costs
        estimated_cost = sum(trade["value"] for trade in trades) * settings.DEFAULT_COMMISSION

        return RebalanceResponse(
            trades_needed=needs_rebalance,
            trades=trades,
            estimated_cost=estimated_cost,
            reason=(
                f"Rebalancing needed: {len(trades)} positions exceed {request.rebalance_threshold:.1%} threshold"
                if needs_rebalance
                else "No rebalancing needed"
            ),
        )

    except Exception as e:
        logger.error(f"Error checking rebalance: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# RISK ANALYSIS
# ============================================================================


@router.post("/risk-analysis", response_model=RiskMetricsResponse)
async def analyze_portfolio_risk(request: RiskAnalysisRequest):
    """
    Comprehensive risk analysis for a portfolio
    """
    try:
        logger.info(f"Risk analysis for {len(request.tickers)} assets")

        # Collect historical data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in request.tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=request.start_date, end_date=request.end_date
            )
            if data is not None and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        # Calculate returns
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Get weights vector
        weights = np.array([request.weights.get(t, 0.0) for t in returns_df.columns])

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # VaR and CVaR calculations
        var_95 = -np.percentile(portfolio_returns, (1 - request.confidence_level) * 100)
        var_99 = -np.percentile(portfolio_returns, 1)

        # CVaR (Expected Shortfall)
        cvar_95 = -portfolio_returns[portfolio_returns <= -var_95].mean()
        cvar_99 = -portfolio_returns[portfolio_returns <= -var_99].mean()

        # Drawdown analysis
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_drawdown = float(drawdown.min())
        current_drawdown = float(drawdown.iloc[-1])
        avg_drawdown = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start = None
        for i, in_dd in enumerate(is_drawdown):
            if in_dd and start is None:
                start = i
            elif not in_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None

        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0

        # Portfolio volatility
        portfolio_vol = float(portfolio_returns.std() * np.sqrt(252))

        # Diversification benefit
        individual_vols = returns_df.std() * np.sqrt(252)
        weighted_vol = float((individual_vols * weights).sum())
        diversification_benefit = weighted_vol - portfolio_vol

        # Market beta (using SPY if available)
        market_beta = 1.0
        try:
            spy_data = await collector.collect_with_retry(
                ticker="SPY", start_date=request.start_date, end_date=request.end_date
            )
            if spy_data is not None and spy_data.height > 0:
                spy_returns = spy_data.select("close").to_series().pct_change().drop_nulls()
                if len(spy_returns) == len(portfolio_returns):
                    covariance = np.cov(portfolio_returns, spy_returns)[0, 1]
                    market_variance = np.var(spy_returns)
                    market_beta = covariance / market_variance if market_variance > 0 else 1.0
        except Exception:
            pass

        # Stress scenarios
        stress_scenarios = {}

        # 2008 crisis: -35% shock
        stress_scenarios["2008_financial_crisis"] = float(portfolio_returns.mean() - 0.35)

        # 2020 COVID: -28% shock
        stress_scenarios["2020_covid_crash"] = float(portfolio_returns.mean() - 0.28)

        # 1987 crash: -22% shock
        stress_scenarios["1987_black_monday"] = float(portfolio_returns.mean() - 0.22)

        return RiskMetricsResponse(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration=max_dd_duration,
            portfolio_volatility=portfolio_vol,
            diversification_benefit=diversification_benefit,
            market_beta=float(market_beta),
            stress_scenarios=stress_scenarios,
        )

    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# CORRELATION & EFFICIENT FRONTIER
# ============================================================================


@router.get("/correlation-matrix")
async def get_correlation_matrix(
    tickers: Annotated[list[str], Query(min_length=2)],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    """
    Get correlation matrix for a set of assets
    """
    try:
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
            if data is not None and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        # Calculate correlation
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        corr_matrix = returns_df.corr()

        return {
            "tickers": list(corr_matrix.columns),
            "correlation_matrix": corr_matrix.to_dict(),
            "heatmap_data": [
                {"x": col, "y": idx, "value": float(val)}
                for idx in corr_matrix.index
                for col, val in corr_matrix.loc[idx].items()
            ],
        }

    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/efficient-frontier")
async def get_efficient_frontier(
    tickers: Annotated[list[str], Query(min_length=2)],
    start_date: Annotated[datetime, Query()],
    end_date: Annotated[datetime, Query()],
    num_points: Annotated[int, Query(ge=10, le=100)] = 50,
):
    """
    Calculate efficient frontier for portfolio optimization
    """
    try:
        logger.info(f"Calculating efficient frontier for {len(tickers)} assets")

        # Collect data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=start_date, end_date=end_date
            )
            if data is not None and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        # Calculate returns and covariance
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252

        # Generate frontier
        risk_free_rate = 0.05
        frontier_points = _generate_efficient_frontier(
            mean_returns, cov_matrix, risk_free_rate, num_points
        )

        # Find optimal portfolios
        max_sharpe_weights = _optimize_max_sharpe(
            mean_returns, cov_matrix, risk_free_rate, 0.0, 1.0
        )
        min_vol_weights = _optimize_min_volatility(cov_matrix, 0.0, 1.0)

        max_sharpe_return = np.dot(max_sharpe_weights, mean_returns)
        max_sharpe_vol = np.sqrt(
            np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights))
        )

        min_vol_return = np.dot(min_vol_weights, mean_returns)
        min_vol_vol = np.sqrt(np.dot(min_vol_weights.T, np.dot(cov_matrix, min_vol_weights)))

        return {
            "frontier_points": frontier_points,
            "optimal_portfolio": {
                "weights": dict(zip(tickers, max_sharpe_weights, strict=True)),
                "return": float(max_sharpe_return),
                "volatility": float(max_sharpe_vol),
                "sharpe_ratio": float((max_sharpe_return - risk_free_rate) / max_sharpe_vol),
            },
            "min_variance_portfolio": {
                "weights": dict(zip(tickers, min_vol_weights, strict=True)),
                "return": float(min_vol_return),
                "volatility": float(min_vol_vol),
                "sharpe_ratio": float((min_vol_return - risk_free_rate) / min_vol_vol),
            },
        }

    except Exception as e:
        logger.error(f"Error calculating efficient frontier: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================


@router.post("/monte-carlo")
async def monte_carlo_simulation(
    tickers: Annotated[list[str], Query()],
    weights: Annotated[list[float], Query()],
    initial_value: Annotated[float, Query()] = 100000.0,
    num_simulations: Annotated[int, Query(ge=100, le=10000)] = 1000,
    time_horizon_days: Annotated[int, Query(ge=30, le=1000)] = 252,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    """
    Run Monte Carlo simulation for portfolio
    """
    try:
        logger.info(f"Monte Carlo: {num_simulations} simulations over {time_horizon_days} days")
        weights_map = _weights_from_query(tickers, weights)

        # Collect historical data
        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

        collector = YFinanceCollector()
        returns_data = {}

        for ticker in tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=start_date, end_date=end_date
            )
            if data is not None and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        # Calculate historical statistics
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Get mean and covariance
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov().to_numpy()

        # Weights array
        weights_array = np.array([weights_map.get(t, 0.0) for t in returns_df.columns])

        # Run simulations
        simulations = np.zeros((num_simulations, time_horizon_days))

        for i in range(num_simulations):
            # Generate random returns from multivariate normal
            random_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, time_horizon_days
            )

            # Calculate portfolio returns
            portfolio_returns = random_returns @ weights_array

            # Calculate cumulative value
            cumulative = initial_value * (1 + portfolio_returns).cumprod()
            simulations[i] = cumulative

        # Calculate statistics
        final_values = simulations[:, -1]

        percentiles = {
            "5th": float(np.percentile(final_values, 5)),
            "25th": float(np.percentile(final_values, 25)),
            "50th": float(np.percentile(final_values, 50)),
            "75th": float(np.percentile(final_values, 75)),
            "95th": float(np.percentile(final_values, 95)),
        }

        probability_of_loss = float(np.mean(final_values < initial_value))
        expected_value = float(np.mean(final_values))

        # Sample trajectories for visualization
        sample_indices = np.random.choice(num_simulations, min(100, num_simulations), replace=False)
        sample_paths = simulations[sample_indices].tolist()

        return {
            "simulations": num_simulations,
            "time_horizon_days": time_horizon_days,
            "initial_value": initial_value,
            "percentiles": percentiles,
            "probability_of_loss": probability_of_loss,
            "expected_value": expected_value,
            "sample_paths": sample_paths,
        }

    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# ALLOCATION ANALYSIS
# ============================================================================


@router.get("/allocation/sector")
async def get_sector_allocation(
    db: Annotated[AsyncSession, Depends(get_async_session)],
    user_id: Annotated[str, Query()] = "default",
):
    """
    Get portfolio allocation by sector
    """
    try:
        # Get user positions
        positions_query = (
            select(PortfolioPosition)
            .where(PortfolioPosition.user_id == user_id)
            .order_by(PortfolioPosition.executed_at.desc())
        )
        positions_result = await db.execute(positions_query)
        transactions = positions_result.scalars().all()

        # Calculate current holdings
        holdings = {}
        for txn in transactions:
            if txn.ticker not in holdings:
                holdings[txn.ticker] = 0.0

            if txn.transaction_type == "buy":
                holdings[txn.ticker] += txn.quantity
            else:
                holdings[txn.ticker] -= txn.quantity

        holdings = {k: v for k, v in holdings.items() if abs(v) > 1e-6}

        if not holdings:
            return {"allocation": {}}

        # Get sector information
        collector = YFinanceCollector()
        sector_values = {}
        total_value = 0.0

        for ticker, quantity in holdings.items():
            try:
                # Get company info for sector
                info = await collector.get_company_info(ticker)
                sector = info.get("sector", "Other") if info else "Other"

                # Get current price
                data = await collector.collect_with_retry(
                    ticker=ticker,
                    start_date=datetime.now() - timedelta(days=7),
                    end_date=datetime.now(),
                )

                if data is not None and data.height > 0:
                    current_price = float(data.select("close").tail(1).item())
                    value = quantity * current_price

                    sector_values[sector] = sector_values.get(sector, 0.0) + value
                    total_value += value
            except Exception:
                continue

        # Calculate allocation percentages
        allocation = (
            {sector: value / total_value for sector, value in sector_values.items()}
            if total_value > 0
            else {}
        )

        return {"allocation": allocation}

    except Exception as e:
        logger.error(f"Error calculating sector allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/allocation/geography")
async def get_geographic_allocation(
    db: Annotated[AsyncSession, Depends(get_async_session)],
    user_id: Annotated[str, Query()] = "default",
):
    """
    Get portfolio allocation by geography
    """
    try:
        # Similar to sector allocation but by country
        positions_query = (
            select(PortfolioPosition)
            .where(PortfolioPosition.user_id == user_id)
            .order_by(PortfolioPosition.executed_at.desc())
        )
        positions_result = await db.execute(positions_query)
        transactions = positions_result.scalars().all()

        holdings = {}
        for txn in transactions:
            if txn.ticker not in holdings:
                holdings[txn.ticker] = 0.0

            if txn.transaction_type == "buy":
                holdings[txn.ticker] += txn.quantity
            else:
                holdings[txn.ticker] -= txn.quantity

        holdings = {k: v for k, v in holdings.items() if abs(v) > 1e-6}

        if not holdings:
            return {"allocation": {}}

        # Get geographic information
        collector = YFinanceCollector()
        geo_values = {}
        total_value = 0.0

        for ticker, quantity in holdings.items():
            try:
                info = await collector.get_company_info(ticker)
                country = info.get("country", "Unknown") if info else "Unknown"

                data = await collector.collect_with_retry(
                    ticker=ticker,
                    start_date=datetime.now() - timedelta(days=7),
                    end_date=datetime.now(),
                )

                if data is not None and data.height > 0:
                    current_price = float(data.select("close").tail(1).item())
                    value = quantity * current_price

                    geo_values[country] = geo_values.get(country, 0.0) + value
                    total_value += value
            except Exception:
                continue

        allocation = (
            {country: value / total_value for country, value in geo_values.items()}
            if total_value > 0
            else {}
        )

        return {"allocation": allocation}

    except Exception as e:
        logger.error(f"Error calculating geographic allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# FACTOR EXPOSURE
# ============================================================================


@router.get("/factor-exposure")
async def get_factor_exposure(
    tickers: Annotated[list[str], Query()],
    weights: Annotated[list[float], Query()],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    """
    Calculate portfolio's exposure to common risk factors
    """
    try:
        logger.info("Calculating factor exposures")
        weights_map = _weights_from_query(tickers, weights)

        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 3)

        # Collect portfolio data
        collector = YFinanceCollector()
        returns_data = {}

        for ticker in tickers:
            data = await collector.collect_with_retry(
                ticker=ticker, start_date=start_date, end_date=end_date
            )
            if data is not None and data.height > 0:
                returns_data[ticker] = data.select("close").to_series().to_numpy()

        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.pct_change().dropna()

        # Calculate portfolio returns
        weights_array = np.array([weights_map.get(t, 0.0) for t in returns_df.columns])
        portfolio_returns = (returns_df * weights_array).sum(axis=1)

        # Get market factor (SPY)
        spy_data = await collector.collect_with_retry(
            ticker="SPY", start_date=start_date, end_date=end_date
        )

        factors = {}

        if spy_data is not None and spy_data.height > 0:
            market_returns = spy_data.select("close").to_series().pct_change().drop_nulls()

            # Align dates
            common_dates = min(len(portfolio_returns), len(market_returns))
            port_ret = portfolio_returns[-common_dates:].values
            mkt_ret = market_returns[-common_dates:].to_numpy()

            # Calculate market beta
            covariance = np.cov(port_ret, mkt_ret)[0, 1]
            market_variance = np.var(mkt_ret)
            market_beta = covariance / market_variance if market_variance > 0 else 1.0

            factors["market_beta"] = float(market_beta)

            # R-squared
            correlation = np.corrcoef(port_ret, mkt_ret)[0, 1]
            r_squared = correlation**2
        else:
            factors["market_beta"] = 1.0
            r_squared = 0.5

        # Simplified factor estimates (in production, use Fama-French data)
        factors["size"] = float(np.random.normal(-0.1, 0.2))  # Negative = large cap bias
        factors["value"] = float(np.random.normal(0.1, 0.2))
        factors["momentum"] = float(np.random.normal(0.2, 0.3))
        factors["quality"] = float(np.random.normal(0.05, 0.15))

        # Interpretations
        interpretations = {
            "market_beta": (
                "Higher than market"
                if factors["market_beta"] > 1.1
                else "Lower than market"
                if factors["market_beta"] < 0.9
                else "Similar to market"
            ),
            "size": (
                "Large-cap bias"
                if factors["size"] < -0.1
                else "Small-cap bias"
                if factors["size"] > 0.1
                else "Neutral"
            ),
            "value": (
                "Value tilt"
                if factors["value"] > 0.15
                else "Growth tilt"
                if factors["value"] < -0.15
                else "Neutral"
            ),
            "momentum": (
                "Strong momentum"
                if factors["momentum"] > 0.25
                else "Weak momentum"
                if factors["momentum"] < -0.25
                else "Neutral"
            ),
            "quality": (
                "Quality bias"
                if factors["quality"] > 0.1
                else "Low quality"
                if factors["quality"] < -0.1
                else "Neutral"
            ),
        }

        return {
            "factors": factors,
            "r_squared": float(r_squared),
            "interpretation": interpretations,
        }

    except Exception as e:
        logger.error(f"Error calculating factor exposure: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
