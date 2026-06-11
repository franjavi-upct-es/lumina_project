# backend/api/routes/portfolio.py
"""Portfolio-state HTTP endpoint.

Returns the full account snapshot — cash, equity, open positions, and
the running peak-to-trough drawdown. The frontend's
``usePortfolio`` hook polls this every 3 seconds; the broker is queried
once per call so the latency is dominated by the broker's own RTT.

Why does drawdown computation live here and not in the broker?
==============================================================
The :class:`AlpacaBroker` does not natively track peak equity, and we
don't want the :class:`PaperBroker` to maintain different state from
the live broker just to fill that gap. Instead, the API persists the
peak in Redis under the key ``portfolio:peak_equity`` and updates it
monotonically: on every call, ``peak ← max(peak_redis, current_equity)``.

This has three useful properties:
    1. The peak survives API restarts (it's in Redis, not in process memory).
    2. The peak resets implicitly when the operator deletes the key,
       which is exactly what an operator does after a flash-crash event.
    3. The drawdown is consistent across multiple replicas of the API.
"""

from datetime import UTC, datetime, timedelta

<<<<<<< HEAD
from fastapi import APIRouter, Depends
=======
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field
from scipy.optimize import minimize
from sqlalchemy.ext.asyncio import AsyncSession
>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e

from backend.api.deps import get_broker, get_redis, get_timescale, require_api_key
from backend.api.schemas import (
    EquityPoint,
    PortfolioHistoryResponse,
    PortfolioResponse,
    PositionResponse,
)
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import TimescaleStore
from backend.execution.broker.base import BaseBroker

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# Redis key for the persisted peak equity. Stored as plain bytes-encoded
# floats so it survives encoding changes; an empty value reads as
# "first observation" and the current equity becomes the new peak.
_PEAK_EQUITY_KEY: str = "portfolio:peak_equity"


async def _read_peak_equity(redis: RedisCache) -> float | None:
    """Read the persisted peak; returns ``None`` if not yet recorded."""
    raw = await redis.client.get(_PEAK_EQUITY_KEY)
    if raw is None:
        return None
    try:
        return float(raw.decode("utf-8"))
    except (AttributeError, ValueError):
        return None


async def _write_peak_equity(redis: RedisCache, value: float) -> None:
    """Persist the peak. No TTL: peak is a long-lived monotonic counter."""
    await redis.client.set(_PEAK_EQUITY_KEY, str(value))


@router.get(
    "",
    response_model=PortfolioResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_portfolio(
    broker: BaseBroker = Depends(get_broker),
    redis: RedisCache = Depends(get_redis),
) -> PortfolioResponse:
    """Return the live account snapshot with computed drawdown."""
    account = await broker.get_account()

    # Update the running peak. We use ``max`` rather than a CAS loop
    # because the race window (multi-replica API + concurrent requests)
    # is bounded by milliseconds and the worst case is a one-call-late
    # peak update — never a *lower* peak being persisted than reality.
    prior_peak = await _read_peak_equity(redis)
    new_peak = max(prior_peak, account.equity) if prior_peak is not None else account.equity
    if prior_peak is None or new_peak > prior_peak:
        await _write_peak_equity(redis, new_peak)

    drawdown = 1.0 - account.equity / new_peak if new_peak > 0 else 0.0
    # Clamp into [0, 1] to defend against floating-point noise just above
    # the peak (which would otherwise yield a tiny negative drawdown).
    drawdown = max(0.0, min(1.0, drawdown))

    positions = [
        PositionResponse(
            ticker=p.ticker,
            qty=p.qty,
            avg_entry_price=p.avg_entry_price,
            unrealized_pnl=p.unrealized_pnl,
            market_value=p.market_value,
        )
        for p in account.positions.values()
    ]

    return PortfolioResponse(
        equity=account.equity,
        cash=account.cash,
        buying_power=account.buying_power,
        positions=positions,
        peak_equity=new_peak,
        drawdown_pct=drawdown,
    )


@router.get(
    "/history",
    response_model=PortfolioHistoryResponse,
    dependencies=[Depends(require_api_key)],
)
async def get_portfolio_history(
    range: str = "30D",
    ts: TimescaleStore = Depends(get_timescale),
) -> PortfolioHistoryResponse:
    """Return the historical portfolio equity."""
    days = 30
    if range == "1D":
        days = 1
    elif range == "7D":
        days = 7
    elif range == "90D":
        days = 90
    elif range == "YTD":
        now = datetime.now(UTC)
        days = (now - datetime(now.year, 1, 1, tzinfo=UTC)).days
    elif range == "ALL":
        days = 3650

    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    # Choose resolution based on range to prevent too many points
    interval = timedelta(hours=1) if days <= 7 else timedelta(days=1)

    rows = await ts.get_portfolio_history(start, end, interval)

    if not rows:
        return PortfolioHistoryResponse(history=[])

<<<<<<< HEAD
    history = [
        EquityPoint(
            time=r["time_bucket"],
            equity=r["equity"],
            benchmark=None,
=======
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
    db: Annotated[AsyncSession, Depends(get_async_session)] = "default",  # type: ignore
):
    """
    Get current portfolio metrics and performance
    """
    raise HTTPException(status_code=501, detail="Not implemented")


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
        estimated_cost = sum(trade["value"] for trade in trades) * settings.DEFAULT_COMMISSION  # type: ignore

        return RebalanceResponse(
            trades_needed=needs_rebalance,
            trades=trades,
            estimated_cost=estimated_cost,
            reason=(
                f"Rebalancing needed: {len(trades)} positions exceed {request.rebalance_threshold:.1%} threshold"
                if needs_rebalance
                else "No rebalancing needed"
            ),
>>>>>>> 994b45ea5c7f16817f4caea4d941fa54c203899e
        )
        for r in rows
    ]
    return PortfolioHistoryResponse(history=history)
