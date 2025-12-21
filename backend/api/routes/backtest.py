# backend/api/routes/backtest.py
"""
Backtesting endpoints for strategy testing and optimization
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4
from loguru import logger
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from workers.backtest_tasks import run_backtest_task
from config.settings import get_settings
from db.models import get_async_session

router = APIRouter()
settings = get_settings()


# Request Models
class BacktestRequest(BaseModel):
    strategy_name: str
    strategy_code: str = Field(..., description="Python code for the strategy")

    # Tickers and dates
    tickers: List[str] = Field(..., min_items=1, max_items=50)
    start_date: datetime
    end_date: datetime

    # Capital and sizing
    initial_capital: float = Field(100000.0, ge=1000.0, le=10000000.0)
    position_size: float = Field(
        0.1, ge=0.01, le=1.0, description="Fraction of capital per position"
    )
    max_positions: int = Field(10, ge=1, le=50)

    # Transaction costs
    commission: float = Field(
        0.001, ge=0.0, le=0.1, description="Commission rate (0.001 = 0.1%)"
    )
    slippage: float = Field(0.0005, ge=0.0, le=0.05, description="Slippage rate")

    # Risk management
    stop_loss: Optional[float] = Field(
        None, ge=0.01, le=0.5, description="Stop loss percentage"
    )
    take_profit: Optional[float] = Field(
        None, ge=0.01, le=2.0, description="Take profit percentage"
    )

    # Backtesting options
    benchmark: str = "SPY"
    rebalance_frequency: str = Field("daily", regex="^(daily|weekly|monthly)$")

    # Execution
    async_execution: bool = True


class WalkForwardRequest(BaseModel):
    """Request for walk-forward optimization"""

    strategy_name: str
    strategy_code: str
    tickers: List[str]
    start_date: datetime
    end_date: datetime

    # Walk-forward parameters
    train_period_days: int = Field(252, ge=30, le=1000)  # 1 year default
    test_period_days: int = Field(63, ge=10, le=252)  # 3 months default
    step_days: int = Field(21, ge=1, le=126)  # 1 month default

    # Parameter ranges to optimize
    param_ranges: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Parameter ranges for optimization"
    )

    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005


class OptimizationRequest(BaseModel):
    """Request for parameter optimization"""

    strategy_name: str
    strategy_code: str
    tickers: List[str]
    start_date: datetime
    end_date: datetime

    # Parameters to optimize
    param_grid: Dict[str, List[Any]] = Field(
        ..., description="Grid of parameters to test"
    )

    # Optimization metric
    optimization_metric: str = Field(
        "sharpe_ratio", regex="^(sharpe_ratio|sortino_ratio|calmar_ratio|total_return)$"
    )

    initial_capital: float = 100000.0
    commission: float = 0.001


# Response Models
class BacktestJobResponse(BaseModel):
    job_id: str
    strategy_name: str
    status: str
    message: str
    estimated_time_minutes: Optional[int] = None


class BacktestResultsResponse(BaseModel):
    backtest_id: str
    strategy_name: str
    tickers: List[str]
    start_date: datetime
    end_date: datetime

    # Capital
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Benchmark comparison
    benchmark: str
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float

    # Additional metrics
    recovery_factor: Optional[float] = None
    payoff_ratio: Optional[float] = None

    # Detailed data
    equity_curve: Optional[List[Dict[str, Any]]] = None
    trades: Optional[List[Dict[str, Any]]] = None
    monthly_returns: Optional[Dict[str, float]] = None


class BacktestListResponse(BaseModel):
    backtests: List[Dict[str, Any]]
    total: int


# In-memory job tracking
backtest_jobs = {}


@router.post("/run", response_model=BacktestJobResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run a backtest with custom strategy

    **Strategy Code Example:**
    ```python
    def strategy(data, features):
        signals = []
        for i in range(len(data)):
            if features['rsi_14'][i] < 30:
                signals.append('BUY')
            elif features['rsi_14'][i] > 70:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        return signals
    ```

    **Returns:** Job ID for tracking progress
    """
    try:
        job_id = str(uuid4())

        logger.info(f"Received backtest request: {request.strategy_name}")

        # Validate strategy code
        _validate_strategy_code(request.strategy_code)

        if request.async_execution:
            # Async execution with Celery
            task = run_backtest_task.delay(
                job_id=job_id,
                strategy_name=request.strategy_name,
                strategy_code=request.strategy_code,
                config=request.dict(),
            )

            backtest_jobs[job_id] = {
                "task_id": task.id,
                "strategy_name": request.strategy_name,
                "status": "queued",
                "created_at": datetime.now(),
            }

            return BacktestJobResponse(
                job_id=job_id,
                strategy_name=request.strategy_name,
                status="queued",
                message="Backtest job submitted. Check /backtest/jobs/{job_id} for status.",
                estimated_time_minutes=5,
            )
        else:
            # Synchronous execution
            background_tasks.add_task(_run_backtest_sync, job_id, request)

            return BacktestJobResponse(
                job_id=job_id,
                strategy_name=request.strategy_name,
                status="running",
                message="Backtest started in background",
                estimated_time_minutes=5,
            )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error initiating backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_backtest_job_status(job_id: str):
    """
    Get status of a backtest job
    """
    if job_id not in backtest_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = backtest_jobs[job_id]

    # Check Celery task status
    from workers.celery_app import celery_app

    task = celery_app.AsyncResult(job["task_id"])

    status = {
        "job_id": job_id,
        "strategy_name": job["strategy_name"],
        "status": task.state,
        "created_at": job["created_at"].isoformat(),
    }

    if task.state == "PROGRESS":
        status["progress"] = task.info
    elif task.state == "SUCCESS":
        status["result"] = task.result
        status["backtest_id"] = task.result.get("backtest_id")
    elif task.state == "FAILURE":
        status["error"] = str(task.info)

    return status


@router.get("/results/{backtest_id}", response_model=BacktestResultsResponse)
async def get_backtest_results(
    backtest_id: str,
    include_equity_curve: bool = Query(True),
    include_trades: bool = Query(False),
    include_monthly_returns: bool = Query(True),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get detailed results of a completed backtest
    """
    try:
        from db.models import BacktestResult, BacktestTrade

        # Query backtest results
        query = select(BacktestResult).where(BacktestResult.backtest_id == backtest_id)
        result = await db.execute(query)
        backtest = result.scalar_one_or_none()

        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")

        # Build response
        response = BacktestResultsResponse(
            backtest_id=str(backtest.backtest_id),
            strategy_name=backtest.strategy_name,
            tickers=backtest.tickers,
            start_date=backtest.start_date,
            end_date=backtest.end_date,
            initial_capital=backtest.initial_capital,
            final_capital=backtest.final_capital,
            total_return=backtest.total_return,
            annualized_return=backtest.annualized_return,
            volatility=backtest.volatility,
            sharpe_ratio=backtest.sharpe_ratio,
            sortino_ratio=backtest.sortino_ratio,
            calmar_ratio=backtest.calmar_ratio,
            max_drawdown=backtest.max_drawdown,
            total_trades=backtest.num_trades,
            winning_trades=int(backtest.num_trades * backtest.win_rate)
            if backtest.win_rate
            else 0,
            losing_trades=int(backtest.num_trades * (1 - backtest.win_rate))
            if backtest.win_rate
            else 0,
            win_rate=backtest.win_rate or 0.0,
            avg_win=backtest.avg_trade
            if backtest.avg_trade and backtest.avg_trade > 0
            else 0.0,
            avg_loss=abs(backtest.avg_trade)
            if backtest.avg_trade and backtest.avg_trade < 0
            else 0.0,
            profit_factor=backtest.profit_factor or 0.0,
            benchmark="SPY",  # From config
            benchmark_return=0.0,  # Calculate separately
            alpha=0.0,  # Calculate separately
            beta=1.0,  # Calculate separately
            information_ratio=0.0,  # Calculate separately
        )

        # Include equity curve if requested
        if include_equity_curve and backtest.config:
            equity_data = backtest.config.get("equity_curve", [])
            response.equity_curve = equity_data

        # Include trades if requested
        if include_trades:
            trades_query = (
                select(BacktestTrade)
                .where(BacktestTrade.backtest_id == backtest_id)
                .order_by(BacktestTrade.entry_time)
            )

            trades_result = await db.execute(trades_query)
            trades = trades_result.scalars().all()

            response.trades = [
                {
                    "trade_id": str(t.trade_id),
                    "ticker": t.ticker,
                    "direction": t.direction,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent,
                }
                for t in trades
            ]

        # Include monthly returns if requested
        if include_monthly_returns and backtest.config:
            monthly_data = backtest.config.get("monthly_returns", {})
            response.monthly_returns = monthly_data

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=BacktestListResponse)
async def list_backtests(
    strategy_name: Optional[str] = None,
    ticker: Optional[str] = None,
    min_sharpe: Optional[float] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List all backtests with filtering
    """
    try:
        from db.models import BacktestResult
        from sqlalchemy import func

        # Build query
        query = select(BacktestResult)

        # Apply filters
        if strategy_name:
            query = query.where(
                BacktestResult.strategy_name.ilike(f"%{strategy_name}%")
            )
        if ticker:
            query = query.where(BacktestResult.tickers.contains([ticker]))
        if min_sharpe is not None:
            query = query.where(BacktestResult.sharpe_ratio >= min_sharpe)

        # Apply pagination and ordering
        query = (
            query.offset(offset).limit(limit).order_by(BacktestResult.created_at.desc())
        )

        result = await db.execute(query)
        backtests = result.scalars().all()

        # Count total
        count_query = select(func.count(BacktestResult.backtest_id))
        if strategy_name:
            count_query = count_query.where(
                BacktestResult.strategy_name.ilike(f"%{strategy_name}%")
            )
        if ticker:
            count_query = count_query.where(BacktestResult.tickers.contains([ticker]))
        if min_sharpe is not None:
            count_query = count_query.where(BacktestResult.sharpe_ratio >= min_sharpe)

        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Format response
        backtests_list = [
            {
                "backtest_id": str(b.backtest_id),
                "strategy_name": b.strategy_name,
                "tickers": b.tickers,
                "start_date": b.start_date.isoformat(),
                "end_date": b.end_date.isoformat(),
                "total_return": b.total_return,
                "sharpe_ratio": b.sharpe_ratio,
                "max_drawdown": b.max_drawdown,
                "num_trades": b.num_trades,
                "created_at": b.created_at.isoformat(),
            }
            for b in backtests
        ]

        return BacktestListResponse(backtests=backtests_list, total=total)

    except Exception as e:
        logger.error(f"Error listing backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/results/{backtest_id}")
async def delete_backtest(
    backtest_id: str,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Delete a backtest result
    """
    try:
        from db.models import BacktestResult

        # Query backtest
        query = select(BacktestResult).where(BacktestResult.backtest_id == backtest_id)
        result = await db.execute(query)
        backtest = result.scalar_one_or_none()

        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")

        # Delete (cascade will delete trades)
        await db.delete(backtest)
        await db.commit()

        logger.info(f"Backtest {backtest_id} deleted")

        return {"message": f"Backtest {backtest_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting backtest: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/walk-forward")
async def run_walk_forward_optimization(request: WalkForwardRequest):
    """
    Run walk-forward optimization

    **Walk-Forward Process:**
    1. Split data into train/test windows
    2. Optimize parameters on training window
    3. Test on out-of-sample window
    4. Roll forward and repeat
    5. Aggregate results

    **Returns:** Job ID for tracking
    """
    try:
        job_id = str(uuid4())

        logger.info(f"Walk-forward optimization: {request.strategy_name}")

        # Calculate number of windows
        total_days = (request.end_date - request.start_date).days
        num_windows = (total_days - request.train_period_days) // request.step_days

        logger.info(f"Walk-forward will test {num_windows} windows")

        # Submit walk-forward task
        from workers.backtest_tasks import walk_forward_optimization_task

        task = walk_forward_optimization_task.delay(
            job_id=job_id,
            strategy_name=request.strategy_name,
            strategy_code=request.strategy_code,
            tickers=request.tickers,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            train_period_days=request.train_period_days,
            test_period_days=request.test_period_days,
            step_days=request.step_days,
            param_ranges=request.param_ranges,
            initial_capital=request.initial_capital,
            commission=request.commission,
            slippage=request.slippage,
        )

        backtest_jobs[job_id] = {
            "task_id": task.id,
            "strategy_name": request.strategy_name,
            "type": "walk_forward",
            "status": "queued",
            "created_at": datetime.now(),
        }

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Walk-forward optimization started",
            "num_windows": num_windows,
            "estimated_time_minutes": num_windows * 2,
        }

    except Exception as e:
        logger.error(f"Error in walk-forward optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_parameters(request: OptimizationRequest):
    """
    Optimize strategy parameters using grid search

    **Example Parameter Grid:**
    ```json
    {
        "rsi_period": [7, 14, 21],
        "rsi_oversold": [20, 25, 30],
        "rsi_overbought": [70, 75, 80]
    }
    ```

    **Returns:** Best parameters and performance
    """
    try:
        job_id = str(uuid4())

        logger.info(f"Parameter optimization: {request.strategy_name}")

        # Calculate total combinations
        total_combinations = 1
        for param_values in request.param_grid.values():
            total_combinations *= len(param_values)

        logger.info(f"Testing {total_combinations} parameter combinations")

        # Submit optimization task
        from workers.backtest_tasks import optimize_parameters_task

        task = optimize_parameters_task.delay(
            job_id=job_id,
            strategy_name=request.strategy_name,
            strategy_code=request.strategy_code,
            tickers=request.tickers,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            param_grid=request.param_grid,
            optimization_metric=request.optimization_metric,
            initial_capital=request.initial_capital,
            commission=request.commission,
        )

        backtest_jobs[job_id] = {
            "task_id": task.id,
            "strategy_name": request.strategy_name,
            "type": "optimization",
            "status": "queued",
            "created_at": datetime.now(),
        }

        return {
            "job_id": job_id,
            "status": "queued",
            "total_combinations": total_combinations,
            "estimated_time_minutes": total_combinations * 0.5,
            "message": "Optimization started",
        }

    except Exception as e:
        logger.error(f"Error in parameter optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def compare_strategies(
    backtest_ids: List[str] = Query(..., description="List of backtest IDs to compare"),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Compare multiple backtest results side-by-side
    """
    try:
        if len(backtest_ids) < 2:
            raise HTTPException(
                status_code=400, detail="At least 2 backtests required for comparison"
            )

        from db.models import BacktestResult

        # Fetch all backtests
        query = select(BacktestResult).where(
            BacktestResult.backtest_id.in_(backtest_ids)
        )
        result = await db.execute(query)
        backtests = result.scalars().all()

        if len(backtests) != len(backtest_ids):
            raise HTTPException(
                status_code=404, detail="One or more backtests not found"
            )

        # Build comparison
        comparison = {
            "strategies": [],
            "metrics_comparison": {},
            "best_by_metric": {},
        }

        metrics = [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ]

        for backtest in backtests:
            strategy_data = {
                "backtest_id": str(backtest.backtest_id),
                "strategy_name": backtest.strategy_name,
                "metrics": {
                    "total_return": backtest.total_return,
                    "annualized_return": backtest.annualized_return,
                    "sharpe_ratio": backtest.sharpe_ratio,
                    "sortino_ratio": backtest.sortino_ratio,
                    "max_drawdown": backtest.max_drawdown,
                    "win_rate": backtest.win_rate,
                    "profit_factor": backtest.profit_factor,
                },
            }
            comparison["strategies"].append(strategy_data)

        # Find best by each metric
        for metric in metrics:
            values = [
                b.__dict__[metric]
                for b in backtests
                if b.__dict__.get(metric) is not None
            ]
            if values:
                if metric == "max_drawdown":
                    best_idx = values.index(min(values))  # Lower is better
                else:
                    best_idx = values.index(max(values))  # Higher is better

                comparison["best_by_metric"][metric] = {
                    "backtest_id": str(backtests[best_idx].backtest_id),
                    "strategy_name": backtests[best_idx].strategy_name,
                    "value": values[best_idx],
                }

        return comparison

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}/trades")
async def get_backtest_trades(
    backtest_id: str,
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get detailed trade list from a backtest
    """
    try:
        from db.models import BacktestTrade

        # Query trades
        query = (
            select(BacktestTrade)
            .where(BacktestTrade.backtest_id == backtest_id)
            .order_by(BacktestTrade.entry_time.desc())
            .limit(limit)
        )

        result = await db.execute(query)
        trades = result.scalars().all()

        if not trades:
            return {"trades": [], "total": 0}

        # Format trades
        trades_list = [
            {
                "trade_id": str(t.trade_id),
                "ticker": t.ticker,
                "direction": t.direction,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_percent": t.pnl_percent,
                "commission": t.commission,
                "slippage": t.slippage,
            }
            for t in trades
        ]

        return {"trades": trades_list, "total": len(trades)}

    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}/equity-curve")
async def get_equity_curve(
    backtest_id: str,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get equity curve data for visualization
    """
    try:
        from db.models import BacktestResult

        # Query backtest
        query = select(BacktestResult).where(BacktestResult.backtest_id == backtest_id)
        result = await db.execute(query)
        backtest = result.scalar_one_or_none()

        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")

        # Extract equity curve from config
        equity_curve = []
        if backtest.config and "equity_curve" in backtest.config:
            equity_curve = backtest.config["equity_curve"]
        else:
            # Generate basic equity curve from trades
            from db.models import BacktestTrade

            trades_query = (
                select(BacktestTrade)
                .where(BacktestTrade.backtest_id == backtest_id)
                .order_by(BacktestTrade.exit_time)
            )

            trades_result = await db.execute(trades_query)
            trades = trades_result.scalars().all()

            equity = backtest.initial_capital
            for trade in trades:
                if trade.exit_time and trade.pnl:
                    equity += trade.pnl
                    equity_curve.append(
                        {
                            "date": trade.exit_time.isoformat(),
                            "equity": equity,
                            "returns": trade.pnl / (equity - trade.pnl)
                            if equity != trade.pnl
                            else 0,
                        }
                    )

        return {"equity_curve": equity_curve}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching equity curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}/drawdown")
async def get_drawdown_analysis(
    backtest_id: str,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get detailed drawdown analysis
    """
    try:
        from db.models import BacktestResult

        # Get equity curve
        equity_response = await get_equity_curve(backtest_id, db)
        equity_curve = equity_response["equity_curve"]

        if not equity_curve:
            return {
                "max_drawdown": 0.0,
                "avg_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "drawdown_periods": [],
            }

        # Calculate drawdowns
        equity_values = [point["equity"] for point in equity_curve]
        dates = [point["date"] for point in equity_curve]

        running_max = np.maximum.accumulate(equity_values)
        drawdowns = (equity_values - running_max) / running_max

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0

        for i, dd in enumerate(drawdowns):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append(
                    {
                        "start_date": dates[start_idx],
                        "end_date": dates[i],
                        "peak": float(running_max[start_idx]),
                        "trough": float(min(equity_values[start_idx : i + 1])),
                        "drawdown": float(min(drawdowns[start_idx : i + 1])),
                        "duration_days": i - start_idx,
                    }
                )

        # Calculate statistics
        max_dd = float(min(drawdowns)) if len(drawdowns) > 0 else 0.0
        avg_dd = (
            float(np.mean([dd for dd in drawdowns if dd < 0]))
            if any(dd < 0 for dd in drawdowns)
            else 0.0
        )
        max_dd_duration = (
            max([p["duration_days"] for p in drawdown_periods])
            if drawdown_periods
            else 0
        )

        return {
            "max_drawdown": max_dd,
            "avg_drawdown": avg_dd,
            "max_drawdown_duration": max_dd_duration,
            "num_drawdown_periods": len(drawdown_periods),
            "drawdown_periods": sorted(drawdown_periods, key=lambda x: x["drawdown"])[
                :10
            ],  # Top 10 worst drawdowns
        }

    except Exception as e:
        logger.error(f"Error calculating drawdowns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}/monthly-returns")
async def get_monthly_returns(
    backtest_id: str,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Get monthly returns breakdown
    """
    try:
        from db.models import BacktestResult, BacktestTrade

        # Query backtest
        query = select(BacktestResult).where(BacktestResult.backtest_id == backtest_id)
        result = await db.execute(query)
        backtest = result.scalar_one_or_none()

        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")

        # Check if monthly returns are stored in config
        if backtest.config and "monthly_returns" in backtest.config:
            return {"monthly_returns": backtest.config["monthly_returns"]}

        # Calculate from trades
        trades_query = (
            select(BacktestTrade)
            .where(BacktestTrade.backtest_id == backtest_id)
            .where(BacktestTrade.exit_time.isnot(None))
            .order_by(BacktestTrade.exit_time)
        )

        trades_result = await db.execute(trades_query)
        trades = trades_result.scalars().all()

        if not trades:
            return {"monthly_returns": {}}

        # Group by month
        monthly_pnl = {}
        for trade in trades:
            month_key = trade.exit_time.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0.0
            monthly_pnl[month_key] += trade.pnl or 0.0

        # Calculate returns
        equity = backtest.initial_capital
        monthly_returns = {}

        for month in sorted(monthly_pnl.keys()):
            pnl = monthly_pnl[month]
            ret = (pnl / equity) * 100 if equity > 0 else 0.0
            monthly_returns[month] = round(ret, 2)
            equity += pnl

        return {"monthly_returns": monthly_returns}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating monthly returns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def _validate_strategy_code(code: str):
    """
    Validate strategy code for security
    """
    # Check for dangerous operations
    forbidden = [
        "import os",
        "import sys",
        "import subprocess",
        "eval(",
        "exec(",
        "__import__",
        "open(",
        "file(",
        "input(",
        "compile(",
        "globals(",
        "locals(",
    ]

    for forbidden_str in forbidden:
        if forbidden_str in code:
            raise ValueError(f"Forbidden operation detected: {forbidden_str}")

    # Code should define a function called 'strategy'
    if "def strategy(" not in code:
        raise ValueError("Strategy code must define a function called 'strategy'")


async def _run_backtest_sync(job_id: str, request: BacktestRequest):
    """
    Synchronous backtest execution
    """
    try:
        backtest_jobs[job_id]["status"] = "running"

        # Import required modules
        from data_engine.collectors.yfinance_collector import YFinanceCollector
        from data_engine.transformers.feature_engineering import FeatureEngineer
        import polars as pl

        # Collect data for all tickers
        collector = YFinanceCollector()
        all_data = {}

        for ticker in request.tickers:
            data = await collector.collect_with_retry(
                ticker=ticker,
                start_date=request.start_date,
                end_date=request.end_date,
            )

            if data is not None:
                # Engineer features
                fe = FeatureEngineer()
                enriched_data = fe.create_all_features(data)
                all_data[ticker] = enriched_data

        if not all_data:
            raise ValueError("No data collected for any ticker")

        # Compile strategy
        namespace = {}
        exec(request.strategy_code, namespace)
        strategy_func = namespace.get("strategy")

        if not strategy_func:
            raise ValueError("Strategy function not found in code")

        # Initialize backtest state
        equity = request.initial_capital
        positions = {}
        trades = []
        equity_curve = [{"date": request.start_date.isoformat(), "equity": equity}]

        # Run backtest (simplified version)
        for ticker, data in all_data.items():
            data_pd = data.to_pandas()

            # Generate signals
            try:
                signals = strategy_func(data_pd, data_pd)
            except Exception as e:
                logger.error(f"Error in strategy execution: {e}")
                continue

            # Execute trades based on signals
            for i, signal in enumerate(signals):
                if i == 0:
                    continue

                current_price = data_pd.iloc[i]["close"]

                if signal == "BUY" and ticker not in positions:
                    # Open long position
                    position_value = equity * request.position_size
                    shares = position_value / current_price
                    cost = position_value * (1 + request.commission + request.slippage)

                    if equity >= cost:
                        positions[ticker] = {
                            "shares": shares,
                            "entry_price": current_price,
                            "entry_time": data_pd.iloc[i].name,
                        }
                        equity -= cost

                elif signal == "SELL" and ticker in positions:
                    # Close position
                    position = positions[ticker]
                    proceeds = (
                        position["shares"]
                        * current_price
                        * (1 - request.commission - request.slippage)
                    )

                    pnl = proceeds - (position["shares"] * position["entry_price"])
                    equity += proceeds

                    trades.append(
                        {
                            "ticker": ticker,
                            "direction": "long",
                            "entry_time": position["entry_time"],
                            "exit_time": data_pd.iloc[i].name,
                            "entry_price": position["entry_price"],
                            "exit_price": current_price,
                            "shares": position["shares"],
                            "pnl": pnl,
                        }
                    )

                    del positions[ticker]

                # Record equity
                equity_curve.append(
                    {
                        "date": data_pd.iloc[i].name.isoformat(),
                        "equity": equity,
                    }
                )

        # Calculate metrics
        total_return = (equity - request.initial_capital) / request.initial_capital

        returns = np.diff([point["equity"] for point in equity_curve])
        returns = returns / [point["equity"] for point in equity_curve[:-1]]

        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        sharpe = (np.mean(returns) * 252) / volatility if volatility > 0 else 0.0

        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        # Store results (in production, save to database)
        backtest_id = str(uuid4())

        backtest_jobs[job_id]["status"] = "completed"
        backtest_jobs[job_id]["backtest_id"] = backtest_id
        backtest_jobs[job_id]["results"] = {
            "initial_capital": request.initial_capital,
            "final_capital": equity,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "volatility": volatility,
            "num_trades": len(trades),
            "win_rate": win_rate,
            "equity_curve": equity_curve,
        }

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        backtest_jobs[job_id]["status"] = "failed"
        backtest_jobs[job_id]["error"] = str(e)
