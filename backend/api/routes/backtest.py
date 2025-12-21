# backend/api/routes/backtest.py
"""
Backtesting endpoints for strategy testing and optimization
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4
from loguru import logger

from workers.backtest_tasks import run_backtest_task
from config.settings import get_settings

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
):
    """
    Get detailed results of a completed backtest
    """
    try:
        # TODO: Query from database
        raise HTTPException(status_code=404, detail="Backtest not found")

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
):
    """
    List all backtests with filtering
    """
    try:
        # TODO: Query from database with filters
        backtests = []

        return BacktestListResponse(backtests=backtests, total=len(backtests))

    except Exception as e:
        logger.error(f"Error listing backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/results/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """
    Delete a backtest result
    """
    try:
        # TODO: Delete from database
        return {"message": f"Backtest {backtest_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting backtest: {e}")
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

        # TODO: Implement walk-forward task

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Walk-forward optimization started",
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

        # TODO: Implement optimization task

        return {
            "job_id": job_id,
            "status": "queued",
            "total_combinations": total_combinations,
            "estimated_time_minutes": total_combinations * 0.5,
        }

    except Exception as e:
        logger.error(f"Error in parameter optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def compare_strategies(
    backtest_ids: List[str] = Query(..., description="List of backtest IDs to compare"),
):
    """
    Compare multiple backtest results side-by-side
    """
    try:
        if len(backtest_ids) < 2:
            raise HTTPException(
                status_code=400, detail="At least 2 backtests required for comparison"
            )

        # TODO: Fetch and compare results

        return {"comparison": "TODO: Implement comparison"}

    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}/trades")
async def get_backtest_trades(backtest_id: str, limit: int = Query(100, ge=1, le=1000)):
    """
    Get detailed trade list from a backtest
    """
    try:
        # TODO: Query trades from database
        return {"trades": []}

    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}/equity-curve")
async def get_equity_curve(backtest_id: str):
    """
    Get equity curve data for visualization
    """
    try:
        # TODO: Query equity curve
        return {"equity_curve": []}

    except Exception as e:
        logger.error(f"Error fetching equity curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}/drawdown")
async def get_drawdown_analysis(backtest_id: str):
    """
    Get detailed drawdown analysis
    """
    try:
        # TODO: Calculate drawdowns
        return {
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "drawdown_periods": [],
        }

    except Exception as e:
        logger.error(f"Error calculating drawdowns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{backtest_id}/monthly-returns")
async def get_monthly_returns(backtest_id: str):
    """
    Get monthly returns breakdown
    """
    try:
        # TODO: Calculate monthly returns
        return {"monthly_returns": {}}

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

        # TODO: Implement actual backtesting

        backtest_jobs[job_id]["status"] = "completed"

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        backtest_jobs[job_id]["status"] = "failed"
        backtest_jobs[job_id]["error"] = str(e)
