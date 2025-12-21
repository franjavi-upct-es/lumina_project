# backend/workers/backtest_tasks.py
"""
Celery tasks for backtesting strategies
"""

from celery import shared_task
from typing import Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
import numpy as np
import pandas as pd
from uuid import uuid4
import itertools

from data_engine.collectors.yfinance_collector import YFinanceCollector
from data_engine.transformers.feature_engineering import FeatureEngineer
from config.settings import get_settings

settings = get_settings()


@shared_task(bind=True, name="workers.backtest_tasks.run_backtest_task")
def run_backtest_task(
    self, job_id: str, strategy_name: str, strategy_code: str, config: Dict[str, Any]
):
    """
    Execute a backtest asynchronously
    """
    try:
        logger.info(f"Starting backtest job {job_id}: {strategy_name}")

        # Update progress
        self.update_state(
            state="PROGRESS", meta={"step": "data_collection", "progress": 10}
        )

        # Collect data
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        collector = YFinanceCollector()
        all_data = {}

        for ticker in config["tickers"]:
            data = loop.run_until_complete(
                collector.collect_with_retry(
                    ticker=ticker,
                    start_date=datetime.fromisoformat(config["start_date"]),
                    end_date=datetime.fromisoformat(config["end_date"]),
                )
            )

            if data is not None:
                fe = FeatureEngineer()
                enriched_data = fe.create_all_features(data)
                all_data[ticker] = enriched_data

        loop.close()

        if not all_data:
            raise ValueError("No data collected")

        self.update_state(
            state="PROGRESS", meta={"step": "strategy_execution", "progress": 40}
        )

        # Compile strategy
        namespace = {}
        exec(strategy_code, namespace)
        strategy_func = namespace.get("strategy")

        if not strategy_func:
            raise ValueError("Strategy function not found")

        # Initialize backtest
        equity = config["initial_capital"]
        positions = {}
        trades = []
        equity_curve = []

        # Execute backtest
        for ticker, data in all_data.items():
            data_pd = data.to_pandas()

            try:
                signals = strategy_func(data_pd, data_pd)
            except Exception as e:
                logger.error(f"Strategy error for {ticker}: {e}")
                continue

            for i, signal in enumerate(signals):
                if i == 0:
                    continue

                current_price = data_pd.iloc[i]["close"]
                current_time = data_pd.index[i]

                # BUY signal
                if signal == "BUY" and ticker not in positions:
                    position_value = equity * config["position_size"]
                    shares = position_value / current_price

                    commission = position_value * config["commission"]
                    slippage = position_value * config["slippage"]
                    cost = position_value + commission + slippage

                    if equity >= cost and len(positions) < config["max_positions"]:
                        positions[ticker] = {
                            "shares": shares,
                            "entry_price": current_price,
                            "entry_time": current_time,
                        }
                        equity -= cost

                # SELL signal
                elif signal == "SELL" and ticker in positions:
                    position = positions[ticker]
                    proceeds = position["shares"] * current_price

                    commission = proceeds * config["commission"]
                    slippage = proceeds * config["slippage"]
                    net_proceeds = proceeds - commission - slippage

                    pnl = net_proceeds - (position["shares"] * position["entry_price"])
                    pnl_percent = (
                        pnl / (position["shares"] * position["entry_price"]) * 100
                    )

                    equity += net_proceeds

                    trades.append(
                        {
                            "ticker": ticker,
                            "direction": "long",
                            "entry_time": position["entry_time"],
                            "exit_time": current_time,
                            "entry_price": position["entry_price"],
                            "exit_price": current_price,
                            "quantity": position["shares"],
                            "pnl": pnl,
                            "pnl_percent": pnl_percent,
                            "commission": commission,
                            "slippage": slippage,
                        }
                    )

                    del positions[ticker]

                # Check stop loss / take profit
                for tick, pos in list(positions.items()):
                    current_return = (current_price - pos["entry_price"]) / pos[
                        "entry_price"
                    ]

                    if (
                        config.get("stop_loss")
                        and current_return <= -config["stop_loss"]
                    ):
                        # Stop loss triggered
                        proceeds = pos["shares"] * current_price
                        commission = proceeds * config["commission"]
                        slippage = proceeds * config["slippage"]
                        net_proceeds = proceeds - commission - slippage

                        pnl = net_proceeds - (pos["shares"] * pos["entry_price"])
                        equity += net_proceeds

                        trades.append(
                            {
                                "ticker": tick,
                                "direction": "long",
                                "entry_time": pos["entry_time"],
                                "exit_time": current_time,
                                "entry_price": pos["entry_price"],
                                "exit_price": current_price,
                                "quantity": pos["shares"],
                                "pnl": pnl,
                                "pnl_percent": (
                                    pnl / (pos["shares"] * pos["entry_price"])
                                )
                                * 100,
                                "commission": commission,
                                "slippage": slippage,
                                "exit_reason": "stop_loss",
                            }
                        )

                        del positions[tick]

                    elif (
                        config.get("take_profit")
                        and current_return >= config["take_profit"]
                    ):
                        # Take profit triggered
                        proceeds = pos["shares"] * current_price
                        commission = proceeds * config["commission"]
                        slippage = proceeds * config["slippage"]
                        net_proceeds = proceeds - commission - slippage

                        pnl = net_proceeds - (pos["shares"] * pos["entry_price"])
                        equity += net_proceeds

                        trades.append(
                            {
                                "ticker": tick,
                                "direction": "long",
                                "entry_time": pos["entry_time"],
                                "exit_time": current_time,
                                "entry_price": pos["entry_price"],
                                "exit_price": current_price,
                                "quantity": pos["shares"],
                                "pnl": pnl,
                                "pnl_percent": (
                                    pnl / (pos["shares"] * pos["entry_price"])
                                )
                                * 100,
                                "commission": commission,
                                "slippage": slippage,
                                "exit_reason": "take_profit",
                            }
                        )

                        del positions[tick]

                # Record equity
                equity_curve.append(
                    {
                        "date": current_time.isoformat(),
                        "equity": equity,
                    }
                )

        self.update_state(
            state="PROGRESS", meta={"step": "metrics_calculation", "progress": 80}
        )

        # Calculate metrics
        final_capital = equity
        total_return = (final_capital - config["initial_capital"]) / config[
            "initial_capital"
        ]

        # Time-based metrics
        days = (
            datetime.fromisoformat(config["end_date"])
            - datetime.fromisoformat(config["start_date"])
        ).days
        years = days / 365.25
        annualized_return = (
            (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        )

        # Risk metrics
        equity_values = [point["equity"] for point in equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]

        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0

        mean_return = np.mean(returns) if len(returns) > 0 else 0.0
        sharpe_ratio = (mean_return * 252) / volatility if volatility > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = (
            np.std(downside_returns) * np.sqrt(252)
            if len(downside_returns) > 0
            else 0.0
        )
        sortino_ratio = (mean_return * 252) / downside_std if downside_std > 0 else 0.0

        # Drawdown
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = (equity_values - running_max) / running_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        # Trade statistics
        num_trades = len(trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] < 0]

        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0

        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = (
            np.mean([abs(t["pnl"]) for t in losing_trades]) if losing_trades else 0.0
        )

        total_wins = sum([t["pnl"] for t in winning_trades]) if winning_trades else 0.0
        total_losses = (
            abs(sum([t["pnl"] for t in losing_trades])) if losing_trades else 0.0
        )
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Monthly returns
        monthly_returns = {}
        for trade in trades:
            if "exit_time" in trade and trade["exit_time"]:
                month_key = pd.to_datetime(trade["exit_time"]).strftime("%Y-%m")
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = 0.0
                monthly_returns[month_key] += trade["pnl"]

        self.update_state(state="PROGRESS", meta={"step": "completed", "progress": 100})

        # Save results (in production, save to database)
        backtest_id = str(uuid4())

        result = {
            "backtest_id": backtest_id,
            "job_id": job_id,
            "strategy_name": strategy_name,
            "tickers": config["tickers"],
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "initial_capital": config["initial_capital"],
            "final_capital": final_capital,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "equity_curve": equity_curve,
            "trades": trades,
            "monthly_returns": monthly_returns,
            "completed_at": datetime.now().isoformat(),
        }

        logger.success(f"Backtest {backtest_id} completed successfully")
        return result

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@shared_task(bind=True, name="workers.backtest_tasks.walk_forward_optimization_task")
def walk_forward_optimization_task(
    self,
    job_id: str,
    strategy_name: str,
    strategy_code: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    train_period_days: int,
    test_period_days: int,
    step_days: int,
    param_ranges: Dict[str, Dict[str, Any]],
    initial_capital: float,
    commission: float,
    slippage: float,
):
    """
    Perform walk-forward optimization
    """
    try:
        logger.info(f"Starting walk-forward optimization: {job_id}")

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        results = []
        current_start = start_dt
        window_num = 0

        while (
            current_start + timedelta(days=train_period_days + test_period_days)
            <= end_dt
        ):
            window_num += 1

            train_start = current_start
            train_end = current_start + timedelta(days=train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_period_days)

            logger.info(
                f"Window {window_num}: Train {train_start} to {train_end}, Test {test_start} to {test_end}"
            )

            self.update_state(
                state="PROGRESS",
                meta={
                    "step": f"window_{window_num}",
                    "window": window_num,
                    "train_period": f"{train_start.date()} to {train_end.date()}",
                    "test_period": f"{test_start.date()} to {test_end.date()}",
                },
            )

            # Optimize on training period
            if param_ranges:
                # Run parameter optimization
                best_params = _optimize_on_period(
                    strategy_code=strategy_code,
                    tickers=tickers,
                    start_date=train_start,
                    end_date=train_end,
                    param_ranges=param_ranges,
                    initial_capital=initial_capital,
                    commission=commission,
                    slippage=slippage,
                )
            else:
                best_params = {}

            # Test on out-of-sample period
            test_config = {
                "tickers": tickers,
                "start_date": test_start.isoformat(),
                "end_date": test_end.isoformat(),
                "initial_capital": initial_capital,
                "position_size": 0.1,
                "max_positions": 10,
                "commission": commission,
                "slippage": slippage,
            }

            test_result = run_backtest_task(
                job_id=f"{job_id}_window_{window_num}",
                strategy_name=f"{strategy_name}_window_{window_num}",
                strategy_code=strategy_code,
                config=test_config,
            )

            results.append(
                {
                    "window": window_num,
                    "train_period": {
                        "start": train_start.isoformat(),
                        "end": train_end.isoformat(),
                    },
                    "test_period": {
                        "start": test_start.isoformat(),
                        "end": test_end.isoformat(),
                    },
                    "best_params": best_params,
                    "test_metrics": {
                        "total_return": test_result["total_return"],
                        "sharpe_ratio": test_result["sharpe_ratio"],
                        "max_drawdown": test_result["max_drawdown"],
                        "num_trades": test_result["num_trades"],
                    },
                }
            )

            # Move forward
            current_start += timedelta(days=step_days)

        # Aggregate results
        avg_return = np.mean([r["test_metrics"]["total_return"] for r in results])
        avg_sharpe = np.mean([r["test_metrics"]["sharpe_ratio"] for r in results])
        avg_drawdown = np.mean([r["test_metrics"]["max_drawdown"] for r in results])

        summary = {
            "job_id": job_id,
            "strategy_name": strategy_name,
            "num_windows": len(results),
            "aggregate_metrics": {
                "avg_return": avg_return,
                "avg_sharpe": avg_sharpe,
                "avg_drawdown": avg_drawdown,
                "consistency": np.std(
                    [r["test_metrics"]["total_return"] for r in results]
                ),
            },
            "windows": results,
            "completed_at": datetime.now().isoformat(),
        }

        logger.success(f"Walk-forward optimization completed: {job_id}")
        return summary

    except Exception as e:
        logger.error(f"Walk-forward optimization failed: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@shared_task(bind=True, name="workers.backtest_tasks.optimize_parameters_task")
def optimize_parameters_task(
    self,
    job_id: str,
    strategy_name: str,
    strategy_code: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    param_grid: Dict[str, List[Any]],
    optimization_metric: str,
    initial_capital: float,
    commission: float,
):
    """
    Optimize strategy parameters using grid search
    """
    try:
        logger.info(f"Starting parameter optimization: {job_id}")

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        total_combinations = len(combinations)
        logger.info(f"Testing {total_combinations} parameter combinations")

        results = []

        for idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            self.update_state(
                state="PROGRESS",
                meta={
                    "step": "optimization",
                    "progress": int((idx / total_combinations) * 100),
                    "current_combination": idx + 1,
                    "total_combinations": total_combinations,
                    "current_params": params,
                },
            )

            # Inject parameters into strategy code
            modified_code = _inject_parameters(strategy_code, params)

            # Run backtest
            config = {
                "tickers": tickers,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
                "position_size": 0.1,
                "max_positions": 10,
                "commission": commission,
                "slippage": 0.0005,
            }

            try:
                result = run_backtest_task(
                    job_id=f"{job_id}_combo_{idx}",
                    strategy_name=f"{strategy_name}_combo_{idx}",
                    strategy_code=modified_code,
                    config=config,
                )

                metric_value = result.get(optimization_metric, 0.0)

                results.append(
                    {
                        "params": params,
                        "metric_value": metric_value,
                        "metrics": {
                            "total_return": result["total_return"],
                            "sharpe_ratio": result["sharpe_ratio"],
                            "max_drawdown": result["max_drawdown"],
                            "num_trades": result["num_trades"],
                        },
                    }
                )

            except Exception as e:
                logger.error(f"Backtest failed for params {params}: {e}")
                results.append(
                    {
                        "params": params,
                        "metric_value": -np.inf,
                        "error": str(e),
                    }
                )

        # Find best parameters
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            raise ValueError("All parameter combinations failed")

        best_result = max(valid_results, key=lambda x: x["metric_value"])

        # Sort all results
        sorted_results = sorted(
            valid_results,
            key=lambda x: x["metric_value"],
            reverse=True,
        )

        summary = {
            "job_id": job_id,
            "strategy_name": strategy_name,
            "optimization_metric": optimization_metric,
            "total_combinations": total_combinations,
            "successful_combinations": len(valid_results),
            "best_parameters": best_result["params"],
            "best_metric_value": best_result["metric_value"],
            "best_metrics": best_result["metrics"],
            "top_10_results": sorted_results[:10],
            "completed_at": datetime.now().isoformat(),
        }

        logger.success(f"Parameter optimization completed: {job_id}")
        return summary

    except Exception as e:
        logger.error(f"Parameter optimization failed: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


# Helper functions
def _optimize_on_period(
    strategy_code: str,
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    param_ranges: Dict[str, Dict[str, Any]],
    initial_capital: float,
    commission: float,
    slippage: float,
) -> Dict[str, Any]:
    """
    Optimize parameters on a specific period
    """
    # Generate parameter grid
    param_grid = {}
    for param_name, param_config in param_ranges.items():
        if param_config["type"] == "range":
            param_grid[param_name] = list(
                np.arange(
                    param_config["start"],
                    param_config["end"],
                    param_config.get("step", 1),
                )
            )
        elif param_config["type"] == "list":
            param_grid[param_name] = param_config["values"]

    # Run optimization
    result = optimize_parameters_task(
        job_id=f"opt_{start_date.date()}",
        strategy_name="optimization",
        strategy_code=strategy_code,
        tickers=tickers,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        param_grid=param_grid,
        optimization_metric="sharpe_ratio",
        initial_capital=initial_capital,
        commission=commission,
    )

    return result["best_parameters"]


def _inject_parameters(strategy_code: str, params: Dict[str, Any]) -> str:
    """
    Inject parameters into strategy code
    """
    # Add parameter definitions at the top of the strategy
    param_definitions = "\n".join([f"{k} = {repr(v)}" for k, v in params.items()])

    # Find where to inject (after imports, before function definition)
    lines = strategy_code.split("\n")
    inject_index = 0

    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            inject_index = i
            break

    modified_lines = lines[:inject_index] + [param_definitions] + lines[inject_index:]

    return "\n".join(modified_lines)
