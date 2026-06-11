"""Redis-backed worker for dashboard-submitted backtests."""

from __future__ import annotations

import asyncio
import json
import math
import signal
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow
import numpy as np
from loguru import logger

from backend.api.schemas import BacktestRequest
from backend.cognition.agent.ppo_agent import PPOAgent
from backend.cognition.agent.runtime import load_agent, pick_device
from backend.config.constants import NEXUS_OUTPUT_DIM
from backend.config.logging import configure_logging
from backend.config.settings import Settings, get_settings
from backend.data_engine.storage.redis_cache import RedisCache
from backend.data_engine.storage.timescale import TimescaleStore
from backend.simulation.environments.base_env import EnvConfig, LuminaTradingEnv
from backend.simulation.generators.synthetic_data import jump_diffusion_episode

_REQUEST_PREFIX = "backtest:request:"
_RESULT_PREFIX = "backtest:result:"
_LOCK_PREFIX = "backtest:lock:"
_RESULT_TTL_SECONDS = 24 * 3600
_LOCK_TTL_SECONDS = 6 * 3600


@dataclass(slots=True)
class EpisodeMetrics:
    ticker: str
    sharpe: float
    max_drawdown: float
    total_return: float
    steps: int
    synthetic: bool


@dataclass(slots=True)
class BacktestMetrics:
    sharpe: float
    max_drawdown: float
    total_return: float
    steps: int
    synthetic_tickers: int

    def as_result_payload(self) -> dict[str, float | int | str]:
        return {
            "status": "completed",
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "steps": self.steps,
            "synthetic_tickers": self.synthetic_tickers,
        }


class _SingleEpisode:
    def __init__(self, episode: dict[str, Any]) -> None:
        self.episode = episode

    def __iter__(self):
        yield self.episode


class BacktestWorker:
    """Poll Redis for API-submitted backtest jobs and execute them."""

    def __init__(
        self,
        *,
        redis: RedisCache,
        timescale: TimescaleStore,
        agent: PPOAgent,
        settings: Settings,
    ) -> None:
        self.redis = redis
        self.timescale = timescale
        self.agent = agent
        self.settings = settings
        self._running = False

    async def run(self, stop_event: asyncio.Event) -> None:
        self._running = True
        logger.info("BacktestWorker started")
        while not stop_event.is_set():
            try:
                processed = await self.run_once()
                if processed:
                    logger.info("BacktestWorker processed {} job(s)", processed)
            except Exception:
                logger.exception("BacktestWorker poll cycle failed")
            await asyncio.sleep(self.settings.BACKTEST_WORKER_POLL_SECONDS)
        self._running = False
        logger.info("BacktestWorker stopped")

    async def run_once(self) -> int:
        processed = 0
        async for raw_key in self.redis.client.scan_iter(match=f"{_REQUEST_PREFIX}*", count=100):
            key = _decode(raw_key)
            if await self._process_key(key):
                processed += 1
        return processed

    async def _process_key(self, key: str) -> bool:
        run_id = key.removeprefix(_REQUEST_PREFIX)
        lock_key = _LOCK_PREFIX + run_id
        claimed = await self.redis.client.set(lock_key, b"1", ex=_LOCK_TTL_SECONDS, nx=True)
        if not claimed:
            return False

        try:
            raw = await self.redis.client.get(key)
            if raw is None:
                return False
            req = BacktestRequest.model_validate(json.loads(_decode(raw)))
            await self._set_result(run_id, {"status": "running"})
            await self.timescale.upsert_backtest_run(run_id, "running")

            metrics = await execute_backtest(
                run_id=run_id,
                req=req,
                timescale=self.timescale,
                agent=self.agent,
                settings=self.settings,
            )
            payload = metrics.as_result_payload()
            await self._set_result(run_id, payload)
            await self.timescale.upsert_backtest_run(
                run_id,
                "completed",
                metrics.sharpe,
                metrics.max_drawdown,
                metrics.total_return,
            )
            await self.redis.client.delete(key)
            logger.success("Backtest {} completed", run_id)
            return True
        except Exception as exc:
            logger.exception("Backtest {} failed", run_id)
            await self._set_result(run_id, {"status": "failed", "failure_reason": str(exc)})
            await self.timescale.upsert_backtest_run(run_id, "failed")
            await self.redis.client.delete(key)
            return True
        finally:
            await self.redis.client.delete(lock_key)

    async def _set_result(self, run_id: str, payload: dict[str, Any]) -> None:
        await self.redis.client.set(
            _RESULT_PREFIX + run_id,
            json.dumps({"run_id": run_id, **payload}),
            ex=_RESULT_TTL_SECONDS,
        )


async def execute_backtest(
    *,
    run_id: str,
    req: BacktestRequest,
    timescale: TimescaleStore,
    agent: PPOAgent,
    settings: Settings,
) -> BacktestMetrics:
    """Run one dashboard backtest request and log it to MLflow."""
    tickers = [ticker.strip().upper() for ticker in req.tickers if ticker.strip()]
    if not tickers:
        raise ValueError("backtest request has no tickers")

    start = _ensure_utc(req.start)
    end = _ensure_utc(req.end)
    if end <= start:
        raise ValueError("backtest end must be after start")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_BACKTEST_EXPERIMENT_NAME)

    episode_metrics: list[EpisodeMetrics] = []
    with mlflow.start_run(run_name=f"backtest_{run_id}"):
        mlflow.log_params(
            {
                "run_id": run_id,
                "tickers": ",".join(tickers),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "initial_capital": float(req.initial_capital),
            }
        )
        for ticker in tickers:
            rows = await timescale.get_historical_window_rows(ticker, start, end, freq="1d")
            episode = _episode_from_rows(ticker, rows)
            if episode is None:
                if not settings.ALLOW_SYNTHETIC_SIMULATION_FALLBACK:
                    raise RuntimeError(
                        f"Not enough historical OHLCV for {ticker} between "
                        f"{start.date()} and {end.date()}. Run a backfill first or set "
                        "ALLOW_SYNTHETIC_SIMULATION_FALLBACK=true for a clearly synthetic smoke test."
                    )
                episode = _synthetic_episode(
                    ticker=ticker,
                    n_steps=settings.BACKTEST_SYNTHETIC_STEPS,
                    seed=_seed_for(run_id, ticker),
                )
            metrics = _run_episode(
                ticker=ticker,
                episode=episode,
                agent=agent,
                initial_capital=req.initial_capital,
            )
            episode_metrics.append(metrics)
            mlflow.log_metrics(
                {
                    f"{ticker}_sharpe": metrics.sharpe,
                    f"{ticker}_max_drawdown": metrics.max_drawdown,
                    f"{ticker}_total_return": metrics.total_return,
                    f"{ticker}_steps": metrics.steps,
                    f"{ticker}_synthetic": float(metrics.synthetic),
                }
            )

        aggregate = _aggregate_metrics(episode_metrics)
        mlflow.log_metrics(
            {
                "sharpe": aggregate.sharpe,
                "max_drawdown": aggregate.max_drawdown,
                "total_return": aggregate.total_return,
                "steps": aggregate.steps,
                "synthetic_tickers": aggregate.synthetic_tickers,
            }
        )
        return aggregate


def _episode_from_rows(ticker: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(rows) < 3:
        return None
    sorted_rows = sorted(rows, key=lambda row: row["time"])
    prices = np.asarray([float(row["close"]) for row in sorted_rows], dtype=np.float32)
    if prices.size < 3 or np.any(prices <= 0):
        return None

    rng = np.random.default_rng(_seed_for("historical", ticker))
    log_returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
    return {
        "prices": prices,
        "market_states": rng.standard_normal((len(prices), NEXUS_OUTPUT_DIM)).astype(np.float32)
        * 0.1,
        "volatility": np.abs(log_returns).astype(np.float32),
        "uncertainties": np.full(len(prices), 0.2, dtype=np.float32),
        "open": np.asarray([float(row["open"]) for row in sorted_rows], dtype=np.float32),
        "high": np.asarray([float(row["high"]) for row in sorted_rows], dtype=np.float32),
        "low": np.asarray([float(row["low"]) for row in sorted_rows], dtype=np.float32),
        "close": prices,
        "volume": np.asarray([float(row["volume"] or 0) for row in sorted_rows], dtype=np.float32),
        "ticker": ticker,
        "synthetic": False,
    }


def _synthetic_episode(*, ticker: str, n_steps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    episode = jump_diffusion_episode(max(3, n_steps), rng=rng)
    episode["ticker"] = ticker
    return episode


def _run_episode(
    *,
    ticker: str,
    episode: dict[str, Any],
    agent: PPOAgent,
    initial_capital: float,
) -> EpisodeMetrics:
    env = LuminaTradingEnv(
        _SingleEpisode(episode),
        config=EnvConfig(initial_capital=float(initial_capital)),
    )
    obs, _ = env.reset()
    equities = [float(initial_capital)]
    rewards: list[float] = []
    done = False
    while not done:
        action, _log_prob, _value, _uncertainty, _vetoed = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        equities.append(float(info["equity"]))
        rewards.append(float(reward))
        done = bool(terminated or truncated)

    return EpisodeMetrics(
        ticker=ticker,
        sharpe=_sharpe_from_equity(equities),
        max_drawdown=_max_drawdown(equities),
        total_return=equities[-1] / max(equities[0], 1e-9) - 1.0,
        steps=len(rewards),
        synthetic=bool(episode.get("synthetic", False)),
    )


def _aggregate_metrics(items: list[EpisodeMetrics]) -> BacktestMetrics:
    if not items:
        raise ValueError("no backtest episodes executed")
    return BacktestMetrics(
        sharpe=float(np.mean([item.sharpe for item in items])),
        max_drawdown=float(max(item.max_drawdown for item in items)),
        total_return=float(np.mean([item.total_return for item in items])),
        steps=int(sum(item.steps for item in items)),
        synthetic_tickers=sum(1 for item in items if item.synthetic),
    )


def _sharpe_from_equity(equities: list[float]) -> float:
    if len(equities) < 3:
        return 0.0
    arr = np.asarray(equities, dtype=np.float64)
    returns = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
    if returns.size < 2:
        return 0.0
    std = float(returns.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float(returns.mean() / std * math.sqrt(252.0))


def _max_drawdown(equities: list[float]) -> float:
    arr = np.asarray(equities, dtype=np.float64)
    peaks = np.maximum.accumulate(arr)
    drawdowns = 1.0 - arr / np.maximum(peaks, 1e-9)
    return float(np.max(drawdowns)) if drawdowns.size else 0.0


def _seed_for(run_id: str, ticker: str) -> int:
    return zlib.crc32(f"{run_id}:{ticker}".encode()) & 0xFFFF_FFFF


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _decode(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


async def _amain() -> None:
    configure_logging()
    settings = get_settings()
    redis = RedisCache()
    timescale = TimescaleStore()
    await redis.connect()
    await timescale.connect()
    agent = load_agent(settings, pick_device())

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, request_stop)

    worker = BacktestWorker(
        redis=redis,
        timescale=timescale,
        agent=agent,
        settings=settings,
    )
    try:
        await worker.run(stop_event)
    finally:
        await redis.disconnect()
        await timescale.disconnect()


def main() -> int:
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("Backtest worker crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
