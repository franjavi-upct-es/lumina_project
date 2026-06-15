"""Live agent loop driven by assembled market states."""

from __future__ import annotations

import asyncio
import json
import signal
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dt_time
from typing import Any, cast

import numpy as np
from loguru import logger
from prometheus_client import Counter, Histogram

from backend.cognition.agent.ppo_agent import PPOAgent
from backend.cognition.agent.runtime import load_agent, pick_device
from backend.config.constants import NEXUS_OUTPUT_DIM
from backend.config.settings import Settings, get_settings
from backend.data_engine.storage.redis_cache import RedisCache, k_tick_latest
from backend.execution.broker.base import BaseBroker
from backend.execution.broker.factory import get_broker
from backend.execution.orchestrator import ExecutionOrchestrator
from backend.execution.safety.arbitrator import SafetyArbitrator
from backend.execution.safety.kill_switch import (
    KillSwitch,
    LocalKillSwitch,
    LocalKillSwitchListener,
)
from backend.execution.safety.risk_monitor import RiskMonitor

AGENT_LOOP_LATENCY = Histogram(
    "agent_loop_latency_seconds",
    "Live state -> agent -> broker latency",
    labelnames=("stage",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
AGENT_LOOP_SKIPS = Counter(
    "agent_loop_skips_total",
    "Live agent cycles skipped because required state was missing",
    labelnames=("ticker", "reason"),
)


def k_market_state(ticker: str) -> str:
    return f"state:market:{ticker}"


def _session_progress(now: datetime) -> float:
    """Approximate US cash-session progress in UTC, clamped to [0, 1]."""
    session_start = datetime.combine(now.date(), dt_time(14, 30), tzinfo=UTC)
    session_end = datetime.combine(now.date(), dt_time(21, 0), tzinfo=UTC)
    if now <= session_start:
        return 0.0
    if now >= session_end:
        return 1.0
    return (now - session_start).total_seconds() / (session_end - session_start).total_seconds()


@dataclass(slots=True)
class PortfolioSnapshot:
    state: np.ndarray
    equity: float
    peak_equity: float
    current_position: float
    consecutive_losses: int


class LiveAgentLoop:
    """One state-market-driven decision loop, reusable across tickers."""

    def __init__(
        self,
        *,
        agent: PPOAgent,
        orchestrator: ExecutionOrchestrator,
        broker: BaseBroker,
        redis: RedisCache,
        settings: Settings,
        latch: LocalKillSwitch | None = None,
    ) -> None:
        self.agent = agent
        self.orchestrator = orchestrator
        self.broker = broker
        self.redis = redis
        self.settings = settings
        self.latch = latch or LocalKillSwitch.instance()
        self._running = False
        self._agent_lock = asyncio.Lock()
        self._peak_equity = settings.INITIAL_CAPITAL
        self._last_equity: float | None = None
        self._consecutive_losses = 0

    async def run(self, ticker: str) -> None:
        self._running = True
        logger.info("LiveAgentLoop started for {}", ticker)
        while self._running:
            t0 = time.perf_counter()
            try:
                await self._cycle(ticker, t0)
            except Exception as exc:
                logger.error("Agent loop error on {}: {}", ticker, exc)
            await asyncio.sleep(self.settings.AGENT_TICK_INTERVAL_SECONDS)

    async def stop(self) -> None:
        self._running = False

    async def _cycle(self, ticker: str, t0: float | None = None) -> None:
        t0 = t0 or time.perf_counter()
        if self.latch.is_liquidate():
            AGENT_LOOP_SKIPS.labels(ticker=ticker, reason="kill_switch").inc()
            return

        market_state = await self._read_market_state(ticker)
        if market_state is None:
            AGENT_LOOP_SKIPS.labels(ticker=ticker, reason="market_state").inc()
            return
        t_state = time.perf_counter()
        AGENT_LOOP_LATENCY.labels(stage="state_fetch").observe(t_state - t0)

        latest_price = await self._latest_price(ticker)
        if latest_price is None:
            AGENT_LOOP_SKIPS.labels(ticker=ticker, reason="latest_price").inc()
            return
        self._update_broker_price(ticker, latest_price)

        portfolio = await self._portfolio_state(ticker, latest_price)
        full_state = np.concatenate([market_state, portfolio.state]).astype(np.float32)
        async with self._agent_lock:
            action, _log_prob, _value, uncertainty, vetoed = self.agent.act(
                full_state,
                deterministic=True,
            )
        t_agent = time.perf_counter()
        AGENT_LOOP_LATENCY.labels(stage="agent").observe(t_agent - t_state)

        result = await self.orchestrator.execute(
            ticker=ticker,
            proposed_action=action,
            uncertainty=uncertainty,
            latest_price=latest_price,
            peak_equity=portfolio.peak_equity,
            consecutive_losses=portfolio.consecutive_losses,
        )
        await self._update_loss_streak()
        total_s = time.perf_counter() - t0
        AGENT_LOOP_LATENCY.labels(stage="total").observe(total_s)

        ts = datetime.now(UTC).isoformat()
        payload = {
            "ticker": ticker,
            "action": action.tolist() if hasattr(action, "tolist") else list(action),
            "uncertainty": float(uncertainty),
            "vetoed": bool(vetoed) or not result.decision.approved,
            "gate_active": bool(vetoed) or not result.decision.approved,
            "final_action": float(result.final_action),
            "current_position": portfolio.current_position,
            "equity": portfolio.equity,
            "peak_equity": portfolio.peak_equity,
            "latest_price": latest_price,
            "consecutive_losses": self._consecutive_losses,
            "ts": ts,
            "latency_ms": total_s * 1000,
        }
        await self.redis.client.set("agent:last_action", json.dumps(payload))
        await cast(Awaitable[Any], self.redis.client.lpush("agent:history", json.dumps(payload)))
        await cast(Awaitable[Any], self.redis.client.ltrim("agent:history", 0, 999))
        await self.redis.client.publish(
            "channel:agent.action",
            json.dumps({"type": "action", "ts": ts, "payload": payload}),
        )

    async def _read_market_state(self, ticker: str) -> np.ndarray | None:
        raw = await self.redis.client.get(k_market_state(ticker))
        if raw is None:
            return None
        state = np.frombuffer(cast(bytes, raw), dtype=np.float32).copy()
        if state.shape != (NEXUS_OUTPUT_DIM,):
            logger.error("Bad market state shape for {}: {}", ticker, state.shape)
            return None
        return state

    async def _latest_price(self, ticker: str) -> float | None:
        raw = await self.redis.client.get(k_tick_latest(ticker))
        if raw is None:
            return None
        tick = json.loads(raw)
        price = tick.get("c")
        if price is None:
            return None
        price_f = float(price)
        return price_f if price_f > 0 else None

    def _update_broker_price(self, ticker: str, price: float) -> None:
        update = getattr(self.broker, "update_price", None)
        if callable(update):
            update(ticker, price)

    async def _portfolio_state(self, ticker: str, latest_price: float) -> PortfolioSnapshot:
        acct = await self.broker.get_account()
        self._peak_equity = max(self._peak_equity, acct.equity)
        drawdown = 1.0 - acct.equity / self._peak_equity if self._peak_equity > 0 else 0.0
        position = acct.positions.get(ticker)
        qty = position.qty if position else 0.0
        current_position = qty * latest_price / acct.equity if acct.equity > 0 else 0.0
        state = np.asarray(
            [
                np.clip(current_position, -1.0, 1.0),
                acct.equity / self.settings.INITIAL_CAPITAL
                if self.settings.INITIAL_CAPITAL > 0
                else 1.0,
                np.clip(drawdown, 0.0, 1.0),
                _session_progress(datetime.now(UTC)),
            ],
            dtype=np.float32,
        )
        return PortfolioSnapshot(
            state=state,
            equity=acct.equity,
            peak_equity=self._peak_equity,
            current_position=current_position,
            consecutive_losses=self._consecutive_losses,
        )

    async def _update_loss_streak(self) -> None:
        acct = await self.broker.get_account()
        if self._last_equity is not None:
            if acct.equity < self._last_equity - 1e-6:
                self._consecutive_losses += 1
            elif acct.equity > self._last_equity + 1e-6:
                self._consecutive_losses = 0
        self._last_equity = acct.equity
        self._peak_equity = max(self._peak_equity, acct.equity)


async def _amain() -> None:
    from backend.config.logging import configure_logging

    configure_logging()
    settings = get_settings()
    device = pick_device()
    redis = RedisCache()
    await redis.connect()
    broker = get_broker(settings)
    kill_switch = KillSwitch(redis)
    listener = LocalKillSwitchListener(redis)
    await listener.start()
    agent = load_agent(settings, device)
    orchestrator = ExecutionOrchestrator(
        broker=broker,
        arbitrator=SafetyArbitrator(),
        kill_switch=kill_switch,
    )
    live_loop = LiveAgentLoop(
        agent=agent,
        orchestrator=orchestrator,
        broker=broker,
        redis=redis,
        settings=settings,
    )
    risk_monitor = RiskMonitor(
        broker=broker,
        kill_switch=kill_switch,
        max_drawdown_soft=settings.MAX_DRAWDOWN_LIMIT * 0.75,
        max_drawdown_hard=settings.MAX_DRAWDOWN_LIMIT,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, request_stop)

    tasks = [
        asyncio.create_task(live_loop.run(ticker), name=f"agent_loop:{ticker}")
        for ticker in settings.LIVE_TICKERS
    ]
    tasks.append(asyncio.create_task(risk_monitor.run(), name="risk_monitor"))
    try:
        await stop_event.wait()
    finally:
        await live_loop.stop()
        await risk_monitor.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await listener.stop()
        await redis.disconnect()


def main() -> int:
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("Live agent loop crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
