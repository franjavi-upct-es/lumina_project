# backend/integration/e2e_loop.py
"""End-to-end inference loop with a strict latency budget.

This is the canonical "Reflex Arc" from
``Lumina_V3_Deep_Fusion_Architecture.md`` §6. Every cycle must complete
within ~100 ms:

      tick → feature fetch → fusion forward → agent forward → order

Each stage is instrumented via Prometheus histograms (label ``stage``).
A warning is logged whenever the *total* exceeds ``LATENCY_BUDGET_MS``;
``scripts/benchmark_e2e.py`` enforces this as a hard CI gate.

In-process kill-switch latch
============================
Per audit finding 3.2 we check the process-local
:class:`LocalKillSwitch` on entry to every cycle. The check is O(100 ns)
(see the class docstring), so adding it is free. If the latch is armed
the cycle short-circuits BEFORE any encoder forward pass or broker
call — propagation from "operator pressed the API button" to "loop
stops" is bounded by ~3-4 ms p99 (Redis pub/sub fan-out + asyncio task
wake-up), down from up to 5 s with the old slow-polling-only path.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime

import numpy as np
import torch
from loguru import logger
from prometheus_client import Counter, Histogram

from backend.cognition.agent.ppo_agent import PPOAgent
from backend.config.settings import get_settings
from backend.data_engine.storage.redis_cache import RedisCache, k_tick_latest
from backend.execution.orchestrator import ExecutionOrchestrator
from backend.execution.safety.kill_switch import LocalKillSwitch
from backend.feature_store.client import FeatureStoreClient
from backend.fusion.nexus import DeepFusionNexus

E2E_LATENCY = Histogram(
    "e2e_loop_latency_seconds",
    "Full reflex-arc latency, broken down by stage",
    labelnames=("stage",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
E2E_KILL_SWITCH_SHORT_CIRCUITS = Counter(
    "e2e_loop_kill_switch_short_circuits_total",
    "Cycles that short-circuited because the local kill-switch latch was armed",
)

LATENCY_BUDGET_MS: float = 100.0
"""Hard latency target from the architecture spec."""


class EndToEndLoop:
    """Single-ticker decision loop. Run one of these per asset."""

    def __init__(
        self,
        agent: PPOAgent,
        nexus: DeepFusionNexus,
        feature_client: FeatureStoreClient,
        orchestrator: ExecutionOrchestrator,
        redis: RedisCache,
        device: str = "cuda",
        tick_interval_s: float = 1.0,
        latch: LocalKillSwitch | None = None,
    ):
        self.agent = agent
        self.nexus = nexus.to(device).eval()
        self.fc = feature_client
        self.orch = orchestrator
        self.redis = redis
        self.device = device
        self.interval = tick_interval_s
        settings = get_settings()
        self._peak_equity = settings.INITIAL_CAPITAL
        self._last_equity: float | None = None
        self._consecutive_losses = 0
        # The latch defaults to the process-wide singleton so different
        # EndToEndLoop instances (one per ticker) share the same latch
        # without explicit wiring. Tests can inject a fresh instance.
        self.latch = latch or LocalKillSwitch.instance()
        self._running = False

    # ------------------------------------------------------------------
    async def run(self, ticker: str) -> None:
        """Main loop. Runs until ``stop()`` is called."""
        self._running = True
        logger.info(f"E2E loop started for {ticker}")
        while self._running:
            t0 = time.perf_counter()
            try:
                await self._cycle(ticker, t0)
            except Exception as exc:
                logger.error(f"E2E cycle error on {ticker}: {exc}")
            await asyncio.sleep(self.interval)

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    async def _cycle(self, ticker: str, t0: float) -> None:
        """One iteration of the reflex arc."""
        # --- Stage 0: in-process kill-switch latch -----------------------
        # First thing every cycle. If LIQUIDATE_ALL has been signalled
        # we stop here — no fetch, no fusion, no agent, no order. The
        # execution-side liquidation is the orchestrator's job (driven
        # by the SafetyArbitrator's kill-switch rule), not ours.
        if self.latch.is_liquidate():
            E2E_KILL_SWITCH_SHORT_CIRCUITS.inc()
            return

        # --- Stage 1: Feature fetch (target: ≤ 5 ms) -------------------
        bundle = await self.fc.get_bundle(ticker)
        t_fetch = time.perf_counter()
        E2E_LATENCY.labels(stage="feature_fetch").observe(t_fetch - t0)
        if len(bundle) < 3:
            return  # not enough modalities yet — silently skip

        # --- Stage 2: Fusion forward (target: ≤ 25 ms incl. uncertainty)
        p = torch.from_numpy(bundle["price_emb"]).unsqueeze(0).to(self.device)
        s = torch.from_numpy(bundle["semantic_emb"]).unsqueeze(0).to(self.device)
        g = torch.from_numpy(bundle["graph_emb"]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            state_mean, _state_std = self.nexus.encode_with_uncertainty(
                p,
                s,
                g,
                n_samples=10,
            )
        t_fusion = time.perf_counter()
        E2E_LATENCY.labels(stage="fusion").observe(t_fusion - t_fetch)

        latest_price = await self._latest_price(ticker)
        if latest_price is None:
            return
        self._update_broker_price(ticker, latest_price)
        acct = await self.orch.broker.get_account()
        self._peak_equity = max(self._peak_equity, acct.equity)
        position = acct.positions.get(ticker)
        position_fraction = (
            (position.qty * latest_price / acct.equity) if position and acct.equity > 0 else 0.0
        )
        drawdown = 1.0 - acct.equity / self._peak_equity if self._peak_equity > 0 else 0.0

        # --- Stage 3: Agent forward (target: ≤ 10 ms) ------------------
        state_np = state_mean.cpu().numpy().squeeze(0)
        portfolio_state = np.asarray(
            [
                np.clip(position_fraction, -1.0, 1.0),
                acct.equity / get_settings().INITIAL_CAPITAL,
                np.clip(drawdown, 0.0, 1.0),
                0.0,
            ],
            dtype=np.float32,
        )
        full_state = np.concatenate([state_np, portfolio_state]).astype(np.float32)
        action, _log_prob, _value, action_uncertainty, vetoed = self.agent.act(
            full_state,
            deterministic=True,
        )
        t_agent = time.perf_counter()
        E2E_LATENCY.labels(stage="agent").observe(t_agent - t_fusion)

        # --- Stage 4: Order build + submit (target: ≤ 60 ms) -----------
        result = await self.orch.execute(
            ticker=ticker,
            proposed_action=action,
            uncertainty=action_uncertainty,
            latest_price=latest_price,
            peak_equity=self._peak_equity,
            consecutive_losses=self._consecutive_losses,
        )
        await self._update_loss_streak()
        total_s = time.perf_counter() - t0
        E2E_LATENCY.labels(stage="total").observe(total_s)

        if total_s * 1000 > LATENCY_BUDGET_MS:
            logger.warning(
                f"Latency budget exceeded: {total_s * 1000:.1f}ms > {LATENCY_BUDGET_MS}ms",
            )

        # --- Persist + publish for the dashboard -----------------------
        ts = datetime.now(UTC).isoformat()
        payload = {
            "ticker": ticker,
            "action": action.tolist() if hasattr(action, "tolist") else list(action),
            "uncertainty": float(action_uncertainty),
            "vetoed": bool(vetoed),
            "final_action": (
                result.final_action.tolist()
                if hasattr(result.final_action, "tolist")
                else result.final_action
            ),
            "ts": ts,
            "latency_ms": total_s * 1000,
        }
        await self.redis.client.set("agent:last_action", json.dumps(payload))
        await self.redis.client.lpush("agent:history", json.dumps(payload))
        await self.redis.client.ltrim("agent:history", 0, 999)
        await self.redis.client.publish(
            "channel:agent.action",
            json.dumps({"type": "action", "ts": ts, "payload": payload}),
        )

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
        update = getattr(self.orch.broker, "update_price", None)
        if callable(update):
            update(ticker, price)

    async def _update_loss_streak(self) -> None:
        acct = await self.orch.broker.get_account()
        if self._last_equity is not None:
            if acct.equity < self._last_equity - 1e-6:
                self._consecutive_losses += 1
            elif acct.equity > self._last_equity + 1e-6:
                self._consecutive_losses = 0
        self._last_equity = acct.equity
        self._peak_equity = max(self._peak_equity, acct.equity)
