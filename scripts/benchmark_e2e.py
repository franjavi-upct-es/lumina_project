# scripts/benchmark_e2e.py
"""CLI: benchmark the end-to-end loop latency.

Methodology
-----------
1. Build the full reflex-arc stack in process:
       - PaperBroker (no network)
       - DeepFusionNexus + PolicyNetwork loaded from ``models/``
       - RedisCache pre-populated with synthetic embeddings for SPY
2. Run ``EndToEndLoop._cycle`` 10,000 times, recording the latency of
   each stage via ``time.perf_counter``.
3. Print p50/p95/p99 per stage plus total.
4. Fail (exit 1) if the total p99 exceeds the 100 ms budget from the
   architecture spec.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from backend.cognition.agent.policy_network import PolicyNetwork
from backend.cognition.agent.ppo_agent import PPOAgent
from backend.cognition.agent.uncertainty_gate import UncertaintyGate
from backend.config.constants import (
    ACTION_DIM,
    DIM_GRAPH,
    DIM_PRICE,
    DIM_SEMANTIC,
    NEXUS_OUTPUT_DIM,
)
from backend.config.logging import configure_logging
from backend.data_engine.storage.redis_cache import get_redis_cache
from backend.execution.broker.paper_adapter import PaperBroker
from backend.execution.orchestrator import ExecutionOrchestrator
from backend.execution.safety.arbitrator import SafetyArbitrator
from backend.execution.safety.kill_switch import KillSwitch, LocalKillSwitch
from backend.feature_store.client import FeatureStoreClient
from backend.fusion.nexus import DeepFusionNexus
from backend.integration.e2e_loop import LATENCY_BUDGET_MS, EndToEndLoop

_ITERATIONS: int = 10_000
_TICKER: str = "SPY"


def _pctiles(latencies_ms: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(latencies_ms)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


async def _populate_synthetic_features(redis, ticker: str) -> None:
    rng = np.random.default_rng(42)
    await redis.set_embedding("price", ticker, rng.standard_normal(DIM_PRICE).astype(np.float32))
    await redis.set_embedding(
        "semantic", ticker, rng.standard_normal(DIM_SEMANTIC).astype(np.float32)
    )
    await redis.set_embedding("graph", ticker, rng.standard_normal(DIM_GRAPH).astype(np.float32))
    await redis.publish_tick(
        ticker,
        {
            "t": int(time.time() * 1000),
            "o": 450.0,
            "h": 451.0,
            "l": 449.0,
            "c": 450.0,
            "v": 1_000_000,
        },
    )


async def _benchmark() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Benchmarking on device: {device}")

    # 1. Load Checkpoints
    agent_path = Path("models/agent/final.pt")
    fusion_path = Path("models/fusion/best.pt")

    # Note: PolicyNetwork state_dim is NEXUS_OUTPUT_DIM + 4 (portfolio state)
    policy = PolicyNetwork(state_dim=NEXUS_OUTPUT_DIM + 4, action_dim=ACTION_DIM)
    if agent_path.exists():
        policy.load_state_dict(torch.load(agent_path, map_location=device, weights_only=True))
    else:
        logger.warning(f"Agent checkpoint missing at {agent_path}. Using random weights.")
    agent = PPOAgent(policy=policy, uncertainty_gate=UncertaintyGate(), device=device)

    nexus = DeepFusionNexus()
    if fusion_path.exists():
        fusion_data = torch.load(fusion_path, map_location=device, weights_only=True)
        nexus.load_state_dict(
            fusion_data.get("model", fusion_data) if isinstance(fusion_data, dict) else fusion_data
        )
    else:
        logger.warning(f"Fusion checkpoint missing at {fusion_path}. Using random weights.")
    nexus.to(device).eval()

    # 2. Build Stack
    redis = get_redis_cache()
    await redis.connect()
    await _populate_synthetic_features(redis, _TICKER)

    fc = FeatureStoreClient(mode="online", redis=redis)
    broker = PaperBroker()
    broker.update_price(_TICKER, 450.0)

    orchestrator = ExecutionOrchestrator(
        broker=broker,
        arbitrator=SafetyArbitrator(),
        kill_switch=KillSwitch(redis),
    )

    loop = EndToEndLoop(
        agent=agent,
        nexus=nexus,
        feature_client=fc,
        orchestrator=orchestrator,
        redis=redis,
        device=device,
        latch=LocalKillSwitch.instance(),
    )

    # 3. Warm-up
    logger.info("Warming up...")
    for _ in range(10):
        await loop._cycle(_TICKER, time.perf_counter())

    # 4. Measure
    logger.info(f"Running {_ITERATIONS} iterations...")
    latencies: list[float] = []

    # We'll use a simplified version of _cycle to avoid Prometheus overhead if necessary,
    # but EndToEndLoop._cycle is what we want to benchmark.
    for _ in range(_ITERATIONS):
        t0 = time.perf_counter()
        await loop._cycle(_TICKER, t0)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50, p95, p99 = _pctiles(latencies)
    logger.info(f"Total Latency: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")

    # Save to CSV for QA report
    stamp = time.strftime("%Y%m%dT%H%M%SZ")
    csv_path = Path(f"reports/benchmark_e2e_{stamp}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("metric,value_ms\n")
        f.write(f"p50,{p50:.3f}\n")
        f.write(f"p95,{p95:.3f}\n")
        f.write(f"p99,{p99:.3f}\n")
    logger.info(f"Benchmark results saved to {csv_path}")

    await redis.disconnect()

    if p99 > LATENCY_BUDGET_MS:
        logger.error(f"p99 latency {p99:.2f}ms exceeds budget {LATENCY_BUDGET_MS}ms")
        return 1

    logger.success("Benchmark passed.")
    return 0


def main() -> int:
    configure_logging()
    try:
        return asyncio.run(_benchmark())
    except Exception as e:
        logger.exception(f"Benchmark failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
