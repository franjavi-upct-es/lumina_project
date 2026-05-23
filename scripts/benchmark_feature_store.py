# scripts/benchmark_feature_store.py
"""CLI: benchmark the Feature Store hot path.

Methodology
-----------
1. Pre-populate Redis with random ``DIM_PRICE / DIM_SEMANTIC / DIM_GRAPH``
   float32 embeddings for every ticker in ``TARGET_TICKERS`` — that is,
   ``len(TARGET_TICKERS) × 3`` keys.
2. Time 10,000 single-ticker ``FeatureStoreClient.get_bundle`` calls
   — the common case during the inference loop.
3. Time 10,000 batched ``FeatureStoreClient.mget_embeddings`` calls
   over the full universe — used by the State Assembler.
4. Print p50/p95/p99 percentiles and fail (exit 1) if either p99
   violates its budget:
       single-ticker get_bundle : p99 ≤ 5 ms
       universe mget            : p99 ≤ 20 ms
"""

from __future__ import annotations

import asyncio
import sys
import time

import numpy as np
from loguru import logger

from backend.config.constants import DIM_GRAPH, DIM_PRICE, DIM_SEMANTIC, TARGET_TICKERS
from backend.config.logging import configure_logging
from backend.data_engine.storage.redis_cache import get_redis_cache
from backend.feature_store.client import FeatureStoreClient

_ITERATIONS: int = 10_000
_BUDGET_BUNDLE_P99_MS: float = 5.0
_BUDGET_MGET_P99_MS: float = 20.0


def _pctiles(latencies_ms: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(latencies_ms)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


async def _populate(redis, tickers: list[str]) -> None:
    rng = np.random.default_rng(0)
    for t in tickers:
        await redis.set_embedding("price", t, rng.standard_normal(DIM_PRICE).astype(np.float32))
        await redis.set_embedding(
            "semantic",
            t,
            rng.standard_normal(DIM_SEMANTIC).astype(np.float32),
        )
        await redis.set_embedding("graph", t, rng.standard_normal(DIM_GRAPH).astype(np.float32))


async def _benchmark() -> int:
    tickers = sorted(TARGET_TICKERS)
    redis = get_redis_cache()
    await redis.connect()
    await _populate(redis, tickers)

    client = FeatureStoreClient(mode="online", redis=redis)

    # Single-ticker bundle ----------------------------------------------
    bundle_latencies: list[float] = []
    for _ in range(_ITERATIONS):
        t0 = time.perf_counter()
        await client.get_bundle(tickers[0])
        bundle_latencies.append((time.perf_counter() - t0) * 1000.0)
    p50, p95, p99 = _pctiles(bundle_latencies)
    logger.info(f"get_bundle: p50={p50:.3f}ms  p95={p95:.3f}ms  p99={p99:.3f}ms")

    # Universe MGET -----------------------------------------------------
    mget_latencies: list[float] = []
    for _ in range(_ITERATIONS):
        t0 = time.perf_counter()
        await client.mget_embeddings("price_emb", tickers)
        mget_latencies.append((time.perf_counter() - t0) * 1000.0)
    m50, m95, m99 = _pctiles(mget_latencies)
    logger.info(f"mget:       p50={m50:.3f}ms  p95={m95:.3f}ms  p99={m99:.3f}ms")

    exit_code = 0
    if p99 > _BUDGET_BUNDLE_P99_MS:
        logger.error(f"get_bundle p99 {p99:.2f}ms exceeds budget {_BUDGET_BUNDLE_P99_MS}ms")
        exit_code = 1
    if m99 > _BUDGET_MGET_P99_MS:
        logger.error(f"mget p99 {m99:.2f}ms exceeds budget {_BUDGET_MGET_P99_MS}ms")
        exit_code = 1
    await redis.disconnect()
    return exit_code


def main() -> int:
    configure_logging()
    return asyncio.run(_benchmark())


if __name__ == "__main__":
    sys.exit(main())
