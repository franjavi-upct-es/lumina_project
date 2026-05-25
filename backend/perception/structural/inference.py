# backend/perception/structural/inference.py
"""Graph encoder inference — runs daily."""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data

from backend.data_engine.storage.redis_cache import RedisCache
from backend.perception.common.embedding_writer import EmbeddingWriter
from backend.perception.structural.gat_model import GraphEncoder


class GraphInferenceService:
    def __init__(self, model: GraphEncoder, redis: RedisCache, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = device
        self.redis = redis
        self.writer = EmbeddingWriter(redis)

    @torch.no_grad()
    async def run_once(self, data: Data, tickers_order: list[str]) -> None:
        if data.x.shape[0] == 0:
            logger.warning("Graph inference skipped: empty node set (no price data yet?)")
            return
        data = data.to(self.device)
        z = self.model(data.x, data.edge_index, data.edge_attr)
        z_np = z.cpu().numpy().astype(np.float32)
        if z_np.shape[0] != len(tickers_order):
            logger.error(f"Node count mismatch: {z_np.shape[0]} vs {len(tickers_order)}")
            return
        for ticker, vec in zip(tickers_order, z_np, strict=True):
            await self.writer.write("graph", ticker, vec)
        logger.info(f"Graph embeddings updated for {len(tickers_order)} nodes")


_GRAPH_CHECKPOINT = "models/structural/best.pt"
_DAILY_INTERVAL_S = 24 * 3600


def _pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda") + 1
        return "cuda"
    except RuntimeError as exc:
        logger.warning(f"CUDA reports available but probe failed ({exc}); falling back to CPU.")
        return "cpu"


async def _amain() -> None:
    import asyncio
    import signal
    from datetime import UTC, datetime
    from pathlib import Path

    from backend.data_engine.storage.timescale import TimescaleStore
    from backend.perception.structural.graph_builder import build_graph_data

    device = _pick_device()
    model = GraphEncoder()
    ckpt_path = Path(_GRAPH_CHECKPOINT)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded Graph encoder weights from {ckpt_path} (device={device})")
    else:
        logger.warning(
            f"Graph checkpoint {ckpt_path} not found — running with random weights. "
            "Train via backend.perception.structural.trainer first."
        )

    redis = RedisCache()
    timescale = TimescaleStore()
    await redis.connect()
    await timescale.connect()
    service = GraphInferenceService(model, redis, device=device)
    logger.info("GraphInferenceService starting (daily cadence)")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        while not stop_event.is_set():
            try:
                data = await build_graph_data(datetime.now(UTC), timescale)
                tickers_order = list(getattr(data, "ticker", []))
                await service.run_once(data, tickers_order)
            except Exception as exc:
                logger.exception(f"Graph inference iteration failed: {exc}")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=_DAILY_INTERVAL_S)
            except TimeoutError:
                pass
    finally:
        await redis.disconnect()
        await timescale.disconnect()


def main() -> int:
    import asyncio
    import sys

    from backend.config.logging import configure_logging

    configure_logging()
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("GraphInferenceService crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
