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
        data = data.to(self.device)
        z = self.model(data.x, data.edge_index, data.edge_attr)
        z_np = z.cpu().numpy().astype(np.float32)
        if z_np.shape[0] != len(tickers_order):
            logger.error(f"Node count mismatch: {z_np.shape[0]} vs {len(tickers_order)}")
            return
        for ticker, vec in zip(tickers_order, z_np, strict=True):
            await self.writer.write("graph", ticker, vec)
        logger.info(f"Graph embeddings updated for {len(tickers_order)} nodes")
