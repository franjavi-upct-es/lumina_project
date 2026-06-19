# backend/perception/semantic/inference.py
"""Live semantic encoder."""

from __future__ import annotations

import json

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer

from backend.config.constants import DIM_SEMANTIC
from backend.config.settings import get_settings
from backend.data_engine.storage.redis_cache import RedisCache
from backend.perception.common.embedding_writer import EmbeddingWriter
from backend.perception.semantic.distilled_llm import TEACHER_MODEL, DistilledFinancialEncoder


class SemanticInferenceService:
    def __init__(
        self,
        model: DistilledFinancialEncoder,
        redis: RedisCache,
        device: str = "cuda",
        max_len: int = 256,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
        self.redis = redis
        self.writer = EmbeddingWriter(redis)
        self._running = False

    async def run(self) -> None:
        self._running = True
        pubsub = self.redis.client.pubsub()
        await pubsub.subscribe("channel:news.global")
        try:
            # Poll rather than block in ``listen()``: redis-py 8.x defaults
            # ``socket_timeout`` to 5 s, so a blocking read raises
            # ``TimeoutError`` on an idle channel. The 1 s poll also keeps
            # ``_running`` responsive for graceful shutdown.
            while self._running:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None or msg["type"] != "message":
                    continue
                try:
                    await self._handle(json.loads(msg["data"]))
                except Exception as exc:
                    logger.error(f"Semantic inference error: {exc}")
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()

    async def stop(self) -> None:
        self._running = False

    async def seed_no_news_embeddings(self, tickers: list[str]) -> None:
        vec = np.zeros(DIM_SEMANTIC, dtype=np.float32)
        for ticker in tickers:
            await self.writer.write("semantic", ticker, vec)

    async def _handle(self, event: dict) -> None:
        text = f"{event.get('headline', '')} {event.get('body', '')}"
        tickers = event.get("tickers") or []
        if not tickers or not text.strip():
            return
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            emb, _ = self.model(enc["input_ids"], enc["attention_mask"])
        vec = emb.cpu().numpy().squeeze(0).astype(np.float32)
        for t in tickers:
            await self.writer.write("semantic", t, vec)


_SEMANTIC_CHECKPOINT = "models/semantic/best.pt"


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
    from pathlib import Path

    settings = get_settings()
    device = _pick_device()
    model = DistilledFinancialEncoder()
    ckpt_path = Path(_SEMANTIC_CHECKPOINT)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
        logger.info(f"Loaded semantic encoder weights from {ckpt_path} (device={device})")
    elif settings.ALLOW_RANDOM_MODELS:
        logger.warning(
            f"Semantic checkpoint {ckpt_path} not found — running with random weights. "
            "Train via backend.perception.semantic.trainer first."
        )
    else:
        raise FileNotFoundError(
            f"Semantic checkpoint {ckpt_path} not found. "
            "Set ALLOW_RANDOM_MODELS=true only for synthetic smoke tests."
        )

    redis = RedisCache()
    await redis.connect()
    service = SemanticInferenceService(model, redis, device=device)
    await service.seed_no_news_embeddings(settings.LIVE_TICKERS)
    logger.info("SemanticInferenceService starting (subscribed to channel:news.global)")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(service.stop()))

    try:
        await service.run()
    finally:
        await redis.disconnect()


def main() -> int:
    import asyncio

    from backend.config.logging import configure_logging

    configure_logging()
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("SemanticInferenceService crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
