# backend/perception/temporal/inference.py
"""Live TFT inference service."""

from __future__ import annotations

from collections import defaultdict, deque

import polars as pl
import torch
from loguru import logger

from backend.config.constants import OHLCV_WINDOW_MINUTES, TARGET_TICKERS
from backend.config.settings import get_settings
from backend.data_engine.storage.redis_cache import RedisCache
from backend.perception.common.embedding_writer import EmbeddingWriter
from backend.perception.temporal.preprocessor import preprocess_ohlcv_window
from backend.perception.temporal.tft_model import TemporalFusionTransformer


class TFTInferenceService:
    def __init__(
        self,
        model: TemporalFusionTransformer,
        redis: RedisCache,
        device: str = "cuda",
        window: int = OHLCV_WINDOW_MINUTES,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.window = window
        self.redis = redis
        self.writer = EmbeddingWriter(redis)
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._running = False

    async def run(self, tickers: list[str] | None = None) -> None:
        self._running = True
        tickers = tickers or list(TARGET_TICKERS)
        async for ticker, tick in self.redis.subscribe_ticks(tickers):
            if not self._running:
                break
            try:
                await self._handle(ticker, tick)
            except Exception as exc:
                logger.error(f"TFT inference error on {ticker}: {exc}")

    async def stop(self) -> None:
        self._running = False

    async def _handle(self, ticker: str, tick: dict) -> None:
        self._buffers[ticker].append(
            [
                float(tick["o"]),
                float(tick["h"]),
                float(tick["l"]),
                float(tick["c"]),
                float(tick["v"]),
            ]
        )
        buf = self._buffers[ticker]
        if len(buf) < self.window:
            return
        df = pl.DataFrame(
            list(buf),
            schema=["open", "high", "low", "close", "volume"],
            orient="row",
        )
        x = preprocess_ohlcv_window(df, ticker).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb, _ = self.model(x)
        await self.writer.write("price", ticker, emb.cpu().numpy().squeeze(0))


_TFT_CHECKPOINT = "models/temporal/best.pt"


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
    from pathlib import Path

    settings = get_settings()
    device = _pick_device()
    model = TemporalFusionTransformer()
    ckpt_path = Path(_TFT_CHECKPOINT)
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            model.load_state_dict(state_dict)
            logger.info(f"Loaded TFT weights from {ckpt_path} (device={device})")
        except RuntimeError as exc:
            if not settings.ALLOW_RANDOM_MODELS:
                raise
            logger.error(
                f"TFT checkpoint {ckpt_path} is incompatible: {exc}. Using random weights."
            )
    elif settings.ALLOW_RANDOM_MODELS:
        logger.warning(
            f"TFT checkpoint {ckpt_path} not found — running with random weights. "
            "Train via backend.perception.temporal.trainer first."
        )
    else:
        raise FileNotFoundError(
            f"TFT checkpoint {ckpt_path} not found. "
            "Set ALLOW_RANDOM_MODELS=true only for synthetic smoke tests."
        )

    redis = RedisCache()
    await redis.connect()
    service = TFTInferenceService(model, redis, device=device)
    logger.info("TFTInferenceService starting (window={}).", service.window)
    stop_event = asyncio.Event()
    stop_tasks: set[asyncio.Task[None]] = set()

    import signal

    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        stop_event.set()
        task = asyncio.create_task(service.stop())
        stop_tasks.add(task)
        task.add_done_callback(stop_tasks.discard)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _request_stop)

    try:
        await service.run(settings.LIVE_TICKERS or list(TARGET_TICKERS))
    finally:
        await redis.disconnect()


def main() -> int:
    import asyncio

    from backend.config.logging import configure_logging

    configure_logging()
    try:
        asyncio.run(_amain())
    except Exception:
        logger.exception("TFTInferenceService crashed")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
