# backend/api/routes/agent.py
"""Agent routes: status, history, WebSocket stream."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from backend.api.deps import get_redis, require_api_key
from backend.api.schemas import AgentStatusResponse
from backend.data_engine.storage.redis_cache import RedisCache

import numpy as np

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.get("/status", response_model=AgentStatusResponse, dependencies=[Depends(require_api_key)])
async def get_status(redis: RedisCache = Depends(get_redis)) -> AgentStatusResponse:
    data_raw = await redis.client.get("agent:last_action")
    
    # Also fetch attention weights for the dashboard heatmap
    # Hardcoded to SPY for the general status for now, could be parameterized
    attn_raw = await redis.client.get("state:attention:SPY")
    attn_list = None
    if attn_raw:
        attn_list = np.frombuffer(attn_raw, dtype=np.float32).tolist()

    if data_raw:
        d = json.loads(data_raw)
        return AgentStatusResponse(
            current_action=d.get("action", 0.0),
            uncertainty=d.get("uncertainty", 0.0),
            gate_active=d.get("gate_active", False),
            last_update=datetime.fromisoformat(d["ts"]),
            consecutive_vetoes=d.get("consecutive_vetoes", 0),
            attention_weights=attn_list
        )
    return AgentStatusResponse(
        current_action=0.0,
        uncertainty=0.0,
        gate_active=False,
        last_update=datetime.now(UTC),
        attention_weights=attn_list
    )


@router.get("/history", dependencies=[Depends(require_api_key)])
async def get_history(limit: int = 100, redis: RedisCache = Depends(get_redis)):
    entries = await redis.client.lrange("agent:history", 0, limit - 1)
    return [json.loads(e) for e in entries]


@router.websocket("/stream")
async def stream_agent(ws: WebSocket, redis: RedisCache = Depends(get_redis)):
    await ws.accept()
    pubsub = redis.client.pubsub()
    await pubsub.subscribe("channel:agent.action")
    heartbeat_task = None
    try:

        async def heartbeat():
            while True:
                await asyncio.sleep(15)
                await ws.send_json({"type": "heartbeat", "ts": datetime.now(UTC).isoformat()})

        heartbeat_task = asyncio.create_task(heartbeat())
        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            await ws.send_text(msg["data"].decode("utf-8"))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
        await pubsub.unsubscribe()
        await pubsub.aclose()
