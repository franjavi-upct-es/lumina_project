# backend/api/routes/agent.py
"""Agent routes: status, history, WebSocket stream."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from backend.api.deps import authorize_websocket, get_redis, require_api_key
from backend.api.schemas import AgentStatusResponse
from backend.data_engine.storage.redis_cache import RedisCache

router = APIRouter(prefix="/api/agent", tags=["agent"])


def _scalar_action(payload: dict) -> float:
    """Return the dashboard-friendly target fraction from a loop payload."""
    final_action = payload.get("final_action")
    if isinstance(final_action, int | float):
        return float(final_action)

    action = payload.get("action", 0.0)
    if isinstance(action, int | float):
        return float(action)
    if isinstance(action, list) and action:
        return float(action[0])
    return 0.0


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
            current_action=_scalar_action(d),
            uncertainty=d.get("uncertainty", 0.0),
            gate_active=bool(d.get("gate_active", d.get("vetoed", False))),
            last_update=datetime.fromisoformat(d["ts"]),
            consecutive_vetoes=d.get("consecutive_vetoes", 0),
            attention_weights=attn_list,
            has_action=True,
        )
    return AgentStatusResponse(
        current_action=0.0,
        uncertainty=0.0,
        gate_active=False,
        last_update=datetime.now(UTC),
        attention_weights=attn_list,
        has_action=False,
    )


@router.get("/history", dependencies=[Depends(require_api_key)])
async def get_history(limit: int = 100, redis: RedisCache = Depends(get_redis)):
    entries = await redis.client.lrange("agent:history", 0, limit - 1)
    return [json.loads(e) for e in entries]


@router.websocket("/stream")
async def stream_agent(ws: WebSocket, redis: RedisCache = Depends(get_redis)):
    if not await authorize_websocket(ws):
        return
    await ws.accept()
    pubsub = redis.client.pubsub()
    await pubsub.subscribe("channel:agent.action")
    try:
        # Poll with a short timeout rather than blocking in ``listen()``:
        # redis-py 8.x defaults ``socket_timeout`` to 5 s, so a blocking
        # read raises ``TimeoutError`` on an idle channel. ``get_message``
        # returns ``None`` on timeout instead, so idle never crashes the
        # stream. The same loop drives the 15 s heartbeat, which doubles as
        # disconnect detection (the send raises once the client is gone).
        last_heartbeat = asyncio.get_running_loop().time()
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg is not None and msg["type"] == "message":
                await ws.send_text(msg["data"].decode("utf-8"))
            now = asyncio.get_running_loop().time()
            if now - last_heartbeat >= 15:
                # ``payload`` is required by the AgentStreamMessage contract;
                # send an empty one so the client classifies this as a
                # heartbeat envelope rather than coercing it into an action.
                await ws.send_json(
                    {"type": "heartbeat", "ts": datetime.now(UTC).isoformat(), "payload": {}}
                )
                last_heartbeat = now
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await pubsub.unsubscribe()
        await pubsub.aclose()
