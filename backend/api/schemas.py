# backend/api/schemas.py
"""Pydantic models that describe the public HTTP surface of the API.

Every wire-format object the API exchanges lives here. The convention
is one class per request/response shape, named ``XxxRequest`` or
``XxxResponse``. Internal data classes used inside services live with
their service module, not here.

Why a separate schemas module instead of nested classes per route?
==================================================================
* The schemas are *contract*. Stable identifiers + a single import
  path make it trivial to share them with the TypeScript client
  (``frontend/src/types/*``) — the TypeScript interfaces mirror these
  Pydantic models field-for-field.
* OpenAPI generation (``/openapi.json``) is cleaner when models live
  outside the routers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ----- Health & monitoring --------------------------------------------------
HealthStatus = Literal["ok", "degraded", "down"]


class HealthResponse(BaseModel):
    """Top-level health response returned by ``GET /api/monitoring/health``.

    The ``components`` field is a free-form mapping of subsystem-name to
    subsystem-specific health dictionary. Subsystems known to the API:
    ``redis``, ``timescale``, ``broker``, ``kill_switch``.

    The aggregate ``status`` is derived as follows:
        * "ok"       — every component reports ``connected = True`` AND
                        the kill switch is "NORMAL".
        * "degraded" — at least one component is unreachable OR the
                        kill switch is "CLOSE_ONLY".
        * "down"     — the kill switch is "LIQUIDATE_ALL".
    """

    status: HealthStatus
    components: dict[str, dict]


# ----- Agent ----------------------------------------------------------------
class AgentStatusResponse(BaseModel):
    """Snapshot of the live agent surfaced to the dashboard."""

    current_action: float = Field(ge=-1, le=1)
    uncertainty: float = Field(ge=0, le=1)
    gate_active: bool
    last_update: datetime
    consecutive_vetoes: int = 0
    attention_weights: list[float] | None = None  # [Price, News, Graph]
    has_action: bool = False


class AgentStreamMessage(BaseModel):
    """WebSocket envelope on ``/api/agent/stream``.

    Type taxonomy:
        ``action``    — a new decision has been taken by the agent
        ``veto``      — the safety arbitrator rejected an action
        ``liquidate`` — the kill switch escalated to LIQUIDATE_ALL
        ``heartbeat`` — periodic keep-alive (every 15 s) so the
                        frontend can show "live" / "stale" status.
    """

    type: Literal["action", "veto", "liquidate", "heartbeat"]
    ts: datetime
    payload: dict


# ----- Portfolio ------------------------------------------------------------
class PositionResponse(BaseModel):
    """A single open position. Mirrors :class:`Position` from the broker."""

    ticker: str
    qty: float
    avg_entry_price: float
    unrealized_pnl: float
    market_value: float


class PortfolioResponse(BaseModel):
    """Account-level summary returned by ``GET /api/portfolio``.

    ``drawdown_pct`` is the *peak-to-trough* drawdown expressed as a
    positive fraction in [0, 1]. Computed by the API rather than the
    broker because the broker (especially the live Alpaca adapter) does
    not track peak equity natively. We persist the peak in Redis under
    ``portfolio:peak_equity`` and update it on every ``GET`` call.
    """

    equity: float
    cash: float
    buying_power: float
    positions: list[PositionResponse]
    peak_equity: float
    drawdown_pct: float = Field(ge=0, le=1)


class EquityPoint(BaseModel):
    time: datetime
    equity: float
    benchmark: float | None = None


class PortfolioHistoryResponse(BaseModel):
    history: list[EquityPoint]


# ----- Risk -----------------------------------------------------------------
class KillSwitchRequest(BaseModel):
    state: Literal["NORMAL", "CLOSE_ONLY", "LIQUIDATE_ALL"]
    reason: str = ""


class KillSwitchResponse(BaseModel):
    state: str
    set_at: datetime


# ----- Backtest -------------------------------------------------------------
class BacktestRequest(BaseModel):
    """Submit a backtest job to the worker queue."""

    start: datetime
    end: datetime
    tickers: list[str]
    initial_capital: float = 100_000.0


class BacktestResultResponse(BaseModel):
    """Async result polling. ``status`` walks pending → running → completed."""

    run_id: str
    status: Literal["pending", "running", "completed", "failed"]
    sharpe: float | None = None
    max_drawdown: float | None = None
    total_return: float | None = None
    failure_reason: str | None = None


# ----- Data -----------------------------------------------------------------
class OHLCVResponse(BaseModel):
    """A single OHLCV bar as returned by ``GET /api/data/ohlcv/{ticker}``."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
