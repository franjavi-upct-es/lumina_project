# backend/execution/broker/alpaca_adapter.py
"""Alpaca broker adapter."""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger

from backend.config.settings import get_settings
from backend.execution.broker.base import (
    AccountSnapshot,
    BaseBroker,
    Order,
    Position,
)


class AlpacaBroker(BaseBroker):
    def __init__(self):
        # Lazy import to avoid hard dep during tests.
        from alpaca.trading.client import TradingClient

        s = get_settings()
        self.client = TradingClient(s.ALPACA_API_KEY, s.ALPACA_SECRET_KEY, paper=s.ALPACA_PAPER)
        logger.info(f"AlpacaBroker initialized (paper={s.ALPACA_PAPER})")

    async def submit_order(self, ticker, qty, side, client_order_id):
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=ticker,
            qty=abs(qty),
            side=AlpacaSide.BUY if side == "buy" else AlpacaSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        self.client.submit_order(req)
        return Order(
            ticker=ticker,
            side=side,
            qty=qty,
            client_order_id=client_order_id,
            status="pending",
            submitted_at=datetime.now(UTC),
        )

    async def cancel_order(self, client_order_id: str) -> bool:
        try:
            self.client.cancel_order_by_client_id(client_order_id)
            return True
        except Exception as exc:
            logger.error(f"Cancel failed: {exc}")
            return False

    async def get_account(self) -> AccountSnapshot:
        acct = self.client.get_account()
        positions = await self.get_positions()
        return AccountSnapshot(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            positions=positions,
        )

    async def get_positions(self) -> dict[str, Position]:
        out = {}
        for p in self.client.get_all_positions():
            out[p.symbol] = Position(
                ticker=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                unrealized_pnl=float(p.unrealized_pl),
                market_value=float(p.market_value),
            )
        return out

    async def liquidate_all(self) -> list[Order]:
        self.client.close_all_positions(cancel_orders=True)
        logger.warning("Alpaca: liquidate_all executed")
        return []

    async def health_check(self) -> dict:
        try:
            acct = self.client.get_account()
            return {"connected": True, "account_status": str(acct.status)}
        except Exception as exc:
            return {"connected": False, "error": str(exc)}
