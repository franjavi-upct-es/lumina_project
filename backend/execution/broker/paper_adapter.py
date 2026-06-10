# backend/execution/broker/paper_adapter.py
"""Local paper broker: no network, deterministic fills."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from loguru import logger

from backend.config.settings import get_settings
from backend.execution.broker.base import (
    AccountSnapshot,
    BaseBroker,
    Order,
    OrderSide,
    Position,
)


class PaperBroker(BaseBroker):
    def __init__(self, slippage_bps: float = 2.0, commission_bps: float = 0.0):
        s = get_settings()
        self._cash = s.INITIAL_CAPITAL
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._last_prices: dict[str, float] = {}
        self._slippage_bps = slippage_bps
        self._commission_bps = commission_bps
        logger.info(f"PaperBroker initialized: capital=${s.INITIAL_CAPITAL:,.0f}")

    def update_price(self, ticker: str, price: float) -> None:
        self._last_prices[ticker] = price

    @staticmethod
    def _position(
        ticker: str,
        qty: float,
        avg_entry_price: float,
        market_price: float,
    ) -> Position:
        return Position(
            ticker=ticker,
            qty=qty,
            avg_entry_price=avg_entry_price,
            unrealized_pnl=(market_price - avg_entry_price) * qty,
            market_value=qty * market_price,
        )

    async def submit_order(self, ticker, qty, side, client_order_id):
        price = self._last_prices.get(ticker)
        if price is None:
            return Order(
                ticker=ticker,
                side=side,
                qty=qty,
                client_order_id=client_order_id,
                status="rejected",
            )
        slippage = price * (self._slippage_bps / 1e4) * (1 if side == "buy" else -1)
        fill_price = price + slippage
        commission = abs(qty) * fill_price * (self._commission_bps / 1e4)
        if side == "buy":
            cost = qty * fill_price + commission
            if cost > self._cash:
                return Order(
                    ticker=ticker,
                    side=side,
                    qty=qty,
                    client_order_id=client_order_id,
                    status="rejected",
                )
            self._cash -= cost
            pos = self._positions.get(ticker)
            if pos and pos.qty >= 0:
                new_qty = pos.qty + qty
                new_avg = (pos.qty * pos.avg_entry_price + qty * fill_price) / new_qty
                self._positions[ticker] = self._position(ticker, new_qty, new_avg, fill_price)
            elif pos and pos.qty < 0:
                new_qty = pos.qty + qty
                if abs(new_qty) < 1e-6:
                    del self._positions[ticker]
                elif new_qty < 0:
                    self._positions[ticker] = self._position(
                        ticker, new_qty, pos.avg_entry_price, fill_price
                    )
                else:
                    self._positions[ticker] = self._position(
                        ticker, new_qty, fill_price, fill_price
                    )
            else:
                self._positions[ticker] = self._position(ticker, qty, fill_price, fill_price)
        else:
            self._cash += qty * fill_price - commission
            pos = self._positions.get(ticker)
            if pos and pos.qty > 0:
                new_qty = pos.qty - qty
                if abs(new_qty) < 1e-6:
                    del self._positions[ticker]
                elif new_qty > 0:
                    self._positions[ticker] = self._position(
                        ticker, new_qty, pos.avg_entry_price, fill_price
                    )
                else:
                    self._positions[ticker] = self._position(
                        ticker, new_qty, fill_price, fill_price
                    )
            elif pos and pos.qty <= 0:
                short_qty = abs(pos.qty)
                new_abs_qty = short_qty + qty
                new_avg = (short_qty * pos.avg_entry_price + qty * fill_price) / new_abs_qty
                self._positions[ticker] = self._position(
                    ticker,
                    -new_abs_qty,
                    new_avg,
                    fill_price,
                )
            else:
                self._positions[ticker] = self._position(ticker, -qty, fill_price, fill_price)
        order = Order(
            ticker=ticker,
            side=side,
            qty=qty,
            client_order_id=client_order_id,
            status="filled",
            filled_qty=qty,
            avg_fill_price=fill_price,
            submitted_at=datetime.now(UTC),
            filled_at=datetime.now(UTC),
        )
        self._orders[client_order_id] = order
        return order

    async def cancel_order(self, client_order_id: str) -> bool:
        return client_order_id in self._orders

    async def get_account(self) -> AccountSnapshot:
        marked_positions = {
            ticker: self._position(
                ticker,
                pos.qty,
                pos.avg_entry_price,
                self._last_prices.get(ticker, pos.avg_entry_price),
            )
            for ticker, pos in self._positions.items()
        }
        self._positions = marked_positions
        total_position_value = sum(p.market_value for p in marked_positions.values())
        equity = self._cash + total_position_value
        return AccountSnapshot(
            equity=equity,
            cash=self._cash,
            buying_power=self._cash,
            positions=dict(marked_positions),
        )

    async def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    async def liquidate_all(self) -> list[Order]:
        orders = []
        for ticker, pos in list(self._positions.items()):
            side: OrderSide = "sell" if pos.qty > 0 else "buy"
            order = await self.submit_order(
                ticker, abs(pos.qty), side, f"liq_{uuid.uuid4().hex[:8]}"
            )
            orders.append(order)
        return orders

    async def health_check(self) -> dict:
        return {"connected": True, "cash": self._cash, "positions": len(self._positions)}
