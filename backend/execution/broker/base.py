# backend/execution/broker/base.py
"""Abstract broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

OrderSide = Literal["buy", "sell"]
OrderStatus = Literal["pending", "filled", "partial", "cancelled", "rejected"]


@dataclass
class Order:
    ticker: str
    side: OrderSide
    qty: float
    client_order_id: str
    status: OrderStatus = "pending"
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None


@dataclass
class Position:
    ticker: str
    qty: float
    avg_entry_price: float
    unrealized_pnl: float
    market_value: float


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    positions: dict[str, Position]


class BaseBroker(ABC):
    @abstractmethod
    async def submit_order(
        self, ticker: str, qty: float, side: OrderSide, client_order_id: str
    ) -> Order: ...
    @abstractmethod
    async def cancel_order(self, client_order_id: str) -> bool: ...
    @abstractmethod
    async def get_account(self) -> AccountSnapshot: ...
    @abstractmethod
    async def get_positions(self) -> dict[str, Position]: ...
    @abstractmethod
    async def liquidate_all(self) -> list[Order]: ...
    @abstractmethod
    async def health_check(self) -> dict: ...
