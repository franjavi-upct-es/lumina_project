# backend/backtesting/event_driven.py
"""
Event-driver backtesting engine for realistic strategy simulation
Processes market events sequentially to avoid look-ahead bias
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import PriorityQueue

import numpy as np
import pandas as pd
from loguru import logger


class EventType(Enum):
    """Types of events in the backtesting system"""

    MARKET = "market"  # New market data available
    SIGNAL = "signal"  # Trading signal generated
    ORDER = "order"  # Order to be executed
    FILL = "fill"  # Order filled
    RISK_CHECK = "risk_check"  # Risk management check


@dataclass(order=True)
class Event:
    """Base class for all events"""

    timestamp: datetime
    event_type: EventType = field(compare=False)
    priority: int = field(default=0, compare=True)
    data: dict[str, any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        # Ensure timestamp is timezone-aware
        if self.timestamp.tzinfo is None:
            from datetime import UTC

            self.timestamp = self.timestamp.replace(tzinfo=UTC)


@dataclass
class MarketEvent(Event):
    """Market data update event"""

    ticker: str = ""
    price: float = 0.0
    volume: int = 0

    def __init__(self, timestamp: datetime, ticker: str, price: float, volume: int):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.MARKET,
            priority=1,
            data={"ticker": ticker, "price": price, "volume": volume},
        )
        self.ticker = ticker
        self.price = price
        self.volume = volume


@dataclass
class SignalEvent(Event):
    """Trading signal event"""

    ticker: str = ""
    signal_type: str = ""  # 'BUY', 'SELL', 'HOLD'
    strength: float = 0.0

    def __init__(self, timestamp: datetime, ticker: str, signal_type: str, strength: float = 1.0):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.SIGNAL,
            priority=2,
            data={"ticker": ticker, "signal_type": signal_type, "strength": strength},
        )
        self.ticker = ticker
        self.signal_type = signal_type
        self.strength = strength


@dataclass
class OrderEvent(Event):
    """Order event"""

    ticker: str = ""
    order_type: str = ""  # 'MARKET', 'LIMIT', 'STOP'
    quantity: float = 0.0
    direction: str = ""  # 'BUY', 'SELL'
    limit_price: float | None = None

    def __init__(
        self,
        timestamp: datetime,
        ticker: str,
        order_type: str,
        quantity: float,
        direction: str,
        limit_price: float | None = None,
    ):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.ORDER,
            priority=3,
            data={
                "ticker": ticker,
                "order_type": order_type,
                "quantity": quantity,
                "direction": direction,
                "limit_price": limit_price,
            },
        )
        self.ticker = ticker
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.limit_price = limit_price


@dataclass
class FillEvent(Event):
    """Order fill event"""

    ticker: str = ""
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    direction: str = ""

    def __init__(
        self,
        timestamp: datetime,
        ticker: str,
        quantity: float,
        fill_price: float,
        direction: str,
        commission: float = 0.0,
        slippage: float = 0.0,
    ):
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.FILL,
            priority=4,
            data={
                "ticker": ticker,
                "quantity": quantity,
                "fill_price": fill_price,
                "commission": commission,
                "slippage": slippage,
                "direction": direction,
            },
        )
        self.ticker = ticker
        self.quantity = quantity
        self.fill_price = fill_price
        self.commission = commission
        self.slippage = slippage
        self.direction = direction


class Portfolio:
    """
    Portfolio management for event-driven backtesting
    Tracks positions, cash, and calculates returns
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.inital_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, float] = {}  # ticker -> quantity
        self.avg_costs: dict[str, float] = {}  # ticker -> avg cost

        # Performance tracking
        self.equity_curve: list[dict[str, any]] = []
        self.trades: list[dict[str, any]] = []
        self.current_prices: dict[str, float] = {}

    def update_market_value(self, ticker: str, price: float):
        """Update current market price for a ticker"""
        self.current_prices[ticker] = price

    def get_equity(self) -> float:
        """Calculate total portfolio equity"""
        positions_value = sum(
            self.positions.get(ticker, 0) * self.current_prices.get(ticker, 0)
            for ticker in self.positions
        )
        return self.cash + positions_value

    def get_position(self, ticker: str) -> float:
        """Get current position size for ticker"""
        return self.positions.get(ticker, 0.0)

    def execute_fill(self, fill: FillEvent):
        """Execute a fill and update portfolio"""
        ticker = fill.ticker
        quantity = fill.quantity
        fill_price = fill.fill_price
        direction = fill.direction

        # Calculate total cost including fees
        total_cost = quantity * fill_price + fill.commission

        if direction == "BUY":
            # Update position
            current_pos = self.positions.get(ticker, 0.0)
            current_avg = self.avg_costs.get(ticker, 0.0)

            # Calculate new average cost
            if current_pos + quantity > 0:
                new_avg = (current_pos * current_avg + total_cost) / (current_pos + quantity)
                self.avg_costs[ticker] = new_avg

            self.positions[ticker] = current_pos + quantity
            self.cash -= total_cost

        elif direction == "SELL":
            # Update position
            current_pos = self.positions.get(ticker, 0.0)

            if current_pos >= quantity:
                self.positions[ticker] = current_pos - quantity
                self.cash += quantity * fill_price - fill.commission

                # Record trade
                if ticker in self.avg_costs:
                    pnl = (fill_price - self.avg_costs[ticker]) * quantity - fill.commission
                    self.trades.append(
                        {
                            "timestamp": fill.timestamp,
                            "ticker": ticker,
                            "quantity": quantity,
                            "entry_price": self.avg_costs[ticker],
                            "exit_price": fill_price,
                            "pnl": pnl,
                            "commission": fill.commission,
                        }
                    )

                # Clean up zero positions
                if abs(self.positions[ticker]) < 1e-8:
                    del self.positions[ticker]
                    if ticker in self.avg_costs:
                        del self.avg_costs[ticker]

        else:
            logger.warning(f"Insufficient position to sell {quantity} of {ticker}")

        # Record equity
        self.equity_curve.append(
            {
                "timestamp": fill.timestamp,
                "equity": self.get_equity(),
                "cash": self.cash,
                "position_value": self.get_equity() - self.cash,
            }
        )

    def get_metrics(self) -> dict[str, float]:
        """Calculate portfolio performance metrics"""
        if len(self.equity_curve) < 2:
            return {}

        # Extract equity values
        equity_series = pd.Series([p["equity"] for p in self.equity_curve])
        returns = equity_series.pct_change().dropna()

        # Calculate metrics
        total_return = (self.get_equity() - self.initial_capital) / self.initial_capital

        # Annualized metrics
        num_days = len(equity_series)
        years = num_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Sharpe ratio (assuming 5% risk-free rate)
        risk_free_rate = 0.05
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Drawdown
        cumulative = equity_series / equity_series.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        winning_trades = [t for t in self.trades if t["pnl"] > 0]
        losing_trades = [t for t in self.trades if t["pnl"] < 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t["pnl"]) for t in losing_trades]) if losing_trades else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "final_equity": self.get_equity(),
        }


class EventDrivenBacktester:
    """
    Event-driven backtesting engine
    Processes events sequentially to simulate realistic trading
    """

    def __init__(
        self, initial_capital: float = 100000.0, commission: float = 0.001, slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission
        self.slippage_rate = slippage

        self.event_queue = PriorityQueue()
        self.portfolio = Portfolio(initial_capital)

        # Strategy callbacks
        self.strategy_func: Callable | None = None
        self.risk_manager_func: Callable | None = None

        # Data storage
        self.market_data: dict[str, pd.DataFrame] = {}
        self.current_time: datetime | None = None

    def set_strategy(self, strategy_func: Callable):
        """
        Set the strategy function

        Args:
            strategy_func: Function that takes (event, portfolio) and returns signals
        """
        self.strategy_func = strategy_func

    def set_risk_manager(self, risk_manager_func: Callable):
        """
        Set the risk management function

        Args:
            risk_manager_func: Function that validates orders
        """
        self.risk_manager_func = risk_manager_func

    def load_market_data(self, ticker: str, data: pd.DataFrame):
        """Load market data for a ticker"""
        self.market_data[ticker] = data.copy()
        logger.info(f"Loaded {len(data)} bars for {ticker}")

    def run(self) -> dict[str, any]:
        """
        Run the backtest

        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting event-driven backtest...")

        # Generate market events from data
        self._generate_market_events()

        # Process event queue
        event_count = 0
        while not self.event_queue.empty():
            event = self.event_queue.get()
            self.current_time = event.timestamp

            self._process_event(event)
            event_count += 1

            if event_count % 1000 == 0:
                logger.debug(f"Processed {event_count} events")

        logger.success(f"Backtest complete: {event_count} events processed")

        # Calculate final metrics
        metrics = self.portfolio.get_metrics()

        return {
            "metrics": metrics,
            "equity_curve": self.portfolio.equity_curve,
            "trades": self.portfolio.trades,
            "positions": self.portfolio.positions,
        }

    def _generate_market_events(self):
        """Generate market events from loaded data"""
        # Combine all tickers' data with timestamps
        all_events = []

        for ticker, data in self.market_data.items():
            for idx, row in data.iterrows():
                timestamp = pd.to_datetime(idx)
                event = MarketEvent(
                    timestamp=timestamp,
                    ticker=ticker,
                    price=float(row["close"]),
                    volume=int(row["volume"]) if "volume" in row else 0,
                )
                all_events.append(event)

        # Sort by timestamp and add to queue
        all_events.sort(key=lambda e: e.timestamp)

        for event in all_events:
            self.event_queue.put(event)

        logger.info(f"Generated {len(all_events)} market events")

    def _process_event(self, event: Event):
        """Process a single event"""
        if event.event_type == EventType.MARKET:
            self._process_market_event(event)
        elif event.event_type == EventType.SIGNAL:
            self._process_signal_event(event)
        elif event.event_type == EventType.ORDER:
            self._process_order_event(event)
        elif event.event_type == EventType.FILL:
            self._process_fill_event(event)

    def _process_market_event(self, event: MarketEvent):
        """Process market data event"""
        # Update portfolio with current prices
        self.portfolio.update_market_value(event.ticker, event.price)

        # Generate trading signals if strategy is set
        if self.strategy_func:
            signals = self.strategy_func(event, self.portfolio)

            if signals:
                for signal in signals:
                    self.event_queue.put(signal)

    def _process_signal_event(self, event: SignalEvent):
        """Process trading signal event"""
        # Convert signal to order
        if event.signal_type in ["BUY", "SELL"]:
            # Calculate position size (simple example: 10% of portfolio)
            equity = self.portfolio.get_equity()
            position_value = equity * 0.1

            current_price = self.portfolio.current_prices.get(event.ticker, 0)
            if current_price > 0:
                quantity = position_value / current_price

                # Create order
                order = OrderEvent(
                    timestamp=event.timestamp,
                    ticker=event.ticker,
                    order_type="MARKET",
                    quantity=quantity,
                    direction=event.signal_type,
                )

                self.event_queue.put(order)

    def _process_order_event(self, event: OrderEvent):
        """Process order event"""
        # Check risk management
        if self.risk_manager_func:
            if not self.risk_manager_func(event, self.portfolio):
                logger.warning(f"Order rejected by risk manager: {event.ticker}")
                return

        # Simulate order execution
        current_price = self.portfolio.current_prices.get(event.ticker, 0)

        if current_price <= 0:
            logger.warning(f"No price available for {event.ticker}")
            return

        # Calculate fill price with slippage
        if event.direction == "BUY":
            fill_price = current_price * (1 + self.slippage_rate)
        else:
            fill_price = current_price * (1 - self.slippage_rate)

        # Calculate commission
        commission = event.quantity * fill_price * self.commission_rate

        # Create fill event
        fill = FillEvent(
            timestamp=event.timestamp,
            ticker=event.ticker,
            quantity=event.quantity,
            fill_price=fill_price,
            direction=event.direction,
            commission=commission,
            slippage=abs(fill_price - current_price) * event.quantity,
        )

        self.event_queue.put(fill)

    def _process_fill_event(self, event: FillEvent):
        """Process fill event"""
        self.portfolio.execute_fill(event)


# Example usage functions


def simple_momentum_strategy(event: MarketEvent, portfolio: Portfolio) -> list[SignalEvent]:
    """
    Example momentum strategy
    Buy when price crosses above 20-day MA, sell when crosses below
    """
    signals = []

    # This is simplified - in practice, maintain state with historical prices
    # and calculate indicators properly

    return signals


def simple_risk_manager(order: OrderEvent, portfolio: Portfolio) -> bool:
    """
    Example risk manager
    Checks basic position limits and cash availability
    """
    # Check cash for buy orders
    if order.direction == "BUY":
        current_price = portfolio.current_prices.get(order.ticker, 0)
        required_cash = order.quantity * current_price * 1.01  # Include buffer

        if portfolio.cash < required_cash:
            return False

    # Check position size for sell orders
    if order.direction == "SELL":
        current_position = portfolio.get_position(order.ticker)
        if current_position < order.quantity:
            return False

    return True
