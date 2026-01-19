# backend/backtesting/transaction_costs.py
"""
Realistic transaction cost modeling for backtesting
Includes comissions, slippage, market impact, and spread costs
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class OrderType(Enum):
    """Types of orders"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class AssetClass(Enum):
    """Asset classes with different cost structures"""

    EQUITY = "equity"
    FUTURES = "futures"
    FOREX = "forex"
    CRYPTO = "crypto"
    OPTIONS = "options"


@dataclass
class TransactionCostConfig:
    """Configuration for transaction costs"""

    # Commission rates
    fixed_commission: float = 0.0  # Fixed per trade
    commission_rate: float = 0.001  # 0.1% per trade
    min_commission: float = 1.0  # Minimum commission
    max_commission: float = 100.0  # Maximum commission

    # Spread costs
    spread_bps: float = 5.0  # Bid-ask spread in basis points

    #
    fixed_slippage_bps: float = 2.0  # Fixed slippage in basis points
    price_impact_coefficient: float = 0.1  # Market impact coefficient

    # Asset class
    asset_class: AssetClass = AssetClass.EQUITY

    # Market conditions
    volatility_multiplier: float = 1.0  # Multiplier during high volatility
    liquidity_factor: float = 1.0  # Lower = less liquid


class TransactionCostModel:
    """
    Comprehensive transaction cost model

    Models various types of costs:
    1. Commission costs (broker fees)
    2. Spread costs (bid-ask spread)
    3. Slippage (execution vs expected price)
    4. Market impact (price movement from large orders)
    """

    def __init__(self, config: TransactionCostConfig | None = None):
        self.config = config or TransactionCostConfig()

    def calculate_total_cost(
        self,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        volume: float = 1000000.0,
        volatility: float = 0.01,
        **kwargs,
    ) -> dict[str, float]:
        """
        Calculate total transaction costs

        Args:
            quantity: Number of shares/units
            price: Current price
            order_type: Type of order
            volume: Daily volume
            volatility: Current volatility
            **kwargs: Additional parameters

        Returns:
            Dictionary with breakdown of costs
        """
        notional_value = quantity * price

        # 1. Commission costs
        commission = self.calculate_commission(notional_value)

        # 2. Spread costs
        spread_cost = self.calculate_spread_cost(notional_value, order_type)

        # 3. Slippage
        slippage = self.calculate_slippage(quantity, price, volume, volatility, order_type)

        # 4. Market impact
        market_impact = self.calculate_market_impact(quantity, price, volume, volatility)

        # Total costs
        total_cost = commission + spread_cost + slippage + market_impact

        return {
            "commission": commission,
            "spread_cost": spread_cost,
            "slippage": slippage,
            "market_impact": market_impact,
            "total_cost": total_cost,
            "total_cost_bps": (total_cost / notional_value) * 10000 if notional_value > 0 else 0,
            "notional_value": notional_value,
        }

    def calculate_commission(self, notional_value: float) -> float:
        """
        Calculate commission costs

        Args:
            notional_value: Notional value of trade

        Returns:
            Commission amount
        """
        # Calculate percentage-based commission
        commission = notional_value * self.config.commission_rate

        # Add fixed commission
        commission += self.config.fixed_commission

        # Apply min/max bounds
        commission = max(commission, self.config.min_commission)
        commission = min(commission, self.config.max_commission)

        return commission

    def calculate_spread_cost(
        self,
        notional_value: float,
        order_type: OrderType,
    ) -> float:
        """
        Calculate bid-ask spread cost

        Market orders py the full spread, limit orders pay half (on average)

        Args:
            notional_value: Notional value of trade
            order_type: Type of order

        Returns:
            Spread cost
        """
        # Base spread in dollars
        spread_cost = (self.config.spread_bps / 10000) * notional_value

        # Market orders pay full spread, limit orders pay half
        if order_type == OrderType.MARKET:
            multiplier = 1.0
        elif order_type == OrderType.LIMIT:
            multiplier = 0.5  # On average, limit orders cross half the spread
        else:
            multiplier = 0.75

        # Adjust for asset class
        asset_multiplier = self._get_assest_class_multiplier()

        # Adjust for liquidity
        liquidity_multiplier = 1 / self.config.liquidity_factor

        return spread_cost * multiplier * asset_multiplier * liquidity_multiplier

    def calculate_slippage(
        self, quantity: float, price: float, volume: float, volatility: float, order_type: OrderType
    ) -> float:
        """
        Calculate slippage (difference between expected and execution price)

        Slippage model:
        - Fixed component (always present)
        - Volatility component (higher volatility = more slippage)
        - Volume component (larger trades relative to volume = more slippage)

        Args:
            quantity: Number of shares/units
            price: Current price
            volume: Daily volume
            volatility: Current volatility
            order_type: Type of order

        Returns:
            Slippage cost
        """
        notional_value = quantity * price

        # Fixed slippage
        fixed_slippage = (self.config.fixed_slippage_bps / 10000) * notional_value

        # Volatility-adjusted slippage
        volatility_slippage = volatility * notional_value * self.config.volatility_multiplier

        # Volume-adjusted slippage (participation rate)
        participation_rate = quantity / volume if volume > 0 else 0.01
        volume_slippage = participation_rate * 0.01 * notional_value

        # Market orders have more slippage than limit orders
        if order_type == OrderType.MARKET:
            order_multiplier = 1.5
        elif order_type == OrderType.LIMIT:
            order_multiplier = 0.5
        else:
            order_multiplier = 1.0

        total_slippage = (fixed_slippage + volatility_slippage + volume_slippage) * order_multiplier

        return total_slippage

    def calculate_market_impact(
        self, quantity: float, price: float, volume: float, volatility: float
    ) -> float:
        """
        Calculate market impact (permanent price impact from trade)

        Uses square root model: impact ∝ √(quantity/volume) * volatility

        Args:
            quantity: Number of shares/units
            price: Current price
            volume: Daily volume
            volatility: Current volatility

        Returns:
            Market impact * price
        """
        notional_value = quantity * price

        # Participation rate
        participation_rate = quantity / volume if volume > 0 else 0.01

        # Square root impact model
        impact_factor = np.sqrt(participation_rate) * volatility

        # Scale by coefficient
        market_impact = impact_factor * self.config.price_impact_coefficient * notional_value

        # Adjust for liquidity
        market_impact *= 1 / self.config.liquidity_factor

        return market_impact

    def _get_assest_class_multiplier(self) -> float:
        """Get cost multiplier based on asset class"""
        multipliers = {
            AssetClass.EQUITY: 1.0,
            AssetClass.FUTURES: 0.5,
            AssetClass.FOREX: 0.3,
            AssetClass.CRYPTO: 2.0,
            AssetClass.OPTIONS: 1.5,
        }
        return multipliers.get(self.config.asset_class, 1.0)

    def simulate_execution_price(
        self,
        target_price: float,
        quantity: float,
        volume: float,
        volatility: float,
        order_type: OrderType = OrderType.MARKET,
        is_buy: bool = True,
    ) -> float:
        """
        Simulate actual execution price including all costs

        Args:
            target_price: Target/expected price
            quantity: Number of shares/units
            volume: Daily volume
            volatility: Current volatility
            order_type: Type of order
            is_buy: True for buy orders, False for sell orders

        Returns:
            Simulated execution price
        """
        # Calculate total costs
        costs = self.calculate_total_cost(
            quantity=quantity,
            price=target_price,
            order_type=order_type,
            volume=volume,
            volatility=volatility,
        )

        # Total cost per share
        cost_per_share = costs["total_cost"] / quantity if quantity > 0 else 0

        # Add/subtract costs on buy/sell
        if is_buy:
            execution_price = target_price + cost_per_share
        else:
            execution_price = target_price - cost_per_share

        # Add random component (microstructure noise)
        noise = np.random.normal(0, volatility * target_price * 0.1)
        execution_price += noise

        return execution_price


class TieredCommissionModel(TransactionCostModel):
    """
    Commission model with volume-based tiers
    """

    def __init__(self, tiers: dict[float, float] | None = None, **kwargs):
        """
        Args:
            tiers: Dictionary mapping volume thresholds to commission rates
                   e.g., {0: 0.005, 100000: 0.003, 1000000: 0.001}
        """
        super().__init__(**kwargs)

        self.tiers = tiers or {
            0: 0.005,  # 0.5% for < $100k
            100000: 0.003,  # 0.3% for $100k-$1M
            1000000: 0.001,  # 0.1% for > $1M
        }

    def calculate_commission(self, notional_value: float) -> float:
        """Calculate commission using tiered structure"""
        # Find applicable tier
        applicable_rate = 0
        for threshold, rate in sorted(self.tiers.items()):
            if notional_value >= threshold:
                applicable_rate = rate
            else:
                break

        commission = notional_value * applicable_rate

        # Apply min/max bounds
        commission = max(commission, self.config.min_commission)
        commission = min(commission, self.config.max_commission)

        return commission


class AlmgrenChrissCostModel(TransactionCostModel):
    """
    Almgren-Chriss optimal execution cost model
    More sophisticated model for large institutional trades
    """

    def __init__(
        self,
        permanent_impact_coeff: float = 0.1,
        temporary_impact_coeff: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eta = permanent_impact_coeff  # Permanent impact
        self.gamma = temporary_impact_coeff  # Temporary impact

    def calculate_market_impact(
        self, quantity: float, price: float, volume: float, volatility: float
    ) -> float:
        """
        Calculate impact using Almgren-Chriss model

        I = n * v + γ * σ * sqrt(v/V)

        where:
        - v = trade size
        - V = daily volume
        - σ = volatility
        """
        notional_value = quantity * price
        participation_rate = quantity / volume if volume > 0 else 0.01

        # Permanent impact (linear in trade size)
        permanent = self.eta * notional_value * participation_rate

        # Temporary impact (square root model)
        temporary = self.gamma * volatility * notional_value * np.sqrt(participation_rate)

        return permanent + temporary


def create_cost_model(model_type: str = "standard", **kwargs) -> TransactionCostModel:
    """
    Factory function to create different cost models

    Args:
        model_type: Type of model ('standard', 'tiered', 'almgren_chriss')
        **kwargs: Additional parameters for the model

    Returns:
        TransactionCostModel instance
    """
    if model_type == "tiered":
        return TieredCommissionModel(**kwargs)
    elif model_type == "almgren_chriss":
        return AlmgrenChrissCostModel(**kwargs)
    else:
        return TransactionCostModel(**kwargs)


# Present configurations for different scenarios

RETAIL_EQUITY_CONFIG = TransactionCostConfig(
    fixed_commission=0.0,
    commission_rate=0.0,  # Zero commission (Robinhood style)
    spread_bps=10.0,
    fixed_slippage_bps=5.0,
    asset_class=AssetClass.EQUITY,
    liquidity_factor=1.0,
)

INSTITUTIONAL_EQUITY_CONFIG = TransactionCostConfig(
    fixed_commission=0.0,
    commission_rate=0.0005,  # 0.05% (0.5 bps)
    min_commission=10.0,
    spread_bps=2.0,
    fixed_slippage_bps=1.0,
    price_impact_coefficient=0.05,
    asset_class=AssetClass.EQUITY,
    liquidity_factor=1.5,
)

FUTURES_CONFIG = TransactionCostConfig(
    fixed_commission=2.0,
    commission_rate=0.0,
    spread_bps=1.0,
    fixed_slippage_bps=0.5,
    asset_class=AssetClass.FUTURES,
    liquidity_factor=2.0,
)

CRYPTO_CONFIG = TransactionCostConfig(
    fixed_commission=0.0,
    commission_rate=0.002,  # 0.2% (20 bps)
    spread_bps=20.0,
    fixed_slippage_bps=10.0,
    price_impact_coefficient=0.2,
    asset_class=AssetClass.CRYPTO,
    liquidity_factor=0.5,
    volatility_multiplier=2.0,
)
