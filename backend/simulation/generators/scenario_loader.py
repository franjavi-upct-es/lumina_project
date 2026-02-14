# backend/simulation/generators/scenario_loader.py
"""
Historical Scenario Loader

Loads and replays historical market crashes and anomalies:
- 1987 Black Monday
- 2008 Financial Crisis
- 2010 Flash Crash
- 2020 COVID-19 Crash
- 2022 Crypto Winter

Used for:
- Testing agent robustness on known extreme events
- Curriculum learning progression
- Benchmark comparison

Each scenario includes metadata about the event and can be scaled
or combined with synthetic data for training.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class HistoricalCrash(Enum):
    """Historical market crash scenarios."""

    BLACK_MONDAY_1987 = "black_monday_1987"
    DOTCOM_CRASH_2000 = "dotcom_crash_2000"
    FINANCIAL_CRISIS_2008 = "financial_crisis_2008"
    FLASH_CRASH_2010 = "flash_crash_2010"
    COVID_CRASH_2020 = "covid_crash_2020"
    VOLATILITY_SPIKE_2022 = "volatility_spike_2022"


@dataclass
class ScenarioMetadata:
    """
    Metadata for historical scenario.

    Attributes:
        name: Scenario name
        date: Event date
        max_drawdown: Maximum drawdown during event
        duration_days: Duration in trading days
        peak_volatility: Peak VIX or volatility measure
        description: Event description
    """

    name: str
    date: datetime
    max_drawdown: float
    duration_days: int
    peak_volatility: float
    description: str


class ScenarioLoader:
    """
    Loads historical crash scenarios for training and testing.

    Scenarios can be:
    1. Replayed exactly as they occurred
    2. Scaled in severity
    3. Injected into synthetic data
    4. Combined with adversarial perturbations

    Example:
        >>> loader = ScenarioLoader()
        >>> crash_data = loader.load_scenario(HistoricalCrash.FLASH_CRASH_2010)
        >>> metadata = loader.get_metadata(HistoricalCrash.FLASH_CRASH_2010)
    """

    def __init__(self):
        """Initialize scenario loader."""
        self._scenarios = self._create_scenario_database()
        logger.info(f"ScenarioLoader initialized with {len(self._scenarios)} scenarios")

    def _create_scenario_database(self) -> dict[HistoricalCrash, ScenarioMetadata]:
        """Create database of historical scenarios with metadata."""
        return {
            HistoricalCrash.BLACK_MONDAY_1987: ScenarioMetadata(
                name="Black Monday",
                date=datetime(1987, 10, 19),
                max_drawdown=0.2276,  # -22.76% in one day
                duration_days=1,
                peak_volatility=150.0,  # Estimated VIX equivalent
                description="Largest single-day percentage decline in DJIA history",
            ),
            HistoricalCrash.DOTCOM_CRASH_2000: ScenarioMetadata(
                name="Dot-com Bubble Burst",
                date=datetime(2000, 3, 10),
                max_drawdown=0.7800,  # -78% NASDAQ peak-to-trough
                duration_days=365,
                peak_volatility=45.0,
                description="Technology bubble collapse, NASDAQ fell 78%",
            ),
            HistoricalCrash.FINANCIAL_CRISIS_2008: ScenarioMetadata(
                name="Financial Crisis",
                date=datetime(2008, 9, 15),  # Lehman collapse
                max_drawdown=0.5700,  # -57% S&P 500
                duration_days=252,
                peak_volatility=89.53,  # Peak VIX
                description="Global financial crisis, housing bubble collapse",
            ),
            HistoricalCrash.FLASH_CRASH_2010: ScenarioMetadata(
                name="Flash Crash",
                date=datetime(2010, 5, 6),
                max_drawdown=0.0987,  # -9.87% intraday
                duration_days=1,
                peak_volatility=48.0,
                description="Algorithmic trading-induced rapid crash and recovery",
            ),
            HistoricalCrash.COVID_CRASH_2020: ScenarioMetadata(
                name="COVID-19 Crash",
                date=datetime(2020, 2, 20),
                max_drawdown=0.3400,  # -34% S&P 500
                duration_days=23,  # Fastest bear market ever
                peak_volatility=82.69,  # Peak VIX
                description="Pandemic-induced market crash, fastest bear market",
            ),
            HistoricalCrash.VOLATILITY_SPIKE_2022: ScenarioMetadata(
                name="2022 Market Volatility",
                date=datetime(2022, 1, 3),
                max_drawdown=0.2550,  # -25.5% NASDAQ
                duration_days=180,
                peak_volatility=37.0,
                description="Fed rate hikes, tech sell-off, inflation concerns",
            ),
        }

    def get_metadata(self, crash: HistoricalCrash) -> ScenarioMetadata:
        """
        Get metadata for scenario.

        Args:
            crash: Historical crash identifier

        Returns:
            Scenario metadata
        """
        return self._scenarios[crash]

    def list_scenarios(self) -> list[str]:
        """List available scenarios."""
        return [crash.value for crash in HistoricalCrash]

    def load_scenario(
        self, crash: HistoricalCrash, scale_factor: float = 1.0, num_steps: int = 252
    ) -> pd.DataFrame:
        """
        Load historical crash scenario.

        Note: This generates synthetic data matching the crash characteristics.
        For real historical data, integrate with yfinance or other data sources.

        Args:
            crash: Historical crash identifier
            scale_factor: Scale severity (1.0 = historical, 2.0 = 2x worse)
            num_steps: Number of time steps

        Returns:
            OHLCV DataFrame representing the crash
        """
        metadata = self._scenarios[crash]

        # Generate crash pattern
        if crash == HistoricalCrash.BLACK_MONDAY_1987:
            data = self._generate_single_day_crash(metadata.max_drawdown * scale_factor, num_steps)

        elif crash == HistoricalCrash.FLASH_CRASH_2010:
            data = self._generate_flash_crash(metadata.max_drawdown * scale_factor, num_steps)

        elif crash in [HistoricalCrash.FINANCIAL_CRISIS_2008, HistoricalCrash.COVID_CRASH_2020]:
            data = self._generate_extended_crash(
                metadata.max_drawdown * scale_factor, metadata.duration_days, num_steps
            )

        else:
            # Default: gradual decline
            data = self._generate_gradual_decline(metadata.max_drawdown * scale_factor, num_steps)

        # Add volatility
        data = self._add_volatility(data, metadata.peak_volatility * scale_factor)

        logger.info(
            f"Loaded {metadata.name} scenario: "
            f"drawdown={metadata.max_drawdown:.2%}, "
            f"scaled={scale_factor}"
        )

        return data

    def _generate_single_day_crash(self, crash_magnitude: float, num_steps: int) -> pd.DataFrame:
        """Generate single-day crash (Black Monday style)."""
        initial_price = 100.0
        prices = [initial_price]

        # Steady before crash
        pre_crash = int(num_steps * 0.4)
        for _ in range(pre_crash):
            prices.append(prices[-1] * (1 + np.random.normal(0.0005, 0.01)))

        # Crash day
        crash_price = prices[-1] * (1 - crash_magnitude)
        prices.append(crash_price)

        # Recovery
        recovery_steps = num_steps - len(prices)
        recovery_rate = (prices[-1] * 1.1 - crash_price) / recovery_steps

        for i in range(recovery_steps):
            prices.append(crash_price + recovery_rate * (i + 1))

        return self._prices_to_ohlcv(np.array(prices[:num_steps]))

    def _generate_flash_crash(self, crash_magnitude: float, num_steps: int) -> pd.DataFrame:
        """Generate flash crash (rapid crash and recovery)."""
        initial_price = 100.0
        prices = [initial_price]

        crash_start = int(num_steps * 0.5)
        crash_duration = 20  # 20 steps for crash
        recovery_duration = 40  # 40 steps to recover

        # Normal trading
        for _ in range(crash_start):
            prices.append(prices[-1] * (1 + np.random.normal(0.0, 0.01)))

        # Crash phase
        crash_bottom = prices[-1] * (1 - crash_magnitude)
        for i in range(crash_duration):
            progress = i / crash_duration
            price = prices[crash_start] * (1 - crash_magnitude * progress)
            prices.append(price)

        # Recovery phase
        for i in range(recovery_duration):
            progress = i / recovery_duration
            price = crash_bottom + (prices[crash_start] - crash_bottom) * progress
            prices.append(price)

        # Continue normal
        while len(prices) < num_steps:
            prices.append(prices[-1] * (1 + np.random.normal(0.0, 0.01)))

        return self._prices_to_ohlcv(np.array(prices[:num_steps]))

    def _generate_extended_crash(
        self, max_drawdown: float, duration_days: int, num_steps: int
    ) -> pd.DataFrame:
        """Generate extended crisis (2008/2020 style)."""
        initial_price = 100.0
        prices = [initial_price]

        crash_steps = min(duration_days, int(num_steps * 0.6))

        # Pre-crash buildup
        buildup = int(num_steps * 0.2)
        for _ in range(buildup):
            prices.append(prices[-1] * (1 + np.random.normal(0.0003, 0.008)))

        # Crash phase (exponential decay)
        crash_rate = -np.log(1 - max_drawdown) / crash_steps
        for i in range(crash_steps):
            decline = np.exp(-crash_rate * (i + 1))
            volatility = 0.03 * (1 + i / crash_steps)  # Increasing volatility
            noise = np.random.normal(0, volatility)
            prices.append(initial_price * (1 - max_drawdown * (1 - decline)) * (1 + noise))

        # Recovery phase
        bottom = prices[-1]
        while len(prices) < num_steps:
            # Gradual recovery with volatility
            recovery_factor = (len(prices) - buildup - crash_steps) / (
                num_steps - buildup - crash_steps
            )
            target = bottom * (1 + recovery_factor * 0.5)  # 50% recovery
            prices.append(prices[-1] * 0.95 + target * 0.05 + np.random.normal(0, 0.02))

        return self._prices_to_ohlcv(np.array(prices[:num_steps]))

    def _generate_gradual_decline(self, max_drawdown: float, num_steps: int) -> pd.DataFrame:
        """Generate gradual market decline."""
        initial_price = 100.0
        decline_rate = max_drawdown / num_steps

        prices = []
        for i in range(num_steps):
            price = initial_price * (1 - decline_rate * i) * (1 + np.random.normal(0, 0.015))
            prices.append(price)

        return self._prices_to_ohlcv(np.array(prices))

    def _add_volatility(self, data: pd.DataFrame, peak_volatility: float) -> pd.DataFrame:
        """Add volatility to price data."""
        # Scale volatility relative to normal (VIX ~20)
        vol_multiplier = peak_volatility / 20.0

        for i in range(len(data)):
            close = data.loc[data.index[i], "close"]

            # Widen high-low range
            current_range = data.loc[data.index[i], "high"] - data.loc[data.index[i], "low"]
            new_range = current_range * vol_multiplier

            data.loc[data.index[i], "high"] = close + new_range / 2
            data.loc[data.index[i], "low"] = close - new_range / 2

            # Increase volume with volatility
            data.loc[data.index[i], "volume"] *= vol_multiplier

        return data

    def _prices_to_ohlcv(self, prices: np.ndarray) -> pd.DataFrame:
        """Convert price array to OHLCV DataFrame."""
        ohlcv = []

        for i, close in enumerate(prices):
            if i == 0:
                open_price = close
            else:
                open_price = prices[i - 1]

            # Simple OHLC
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))

            volume = np.random.lognormal(15, 0.5) * 1_000_000

            ohlcv.append(
                {"open": open_price, "high": high, "low": low, "close": close, "volume": volume}
            )

        df = pd.DataFrame(ohlcv)
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

        return df

    def combine_scenarios(
        self, scenarios: list[HistoricalCrash], spacing: int = 100
    ) -> pd.DataFrame:
        """
        Combine multiple historical scenarios with spacing.

        Args:
            scenarios: List of crashes to combine
            spacing: Steps between scenarios

        Returns:
            Combined scenario data
        """
        combined = []

        for crash in scenarios:
            metadata = self._scenarios[crash]
            scenario_data = self.load_scenario(crash, num_steps=metadata.duration_days)
            combined.append(scenario_data)

            # Add spacing (normal market)
            if crash != scenarios[-1]:
                spacing_data = self._generate_normal_market(spacing)
                combined.append(spacing_data)

        # Concatenate all
        result = pd.concat(combined, ignore_index=True)
        result.index = pd.date_range(start="2020-01-01", periods=len(result), freq="D")

        logger.info(f"Combined {len(scenarios)} scenarios, total steps: {len(result)}")

        return result

    def _generate_normal_market(self, num_steps: int) -> pd.DataFrame:
        """Generate normal market conditions."""
        initial_price = 100.0
        prices = [initial_price]

        for _ in range(num_steps - 1):
            ret = np.random.normal(0.0005, 0.01)  # ~0.05% mean, 1% daily vol
            prices.append(prices[-1] * (1 + ret))

        return self._prices_to_ohlcv(np.array(prices))
