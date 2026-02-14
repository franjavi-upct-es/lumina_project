# backend/simulation/generators/adversarial.py
"""
Adversarial Scenario Generator - The Nightmare Machine

Creates extreme market scenarios specifically designed to stress-test
and break RL agents. Part of Phase B (Domain Randomization) training.

Philosophy:
"The agent is trained in a simulation environment that generates
'Nightmare Scenarios' - artificial flash crashes, liquidity droughts,
inverse correlation breakdowns, and data outages - specifically
designed to break the agent. It learns survival first, profit second."

Adversarial Warps:
- Warp 1 (Volatility): 2x, 3x, 5x volatility multipliers
- Warp 2 (Noise): Spread widening, slippage spikes
- Warp 3 (Blackout): Data feed outages, missing candles
- Warp 4 (Correlation Break): Inverse correlations
- Warp 5 (Liquidity Drought): Extreme bid-ask spreads

The goal: "Can the agent survive a VIX of 80?"
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class ScenarioType(Enum):
    """Types of adversarial scenarios."""

    FLASH_CRASH = "flash_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_DROUGHT = "liquidity_drought"
    DATA_OUTAGE = "data_outage"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    TRENDING_REVERSAL = "trending_reversal"
    GAP_OPENING = "gap_opening"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class NightmareScenario:
    """
    Configuration for an adversarial scenario.

    Attributes:
        scenario_type: Type of nightmare scenario
        intensity: Intensity multiplier (1.0 = normal, 5.0 = extreme)
        duration: Duration in steps
        start_step: When scenario begins (None = random)
        params: Additional scenario-specific parameters
    """

    scenario_type: ScenarioType
    intensity: float = 2.0
    duration: int = 10
    start_step: int | None = None
    params: dict | None = None


class AdversarialScenarioGenerator:
    """
    Generates adversarial market scenarios for robust RL training.

    Takes clean historical or synthetic data and applies various
    "warps" to create nightmare scenarios.

    Example:
        >>> generator = AdversarialScenarioGenerator()
        >>> clean_data = load_historical_data()
        >>>
        >>> # Apply flash crash
        >>> nightmare = NightmareScenario(
        >>>     scenario_type=ScenarioType.FLASH_CRASH,
        >>>     intensity=3.0,
        >>>     duration=20
        >>> )
        >>> warped_data = generator.apply_scenario(clean_data, nightmare)
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize adversarial generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        logger.info("AdversarialScenarioGenerator initialized")

    def apply_scenario(self, data: pd.DataFrame, scenario: NightmareScenario) -> pd.DataFrame:
        """
        Apply adversarial scenario to data.

        Args:
            data: Clean OHLCV data
            scenario: Scenario configuration

        Returns:
            Warped data with scenario applied
        """
        # Copy data
        warped = data.copy()

        # Determine start step
        if scenario.start_step is None:
            max_start = len(data) - scenario.duration
            start = np.random.randint(0, max(max_start, 1))
        else:
            start = scenario.start_step

        end = min(start + scenario.duration, len(data))

        # Apply scenario-specific transformation
        if scenario.scenario_type == ScenarioType.FLASH_CRASH:
            warped = self._apply_flash_crash(warped, start, end, scenario.intensity)

        elif scenario.scenario_type == ScenarioType.VOLATILITY_SPIKE:
            warped = self._apply_volatility_spike(warped, start, end, scenario.intensity)

        elif scenario.scenario_type == ScenarioType.LIQUIDITY_DROUGHT:
            warped = self._apply_liquidity_drought(warped, start, end, scenario.intensity)

        elif scenario.scenario_type == ScenarioType.DATA_OUTAGE:
            warped = self._apply_data_outage(warped, start, end, scenario.intensity)

        elif scenario.scenario_type == ScenarioType.CORRELATION_BREAKDOWN:
            warped = self._apply_correlation_breakdown(warped, start, end, scenario.intensity)

        elif scenario.scenario_type == ScenarioType.TRENDING_REVERSAL:
            warped = self._apply_trending_reversal(warped, start, end, scenario.intensity)

        elif scenario.scenario_type == ScenarioType.GAP_OPENING:
            warped = self._apply_gap_opening(warped, start, end, scenario.intensity)

        elif scenario.scenario_type == ScenarioType.CIRCUIT_BREAKER:
            warped = self._apply_circuit_breaker(warped, start, end, scenario.intensity)

        logger.debug(
            f"Applied {scenario.scenario_type.value} scenario: "
            f"steps [{start}, {end}), intensity={scenario.intensity}"
        )

        return warped

    def _apply_flash_crash(
        self, data: pd.DataFrame, start: int, end: int, intensity: float
    ) -> pd.DataFrame:
        """
        Simulate flash crash: sudden price drop and recovery.

        Like the 2010 Flash Crash: -9% in 5 minutes, recovery in 20 minutes.
        """
        crash_magnitude = -0.10 * intensity  # -10% per intensity
        duration = end - start
        recovery_point = int(duration * 0.3)  # Recover after 30% of duration

        for i in range(start, end):
            relative_pos = (i - start) / duration

            if relative_pos < 0.3:  # Crash phase
                multiplier = 1 + crash_magnitude * (relative_pos / 0.3)
            else:  # Recovery phase
                recovery_progress = (relative_pos - 0.3) / 0.7
                multiplier = 1 + crash_magnitude * (1 - recovery_progress)

            data.loc[data.index[i], ["open", "high", "low", "close"]] *= multiplier

            # Spike volume during crash
            if relative_pos < 0.3:
                data.loc[data.index[i], "volume"] *= 3 * intensity

        return data

    def _apply_volatility_spike(
        self, data: pd.DataFrame, start: int, end: int, intensity: float
    ) -> pd.DataFrame:
        """
        Multiply volatility by intensity factor.

        Simulates VIX spike from 20 to 60+ (intensity=3.0).
        """
        for i in range(start, end):
            close = data.loc[data.index[i], "close"]

            # Add noise proportional to intensity
            noise = np.random.normal(0, 0.02 * intensity)
            new_close = close * (1 + noise)

            # Expand OHLC range
            range_multiplier = 1 + (intensity - 1) * 0.5
            current_range = data.loc[data.index[i], "high"] - data.loc[data.index[i], "low"]
            expanded_range = current_range * range_multiplier

            data.loc[data.index[i], "high"] = new_close + expanded_range / 2
            data.loc[data.index[i], "low"] = new_close - expanded_range / 2
            data.loc[data.index[i], "close"] = new_close

            # Volume increases with volatility
            data.loc[data.index[i], "volume"] *= 1 + intensity * 0.5

        return data

    def _apply_liquidity_drought(
        self, data: pd.DataFrame, start: int, end: int, intensity: float
    ) -> pd.DataFrame:
        """
        Simulate liquidity drought: wide spreads, low volume.

        Adds 'spread' column representing bid-ask spread.
        """
        if "spread" not in data.columns:
            data["spread"] = 0.001  # 10 bps default

        for i in range(start, end):
            # Widen spread
            data.loc[data.index[i], "spread"] *= intensity

            # Reduce volume
            data.loc[data.index[i], "volume"] /= intensity

            # Add price impact (slippage)
            close = data.loc[data.index[i], "close"]
            impact = close * data.loc[data.index[i], "spread"]

            # Widen high-low range due to slippage
            data.loc[data.index[i], "high"] += impact
            data.loc[data.index[i], "low"] -= impact

        return data

    def _apply_data_outage(
        self, data: pd.DataFrame, start: int, end: int, intensity: float
    ) -> pd.DataFrame:
        """
        Simulate data feed outage: missing candles.

        Intensity determines dropout probability.
        """
        dropout_prob = min(intensity * 0.2, 0.8)  # Max 80% dropout

        for i in range(start, end):
            if np.random.random() < dropout_prob:
                # Missing data - forward fill from previous
                if i > 0:
                    data.loc[data.index[i]] = data.loc[data.index[i - 1]]
                    data.loc[data.index[i], "volume"] = 0  # Mark as missing

        return data

    def _apply_correlation_breakdown(
        self, data: pd.DataFrame, start: int, end: int, intensity: float
    ) -> pd.DataFrame:
        """
        Simulate correlation breakdown.

        Only affects multi-asset environments.
        This is a marker for the environment to invert correlations.
        """
        # Add metadata column
        if "correlation_warp" not in data.columns:
            data["correlation_warp"] = 1.0

        for i in range(start, end):
            # Negative intensity inverts correlations
            data.loc[data.index[i], "correlation_warp"] = -intensity

        return data

    def _apply_trending_reversal(
        self, data: pd.DataFrame, start: int, end: int, intensity: float
    ) -> pd.DataFrame:
        """
        Simulate sudden trend reversal.

        Sharp V-shape or inverse V-shape move.
        """
        # Determine if up-trend or down-trend
        if start > 0:
            recent_returns = data.loc[data.index[max(0, start - 10) : start], "close"].pct_change()
            trend_up = recent_returns.mean() > 0
        else:
            trend_up = np.random.random() > 0.5

        # Reverse the trend
        reversal_magnitude = 0.05 * intensity * (-1 if trend_up else 1)

        for i in range(start, end):
            progress = (i - start) / (end - start)
            move = reversal_magnitude * progress

            data.loc[data.index[i], ["open", "high", "low", "close"]] *= 1 + move

        return data

    def _apply_gap_opening(
        self, data: pd.DataFrame, start: int, end: int, intensity: float
    ) -> pd.DataFrame:
        """
        Simulate gap opening (news over weekend).

        Open differs significantly from previous close.
        """
        if start > 0:
            prev_close = data.loc[data.index[start - 1], "close"]
            gap_size = 0.03 * intensity * np.random.choice([-1, 1])

            new_open = prev_close * (1 + gap_size)

            data.loc[data.index[start], "open"] = new_open

            # Adjust OHLC to maintain consistency
            if new_open > data.loc[data.index[start], "high"]:
                data.loc[data.index[start], "high"] = new_open
            if new_open < data.loc[data.index[start], "low"]:
                data.loc[data.index[start], "low"] = new_open

        return data

    def _apply_circuit_breaker(
        self, data: pd.DataFrame, start: int, end: int, intensity: float
    ) -> pd.DataFrame:
        """
        Simulate trading halt / circuit breaker.

        Price frozen for several periods.
        """
        if start > 0:
            freeze_price = data.loc[data.index[start - 1], "close"]

            for i in range(start, end):
                data.loc[data.index[i], ["open", "high", "low", "close"]] = freeze_price
                data.loc[data.index[i], "volume"] = 0

        return data

    def generate_random_scenario(
        self, max_intensity: float = 3.0, max_duration: int = 50
    ) -> NightmareScenario:
        """
        Generate random nightmare scenario.

        Args:
            max_intensity: Maximum intensity
            max_duration: Maximum duration

        Returns:
            Random scenario configuration
        """
        scenario_type = np.random.choice(list(ScenarioType))
        intensity = np.random.uniform(1.5, max_intensity)
        duration = np.random.randint(10, max_duration)

        return NightmareScenario(
            scenario_type=scenario_type, intensity=intensity, duration=duration
        )

    def generate_nightmare_episode(
        self, base_data: pd.DataFrame, num_scenarios: int = 3, max_intensity: float = 5.0
    ) -> pd.DataFrame:
        """
        Generate complete nightmare episode with multiple scenarios.

        Args:
            base_data: Clean base data
            num_scenarios: Number of scenarios to inject
            max_intensity: Maximum scenario intensity

        Returns:
            Data with multiple nightmare scenarios
        """
        warped = base_data.copy()

        # Ensure scenarios don't overlap
        total_steps = len(base_data)
        used_ranges = []

        for _ in range(num_scenarios):
            scenario = self.generate_random_scenario(max_intensity=max_intensity)

            # Find non-overlapping start position
            attempts = 0
            while attempts < 10:
                start = np.random.randint(0, total_steps - scenario.duration)
                end = start + scenario.duration

                # Check overlap
                overlap = False
                for used_start, used_end in used_ranges:
                    if not (end <= used_start or start >= used_end):
                        overlap = True
                        break

                if not overlap:
                    scenario.start_step = start
                    warped = self.apply_scenario(warped, scenario)
                    used_ranges.append((start, end))
                    break

                attempts += 1

        logger.info(f"Generated nightmare episode with {len(used_ranges)} scenarios")

        return warped


def create_domain_randomization_params() -> dict[str, tuple[float, float]]:
    """
    Create default domain randomization parameter ranges for Phase B.

    Returns:
        Dictionary mapping parameter names to (min, max) ranges
    """
    return {
        "volatility_multiplier": (1.0, 5.0),
        "spread_multiplier": (1.0, 3.0),
        "data_dropout_prob": (0.0, 0.1),
        "noise_std": (0.0, 0.02),
        "jump_intensity": (0.0, 20.0),
    }
