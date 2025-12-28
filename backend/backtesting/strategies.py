# backend/backtesting/strategies.py
"""
Pre-built trading strategies for backtesting
Collection of common strategies with configurable parameters
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from loguru import logger


class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    """
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy

        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame
    ) -> List[str]:
        """
        Generate trading signals

        Args:
            data: Price data with OHLCV
            features: Engineered features

        Returns:
            List of signals: 'BUY', 'SELL', 'HOLD'
        """
        pass
    
    def get_code(self) -> str:
        """
        Get strategy as executable code for API

        Returns:
            Python code string
        """
        return f"""
def strategy(data, features):
    # {self.name}
    # Parameters: {self.params}
    signals = []
    # Implementation here
    return signals
"""


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) based strategy
    Buy when RSI < oversold, Sell when RSI > overbought
    """
    
    def __init__(
            self,
            rsi_period: int = 14,
            oversold: float = 30,
            overbought: float = 70,
            **kwargs,
    ):
        params = {
            "rsi_period": rsi_period,
            "oversold": oversold,
            "overbought": overbought,
        }
        super().__init__("RSI Strategy", params)
    
    def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame
    ) -> List[str]:
        """Generate RSI-based signals"""
        signals = []
        
        rsi_col = f"rsi_{self.params['rsi_period']}"
        
        if rsi_col not in features.columns:
            logger.warning(f"RSI column {rsi_col} not found, returning HOLD")
            return ["HOLD"] * len(data)
        
        for i in range(len(data)):
            if i < self.params["rsi_period"]:
                signals.append("HOLD")
                continue
            
            rsi = features[rsi_col].iloc[i]
            
            if pd.isna(rsi):
                signals.append("HOLD")
            elif rsi < self.params["oversold"]:
                signals.append("BUY")
            elif rsi > self.params["overbought"]:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        return signals
    
    def get_code(self) -> str:
        """Get executable code"""
        return f"""
def strategy(data, features):
    # RSI Strategy
    # Buy when RSI < {self.params['oversold']}, Sell when RSI > {self.params['overbought']}

    signals = []
    rsi_col = 'rsi_{self.params['rsi_period']}'

    for i in range(len(data)):
        if i < {self.params['rsi_period']}:
            signals.append('HOLD')
        else:
            rsi = features[rsi_col].iloc[i] if rsi_col in features.columns else None

            if rsi is None or pd.isna(rsi):
                signals.append('HOLD')
            elif rsi < {self.params['oversold']}:
                signals.append('BUY')
            elif rsi > {self.params['overbought']}:
                signals.append('SELL')
            else:
                signals.append('HOLD')

    return signals
"""


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) strategy
    Buy when MACD crosses above signal, Sell when crosses below
    """
    
    def __init__(self, **kwargs):
        super().__init__("MACD Strategy", {})
    
    def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame
    ) -> List[str]:
        """Generate MACD-based signals"""
        signals = []
        
        if "macd" not in features.columns or "macd_signal" not in features.columns:
            return ["HOLD"] * len(data)
        
        prev_macd = None
        prev_signal = None
        
        for i in range(len(data)):
            macd = features["macd"].iloc[i]
            signal = features["macd_signal"].iloc[i]
            
            if pd.isna(macd) or pd.isna(signal) or prev_macd is None:
                signals.append("HOLD")
            else:
                # Bullish crossover
                if prev_macd <= prev_signal and macd > signal:
                    signals.append("BUY")
                # Bearish crossover
                elif prev_macd >= prev_signal and macd < signal:
                    signals.append("SELL")
                else:
                    signals.append("HOLD")
            
            prev_macd = macd
            prev_signal = signal
        
        return signals
    
    def get_code(self) -> str:
        """Get executable code"""
        return """
def strategy(data, features):
    # MACD Strategy
    # Buy on bullish crossover, Sell on bearish crossover

    signals = []
    prev_macd = None
    prev_signal = None

    for i in range(len(data)):
        macd = features['macd'].iloc[i] if 'macd' in features.columns else None
        signal = features['macd_signal'].iloc[i] if 'macd_signal' in features.columns else None

        if macd is None or signal is None or pd.isna(macd) or pd.isna(signal) or prev_macd is None:
            signals.append('HOLD')
        else:
            # Bullish crossover
            if prev_macd <= prev_signal and macd > signal:
                signals.append('BUY')
            # Bearish crossover
            elif prev_macd >= prev_signal and macd < signal:
                signals.append('SELL')
            else:
                signals.append('HOLD')

        prev_macd = macd
        prev_signal = signal

    return signals
"""


class MovingAverageCrossover(BaseStrategy):
    """
    Moving Average Crossover Strategy
    Buy when fast MA crosses above slow MA, Sell when crosses below
    """
    
    def __init__(self, fast_period: int = 50, slow_period: int = 200, **kwargs):
        params = {"fast_period": fast_period, "slow_period": slow_period}
        super().__init__("MA Crossover Strategy", params)
    
    def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame
    ) -> List[str]:
        """Generate MA crossover signals"""
        signals = []
        
        fast_col = f"sma_{self.params['fast_period']}"
        slow_col = f"sma_{self.params['slow_period']}"
        
        if fast_col not in features.columns or slow_col not in features.columns:
            return ["HOLD"] * len(data)
        
        prev_fast = None
        prev_slow = None
        
        for i in range(len(data)):
            fast = features[fast_col].iloc[i]
            slow = features[slow_col].iloc[i]
            
            if pd.isna(fast) or pd.isna(slow) or prev_fast is None:
                signals.append("HOLD")
            else:
                # Golden cross
                if prev_fast <= prev_slow and fast > slow:
                    signals.append("BUY")
                # Death cross
                elif prev_fast >= prev_slow and fast < slow:
                    signals.append("SELL")
                else:
                    signals.append("HOLD")
            
            prev_fast = fast
            prev_slow = slow
        
        return signals


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy
    Buy when price touches lower band, Sell when touches upper band
    """
    
    def __init__(self, **kwargs):
        super().__init__("Bollinger Bands Strategy", {})
    
    def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame
    ) -> List[str]:
        """Generate Bollinger Bands signals"""
        signals = []
        
        if (
                "bb_upper" not in features.columns
                or "bb_lower" not in features.columns
                or "close" not in data.columns
        ):
            return ["HOLD"] * len(data)
        
        for i in range(len(data)):
            close = data["close"].iloc[i]
            bb_upper = features["bb_upper"].iloc[i]
            bb_lower = features["bb_lower"].iloc[i]
            
            if pd.isna(bb_upper) or pd.isna(bb_lower):
                signals.append("HOLD")
            elif close <= bb_lower:
                signals.append("BUY")
            elif close >= bb_upper:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy
    Buy when price is far below mean, Sell when far above
    """
    
    def __init__(
            self, lookback: int = 20, std_threshold: float = 2.0, **kwargs
    ):
        params = {"lookback": lookback, "std_threshold": std_threshold}
        super().__init__("Mean Reversion Strategy", params)
    
    def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame
    ) -> List[str]:
        """Generate mean reversion signals"""
        signals = []
        
        if "close" not in data.columns:
            return ["HOLD"] * len(data)
        
        closes = data["close"].values
        lookback = self.params["lookback"]
        
        for i in range(len(data)):
            if i < lookback:
                signals.append("HOLD")
                continue
            
            recent = closes[i - lookback: i]
            mean = np.mean(recent)
            std = np.std(recent)
            current = closes[i]
            
            z_score = (current - mean) / std if std > 0 else 0
            
            if z_score < -self.params["std_threshold"]:
                signals.append("BUY")
            elif z_score > self.params["std_threshold"]:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        return signals


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy
    Buy when momentum is positive, Sell when negative
    """
    
    def __init__(self, lookback: int = 20, threshold: float = 0.02, **kwargs):
        params = {"lookback": lookback, "threshold": threshold}
        super().__init__("Momentum Strategy", params)
    
    def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame
    ) -> List[str]:
        """Generate momentum signals"""
        signals = []
        
        if "close" not in data.columns:
            return ["HOLD"] * len(data)
        
        closes = data["close"].values
        lookback = self.params["lookback"]
        
        for i in range(len(data)):
            if i < lookback:
                signals.append("HOLD")
                continue
            
            momentum = (closes[i] - closes[i - lookback]) / closes[i - lookback]
            
            if momentum > self.params["threshold"]:
                signals.append("BUY")
            elif momentum < -self.params["threshold"]:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        return signals


class ComboStrategy(BaseStrategy):
    """
    Combination Strategy
    Combines multiple indicators with voting
    """
    
    def __init__(
            self,
            rsi_oversold: float = 30,
            rsi_overbought: float = 70,
            min_votes: int = 2,
            **kwargs,
    ):
        params = {
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "min_votes": min_votes,
        }
        super().__init__("Combo Strategy", params)
    
    def generate_signals(
            self, data: pd.DataFrame, features: pd.DataFrame
    ) -> List[str]:
        """Generate signals using multiple indicators"""
        signals = []
        
        for i in range(len(data)):
            if i < 20:
                signals.append("HOLD")
                continue
            
            buy_votes = 0
            sell_votes = 0
            
            # RSI vote
            if "rsi_14" in features.columns:
                rsi = features["rsi_14"].iloc[i]
                if not pd.isna(rsi):
                    if rsi < self.params["rsi_oversold"]:
                        buy_votes += 1
                    elif rsi > self.params["rsi_overbought"]:
                        sell_votes += 1
            
            # MACD vote
            if "macd" in features.columns and "macd_signal" in features.columns:
                macd = features["macd"].iloc[i]
                signal = features["macd_signal"].iloc[i]
                if not pd.isna(macd) and not pd.isna(signal):
                    if macd > signal:
                        buy_votes += 1
                    else:
                        sell_votes += 1
            
            # Bollinger Bands vote
            if (
                    "bb_upper" in features.columns
                    and "bb_lower" in features.columns
                    and "close" in data.columns
            ):
                close = data["close"].iloc[i]
                bb_upper = features["bb_upper"].iloc[i]
                bb_lower = features["bb_lower"].iloc[i]
                
                if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    if close <= bb_lower:
                        buy_votes += 1
                    elif close >= bb_upper:
                        sell_votes += 1
            
            # Decision based on votes
            if buy_votes >= self.params["min_votes"]:
                signals.append("BUY")
            elif sell_votes >= self.params["min_votes"]:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        return signals


# Strategy registry
STRATEGIES = {
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "ma_crossover": MovingAverageCrossover,
    "bollinger_bands": BollingerBandsStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "combo": ComboStrategy,
}


def get_strategy(strategy_name: str, **params) -> BaseStrategy:
    """
    Factory function to create strategy instances

    Args:
        strategy_name: Name of strategy
        **params: Strategy parameters

    Returns:
        Strategy instance
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}"
        )
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(**params)


def list_strategies() -> Dict[str, str]:
    """
    List all available strategies

    Returns:
        Dictionary of strategy names and descriptions
    """
    return {
        "rsi": "RSI-based oversold/overbought strategy",
        "macd": "MACD crossover strategy",
        "ma_crossover": "Moving average crossover strategy",
        "bollinger_bands": "Bollinger Bands mean reversion",
        "mean_reversion": "Statistical mean reversion strategy",
        "momentum": "Price momentum strategy",
        "combo": "Multi-indicator combination strategy",
    }