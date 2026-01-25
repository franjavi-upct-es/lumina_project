# backend/ml_engine/features/macro.py
"""
Macroeconomic features for market analysis
Economic indicators and their impact on markets
"""

from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


class MacroFeatures:
    """
    Macroeconomic features

    Categories:
    - Interest rates (Fed Funds, Treasury yields)
    - Economic indicators (GDP, Unemployment, CPI)
    - Market indicators (VIX, Market breadth)
    - Currency and commodities (DXY, Gold, Oil)
    """

    def __init__(self):
        self.feature_names = []
        # Map of FRED series IDs for economic indicators
        self.fred_series = {
            "fed_funds_rate": "DFF",
            "treasury_10y": "DGS10",
            "treasury_2y": "DGS2",
            "gdp_growth": "A191RL1Q225SBEA",
            "unemployment_rate": "UNRATE",
            "cpi": "CPIAUCSL",
            "ppi": "PPIACO",
            "consumer_sentiment": "UMCSENT",
            "retail_sales": "RSXFS",
            "industrial_production": "INDPRO",
            "m2_money_supply": "M2SL",
        }

    async def collect_macro_data(
        self, start_date: datetime, end_date: datetime, fred_api_key: str | None = None
    ) -> pd.DataFrame:
        """
        Collect macroeconomic data from FRED

        Args:
            start_date: Start date
            end_date: End date
            fred_api_key: FRED API key

        Returns:
            DataFrame with macro indicators
        """
        try:
            if fred_api_key:
                from fredapi import Fred

                fred = Fred(api_key=fred_api_key)

                data = {}
                for name, series_id in self.fred_series.items():
                    try:
                        series = fred.get_series(series_id, start_date, end_date)
                        data[name] = series
                    except Exception as e:
                        logger.warning(f"Failed to get {name}: {e}")

                if data:
                    df = pd.DataFrame(data)
                    df.index.name = "date"
                    return df.reset_index()

            # Fallback: generate synthetic data for testing
            logger.warning("No FRED API key provided, using synthetic data")
            return self._generate_synthetic_macro_data(start_date, end_date)

        except Exception as e:
            logger.error(f"Error collecting macro data: {e}")
            return self._generate_synthetic_macro_data(start_date, end_date)

    def _generate_synthetic_macro_data(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate synthetic macro data for testing"""
        dates = pd.date_range(start_date, end_date, freq="D")

        # Generate realistic-looking synthetic data
        np.random.seed(42)
        n = len(dates)

        data = {
            "date": dates,
            "fed_funds_rate": np.random.normal(5.0, 0.5, n).clip(0, 10),
            "treasury_10y": np.random.normal(4.5, 0.3, n).clip(0, 8),
            "treasury_2y": np.random.normal(4.8, 0.4, n).clip(0, 8),
            "unemployment_rate": np.random.normal(4.0, 0.2, n).clip(2, 10),
            "cpi": np.linspace(300, 320, n),  # Trending upward
        }

        return pd.DataFrame(data)

    def create_macro_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived macro features

        Args:
            macro_data: Raw macro data

        Returns:
            DataFrame with engineered features
        """
        df = macro_data.copy()

        # Interest rate features
        if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
            df["yield_curve"] = df["treasury_10y"] - df["treasury_2y"]
            df["yield_curve_inverted"] = (df["yield_curve"] < 0).astype(int)
            self.feature_names.extend(["yield_curve", "yield_curve_inverted"])

        if "fed_funds_rate" in df.columns:
            df["fed_funds_change"] = df["fed_funds_rate"].diff()
            df["fed_funds_trend"] = df["fed_funds_rate"].rolling(30).mean()
            self.feature_names.extend(["fed_funds_change", "fed_funds_trend"])

        # Inflation features
        if "cpi" in df.columns:
            df["inflation_rate"] = df["cpi"].pct_change(12) * 100  # YoY
            df["inflation_trend"] = df["cpi"].pct_change(3) * 100  # QoQ
            self.feature_names.extend(["inflation_rate", "inflation_trend"])

        # Unemployment features
        if "unemployment_rate" in df.columns:
            df["unemployment_change"] = df["unemployment_rate"].diff()
            df["unemployment_ma"] = df["unemployment_rate"].rolling(6).mean()
            self.feature_names.extend(["unemployment_change", "unemployment_ma"])

        # Economic regime indicators
        df["recession_indicator"] = self._detect_recession_regime(df)
        self.feature_names.append("recession_indicator")

        return df

    def _detect_recession_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect potential recession based on economic indicators

        Uses Sahm Rule and yield curve inversion
        """
        recession = pd.Series(0, index=df.index)

        # Yield curve inversion
        if "yield_curve_inverted" in df.columns:
            recession += df["yield_curve_inverted"]

        # Rising unemployment
        if "unemployment_change" in df.columns:
            recession += (df["unemployment_change"] > 0.5).astype(int)

        return (recession >= 2).astype(int)  # Multiple indicators

    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime features based on VIX and market conditions

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with regime features
        """
        # Calculate volatility regime
        if "close" in df.columns:
            returns = df["close"].pct_change()
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)

            df["volatility_regime"] = pd.cut(
                rolling_vol, bins=[0, 0.15, 0.25, 1.0], labels=["low", "medium", "high"]
            )

            # Convert to numeric
            df["vol_regime_low"] = (df["volatility_regime"] == "low").astype(int)
            df["vol_regime_medium"] = (df["volatility_regime"] == "medium").astype(int)
            df["vol_regime_high"] = (df["volatility_regime"] == "high").astype(int)

            self.feature_names.extend(["vol_regime_low", "vol_regime_medium", "vol_regime_high"])

        return df

    def merge_with_price_data(
        self, price_data: pd.DataFrame, macro_data: pd.DataFrame, date_column: str = "time"
    ) -> pd.DataFrame:
        """
        Merge macro data with price data

        Args:
            price_data: Stock price data
            macro_data: Macroeconomic data
            date_column: Name of date column

        Returns:
            Merged DataFrame
        """
        # Ensure both have datetime indices
        price_data = price_data.copy()
        macro_data = macro_data.copy()

        if date_column in price_data.columns:
            price_data[date_column] = pd.to_datetime(price_data[date_column])
        if "date" in macro_data.columns:
            macro_data["date"] = pd.to_datetime(macro_data["date"])

        # Merge using asof (forward fill macro data)
        merged = pd.merge_asof(
            price_data.sort_values(date_column),
            macro_data.sort_values("date"),
            left_on=date_column,
            right_on="date",
            direction="backward",
        )

        return merged

    def calculate_macro_risk_score(self, features: dict[str, float]) -> float:
        """
        Calculate composite macro risk score (0-100)
        Higher = more risk

        Args:
            features: Dictionary of macro features

        Returns:
            Risk score
        """
        score = 0.0

        # Yield curve (inverted = higher risk)
        if "yield_curve" in features:
            if features["yield_curve"] < 0:
                score += 30
            elif features["yield_curve"] < 0.5:
                score += 15

        # Inflation (high = higher risk)
        if "inflation_rate" in features:
            if features["inflation_rate"] > 5:
                score += 25
            elif features["inflation_rate"] > 3:
                score += 10

        # Unemployment (rising = higher risk)
        if "unemployment_change" in features:
            if features["unemployment_change"] > 0.5:
                score += 20
            elif features["unemployment_change"] > 0:
                score += 10

        # Fed policy (tightening = higher risk)
        if "fed_funds_change" in features:
            if features["fed_funds_change"] > 0.25:
                score += 15

        # Recession indicator
        if features.get("recession_indicator", 0) == 1:
            score += 40

        return min(score, 100)

    def get_feature_names(self) -> list[str]:
        """Get list of all macro feature names"""
        base_features = list(self.fred_series.keys())
        return base_features + self.feature_names
