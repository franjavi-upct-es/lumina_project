# backend/ml_engine/features/fundamental.py
"""
Fundamental analysis features
Financial metrics and ratios for ML models
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class FundamentalFeatures:
    """
    Fundamental analysis features

    Categories:
    - Valuation ratios (P/E, P/B, P/S)
    - Profitability metrics (ROE, ROA, margins)
    - Financial health (debt ratios, current ratio)
    - Growth metrics (revenue growth, EPS growth)
    """

    def __init__(self):
        self.feature_names = []

    def create_features_from_info(self, info: dict[str, Any]) -> dict[str, float]:
        """
        Extract fundamental features from company info

        Args:
            info: Company information dictionary from yfinance

        Returns:
            Dictionary of fundamental features
        """
        features = {}

        # Valuation ratios
        features.update(self._extract_valuation_ratios(info))

        # Profitability metrics
        features.update(self._extract_profitability_metrics(info))

        # Financial health metrics
        features.update(self._extract_financial_health(info))

        # Growth metrics
        features.update(self._extract_growth_metrics(info))

        # Market metrics
        features.update(self._extract_market_metrics(info))

        logger.info(f"Created {len(features)} fundamental features")
        return features

    def _extract_valuation_ratios(self, info: dict[str, Any]) -> dict[str, float]:
        """Extract valuation ratios"""
        return {
            "profit_margin": float(info.get("profitMargins", np.nan)),
            "operating_margin": float(info.get("operatingMargins", np.nan)),
            "gross_margin": float(info.get("grossMargins", np.nan)),
            "roe": float(info.get("returnOnEquity", np.nan)),
            "roa": float(info.get("returnOnAssets", np.nan)),
            "roic": float(info.get("returnOnCapital", np.nan))
            if "returnOnCapital" in info
            else np.nan,
        }

    def _extract_profitability_metrics(self, info: dict[str, Any]) -> dict[str, float]:
        """Extract profitability metrics"""
        return {
            "profit_margin": float(info.get("profitMargins", np.nan)),
            "operating_margin": float(info.get("operatingMargins", np.nan)),
            "gross_margin": float(info.get("grossMargins", np.nan)),
            "roe": float(info.get("returnOnEquity", np.nan)),
            "roa": float(info.get("returnOnAssets", np.nan)),
            "roic": float(info.get("returnOnCapital", np.nan))
            if "returnOnCapital" in info
            else np.nan,
        }

    def _extract_financial_health(self, info: dict[str, Any]) -> dict[str, float]:
        """Extract financial health metrics"""
        return {
            "debt_to_equity": float(info.get("debtToEquity", np.nan)),
            "current_ratio": float(info.get("currentRatio", np.nan)),
            "quick_ratio": float(info.get("quickRatio", np.nan)),
            "total_cash": float(info.get("totalCash", np.nan)),
            "total_debt": float(info.get("totalDebt", np.nan)),
            "free_cashflow": float(info.get("freeCashflow", np.nan)),
        }

    def _extract_growth_metrics(self, info: dict[str, Any]) -> dict[str, float]:
        """Extract growth metrics"""
        return {
            "revenue_growth": float(info.get("revenueGrowth", np.nan)),
            "earnings_growth": float(info.get("earningsGrowth", np.nan)),
            "earnings_quarterly_growth": float(info.get("earningsQuarterlyGrowth", np.nan)),
        }

    def _extract_market_metrics(self, info: dict[str, Any]) -> dict[str, float]:
        """Extract market-related metrics"""
        market_cap = float(info.get("marketCap", np.nan))

        return {
            "market_cap": market_cap,
            "market_cap_log": np.log(market_cap) if market_cap > 0 else np.nan,
            "beta": float(info.get("beta", np.nan)),
            "dividend_yield": float(info.get("dividendYield", np.nan)),
            "payout_ratio": float(info.get("payoutRatio", np.nan)),
            "shares_outstanding": float(info.get("sharesOutstanding", np.nan)),
        }

    def create_time_series_features(
        self, data: pd.DataFrame, quarterly_data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Create time-varying fundamental features

        Args:
            data: Price data DataFrame
            quarterly_data: Quarterly fundamental data (earnings, revenue, etc.)

        Returns:
            DataFrame with fundamental time series features
        """
        df = data.copy()

        if quarterly_data is not None:
            # Merge quarterly data
            df = pd.merge_asof(
                df.sort_values("time"),
                quarterly_data.sort_values("date"),
                left_on="time",
                right_on="date",
                direction="backward",
            )

            # Calculate derived metrics
            if "earnings" in df.columns and "price" in df.columns:
                df["earnings_yield"] = df["earnings"] / df["price"]

            if "revenue" in df.columns:
                df["revenue_change"] = df["revenue"].pct_change()

        return df

    def calculate_quality_score(self, features: dict[str, float]) -> float:
        """
        Calculate composite quality score

        Args:
            features: Dictionary of fundamental features

        Returns:
            Quality score (0-100)
        """
        score = 0.0
        weights = 0.0

        # Profitability (40% weight)
        if not np.isnan(features.get("roe", np.nan)):
            roe = features["roe"]
            score += min(max(roe * 100, 0), 40) * 0.4
            weights += 0.4

        # Financial health (30% weight)
        if not np.isnan(features.get("current_ratio", np.nan)):
            cr = features["current_ratio"]
            score += min(max((cr - 1) * 15, 0), 30) * 0.3
            weights += 0.3

        # Growth (30% weight)
        if not np.isnan(features.get("revenue_growth", np.nan)):
            growth = features["revenue_growth"]
            score += min(max(growth * 100, 0), 30) * 0.3
            weights += 0.3

        return score / weights if weights > 0 else 50.0

    def get_feature_names(self) -> list:
        """Get list of all feature names"""
        return [
            # Valuation
            "pe_ratio",
            "forward_pe",
            "peg_ratio",
            "price_to_book",
            "price_to_sales",
            "ev_to_revenue",
            "ev_to_ebitda",
            # Profitability
            "profit_margin",
            "operating_margin",
            "gross_margin",
            "roe",
            "roa",
            "roic",
            # Financial health
            "debt_to_equity",
            "current_ratio",
            "quick_ratio",
            "total_cash",
            "total_debt",
            "free_cashflow",
            # Growth
            "revenue_growth",
            "earnings_growth",
            "earnings_quarterly_growth",
            # Market
            "market_cap",
            "market_cap_log",
            "beta",
            "dividend_yield",
            "payout_ratio",
            "shares_outstanding",
        ]
