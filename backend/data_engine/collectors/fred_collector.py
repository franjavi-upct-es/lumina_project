# backend/data_engine/collectors/fred_collector.py
"""
Federal Reserve Economic Data (FRED) collector
Provides macroeconomic data from the St. Louis Federeal Reserve
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import polars as pl
import requests
from loguru import logger
import asyncio

from backend.data_engine.collectors.base_collector import BaseDataCollector
from backend.config.settings import get_settings

settings = get_settings()


class FredCollector(BaseDataCollector):
    """
    Collector for FRED (Federal Reserve Economic Data)

    Common Series:
    - GDP: Gross Domestic Product
    - UNRATE: Unemployment Rate
    - CPIAUCSL: Consumer Price Index
    - FEDFUNDS: Federal Funds Rate
    - DGS10: 10-Year Treasury Rate
    - T10Y2Y: 10-Year Minus 2-Year Treasury Spread
    - VIXCLS: CBOE Volatility Index (VIX)
    - DEXUSEU: USD/EUR Exchange Rate
    """

    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 120):
        """
        Initialize FRED collector

        Args:
            api_key: FRED API key (defaults to settings)
            rate_limit: Max requests per minute (free tier: 120/min)
        """
        super().__init__(name="FRED", rate_limit=rate_limit)

        self.api_key = api_key or settings.FRED_API_KEY
        self.base_url = "https://api.stlouisfed.org/fred"

        if not self.api_key:
            logger.error("FRED API key not configured")

        # Common series for quick access
        self.common_series = {
            # GDP and Growth
            "gdp": "GDP",
            "real_gdp": "GDPC1",
            "gdp_growth": "A191RL1Q225SBEA",
            # Employment
            "unemployment": "UNRATE",
            "nonfarm_payroll": "PAYEMS",
            "labor_force": "CLF16OV",
            "jobless_claims": "ICSA",
            # Inflation
            "cpi": "CPIAUCSL",
            "core_cpi": "CPILFESL",
            "pce": "PCE",
            "core_pce": "PCEPILFE",
            # Interest Rates
            "fed_funds": "FEDFUNDS",
            "treasury_1m": "DGS1MO",
            "treasury_3m": "DGS3MO",
            "treasury_6m": "DGS6MO",
            "treasury_1y": "DGS1",
            "treasury_2y": "DGS2",
            "treasury_5y": "DGS5",
            "treasury_10y": "DGS10",
            "treasury_30y": "DGS30",
            # Yield Spreads
            "spread_10y_2y": "T10Y2Y",
            "spread_10y_3m": "T10Y3M",
            # Money Supply
            "m1": "M1SL",
            "m2": "M2SL",
            # Market Indicators
            "vix": "VIXCLS",
            "sp500": "SP500",
            # Exchange Rates
            "usd_eur": "DEXUSEU",
            "usd_cny": "DEXCHUS",
            "usd_jpy": "DEXJPUS",
            # Housing
            "housing_starts": "HOUST",
            "home_sales": "HSN1F",
            "case_shiller": "CSUSHPISA",
            # Consumer
            "retail_sales": "RSXFS",
            "consumer_sentiment": "UMCSENT",
            "personal_income": "PI",
            "personal_spending": "PCE",
            # Business
            "industrial_production": "INDPRO",
            "capacity_utilization": "TCU",
            "business_inventories": "BUSINV",
        }

    async def collect(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[pl.DataFrame]:
        """
        Collect economic data series from FRED

        Args:
            ticker: FRED series ID (e.g., 'GDP', 'UNRATE', 'DGS10')
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters
                - frequency: Data frequency ('d', 'w', 'm', 'q', 'a')
                - aggregation_method: Aggregation method ('avg', 'sum', 'eop')

        Returns:
            Polars DataFrame with economic data
        """
        if not self.api_key:
            logger.error("FRED API key not configured")
            return None

        # Check if ticker is a common name
        series_id = self.common_series.get(ticker.lower(), ticker)

        try:
            # Build parameters
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
            }

            # Add rate range if provided
            if start_date:
                params["observation_start"] = start_date.strftime("%Y-%m-%d")
            if end_date:
                params["observation_end"] = end_date.strftime("%Y-%m-%d")

            # Add frequency and aggregation if provided
            if "frequency" in kwargs:
                params["frequency"] = kwargs["frequency"]
            if "aggregation_method" in kwargs:
                params["aggregation_method"] = kwargs["aggregation_method"]

            # Make request
            url = f"{self.base_url}/series/observations"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, params=params, timeout=30)
            )

            if response.status_code != 200:
                logger.error(f"FRED API error: {response.status_code}")
                return None

            data = response.json()

            # Check for errors
            if "error_message" in data:
                logger.error(f"FRED API error: {data['error_message']}")
                return None

            if "observations" not in data:
                logger.error("No observations for series {series_id}")
                return None

            observations = data["observations"]

            # Parse observations
            records = []
            for obs in observations:
                value_str = obs.get("value")

                # Skip missing values (marked as '.')
                if value_str == ".":
                    continue

                try:
                    value = float(value_str)
                    records.append(
                        {"date": obs["date"], "value": value, "series_id": series_id}
                    )
                except (ValueError, TypeError):
                    continue

            if not records:
                logger.warning(f"No valid data for series {series_id}")
                return None

            # Create DataFrame
            df = pl.DataFrame(records)

            # Parse date
            df = df.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("time")
            ).drop("date")

            # Sort by time
            df = df.sort("time")

            # Add metadata
            df = self._add_metadata(df, series_id, "fred")

            logger.success(f"Collected {df.height} observations for {series_id}")
            return df

        except Exception as e:
            logger.error(f"Error collecting FRED series {series_id}: {e}")
            return None

    async def validate_data(self, data: Optional[pl.DataFrame]) -> bool:
        """
        Validate FRED data
        """
        if data is None or data.height == 0:
            return False

        required_columns = ["time", "value"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False

        return True

    async def get_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a FRED series

        Args:
            series_id: FRED series ID

        Returns:
            Dictionary with series metadata
        """
        if not self.api_key:
            return None

        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
            }

            url = f"{self.base_url}/series"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            if "seriess" not in data or not data["seriess"]:
                return None

            series_data = data["seriess"][0]

            info = {
                "id": series_data.get("id"),
                "title": series_data.get("title"),
                "observation_start": series_data.get("observation_start"),
                "observation_end": series_data.get("observation_end"),
                "frequency": series_data.get("frequency"),
                "frequency_short": series_data.get("frequency_short"),
                "units": series_data.get("units"),
                "seasonal_adjustment": series_data.get("seasonal_adjustment"),
                "last_updated": series_data.get("last_updated"),
                "popularity": series_data.get("popularity"),
                "notes": series_data.get("notes"),
            }

            logger.success(f"Fetched info for series {series_id}")
            return info

        except Exception as e:
            logger.error(f"Error fetching series info for {series_id}: {e}")
            return None

    async def search_series(
        self, search_text: str, limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search for FRED series by keyword

        Args:
            search_text: Search query
            limit: Max number of results

        Returns:
            List of matching series
        """
        if not self.api_key:
            return None

        try:
            params = {
                "search_text": search_text,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": limit,
            }

            url = f"{self.base_url}/series/search"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            if "seriess" not in data:
                return None

            results = []
            for series in data["seriess"]:
                results.append(
                    {
                        "id": series.get("id"),
                        "title": series.get("title"),
                        "frequency": series.get("frequency_short"),
                        "units": series.get("units"),
                        "popularity": series.get("popularity"),
                        "observation_start": series.get("observation_start"),
                        "observation_end": series.get("observation_end"),
                    }
                )

            logger.success(f"Found {len(results)} series for '{search_text}'")
            return results

        except Exception as e:
            logger.error(f"Error searching series: {e}")
            return None

    async def get_multiple_series(
        self,
        series_ids: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Get multiple series and merge them into one DataFrame

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame wit all series (wide format)
        """
        if not series_ids:
            return None

        try:
            logger.info(f"Fetching {len(series_ids)} FRED series")

            # Collect all series
            all_series = {}
            for series_id in series_ids:
                data = await self.collect_with_retry(
                    ticker=series_id,
                    start_date=start_date,
                    end_date=end_date,
                )

                if data is not None and data.height > 0:
                    # Keep only time and value columns
                    series_data = data.select(["time", "value"])
                    series_data = series_data.rename({"value": series_id})
                    all_series[series_id] = series_data

            if not all_series:
                logger.warning("No series data collected")
                return None

            # Merge all series on time
            result = None
            for series_id, data in all_series.items():
                if result is None:
                    result = data
                else:
                    result = result.join(data, on="time", how="outer")

            # Short by time
            if result is not None:
                result = result.sort("time")

            logger.success(f"Merged {len(all_series)} series")
            return result

        except Exception as e:
            logger.error(f"Error fetching multiple series: {e}")
            return None

    async def get_economic_indicators_bundle(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Get a bundle of key economic indicators

        Includes:
        - GDP growth
        - Unemployment rate
        - CPI inflation
        - Federal funds rate
        - 10-year treasury
        - VIX

        Returns:
            DataFrame with all indicators
        """
        key_indicators = [
            "UNRATE",  # Unemployment
            "CPIAUCSL",  # CPI
            "FEDFUNDS",  # Fed Funds Rate
            "DGS10",  # 10Y Treasury
            "VIXCLS",  # VIX
            "T10Y2Y",  # Yield Curve
        ]

        return await self.get_multiple_series(
            series_ids=key_indicators,
            start_date=start_date,
            end_date=end_date,
        )

    def get_common_series_list(self) -> Dict[str, str]:
        """
        Get list of common FRED series with descriptions

        Returns:
            Dictionary mapping friendly names to FRED series IDs
        """
        return self.common_series.copy()
