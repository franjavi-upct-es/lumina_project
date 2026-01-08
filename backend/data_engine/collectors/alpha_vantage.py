# backend/data_engine/collectors/alpha_vantage.py
"""
Alpha Vantage data collector for fundamental data and alternative data sources
Provides company fundamentals, earnings, economic indicators
"""

import asyncio
from datetime import datetime
from typing import Any

import polars as pl
import requests
from loguru import logger

from backend.config.settings import get_settings
from backend.data_engine.collectors.base_collector import BaseDataCollector

settings = get_settings()


class AlphaVantageCollector(BaseDataCollector):
    """
    Collector for Alpha Vantage API

    Features:
    - Company fundamentals (balance sheet, income statement, cash flow)
    - Real-time and historical quotes
    - Technical indicators
    - Economic indicators
    - Earnings data
    - IPO calendar
    """

    def __init__(self, api_key: str | None = None, rate_limit: int = 5):
        """
        Initialize Alpha Vantage collector

        Args:
            api_key: Alpha Vantage API key (defaults to settings)
            rate_limit: Max requests per minute (free tier: 5/min, premium: 75/min)
        """
        super().__init__(name="AlphaVantage", rate_limit=rate_limit)

        self.api_key = api_key or settings.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"

        if not self.api_key:
            logger.warning("Alpha Vantage API key not configured")

    async def collect(
        self,
        ticker: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs,
    ) -> pl.DataFrame | None:
        """
        Collect daily price data from Alpha Vantage

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (not used - AV returns full history)
            end_date: End date (not used)
            **kwargs: Additional parameters
                - outputsize: 'compact' (100 days) or 'full' (20+ years)

        Returns:
            Polars DataFrame with OHLCV data
        """
        if not self.api_key:
            logger.error("Alpha Vantage API key not configured")
            return None

        try:
            outputsize = kwargs.get("outputsize", "full")

            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": outputsize,
                "apikey": self.api_key,
            }

            # Make async request
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(self.base_url, params=params, timeout=30)
            )

            if response.status_code != 200:
                logger.error(f"API error: {response.status_code}")
                return None

            data = response.json()

            # Check for errors
            if "Error Message" in data:
                logger.error(f"API returned error: {data['Error Message']}")
                return None

            if "Note" in data:
                logger.warning(f"API rate limit message: {data['Note']}")
                return None

            # Parse time series data
            if "Time Series (Daily)" not in data:
                logger.error("No time series data in response")
                return None

            time_series = data["Time Series (Daily)"]

            # Convert to records
            records = []
            for date_str, values in time_series.items():
                records.append(
                    {
                        "time": datetime.strptime(date_str, "%Y-%m-%d"),
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "adjusted_close": float(values["5. adjusted close"]),
                        "volume": int(values["6. volume"]),
                        "dividend": float(values["7. dividend amount"]),
                        "split_coefficient": float(values["8. split coefficient"]),
                    }
                )

            if not records:
                logger.warning(f"No records parsed for {ticker}")
                return None

            # Create DataFrame
            df = pl.DataFrame(records)

            # Sort by time
            df = df.sort("time")

            # Filter by date range if provided
            if start_date:
                df = df.filter(pl.col("time") >= start_date)
            if end_date:
                df = df.filter(pl.col("time") <= end_date)

            # Add metadata
            df = self._add_metadata(df, ticker, "alpha_vantage")

            logger.success(f"Collected {df.height} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error collecting data for {ticker}: {e}")
            return None

    async def validate_data(self, data: pl.DataFrame | None) -> bool:
        """
        Validate collected data
        """
        if data is None or data.height == 0:
            return False

        required_columns = ["time", "open", "high", "low", "close", "volume"]
        missing = [col for col in required_columns if col not in data.columns]

        if missing:
            logger.error(f"Missing columns: {missing}")
            return False

        # Check for invalid prices
        invalid = data.filter(
            (pl.col("high") < pl.col("low")) | (pl.col("close") < 0) | (pl.col("volume") < 0)
        )

        if invalid.height > 0:
            logger.warning(f"Found {invalid.height} rows with invalid data")

        return True

    async def get_company_overview(self, ticker: str) -> dict[str, Any] | None:
        """
        Get comprehensive company overview and fundamentals

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with company information
        """
        if not self.api_key:
            return None

        try:
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.api_key,
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(self.base_url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            if not data or "Symbol" not in data:
                logger.warning(f"No overview data for {ticker}")
                return None

            # Parse and structure the data
            overview = {
                "ticker": data.get("Symbol"),
                "name": data.get("Name"),
                "description": data.get("Description"),
                "sector": data.get("Sector"),
                "industry": data.get("Industry"),
                "exchange": data.get("Exchange"),
                "currency": data.get("Currency"),
                "country": data.get("Country"),
                # Market data
                "market_cap": self._parse_float(data.get("MarketCapitalization")),
                "ebitda": self._parse_float(data.get("EBITDA")),
                "pe_ratio": self._parse_float(data.get("PERatio")),
                "peg_ratio": self._parse_float(data.get("PEGRatio")),
                "book_value": self._parse_float(data.get("BookValue")),
                "dividend_per_share": self._parse_float(data.get("DividendPerShare")),
                "dividend_yield": self._parse_float(data.get("DividendYield")),
                "eps": self._parse_float(data.get("EPS")),
                "revenue_per_share": self._parse_float(data.get("RevenuePerShareTTM")),
                "profit_margin": self._parse_float(data.get("ProfitMargin")),
                "operating_margin": self._parse_float(data.get("OperatingMarginTTM")),
                "roe": self._parse_float(data.get("ReturnOnEquityTTM")),
                "roa": self._parse_float(data.get("ReturnOnAssetsTTM")),
                "revenue_ttm": self._parse_float(data.get("RevenueTTM")),
                "gross_profit_ttm": self._parse_float(data.get("GrossProfitTTM")),
                "quarterly_earnings_growth": self._parse_float(
                    data.get("QuarterlyEarningsGrowthYOY")
                ),
                "quarterly_revenue_growth": self._parse_float(
                    data.get("QuarterlyRevenueGrowthYOY")
                ),
                "analyst_target_price": self._parse_float(data.get("AnalystTargetPrice")),
                "trailing_pe": self._parse_float(data.get("TrailingPE")),
                "forward_pe": self._parse_float(data.get("ForwardPE")),
                "price_to_sales": self._parse_float(data.get("PriceToSalesRatioTTM")),
                "price_to_book": self._parse_float(data.get("PriceToBookRatio")),
                "ev_to_revenue": self._parse_float(data.get("EVToRevenue")),
                "ev_to_ebitda": self._parse_float(data.get("EVToEBITDA")),
                "beta": self._parse_float(data.get("Beta")),
                "52_week_high": self._parse_float(data.get("52WeekHigh")),
                "52_week_low": self._parse_float(data.get("52WeekLow")),
                "50_day_ma": self._parse_float(data.get("50DayMovingAverage")),
                "200_day_ma": self._parse_float(data.get("200DayMovingAverage")),
                "shares_outstanding": self._parse_float(data.get("SharesOutstanding")),
                "dividend_date": data.get("DividendDate"),
                "ex_dividend_date": data.get("ExDividendDate"),
            }

            logger.success(f"Fetched company overview for {ticker}")
            return overview

        except Exception as e:
            logger.error(f"Error fetching overview for {ticker}: {e}")
            return None

    async def get_income_statement(
        self, ticker: str, quarterly: bool = False
    ) -> pl.DataFrame | None:
        """
        Get income statement data

        Args:
            ticker: Stock ticker symbol
            quarterly: If True, get quarterly data; else annual

        Returns:
            DataFrame with income statement data
        """
        if not self.api_key:
            return None

        try:
            params = {
                "function": "INCOME_STATEMENT",
                "symbol": ticker,
                "apikey": self.api_key,
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(self.base_url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            # Select quarterly or annual reports
            key = "quarterlyReports" if quarterly else "annualReports"
            if key not in data:
                logger.warning(f"No income statement data for {ticker}")
                return None

            reports = data[key]
            if not reports:
                return None

            # Parse reports
            records = []
            for report in reports:
                records.append(
                    {
                        "fiscal_date": report.get("fiscalDateEnding"),
                        "reported_currency": report.get("reportedCurrency"),
                        "total_revenue": self._parse_float(report.get("totalRevenue")),
                        "cost_of_revenue": self._parse_float(report.get("costOfRevenue")),
                        "gross_profit": self._parse_float(report.get("grossProfit")),
                        "operating_expenses": self._parse_float(report.get("operatingExpenses")),
                        "operating_income": self._parse_float(report.get("operatingIncome")),
                        "ebitda": self._parse_float(report.get("ebitda")),
                        "net_income": self._parse_float(report.get("netIncome")),
                        "eps": self._parse_float(report.get("eps")),
                        "research_development": self._parse_float(
                            report.get("researchAndDevelopment")
                        ),
                    }
                )

            df = pl.DataFrame(records)
            df = df.with_columns(pl.col("fiscal_date").str.strptime(pl.Date, "%Y-%m-%d"))

            logger.success(f"Fetched income statement for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching income statement for {ticker}: {e}")
            return None

    async def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> pl.DataFrame | None:
        """
        Get balance sheet data
        """
        if not self.api_key:
            return None

        try:
            params = {
                "function": "BALANCE_SHEET",
                "symbol": ticker,
                "apikey": self.api_key,
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(self.base_url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            key = "quarterlyReports" if quarterly else "annualReports"
            if key not in data:
                return None

            reports = data[key]
            if not reports:
                return None

            records = []
            for report in reports:
                records.append(
                    {
                        "fiscal_date": report.get("fiscalDateEnding"),
                        "total_assets": self._parse_float(report.get("totalAssets")),
                        "total_liabilities": self._parse_float(report.get("totalLiabilities")),
                        "total_shareholder_equity": self._parse_float(
                            report.get("totalShareholderEquity")
                        ),
                        "current_assets": self._parse_float(report.get("totalCurrentAssets")),
                        "current_liabilities": self._parse_float(
                            report.get("totalCurrentLiabilities")
                        ),
                        "cash": self._parse_float(
                            report.get("cashAndCashEquivalentsAtCarryingValue")
                        ),
                        "inventory": self._parse_float(report.get("inventory")),
                        "short_term_debt": self._parse_float(report.get("shortTermDebt")),
                        "long_term_debt": self._parse_float(report.get("longTermDebt")),
                        "retained_earnings": self._parse_float(report.get("retainedEarnings")),
                    }
                )

            df = pl.DataFrame(records)
            df = df.with_columns(pl.col("fiscal_date").str.strptime(pl.Date, "%Y-%m-%d"))

            logger.success(f"Fetched balance sheet for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching balance sheet for {ticker}: {e}")
            return None

    async def get_cash_flow(self, ticker: str, quarterly: bool = False) -> pl.DataFrame | None:
        """
        Get cash flow statement
        """
        if not self.api_key:
            return None

        try:
            params = {
                "function": "CASH_FLOW",
                "symbol": ticker,
                "apikey": self.api_key,
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(self.base_url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            key = "quarterlyReports" if quarterly else "annualReports"
            if key not in data:
                return None

            reports = data[key]
            if not reports:
                return None

            records = []
            for report in reports:
                records.append(
                    {
                        "fiscal_date": report.get("fiscalDateEnding"),
                        "operating_cashflow": self._parse_float(report.get("operatingCashflow")),
                        "capital_expenditures": self._parse_float(
                            report.get("capitalExpenditures")
                        ),
                        "cashflow_from_investment": self._parse_float(
                            report.get("cashflowFromInvestment")
                        ),
                        "cashflow_from_financing": self._parse_float(
                            report.get("cashflowFromFinancing")
                        ),
                        "dividend_payout": self._parse_float(report.get("dividendPayout")),
                        "net_income": self._parse_float(report.get("netIncome")),
                        "depreciation": self._parse_float(report.get("depreciation")),
                        "change_in_cash": self._parse_float(
                            report.get("changeInCashAndCashEquivalents")
                        ),
                    }
                )

            df = pl.DataFrame(records)
            df = df.with_columns(pl.col("fiscal_date").str.strptime(pl.Date, "%Y-%m-%d"))

            logger.success(f"Fetched cash flow for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching cash flow for {ticker}: {e}")
            return None

    async def get_earnings(self, ticker: str) -> pl.DataFrame | None:
        """
        Get earnings data (historical and estimates)
        """
        if not self.api_key:
            return None

        try:
            params = {
                "function": "EARNINGS",
                "symbol": ticker,
                "apikey": self.api_key,
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(self.base_url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            if "quarterlyEarnings" not in data:
                return None

            earnings = data["quarterlyEarnings"]
            if not earnings:
                return None

            records = []
            for earning in earnings:
                records.append(
                    {
                        "fiscal_date": earning.get("fiscalDateEnding"),
                        "reported_date": earning.get("reportedDate"),
                        "reported_eps": self._parse_float(earning.get("reportedEPS")),
                        "estimated_eps": self._parse_float(earning.get("estimatedEPS")),
                        "surprise": self._parse_float(earning.get("surprise")),
                        "surprise_percentage": self._parse_float(earning.get("surprisePercentage")),
                    }
                )

            df = pl.DataFrame(records)

            logger.success(f"Fetched earnings for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {e}")
            return None

    async def get_economic_indicator(
        self, indicator: str, interval: str = "monthly"
    ) -> pl.DataFrame | None:
        """
        Get economic indicators from FRED via Alpha Vantage

        Args:
            indicator: Indicator name (e.g., 'REAL_GDP', 'CPI', 'UNEMPLOYMENT')
            interval: Data interval (monthly, quarterly, annual)

        Returns:
            DataFrame with economic indicator data
        """
        if not self.api_key:
            return None

        try:
            params = {
                "function": indicator,
                "interval": interval,
                "apikey": self.api_key,
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(self.base_url, params=params, timeout=30)
            )

            if response.status_code != 200:
                return None

            data = response.json()

            if "data" not in data:
                logger.warning(f"No data for indicator {indicator}")
                return None

            records = []
            for point in data["data"]:
                records.append(
                    {
                        "date": point.get("date"),
                        "value": self._parse_float(point.get("value")),
                    }
                )

            if not records:
                return None

            df = pl.DataFrame(records)
            df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

            logger.success(f"Fetched economic indicator: {indicator}")
            return df

        except Exception as e:
            logger.error(f"Error fetching indicator {indicator}: {e}")
            return None

    def _parse_float(self, value: Any) -> float | None:
        """
        Safely parse float values
        """
        if value is None or value == "None":
            return None

        try:
            return float(value)
        except (ValueError, TypeError):
            return None
