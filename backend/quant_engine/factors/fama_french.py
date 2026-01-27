# backend/quant_engine/factors/fama_french.py
"""
Fama-French Factor Model Implementation

This module provides functionality to work with Fama-French factor models
for asset pricing and portfolio analysis. Supports:
- 3-Factor Model (Market, SMB, HML)
- 5-Factor Model (adds RMW, CMA)
- 6-Factor Model (adds Momentum - UMD)

Key Features:
- Download and cache factor data from Kenneth French's library
- Factor exposure (beta) estimation via regression
- Alpha and attribution analysis
- Factor-adjusted returns calculation
- Rolling factor exposure analysis

References:
- Fama, E. F., & French, K. R. (1993). Common risk factors in stock and bond returns.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

# Try to import pandas_datareader for Fama-French data
try:
    import pandas_datareader.data as web

    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False
    logger.warning("pandas_datareader not installed. Install with: pip install pandas-datareader")


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================


class FactorModel(Enum):
    """Fama-French factor model types"""

    CAPM = "capm"  # Single factor (Market)
    THREE_FACTOR = "three_factor"  # Market, SMB, HML
    FIVE_FACTOR = "five_factor"  # Market, SMB, HML, RMW, CMA
    SIX_FACTOR = "six_factor"  # Five-factor + Momentum


class Frequency(Enum):
    """Data frequency"""

    DAILY = "daily"
    MONTHLY = "monthly"


@dataclass
class FactorExposure:
    """Results from factor exposure regression"""

    # Factor betas (loadings)
    market_beta: float
    smb_beta: float | None = None  # Size factor
    hml_beta: float | None = None  # Value factor
    rmw_beta: float | None = None  # Profitability factor
    cma_beta: float | None = None  # Investment factor
    mom_beta: float | None = None  # Momentum factor

    # Alpha (intercept)
    alpha: float = 0.0
    alpha_annualized: float = 0.0

    # Statistics
    r_squared: float = 0.0
    adjusted_r_squared: float = 0.0

    # Standard errors and p-values
    alpha_tstat: float = 0.0
    alpha_pvalue: float = 0.0
    beta_tstats: dict[str, float] = None
    beta_pvalues: dict[str, float] = None

    # Model info
    model_type: FactorModel = FactorModel.THREE_FACTOR
    observations: int = 0
    residual_std: float = 0.0

    def __post_init__(self):
        if self.beta_tstats is None:
            self.beta_tstats = {}
        if self.beta_pvalues is None:
            self.beta_pvalues = {}


@dataclass
class FactorAttribution:
    """Factor attribution analysis results"""

    # Total return decomposition
    total_return: float
    factor_return: float  # Return explained by factors
    alpha_return: float  # Return not explained (alpha)

    # Individual factor contributions
    market_contribution: float
    smb_contribution: float | None = None
    hml_contribution: float | None = None
    rmw_contribution: float | None = None
    cma_contribution: float | None = None
    mom_contribution: float | None = None

    # Percentages
    percent_explained: float = 0.0

    # Period info
    start_date: datetime | None = None
    end_date: datetime | None = None


# ============================================================================
# FAMA-FRENCH DATA LOADER
# ============================================================================


class FamaFrenchDataLoader:
    """
    Loader for Fama-French factor data

    Downloads data from Kenneth French's data library and caches locally.
    """

    # Dataset names for pandas_datareader
    DATASETS = {
        Frequency.DAILY: {
            FactorModel.THREE_FACTOR: "F-F_Research_Data_Factors_daily",
            FactorModel.FIVE_FACTOR: "F-F_Research_Data_5_Factors_2x3_daily",
        },
        Frequency.MONTHLY: {
            FactorModel.THREE_FACTOR: "F-F_Research_Data_Factors",
            FactorModel.FIVE_FACTOR: "F-F_Research_Data_5_Factors_2x3",
        },
    }

    # Momentum dataset
    MOMENTUM_DATASETS = {
        Frequency.DAILY: "F-F_Momentum_Factor_daily",
        Frequency.MONTHLY: "F-F_Momentum_Factor",
    }

    def __init__(self, cache_dir: str | Path | None = None):
        """
        Initialize data loader

        Args:
            cache_dir: Directory for caching data (defaults to ~/.lumina/ff_data)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".lumina" / "ff_data"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cache: dict[str, pd.DataFrame] = {}

    def _get_cache_path(self, dataset_name: str) -> Path:
        """Get cache file path for a dataset"""
        safe_name = dataset_name.replace("-", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_name}.parquet"

    def _is_cache_valid(self, cache_path: Path, max_age_days: int = 1) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age < timedelta(days=max_age_days)

    async def get_factors(
        self,
        model: FactorModel = FactorModel.THREE_FACTOR,
        frequency: Frequency = Frequency.DAILY,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        include_momentum: bool = False,
    ) -> pd.DataFrame:
        """
        Get Fama-French factor data

        Args:
            model: Factor model type
            frequency: Data frequency
            start_date: Start date for data
            end_date: End date for data
            include_momentum: Whether to include momentum factor

        Returns:
            DataFrame with factor returns
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._get_factors_sync(
                model, frequency, start_date, end_date, include_momentum
            ),
        )

    def _get_factors_sync(
        self,
        model: FactorModel,
        frequency: Frequency,
        start_date: datetime | None,
        end_date: datetime | None,
        include_momentum: bool,
    ) -> pd.DataFrame:
        """
        Get factor data synchronously
        """
        if not HAS_DATAREADER:
            logger.error("pandas_datareader required for Fama-French data")
            return pd.DataFrame()

        try:
            # Get base factor data
            if model == FactorModel.CAPM:
                dataset_name = self.DATASETS[frequency][FactorModel.THREE_FACTOR]
            else:
                dataset_name = self.DATASETS[frequency].get(
                    model, self.DATASETS[frequency][FactorModel.THREE_FACTOR]
                )

            # Check cache
            cache_key = f"{dataset_name}_{start_date}_{end_date}"
            if cache_key in self._cache:
                df = self._cache[cache_key].copy()
            else:
                # Download from Kenneth French's library
                logger.info(f"Downloading Fama-French data: {dataset_name}")

                ff_data = web.DataReader(dataset_name, "famafrench", start_date, end_date)

                # First table contains the factor returns
                df = ff_data[0].copy()

                # Convert to decimal (data is in percent)
                df = df / 100.0

                # Ensure datetime index
                df.index = pd.to_datetime(df.index.astype(str))

                # Cache
                self._cache[cache_key] = df.copy()

            # Add momentum if requested
            if include_momentum or model == FactorModel.SIX_FACTOR:
                mom_data = self._get_momentum_sync(frequency, start_date, end_date)
                if not mom_data.empty:
                    # Align indices
                    df = df.join(mom_data, how="inner")

            # Standardize column names
            df = self._standardize_columns(df)

            # Filter date range
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]

            return df

        except Exception as e:
            logger.error(f"Error loading Fama-French data: {e}")
            return pd.DataFrame()

    def _get_momentum_sync(
        self,
        frequency: Frequency,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pd.DataFrame:
        """
        Get momentum factor data
        """
        try:
            dataset_name = self.MOMENTUM_DATASETS[frequency]

            mom_data = web.DataReader(dataset_name, "famafrench", start_date, end_date)
            df = mom_data[0].copy()

            # Convert to decimal
            df = df / 100.0

            # Rename column
            if "Mom" in df.columns:
                df = df.rename(columns={"Mom": "MOM"})
            elif "WML" in df.columns:
                df = df.rename(columns={"WML": "MOM"})

            df.index = pd.to_datetime(df.index.astype(str))

            return df

        except Exception as e:
            logger.warning(f"Could not load momentum data: {e}")
            return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names
        """
        column_mapping = {
            "Mkt-RF": "MKT_RF",
            "Mkt-Rf": "MKT_RF",
            "MKT-RF": "MKT_RF",
            "SMB": "SMB",
            "HML": "HML",
            "RMW": "RMW",
            "CMA": "CMA",
            "RF": "RF",
            "Mom": "MOM",
            "WML": "MOM",
        }

        df = df.rename(columns=column_mapping)
        return df


# ============================================================================
# FAMA-FRENCH ANALYZER
# ============================================================================


class FamaFrenchAnalyzer:
    """
    Analyzer for Fama-French factor model regression and attribution

    This class provides methods to:
    - Estimate factor exposures (betas) via OLS regression
    - Calculate alpha (Jensen's alpha)
    - Perform factor attribution analysis
    - Rolling factor exposure estimation

    Example:
        ```python
        analyzer = FamaFrenchAnalyzer()

        # Estimate factor exposures
        exposure = await analyzer.estimate_factor_exposure(
            returns=stock_returns,
            model=FactorModel.THREE_FACTOR,
        )

        print(f"Market Beta: {exposure.market_beta:.3f}")
        print(f"Alpha (annualized): {exposure.alpha_annualized:.2%}")
        ```
    """

    def __init__(self, data_loader: FamaFrenchDataLoader | None = None):
        """
        Initialize analyzer

        Args:
            data_loader: Optional data loader instance
        """
        self.data_loader = data_loader or FamaFrenchDataLoader()

    async def estimate_factor_exposure(
        self,
        returns: pd.Series | np.ndarray,
        dates: pd.DatetimeIndex | None = None,
        model: FactorModel = FactorModel.THREE_FACTOR,
        frequency: Frequency = Frequency.DAILY,
        annualization_factor: int | None = None,
    ) -> FactorExposure:
        """
        Estimate factor exposures using OLS regression

        Runs the regression:
        R_i - R_f = alpha + beta_m * (R_m - R_f) + beta_smb * SMB + beta_hml * HML + ...

        Args:
            returns: Asset returns (should be excess returns or will be converted)
            dates: Date index for returns (required if returns is numpy array)
            model: Factor model to use
            frequency: Data frequency
            annualization_factor: Factor for annualizing alpha (default: 252 for daily)

        Returns:
            FactorExposure object with results
        """
        # Convert to Series if needed
        if isinstance(returns, np.ndarray):
            if dates is None:
                raise ValueError("dates required when returns is numpy array")
            returns = pd.Series(returns, index=dates)

        # Set annualization factor
        if annualization_factor is None:
            annualization_factor = 252 if frequency == Frequency.DAILY else 12

        # Get factor data
        factors = await self.data_loader.get_factors(
            model=model,
            frequency=frequency,
            start_date=returns.index.min(),
            end_date=returns.index.max(),
            include_momentum=(model == FactorModel.SIX_FACTOR),
        )

        if factors.empty:
            logger.error("Could not load factor data")
            return FactorExposure(market_beta=1.0)

        # Align data
        aligned_data = self._align_data(returns, factors)

        if len(aligned_data) < 30:
            logger.warning(f"Only {len(aligned_data)} observations for regression")

        # Run regression
        exposure = self._run_regression(aligned_data, model, annualization_factor)

        return exposure

    def _align_data(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align returns and factor data
        """
        # Create DataFrame with returns
        df = pd.DataFrame({"returns": returns})

        # Join with factors
        df = df.join(factors, how="inner")

        # Drop any NaN
        df = df.dropna()

        return df

    def _run_regression(
        self,
        data: pd.DataFrame,
        model: FactorModel,
        annualization_factor: int,
    ) -> FactorExposure:
        """
        Run OLS regression to estimate factor exposures
        """
        # Calculate excess returns if RF column exists
        if "RF" in data.columns:
            y = data["returns"] - data["RF"]
        else:
            y = data["returns"]

        # Build factor matrix based on model
        if model == FactorModel.CAPM:
            factor_cols = ["MKT_RF"]
        elif model == FactorModel.THREE_FACTOR:
            factor_cols = ["MKT_RF", "SMB", "HML"]
        elif model == FactorModel.FIVE_FACTOR:
            factor_cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
        elif model == FactorModel.SIX_FACTOR:
            factor_cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
        else:
            factor_cols = ["MKT_RF"]

        # Filter to available columns
        available_cols = [col for col in factor_cols if col in data.columns]

        if not available_cols:
            logger.error("No factor columns found in data")
            return FactorExposure(market_beta=1.0)

        X = data[available_cols]
        X = sm.add_constant(X)  # Add intercept

        # Run OLS
        try:
            ols_model = sm.OLS(y, X).fit()
        except Exception as e:
            logger.error(f"OLS regression failed: {e}")
            return FactorExposure(market_beta=1.0)

        # Extract results
        params = ols_model.params
        tvalues = ols_model.tvalues
        pvalues = ols_model.pvalues

        # Build exposure object
        exposure = FactorExposure(
            market_beta=params.get("MKT_RF", 1.0),
            smb_beta=params.get("SMB"),
            hml_beta=params.get("HML"),
            rmw_beta=params.get("RMW"),
            cma_beta=params.get("CMA"),
            mom_beta=params.get("MOM"),
            alpha=params.get("const", 0.0),
            alpha_annualized=params.get("const", 0.0) * annualization_factor,
            r_squared=ols_model.rsquared,
            adjusted_r_squared=ols_model.rsquared_adj,
            alpha_tstat=tvalues.get("const", 0.0),
            alpha_pvalue=pvalues.get("const", 1.0),
            beta_tstats={col: tvalues.get(col, 0.0) for col in available_cols},
            beta_pvalues={col: pvalues.get(col, 1.0) for col in available_cols},
            model_type=model,
            observations=len(y),
            residual_std=np.sqrt(ols_model.mse_resid),
        )

        return exposure

    async def rolling_factor_exposure(
        self,
        returns: pd.Series,
        window: int = 60,
        model: FactorModel = FactorModel.THREE_FACTOR,
        frequency: Frequency = Frequency.DAILY,
        min_periods: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate rolling factor exposures

        Args:
            returns: Asset returns
            window: Rolling window size
            model: Factor model to use
            frequency: Data frequency
            min_periods: Minimum observations for calculation

        Returns:
            DataFrame with rolling betas and alpha
        """
        if min_periods is None:
            min_periods = window // 2

        # Get factor data
        factors = await self.data_loader.get_factors(
            model=model,
            frequency=frequency,
            start_date=returns.index.min(),
            end_date=returns.index.max(),
        )

        if factors.empty:
            return pd.DataFrame()

        # Align data
        aligned = self._align_data(returns, factors)

        # Determine factor columns
        if model == FactorModel.CAPM:
            factor_cols = ["MKT_RF"]
        elif model == FactorModel.THREE_FACTOR:
            factor_cols = ["MKT_RF", "SMB", "HML"]
        elif model == FactorModel.FIVE_FACTOR:
            factor_cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
        else:
            factor_cols = ["MKT_RF", "SMB", "HML"]

        available_cols = [col for col in factor_cols if col in aligned.columns]

        # Calculate excess returns
        if "RF" in aligned.columns:
            excess_returns = aligned["returns"] - aligned["RF"]
        else:
            excess_returns = aligned["returns"]

        # Rolling regression
        results = []

        for i in range(window, len(aligned) + 1):
            if i - window < min_periods:
                continue

            window_data = aligned.iloc[i - window : i]
            window_returns = excess_returns.iloc[i - window : i]

            X = window_data[available_cols]
            X = sm.add_constant(X)
            y = window_returns

            try:
                ols_model = sm.OLS(y, X).fit()

                result = {
                    "date": aligned.index[i - 1],
                    "alpha": ols_model.params.get("const", 0.0),
                    "r_squared": ols_model.rsquared,
                }

                for col in available_cols:
                    result[f"beta_{col.lower()}"] = ols_model.params.get(col, 0.0)

                results.append(result)

            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.set_index("date")

        return df

    async def factor_attribution(
        self,
        returns: pd.Series,
        model: FactorModel = FactorModel.THREE_FACTOR,
        frequency: Frequency = Frequency.DAILY,
    ) -> FactorAttribution:
        """
        Perform factor attribution analysis

        Decomposes portfolio returns into factor contributions.

        Args:
            returns: Asset returns
            model: Factor model to use
            frequency: Data frequency

        Returns:
            FactorAttribution object with decomposition
        """
        # Get factor exposure
        exposure = await self.estimate_factor_exposure(
            returns=returns,
            model=model,
            frequency=frequency,
        )

        # Get factor data
        factors = await self.data_loader.get_factors(
            model=model,
            frequency=frequency,
            start_date=returns.index.min(),
            end_date=returns.index.max(),
        )

        if factors.empty:
            logger.error("Could not load factor data for attribution")
            return FactorAttribution(
                total_return=float(returns.sum()),
                factor_return=0.0,
                alpha_return=float(returns.sum()),
                market_contribution=0.0,
            )

        # Align data
        aligned = self._align_data(returns, factors)

        # Calculate total return
        total_return = float(aligned["returns"].sum())

        # Calculate factor contributions
        market_contribution = exposure.market_beta * float(
            aligned.get("MKT_RF", pd.Series([0])).sum()
        )

        smb_contribution = None
        hml_contribution = None
        rmw_contribution = None
        cma_contribution = None
        mom_contribution = None

        if exposure.smb_beta is not None and "SMB" in aligned.columns:
            smb_contribution = exposure.smb_beta * float(aligned["SMB"].sum())

        if exposure.hml_beta is not None and "HML" in aligned.columns:
            hml_contribution = exposure.hml_beta * float(aligned["HML"].sum())

        if exposure.rmw_beta is not None and "RMW" in aligned.columns:
            rmw_contribution = exposure.rmw_beta * float(aligned["RMW"].sum())

        if exposure.cma_beta is not None and "CMA" in aligned.columns:
            cma_contribution = exposure.cma_beta * float(aligned["CMA"].sum())

        if exposure.mom_beta is not None and "MOM" in aligned.columns:
            mom_contribution = exposure.mom_beta * float(aligned["MOM"].sum())

        # Total factor return
        factor_return = market_contribution
        for contrib in [
            smb_contribution,
            hml_contribution,
            rmw_contribution,
            cma_contribution,
            mom_contribution,
        ]:
            if contrib is not None:
                factor_return += contrib

        # Alpha return
        alpha_return = total_return - factor_return

        # Percent explained
        percent_explained = factor_return / total_return if total_return != 0 else 0.0

        return FactorAttribution(
            total_return=total_return,
            factor_return=factor_return,
            alpha_return=alpha_return,
            market_contribution=market_contribution,
            smb_contribution=smb_contribution,
            hml_contribution=hml_contribution,
            rmw_contribution=rmw_contribution,
            cma_contribution=cma_contribution,
            mom_contribution=mom_contribution,
            percent_explained=percent_explained,
            start_date=aligned.index.min(),
            end_date=aligned.index.max(),
        )

    async def expected_return(
        self,
        exposure: FactorExposure,
        model: FactorModel = FactorModel.THREE_FACTOR,
        frequency: Frequency = Frequency.MONTHLY,
        lookback_years: int = 10,
    ) -> float:
        """
        Calculate expected return using factor model

        E[R_i] = R_f + beta_m * E[MKT_RF] + beta_smb * E[SMB] + ...

        Args:
            exposure: Factor exposure to use
            model: Factor model
            frequency: Frequency for historical averages
            lookback_years: Years of history for average factor returns

        Returns:
            Expected annualized return
        """
        # Get historical factor data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_years * 365)

        factors = await self.data_loader.get_factors(
            model=model,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
        )

        if factors.empty:
            logger.warning("Could not calculate expected return - no factor data")
            return 0.0

        # Calculate average factor returns
        avg_factors = factors.mean()

        # Annualize if monthly
        annualization = 12 if frequency == Frequency.MONTHLY else 252

        # Calculate expected excess return
        expected_excess = exposure.market_beta * avg_factors.get("MKT_RF", 0.0)

        if exposure.smb_beta is not None and "SMB" in avg_factors:
            expected_excess += exposure.smb_beta * avg_factors["SMB"]

        if exposure.hml_beta is not None and "HML" in avg_factors:
            expected_excess += exposure.hml_beta * avg_factors["HML"]

        if exposure.rmw_beta is not None and "RMW" in avg_factors:
            expected_excess += exposure.rmw_beta * avg_factors["RMW"]

        if exposure.cma_beta is not None and "CMA" in avg_factors:
            expected_excess += exposure.cma_beta * avg_factors["CMA"]

        # Annualize
        expected_annual = expected_excess * annualization

        # Add risk-free rate
        rf_annual = avg_factors.get("RF", 0.0) * annualization

        return expected_annual + rf_annual


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def get_factor_exposure(
    returns: pd.Series,
    model: FactorModel = FactorModel.THREE_FACTOR,
) -> FactorExposure:
    """
    Quick function to estimate factor exposure

    Args:
        returns: Asset returns
        model: Factor model to use

    Returns:
        FactorExposure object
    """
    analyzer = FamaFrenchAnalyzer()
    return await analyzer.estimate_factor_exposure(returns=returns, model=model)


async def get_factor_data(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    model: FactorModel = FactorModel.THREE_FACTOR,
    frequency: Frequency = Frequency.DAILY,
) -> pd.DataFrame:
    """
    Quick function to get Fama-French factor data

    Args:
        start_date: Start date
        end_date: End date
        model: Factor model
        frequency: Data frequency

    Returns:
        DataFrame with factor returns
    """
    loader = FamaFrenchDataLoader()
    return await loader.get_factors(
        model=model,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    """Example usage of Fama-French analyzer"""

    async def main():
        # Create sample returns (in practice, use real data)
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates, name="returns")

        # Initialize analyzer
        analyzer = FamaFrenchAnalyzer()

        # Example 1: Estimate factor exposure
        print("\n=== Factor Exposure Analysis ===")
        exposure = await analyzer.estimate_factor_exposure(
            returns=returns,
            model=FactorModel.THREE_FACTOR,
        )

        print(f"Market Beta: {exposure.market_beta:.3f}")
        print(f"SMB Beta: {exposure.smb_beta:.3f}" if exposure.smb_beta else "SMB Beta: N/A")
        print(f"HML Beta: {exposure.hml_beta:.3f}" if exposure.hml_beta else "HML Beta: N/A")
        print(f"Alpha (annualized): {exposure.alpha_annualized:.2%}")
        print(f"R-squared: {exposure.r_squared:.3f}")
        print(f"Observations: {exposure.observations}")

        # Example 2: Factor attribution
        print("\n=== Factor Attribution ===")
        attribution = await analyzer.factor_attribution(
            returns=returns,
            model=FactorModel.THREE_FACTOR,
        )

        print(f"Total Return: {attribution.total_return:.2%}")
        print(f"Factor Return: {attribution.factor_return:.2%}")
        print(f"Alpha Return: {attribution.alpha_return:.2%}")
        print(f"Market Contribution: {attribution.market_contribution:.2%}")

        # Example 3: Get factor data
        print("\n=== Factor Data ===")
        loader = FamaFrenchDataLoader()
        factors = await loader.get_factors(
            model=FactorModel.THREE_FACTOR,
            frequency=Frequency.DAILY,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        if not factors.empty:
            print(f"Factor data shape: {factors.shape}")
            print(f"Columns: {factors.columns.tolist()}")
            print(f"Date range: {factors.index.min()} to {factors.index.max()}")

    asyncio.run(main())
