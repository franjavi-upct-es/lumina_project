# tests/test_data_collection.py
"""
Unit tests for data collection components
Run with: pytest tests/test_data_collection.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from data_engine.collectors.yfinance_collector import YFinanceCollector
from data_engine.transformers.feature_engineering import FeatureEngineer


class TestYFinanceCollector:
    """Tests for YFinance collector"""

    @pytest.fixture
    def collector(self):
        """Create collector instance"""
        return YFinanceCollector(rate_limit=100)

    @pytest.fixture
    def date_range(self):
        """Get test date range"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        return start_date, end_date

    @pytest.mark.asyncio
    async def test_collector_initialization(self, collector):
        """Test collector initializes correctly"""
        assert collector.name == "YFinance"
        assert collector.rate_limit == 100
        assert len(collector._request_timestamps) == 0

    @pytest.mark.asyncio
    async def test_collect_single_ticker(self, collector, date_range):
        """Test collecting data for single ticker"""
        start_date, end_date = date_range

        data = await collector.collect_with_retry(
            ticker="AAPL", start_date=start_date, end_date=end_date
        )

        assert data is not None
        assert data.height > 0
        assert "close" in data.columns
        assert "volume" in data.columns
        assert "ticker" in data.columns

    @pytest.mark.asyncio
    async def test_collect_invalid_ticker(self, collector, date_range):
        """Test collecting invalid ticker returns None"""
        start_date, end_date = date_range

        data = await collector.collect_with_retry(
            ticker="INVALID_TICKER_12345", start_date=start_date, end_date=end_date
        )

        assert data is None or data.height == 0

    @pytest.mark.asyncio
    async def test_data_validation(self, collector, date_range):
        """Test data validation"""
        start_date, end_date = date_range

        data = await collector.collect_with_retry(
            ticker="AAPL", start_date=start_date, end_date=end_date
        )

        # Validate
        is_valid = await collector.validate_data(data)

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_batch_collection(self, collector, date_range):
        """Test batch collection of multiple tickers"""
        start_date, end_date = date_range
        tickers = ["AAPL", "GOOGL", "MSFT"]

        results = await collector.collect_batch(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            max_concurrent=3,
        )

        assert len(results) > 0
        assert len(results) <= len(tickers)

        for ticker, data in results.items():
            assert data.height > 0
            assert "close" in data.columns

    @pytest.mark.asyncio
    async def test_rate_limiting(self, collector):
        """Test rate limiting mechanism"""
        # Make multiple rapid requests
        initial_count = len(collector._request_timestamps)

        for _ in range(5):
            await collector._check_rate_limit()

        # Should have recorded requests
        assert len(collector._request_timestamps) > initial_count

    @pytest.mark.asyncio
    async def test_company_info(self, collector):
        """Test fetching company information"""
        info = await collector.get_company_info("AAPL")

        assert info is not None
        assert "ticker" in info
        assert "name" in info
        assert info["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_health_check(self, collector):
        """Test health check"""
        health = await collector.health_check()

        assert "collector" in health
        assert "status" in health
        assert "timestamp" in health
        assert health["collector"] == "YFinance"


class TestFeatureEngineer:
    """Tests for Feature Engineering"""

    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance"""
        return FeatureEngineer()

    @pytest.fixture
    async def sample_data(self):
        """Get sample data for testing"""
        collector = YFinanceCollector()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        data = await collector.collect_with_retry(
            ticker="AAPL", start_date=start_date, end_date=end_date
        )

        return data

    def test_engineer_initialization(self, engineer):
        """Test feature engineer initializes"""
        assert engineer is not None
        assert hasattr(engineer, "feature_categories")

    @pytest.mark.asyncio
    async def test_create_all_features(self, engineer, sample_data):
        """Test creating all features"""
        enriched = engineer.create_all_features(
            sample_data, add_lags=True, add_rolling=True
        )

        assert enriched is not None
        assert enriched.height == sample_data.height
        assert len(enriched.columns) > len(sample_data.columns)

    @pytest.mark.asyncio
    async def test_price_features(self, engineer, sample_data):
        """Test price feature creation"""
        data_pd = sample_data.to_pandas()
        enriched_pd = engineer._add_price_features(data_pd)

        # Check that price features were added
        assert "returns" in enriched_pd.columns
        assert "log_returns" in enriched_pd.columns
        assert "high_low_range" in enriched_pd.columns
        assert "gap" in enriched_pd.columns

    @pytest.mark.asyncio
    async def test_volume_features(self, engineer, sample_data):
        """Test volume feature creation"""
        data_pd = sample_data.to_pandas()
        enriched_pd = engineer._add_volume_features(data_pd)

        assert "volume_change" in enriched_pd.columns
        assert "vwap" in enriched_pd.columns
        assert "obv" in enriched_pd.columns

    @pytest.mark.asyncio
    async def test_momentum_features(self, engineer, sample_data):
        """Test momentum feature creation"""
        data_pd = sample_data.to_pandas()
        enriched_pd = engineer._add_momentum_features(data_pd)

        assert "rsi_14" in enriched_pd.columns
        assert "stoch_k" in enriched_pd.columns
        assert "williams_r_14" in enriched_pd.columns

    @pytest.mark.asyncio
    async def test_trend_features(self, engineer, sample_data):
        """Test trend feature creation"""
        data_pd = sample_data.to_pandas()
        enriched_pd = engineer._add_tren_features(data_pd)

        assert "sma_20" in enriched_pd.columns
        assert "ema_12" in enriched_pd.columns
        assert "macd" in enriched_pd.columns
        assert "adx_14" in enriched_pd.columns

    @pytest.mark.asyncio
    async def test_volatility_features(self, engineer, sample_data):
        """Test volatility feature creation"""
        data_pd = sample_data.to_pandas()
        enriched_pd = engineer._add_volatility_features(data_pd)

        assert "volatility_20d" in enriched_pd.columns
        assert "atr_14" in enriched_pd.columns
        assert "bb_width_20" in enriched_pd.columns

    @pytest.mark.asyncio
    async def test_get_feature_names_by_category(self, engineer, sample_data):
        """Test getting features by category"""
        enriched = engineer.create_all_features(sample_data)

        # Get features by category
        price_features = engineer.get_feature_names_by_category("price")
        momentum_features = engineer.get_feature_names_by_category("momentum")

        assert len(price_features) > 0
        assert len(momentum_features) > 0
        assert isinstance(price_features, list)

    @pytest.mark.asyncio
    async def test_get_all_feature_names(self, engineer, sample_data):
        """Test getting all feature names"""
        enriched = engineer.create_all_features(sample_data)

        all_features = engineer.get_all_feature_names()

        assert len(all_features) > 0
        assert isinstance(all_features, list)

    @pytest.mark.asyncio
    async def test_no_inf_values(self, engineer, sample_data):
        """Test that features don't contain inf values"""
        enriched = engineer.create_all_features(sample_data)
        enriched_pd = enriched.to_pandas()

        # Check for inf values
        numeric_cols = enriched_pd.select_dtypes(include=["float64", "float32"]).columns

        for col in numeric_cols:
            assert not enriched_pd[col].isin([float("inf"), float("-inf")]).any(), (
                f"Column {col} contains inf values"
            )

    @pytest.mark.asyncio
    async def test_feature_consistency(self, engineer, sample_data):
        """Test that features are consistent across runs"""
        enriched1 = engineer.create_all_features(sample_data)
        enriched2 = engineer.create_all_features(sample_data)

        # Should have same columns
        assert set(enriched1.columns) == set(enriched2.columns)

        # Should have same shape
        assert enriched1.shape == enriched2.shape


# Test fixtures and utilities


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def test_import_modules():
    """Test that all modules can be imported"""
    try:
        from data_engine.collectors.yfinance_collector import YFinanceCollector
        from data_engine.transformers.feature_engineering import FeatureEngineer
        from db.models import init_db, bulk_insert_price_data

        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])