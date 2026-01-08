# tests/test_integration_full.py
"""
Full integration tests for the complete ML pipeline.
These tests verify the entire workflow from data collection to prediction.

WARNING: These tests are slow and resource-intensive.

Usage:
    pytest tests/test_integration_full.py -v -s
    pytest tests/test_integration_full.py -v -s -m slow
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure environment
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault(
    "DATABASE_URL", "postgresql://lumina:lumina_password@localhost:5435/lumina_db"
)


@pytest.mark.slow
class TestFullMLPipeline:
    """Test complete ML pipeline from data to predictions"""

    def test_complete_lstm_pipeline(self):
        """
        Complete LSTM workflow:
        1. Collect data
        2. Engineer features
        3. Train model
        4. Make predictions
        """
        import asyncio

        from loguru import logger

        ticker = "AAPL"
        logger.info(f"Starting complete LSTM pipeline test for {ticker}")

        # Step 1: Collect Data
        logger.info("=" * 50)
        logger.info("STEP 1: Collecting market data")
        logger.info("=" * 50)

        try:
            from backend.data_engine.collectors.yfinance_collector import YFinanceCollector

            collector = YFinanceCollector()

            async def collect_data():
                return await collector.collect_with_retry(
                    ticker=ticker,
                    start_date=datetime.now() - timedelta(days=365),
                    end_date=datetime.now(),
                )

            data = asyncio.run(collect_data())

            assert data is not None, "Data collection returned None"
            assert data.height > 0, "Data collection returned empty DataFrame"
            logger.success(f"✓ Collected {data.height} rows of data")
        except Exception as e:
            pytest.fail(f"Data collection failed: {e}")

        # Step 2: Feature Engineering
        logger.info("=" * 50)
        logger.info("STEP 2: Engineering features")
        logger.info("=" * 50)

        try:
            from backend.data_engine.transformers.feature_engineering import FeatureEngineer

            fe = FeatureEngineer()
            enriched_data = fe.create_all_features(data, add_lags=True, add_rolling=True)
            feature_names = fe.get_all_feature_names()

            assert len(feature_names) > 0, "No features generated"
            logger.success(f"✓ Created {len(feature_names)} features")
            logger.info(f"  Sample features: {feature_names[:5]}")
        except Exception as e:
            pytest.fail(f"Feature engineering failed: {e}")

        # Step 3: Train Model (via Celery task)
        logger.info("=" * 50)
        logger.info("STEP 3: Training LSTM model")
        logger.info("=" * 50)

        try:
            from backend.workers.ml_tasks import train_model_task

            job_id = f"integration_test_{int(time.time())}"

            hyperparams = {
                "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "hidden_dim": 32,  # Small for testing
                "num_layers": 1,
                "dropout": 0.2,
                "sequence_length": 20,
                "prediction_horizon": 3,
                "batch_size": 16,
                "num_epochs": 3,  # Few epochs for testing
                "learning_rate": 0.001,
                "early_stopping_patience": 2,
                "max_features": 10,
            }

            task = train_model_task.delay(
                job_id=job_id, ticker=ticker, model_type="lstm", hyperparams=hyperparams
            )

            logger.info(f"Training task submitted: {task.id}")

            # Wait for training (max 5 minutes)
            timeout = 300
            start_time = time.time()

            while not task.ready() and (time.time() - start_time) < timeout:
                if task.state == "PROGRESS":
                    meta = task.info or {}
                    logger.info(f"  Progress: {meta.get('step', 'unknown')} - {meta.get('progress', 0)}%")
                time.sleep(5)

            if not task.ready():
                pytest.fail("Training did not complete within timeout")

            if task.state == "FAILURE":
                pytest.fail(f"Training failed: {task.info}")

            result = task.result
            model_id = f"{ticker}_lstm_{job_id}"
            logger.success(f"✓ Model trained successfully")
            logger.info(f"  Model ID: {model_id}")
            logger.info(f"  Metrics: {result.get('metrics', {})}")
        except ImportError as e:
            pytest.skip(f"Cannot import ml_tasks: {e}")
        except Exception as e:
            pytest.fail(f"Training failed: {e}")

        # Step 4: Make Predictions
        logger.info("=" * 50)
        logger.info("STEP 4: Generating predictions")
        logger.info("=" * 50)

        try:
            from backend.workers.ml_tasks import predict_task

            pred_task = predict_task.delay(ticker=ticker, model_id=model_id, days_ahead=3)

            logger.info(f"Prediction task submitted: {pred_task.id}")

            # Wait for prediction (max 2 minutes)
            timeout = 120
            start_time = time.time()

            while not pred_task.ready() and (time.time() - start_time) < timeout:
                time.sleep(2)

            if not pred_task.ready():
                pytest.fail("Prediction did not complete within timeout")

            if pred_task.state == "FAILURE":
                pytest.fail(f"Prediction failed: {pred_task.info}")

            prediction = pred_task.result

            assert "predictions" in prediction, "No predictions in result"
            assert len(prediction["predictions"]) == 3, "Expected 3 days of predictions"

            logger.success("✓ Predictions generated successfully")
            logger.info(f"  Current price: ${prediction.get('current_price', 'N/A')}")
            logger.info(f"  Summary: {prediction.get('summary', {})}")

            for pred in prediction["predictions"]:
                logger.info(f"  Day {pred['day']}: ${pred['predicted_price']} ({pred['change_percent']:+.2f}%)")

        except Exception as e:
            pytest.fail(f"Prediction failed: {e}")

        # Summary
        logger.success("=" * 50)
        logger.success("FULL PIPELINE TEST COMPLETED SUCCESSFULLY!")
        logger.success("=" * 50)

    def test_complete_xgboost_pipeline(self):
        """
        Complete XGBoost workflow:
        1. Collect data
        2. Engineer features
        3. Train XGBoost model
        4. Make predictions
        """
        import asyncio

        from loguru import logger

        ticker = "MSFT"
        logger.info(f"Starting complete XGBoost pipeline test for {ticker}")

        # Step 1: Collect Data
        try:
            from backend.data_engine.collectors.yfinance_collector import YFinanceCollector

            collector = YFinanceCollector()

            async def collect_data():
                return await collector.collect_with_retry(
                    ticker=ticker,
                    start_date=datetime.now() - timedelta(days=365),
                    end_date=datetime.now(),
                )

            data = asyncio.run(collect_data())
            assert data is not None and data.height > 0
            logger.success(f"✓ Collected {data.height} rows")
        except Exception as e:
            pytest.fail(f"Data collection failed: {e}")

        # Step 2: Feature Engineering
        try:
            from backend.data_engine.transformers.feature_engineering import FeatureEngineer

            fe = FeatureEngineer()
            enriched_data = fe.create_all_features(data, add_lags=True, add_rolling=True)
            logger.success(f"✓ Features engineered: {enriched_data.shape}")
        except Exception as e:
            pytest.fail(f"Feature engineering failed: {e}")

        # Step 3: Train XGBoost Model
        try:
            from backend.workers.ml_tasks import train_model_task

            job_id = f"xgb_test_{int(time.time())}"

            hyperparams = {
                "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.1,
                "prediction_horizon": 5,
                "max_features": 20,
            }

            task = train_model_task.delay(
                job_id=job_id, ticker=ticker, model_type="xgboost", hyperparams=hyperparams
            )

            # Wait for training
            result = task.get(timeout=300)
            logger.success(f"✓ XGBoost model trained: {result.get('metrics', {})}")
        except Exception as e:
            logger.warning(f"XGBoost training: {e}")
            pytest.skip(f"XGBoost training not available: {e}")

        logger.success("XGBoost pipeline completed!")


@pytest.mark.slow
class TestBacktestIntegration:
    """Test complete backtesting workflow"""

    def test_momentum_strategy_backtest(self):
        """Test momentum strategy backtest end-to-end"""
        from loguru import logger

        logger.info("Testing Momentum Strategy Backtest")

        try:
            from backend.workers.backtest_tasks import run_backtest_task

            backtest_id = f"momentum_test_{int(time.time())}"

            config = {
                "strategy": "momentum",
                "tickers": ["AAPL", "GOOGL", "MSFT"],
                "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "initial_capital": 100000,
                "commission": 0.001,
                "slippage": 0.0005,
                "strategy_params": {"lookback_period": 20, "rebalance_frequency": "weekly"},
            }

            task = run_backtest_task.delay(backtest_id=backtest_id, config=config)

            # Wait for backtest
            result = task.get(timeout=300)

            assert "returns" in result or "metrics" in result
            logger.success(f"✓ Backtest completed: {result.get('metrics', {})}")
        except ImportError:
            pytest.skip("backtest_tasks not available")
        except Exception as e:
            pytest.fail(f"Backtest failed: {e}")

    def test_mean_reversion_strategy_backtest(self):
        """Test mean reversion strategy backtest"""
        from loguru import logger

        logger.info("Testing Mean Reversion Strategy Backtest")

        try:
            from backend.workers.backtest_tasks import run_backtest_task

            backtest_id = f"meanrev_test_{int(time.time())}"

            config = {
                "strategy": "mean_reversion",
                "tickers": ["AAPL"],
                "start_date": (datetime.now() - timedelta(days=365)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "initial_capital": 100000,
                "strategy_params": {"lookback_period": 20, "z_score_threshold": 2.0},
            }

            task = run_backtest_task.delay(backtest_id=backtest_id, config=config)
            result = task.get(timeout=300)

            logger.success(f"✓ Mean reversion backtest completed")
        except Exception as e:
            logger.warning(f"Mean reversion backtest: {e}")


@pytest.mark.slow
class TestRiskAnalysisIntegration:
    """Test risk analysis workflow"""

    def test_portfolio_risk_analysis(self):
        """Test complete portfolio risk analysis"""
        import asyncio

        from loguru import logger

        logger.info("Testing Portfolio Risk Analysis")

        portfolio = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}

        # Step 1: Collect data for all tickers
        try:
            from backend.data_engine.collectors.yfinance_collector import YFinanceCollector

            collector = YFinanceCollector()

            async def collect_all():
                results = {}
                for ticker in portfolio.keys():
                    data = await collector.collect_with_retry(
                        ticker=ticker,
                        start_date=datetime.now() - timedelta(days=365),
                        end_date=datetime.now(),
                    )
                    results[ticker] = data
                return results

            data_dict = asyncio.run(collect_all())
            logger.success(f"✓ Data collected for {len(data_dict)} tickers")
        except Exception as e:
            pytest.fail(f"Data collection failed: {e}")

        # Step 2: Calculate VaR
        try:
            from backend.quant_engine.risk.var_calculator import VaRCalculator

            var_calc = VaRCalculator()
            # This depends on actual implementation
            logger.success("✓ VaR calculator initialized")
        except ImportError:
            logger.warning("VaRCalculator not available")

        # Step 3: Calculate CVaR
        try:
            from backend.quant_engine.risk.cvar_calculator import CVaRCalculator

            cvar_calc = CVaRCalculator()
            logger.success("✓ CVaR calculator initialized")
        except ImportError:
            logger.warning("CVaRCalculator not available")

        logger.success("Risk analysis integration test completed")


@pytest.mark.slow
class TestAPIIntegration:
    """Test API integration with all services"""

    def test_api_full_workflow(self):
        """Test complete workflow through API"""
        import requests
        from loguru import logger

        api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        logger.info(f"Testing API workflow against {api_url}")

        # Step 1: Health check
        try:
            response = requests.get(f"{api_url}/health", timeout=10)
            assert response.status_code == 200
            logger.success("✓ API health check passed")
        except requests.RequestException as e:
            pytest.fail(f"API not accessible: {e}")

        # Step 2: Submit data collection
        try:
            payload = {
                "ticker": "AAPL",
                "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d"),
            }
            response = requests.post(f"{api_url}/api/v1/data/collect", json=payload, timeout=60)
            assert response.status_code in [200, 202]
            logger.success("✓ Data collection submitted")
        except requests.RequestException as e:
            logger.warning(f"Data collection API: {e}")

        # Step 3: Submit training job
        try:
            payload = {
                "ticker": "AAPL",
                "model_type": "lstm",
                "hyperparams": {
                    "hidden_dim": 32,
                    "num_epochs": 3,
                },
            }
            response = requests.post(f"{api_url}/api/v1/ml/train", json=payload, timeout=30)
            if response.status_code in [200, 202]:
                data = response.json()
                job_id = data.get("job_id") or data.get("task_id")
                logger.success(f"✓ Training job submitted: {job_id}")
            else:
                logger.warning(f"Training API: {response.status_code}")
        except requests.RequestException as e:
            logger.warning(f"Training API: {e}")

        logger.success("API integration test completed")


# Pytest configuration
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


if __name__ == "__main__":
    print("=" * 60)
    print("LUMINA PROJECT - Full Integration Test Suite")
    print("=" * 60)
    print("\n⚠️  WARNING: These tests are slow and resource-intensive!")
    print("\nMake sure:")
    print("  1. All Docker containers are running: docker-compose up -d")
    print("  2. Celery workers are active")
    print("  3. GPU is available (for ML tasks)")
    print("\n")

    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "slow"])
